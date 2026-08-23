"""Build a fail-closed evidence bundle for an official production archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

CONCEPTS = {
    "OBJECT_OF_SALE",
    "COMPOSITION",
    "COMPLETENESS",
    "FUEL_OR_IGNITION",
    "NEGATION",
}
CATEGORIES = {"БАД", "Легковоспламеняющиеся"}
GENERIC_PATTERNS = (
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"признак(?:и|ов)? нарушени",
        r"товар (?:следует|нужно|необходимо) (?:заблокировать|запретить)",
        r"нарушение (?:обнаружено|найдено|выявлено)",
        r"violation (?:found|detected)",
    )
)
GENERIC_PATTERNS = tuple(GENERIC_PATTERNS)
UNSUPPORTED_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:на|по) (?:фото|фотографии|изображении|картинке)",
        r"(?:фото|изображение) показывает",
        r"визуально",
        r"не (?:найден|обнаружен|указан|упомянут|виден)",
        r"отсутствие (?:признака|упоминания|информации)",
        r"not (?:found|shown|mentioned|visible)",
        r"image shows",
    )
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"invalid SHA-256 field: {field}")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"non-hex SHA-256 field: {field}") from error
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"JSONL row {line_number} is not an object")
        rows.append(row)
    return rows


def verify_accepted_refit(value: dict[str, Any]) -> None:
    if value.get("schema_version") != "full_data_refit_manifest_v1":
        raise ValueError("experiment 629 manifest schema mismatch")
    if value.get("experiment_id") != 629 or value.get("status") != "accepted":
        raise ValueError("experiment 629 is not accepted")
    if value.get("sealed_training_rows") != 0:
        raise ValueError("experiment 629 used sealed rows")
    _sha256(value.get("frozen_recipe_manifest_sha256"), "frozen_recipe_manifest_sha256")
    fits = value.get("component_fits")
    if not isinstance(fits, list) or not fits:
        raise ValueError("experiment 629 has no component fits")
    names: set[str] = set()
    for fit in fits:
        name = fit.get("component") if isinstance(fit, dict) else None
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("experiment 629 component provenance is incomplete or duplicated")
        names.add(name)
        if not isinstance(fit.get("source_experiment"), int) or fit["source_experiment"] <= 0:
            raise ValueError(f"component {name} has no source experiment")
        if fit.get("fit_count") != 1 or fit.get("integrity_passed") is not True:
            raise ValueError(f"component {name} is not an integrity-checked single fit")
        for field in ("artifact_sha256", "runtime_evidence_sha256", "provenance_sha256"):
            _sha256(fit.get(field), f"{name}.{field}")


def verify_runtime(value: dict[str, Any]) -> None:
    if value.get("schema_version") != "production_runtime_evidence_v1":
        raise ValueError("runtime evidence schema mismatch")
    for field in ("hardware_contract_sha256", "container_contract_sha256"):
        _sha256(value.get(field), field)
    for scope in ("public", "private"):
        measured = value.get(f"measured_{scope}_seconds")
        limit = value.get(f"{scope}_limit_seconds")
        margin = value.get(f"{scope}_margin_seconds")
        if not all(isinstance(item, (int, float)) and math.isfinite(item) for item in (measured, limit, margin)):
            raise ValueError(f"invalid {scope} runtime evidence")
        if measured < 0 or limit <= 0 or margin <= 0:
            raise ValueError(f"nonpositive {scope} runtime margin")
        if not math.isclose(limit - measured, margin, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"inconsistent {scope} runtime margin")


def verify_predictions(
    rows: list[dict[str, Any]], official: dict[str, Any]
) -> None:
    if official.get("schema_version") != "official_submission_identity_v1":
        raise ValueError("official identity manifest schema mismatch")
    official_rows = official.get("rows")
    if not isinstance(official_rows, list) or not official_rows:
        raise ValueError("official identity manifest is empty")
    expected: dict[str, str] = {}
    for row in official_rows:
        identifier = row.get("id") if isinstance(row, dict) else None
        category = row.get("category") if isinstance(row, dict) else None
        if not isinstance(identifier, str) or not identifier or identifier in expected:
            raise ValueError("official IDs must be nonempty and unique")
        if category not in CATEGORIES:
            raise ValueError("unexpected official category")
        expected[identifier] = category
    if set(expected.values()) != CATEGORIES:
        raise ValueError("official input must cover both categories")
    actual: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in actual:
            raise ValueError("prediction IDs must be nonempty and unique")
        actual[identifier] = row
    if set(actual) != set(expected) or len(rows) != len(expected):
        raise ValueError("prediction ID set or row count differs from official input")
    for identifier, row in actual.items():
        if row.get("category") != expected[identifier]:
            raise ValueError(f"category mismatch for ID {identifier}")
        verdict = row.get("verdict")
        score = row.get("score")
        if verdict not in (0, 1):
            raise ValueError(f"verdict outside allowed label domain for ID {identifier}")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
            raise ValueError(f"non-finite score for ID {identifier}")
        _verify_reasoning(row, identifier=identifier)


def _verify_reasoning(row: dict[str, Any], *, identifier: str) -> None:
    source = row.get("card")
    evidence = row.get("evidence")
    concept = row.get("concept")
    explanation = row.get("explanation")
    if not all(isinstance(item, str) for item in (source, evidence, concept, explanation)):
        raise ValueError(f"incomplete reasoning chain for ID {identifier}")
    if evidence == "NO_EVIDENCE":
        if concept != "NO_EVIDENCE" or explanation != "NO_EVIDENCE":
            raise ValueError(f"NO_EVIDENCE must fail closed for ID {identifier}")
        if row.get("char_start") is not None or row.get("char_end") is not None:
            raise ValueError(f"NO_EVIDENCE cannot have offsets for ID {identifier}")
        return
    if concept not in CONCEPTS:
        raise ValueError(f"concept outside closed vocabulary for ID {identifier}")
    start, end = row.get("char_start"), row.get("char_end")
    if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool):
        raise TypeError(f"invalid evidence offsets for ID {identifier}")
    if start < 0 or end <= start or end > len(source) or source[start:end] != evidence:
        raise ValueError(f"evidence is not the exact source substring for ID {identifier}")
    if evidence not in explanation:
        raise ValueError(f"explanation omits its exact quote for ID {identifier}")
    outside_quote = explanation.replace(evidence, " ")
    if any(pattern.search(outside_quote) for pattern in GENERIC_PATTERNS):
        raise ValueError(f"generic violation explanation for ID {identifier}")
    if any(pattern.search(outside_quote) for pattern in UNSUPPORTED_PATTERNS):
        raise ValueError(f"unsupported visual or absence claim for ID {identifier}")


def _safe_member(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe archive member: {name}")
    if name.startswith(".") or any(part.startswith(".") for part in path.parts):
        raise ValueError(f"service file is forbidden: {name}")


def build_manifest(
    *,
    accepted_refit_path: Path,
    official_path: Path,
    predictions_path: Path,
    runtime_path: Path,
    payloads: dict[str, Path],
) -> dict[str, Any]:
    refit = _read_json(accepted_refit_path)
    verify_accepted_refit(refit)
    official = _read_json(official_path)
    rows = _read_jsonl(predictions_path)
    verify_predictions(rows, official)
    runtime = _read_json(runtime_path)
    verify_runtime(runtime)
    members: dict[str, str] = {}
    for name, path in payloads.items():
        _safe_member(name)
        if name in {"production_manifest.json", "predictions.jsonl"}:
            raise ValueError(f"reserved archive member: {name}")
        if name in members or not path.is_file():
            raise ValueError(f"duplicate or missing payload: {name}")
        members[name] = sha256_file(path)
    return {
        "schema_version": "production_submission_manifest_v1",
        "experiment_id": 630,
        "status": "accepted",
        "accepted_629_manifest_sha256": sha256_file(accepted_refit_path),
        "frozen_recipe_manifest_sha256": refit["frozen_recipe_manifest_sha256"],
        "official_identity_manifest_sha256": sha256_file(official_path),
        "predictions_sha256": sha256_file(predictions_path),
        "runtime_evidence_sha256": sha256_file(runtime_path),
        "row_count": len(rows),
        "categories": sorted(CATEGORIES),
        "component_provenance": [
            {
                "component": fit["component"],
                "source_experiment": fit["source_experiment"],
                "artifact_sha256": fit["artifact_sha256"],
                "provenance_sha256": fit["provenance_sha256"],
            }
            for fit in refit["component_fits"]
        ],
        "payload_members": members,
        "allowed_members": sorted([*members, "predictions.jsonl", "production_manifest.json"]),
    }


def assemble_archive(
    *, manifest: dict[str, Any], predictions_path: Path, payloads: dict[str, Path], output: Path
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("production_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        archive.write(predictions_path, "predictions.jsonl")
        for name, path in sorted(payloads.items()):
            archive.write(path, name)


def verify_archive(
    archive_path: Path,
    *,
    accepted_refit_path: Path,
    official_path: Path,
    runtime_path: Path,
) -> dict[str, Any]:
    refit = _read_json(accepted_refit_path)
    verify_accepted_refit(refit)
    official = _read_json(official_path)
    verify_runtime(_read_json(runtime_path))
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP members")
        for name in names:
            _safe_member(name)
        manifest = json.loads(archive.read("production_manifest.json"))
        if manifest.get("schema_version") != "production_submission_manifest_v1":
            raise ValueError("production manifest schema mismatch")
        if manifest.get("experiment_id") != 630 or manifest.get("status") != "accepted":
            raise ValueError("production manifest is not accepted experiment 630")
        if set(names) != set(manifest.get("allowed_members", [])):
            raise ValueError("archive contains a non-allowlisted or missing member")
        if manifest.get("accepted_629_manifest_sha256") != sha256_file(accepted_refit_path):
            raise ValueError("archive is bound to a different experiment 629 result")
        if manifest.get("official_identity_manifest_sha256") != sha256_file(official_path):
            raise ValueError("archive is bound to a different official ID set")
        if manifest.get("runtime_evidence_sha256") != sha256_file(runtime_path):
            raise ValueError("archive is bound to different runtime evidence")
        prediction_bytes = archive.read("predictions.jsonl")
        if hashlib.sha256(prediction_bytes).hexdigest() != manifest.get("predictions_sha256"):
            raise ValueError("prediction checksum mismatch")
        rows = []
        for line in prediction_bytes.decode("utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        verify_predictions(rows, official)
        if manifest.get("row_count") != len(rows) or set(manifest.get("categories", [])) != CATEGORIES:
            raise ValueError("production manifest row count or categories mismatch")
        expected_provenance = sorted(
            (
                fit["component"],
                fit["source_experiment"],
                fit["artifact_sha256"],
                fit["provenance_sha256"],
            )
            for fit in refit["component_fits"]
        )
        recorded_provenance = manifest.get("component_provenance")
        if not isinstance(recorded_provenance, list):
            raise TypeError("component provenance must be a list")
        actual_provenance = sorted(
            (
                item.get("component"),
                item.get("source_experiment"),
                item.get("artifact_sha256"),
                item.get("provenance_sha256"),
            )
            for item in recorded_provenance
            if isinstance(item, dict)
        )
        if actual_provenance != expected_provenance or len(recorded_provenance) != len(expected_provenance):
            raise ValueError("component provenance differs from experiment 629")
        payload_members = manifest.get("payload_members")
        if not isinstance(payload_members, dict):
            raise TypeError("payload member manifest is missing")
        if set(payload_members) != set(names) - {"production_manifest.json", "predictions.jsonl"}:
            raise ValueError("payload member allowlist mismatch")
        for name, expected_hash in payload_members.items():
            _sha256(expected_hash, f"payload_members.{name}")
            if hashlib.sha256(archive.read(name)).hexdigest() != expected_hash:
                raise ValueError(f"payload checksum mismatch: {name}")
    return {
        "schema_version": "production_submission_receipt_v1",
        "decision": "GO",
        "archive_sha256": sha256_file(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "row_count": len(rows),
        "categories": sorted(CATEGORIES),
        "runtime_evidence_sha256": sha256_file(runtime_path),
    }


def _payloads(values: Iterable[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("payload must use ARCHIVE_NAME=LOCAL_PATH")
        name, raw_path = value.split("=", 1)
        if name in result:
            raise ValueError(f"duplicate payload name: {name}")
        result[name] = Path(raw_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--accepted-629", required=True, type=Path)
    build.add_argument("--official-identities", required=True, type=Path)
    build.add_argument("--predictions", required=True, type=Path)
    build.add_argument("--runtime-evidence", required=True, type=Path)
    build.add_argument("--payload", action="append", default=[])
    build.add_argument("--output", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--archive", required=True, type=Path)
    verify.add_argument("--accepted-629", required=True, type=Path)
    verify.add_argument("--official-identities", required=True, type=Path)
    verify.add_argument("--runtime-evidence", required=True, type=Path)
    verify.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        payloads = _payloads(args.payload)
        manifest = build_manifest(
            accepted_refit_path=args.accepted_629,
            official_path=args.official_identities,
            predictions_path=args.predictions,
            runtime_path=args.runtime_evidence,
            payloads=payloads,
        )
        assemble_archive(
            manifest=manifest,
            predictions_path=args.predictions,
            payloads=payloads,
            output=args.output,
        )
    else:
        receipt = verify_archive(
            args.archive,
            accepted_refit_path=args.accepted_629,
            official_path=args.official_identities,
            runtime_path=args.runtime_evidence,
        )
        if args.receipt:
            if args.receipt.exists():
                raise FileExistsError(f"refusing to overwrite receipt: {args.receipt}")
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
