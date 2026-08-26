"""Independently verify exp689 PREPARE outputs and emit an immutable acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

EXPERIMENT = Path(__file__).resolve().parent
SPEC_PATH = EXPERIMENT / "source_prepare_spec_v1.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
S3_REF = re.compile(r"^s3://[^/?#]+/[^?#]+$")
ALLOWED_BUNDLE_PATHS = {
    "experiments/689_qwen35_4b_grounded_transaction_graph_kd/prepare_source_universe.py",
    "experiments/689_qwen35_4b_grounded_transaction_graph_kd/source_prepare_spec_v1.json",
    "experiments/689_qwen35_4b_grounded_transaction_graph_kd/verify_source_prepare.py",
}
SOURCE_CONTRACT_FIELDS = {
    "schema_version", "execution_scope", "source_rows_sha256", "source_row_count",
    "runtime_sha256", "data_sha256", "registry_sha256", "builder_revision_sha256",
    "eligibility_universe_sha256", "stratum_derivation_sha256",
    "image_membership_sha256", "image_transform_sha256", "candidate_generator_sha256",
    "opaque_token_scheme", "label_fields_present", "score_fields_present",
    "first_image_rows_verified", "image_fetch_failures", "image_decode_failures",
    "image_hash_mismatches", "sealed_rows", "public_rows", "self_sha256",
}
SOURCE_ROW_FIELDS = {
    "schema_version", "record_id", "row_token", "component_token", "family_token",
    "stratum", "source_card_sha256", "sources", "first_image", "evidence_candidates",
}
IMAGE_MANIFEST_FIELDS = {
    "record_id", "row_token", "component_token", "fold", "stratum", "reference",
    "original_bytes_sha256", "resized_rgb_sha256", "transformed_jpeg_sha256",
    "transformed_rgb_sha256", "width", "height", "region_sha256",
}
FORBIDDEN_KEYS = {
    "gold", "label", "logit", "prediction", "prob", "probability", "public", "rank",
    "score", "sealed", "target", "verdict", "y_true",
}


class VerificationError(ValueError):
    """Raised when PREPARE evidence cannot support ACCEPT."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"{context}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"{context}: top level must be an object")
    return value


def _jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    raise VerificationError(f"{context}: blank line {line_number}")
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise VerificationError(f"{context}: row {line_number} is not an object")
                result.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"{context}: invalid JSONL: {error}") from error
    return result


def _self_hash(value: dict[str, Any], context: str, field: str = "self_sha256") -> str:
    actual = value.get(field)
    if not isinstance(actual, str) or not HEX64.fullmatch(actual):
        raise VerificationError(f"{context}: missing canonical {field}")
    copy = dict(value)
    copy[field] = None
    if sha256_bytes(canonical_json_bytes(copy)) != actual:
        raise VerificationError(f"{context}: canonical {field} mismatch")
    return actual


def _legacy_hash(value: dict[str, Any], context: str) -> str:
    actual = value.get("contract_sha256")
    if not isinstance(actual, str) or not HEX64.fullmatch(actual):
        raise VerificationError(f"{context}: missing contract_sha256")
    copy = dict(value)
    copy.pop("contract_sha256", None)
    if sha256_bytes(canonical_json_bytes(copy)) != actual:
        raise VerificationError(f"{context}: contract SHA mismatch")
    return actual


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise VerificationError(
            f"{context}: exact fields required; missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _hex(value: Any, context: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise VerificationError(f"{context}: expected lowercase SHA-256")
    return value


def _reject_forbidden(value: Any, context: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normal = str(key).lower().replace("-", "_")
            if normal in FORBIDDEN_KEYS or any(
                normal.startswith(f"{token}_") or normal.endswith(f"_{token}")
                for token in FORBIDDEN_KEYS
            ):
                raise VerificationError(f"{context}: forbidden field {key!r}")
            _reject_forbidden(item, context)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden(item, context)


def _rgb_sha(image: Image.Image) -> str:
    image = image.convert("RGB")
    return sha256_bytes(image.tobytes())


def _regions(image: Image.Image) -> dict[str, str]:
    image = image.convert("RGB")
    mx, my = image.width // 2, image.height // 2
    boxes = {
        "full": (0, 0, image.width, image.height),
        "q00": (0, 0, max(1, mx), max(1, my)),
        "q01": (mx, 0, mx + max(1, mx), max(1, my)),
        "q10": (0, my, max(1, mx), my + max(1, my)),
        "q11": (mx, my, mx + max(1, mx), my + max(1, my)),
    }
    return {
        name: _rgb_sha(image if name == "full" else image.crop(box))
        for name, box in boxes.items()
    }


def _validate_candidate(candidate: dict[str, Any], index: int, sources: list[dict[str, Any]], regions: dict[str, str], context: str) -> None:
    fields = {
        "candidate_id", "candidate_index", "evidence_kind", "source_index",
        "char_start", "char_end", "image_region", "evidence_sha256",
    }
    _exact_keys(candidate, fields, context)
    if candidate["candidate_index"] != index:
        raise VerificationError(f"{context}: nonconsecutive candidate index")
    copy = dict(candidate)
    copy["candidate_id"] = None
    if candidate["candidate_id"] != sha256_bytes(canonical_json_bytes(copy)):
        raise VerificationError(f"{context}: candidate ID mismatch")
    _hex(candidate["evidence_sha256"], f"{context}.evidence_sha256")
    if candidate["evidence_kind"] == "text_span":
        source_index = candidate["source_index"]
        start, end = candidate["char_start"], candidate["char_end"]
        if not isinstance(source_index, int) or not 0 <= source_index < len(sources):
            raise VerificationError(f"{context}: source index mismatch")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
            raise VerificationError(f"{context}: text offsets mismatch")
        text = sources[source_index]["text"]
        if end > len(text) or sha256_text(text[start:end]) != candidate["evidence_sha256"]:
            raise VerificationError(f"{context}: exact text-span SHA mismatch")
        if candidate["image_region"] is not None:
            raise VerificationError(f"{context}: text span binds image")
    elif candidate["evidence_kind"] == "image_region":
        region = candidate["image_region"]
        if region not in regions or candidate["evidence_sha256"] != regions[region]:
            raise VerificationError(f"{context}: decoded image-region SHA mismatch")
        if any(candidate[field] is not None for field in ("source_index", "char_start", "char_end")):
            raise VerificationError(f"{context}: image region binds text")
    else:
        raise VerificationError(f"{context}: unknown evidence kind")


def _validate_bundle(
    bundle_root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    builder_revision: str,
) -> tuple[str, str]:
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise VerificationError("bundle manifest: expected SHA mismatch")
    manifest = _json(manifest_path, "bundle manifest")
    _exact_keys(
        manifest,
        {"schema_version", "builder_revision", "files", "self_sha256"},
        "bundle manifest",
    )
    _self_hash(manifest, "bundle manifest")
    if manifest["schema_version"] != "exp689_source_prepare_bundle_manifest_v1":
        raise VerificationError("bundle manifest: schema mismatch")
    if manifest["builder_revision"] != builder_revision:
        raise VerificationError("bundle manifest: builder revision mismatch")
    files = manifest["files"]
    if not isinstance(files, list) or [item.get("path") for item in files] != sorted(ALLOWED_BUNDLE_PATHS):
        raise VerificationError("bundle manifest: exact source-prep whitelist required")
    for index, item in enumerate(files):
        _exact_keys(item, {"path", "sha256", "size_bytes"}, f"bundle file {index}")
        path = bundle_root / item["path"]
        if not path.is_file() or path.is_symlink():
            raise VerificationError(f"bundle file {index}: missing regular file")
        if path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise VerificationError(f"bundle file {index}: content mismatch")
    builder_path = bundle_root / next(path for path in ALLOWED_BUNDLE_PATHS if path.endswith("prepare_source_universe.py"))
    return expected_manifest_sha256, sha256_file(builder_path)


def _runtime_bindings(runtime_dirs: list[Path], spec: dict[str, Any]) -> list[dict[str, Any]]:
    if len(runtime_dirs) != 5:
        raise VerificationError("exactly five runtime directories are required")
    bindings: list[dict[str, Any]] = []
    for fold, runtime_dir in enumerate(runtime_dirs):
        expected = spec["folds"][str(fold)]
        validation = runtime_dir / "validation.jsonl"
        audit_path = runtime_dir / "runtime_audit.json"
        if sha256_file(validation) != expected["validation_sha256"]:
            raise VerificationError(f"fold {fold}: validation SHA mismatch")
        audit = _json(audit_path, f"fold {fold} runtime audit")
        contract = _legacy_hash(audit, f"fold {fold} runtime audit")
        source_contract = audit.get("source_runtime_contract_sha256", contract)
        if source_contract != expected["source_runtime_contract_sha256"]:
            raise VerificationError(f"fold {fold}: source runtime contract mismatch")
        if audit.get("experiment_id") != "641" or audit.get("outer_fold") != fold:
            raise VerificationError(f"fold {fold}: runtime identity mismatch")
        if audit.get("output_sha256", {}).get("validation.jsonl") != expected["validation_sha256"]:
            raise VerificationError(f"fold {fold}: runtime output binding mismatch")
        bindings.append(
            {
                "fold": fold,
                "runtime_audit_sha256": sha256_file(audit_path),
                "runtime_contract_sha256": contract,
                "source_runtime_contract_sha256": source_contract,
                "validation_sha256": expected["validation_sha256"],
                "validation_rows": expected["validation_rows"],
            }
        )
    return bindings


def _inventory(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise VerificationError("prepare output must be a regular directory")
    paths = sorted(path for path in root.rglob("*") if path.is_file() or path.is_symlink())
    if any(path.is_symlink() for path in paths):
        raise VerificationError("prepare output symlinks are forbidden")
    expected = {"source_rows.jsonl", "source_contract.json", "images/image_manifest.jsonl"} | {
        f"images/G689-{index:03d}.jpg" for index in range(1, 301)
    }
    actual = {path.relative_to(root).as_posix() for path in paths}
    if actual != expected:
        raise VerificationError(
            f"prepare output inventory mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def verify(
    *,
    prepare_dir: Path,
    runtime_dirs: list[Path],
    runtime_archives: list[Path],
    runtime_archive_refs: list[str],
    runtime_archive_sha256: list[str],
    bundle_root: Path,
    bundle_manifest_path: Path,
    bundle_manifest_sha256: str,
    builder_revision: str,
    spec_path: Path = SPEC_PATH,
    terminal_metadata_path: Path,
    terminal_metadata_sha256: str,
    approved_s3_output_ref: str,
    acceptance_path: Path,
) -> dict[str, Any]:
    if acceptance_path.exists():
        raise FileExistsError("refusing to overwrite source-prepare acceptance")
    if not HEX40.fullmatch(builder_revision):
        raise VerificationError("builder revision must be an exact lowercase Git commit")
    if not S3_REF.fullmatch(approved_s3_output_ref):
        raise VerificationError("approved output must be a non-presigned s3:// object reference")
    bundled_spec = bundle_root / next(
        path for path in ALLOWED_BUNDLE_PATHS if path.endswith("source_prepare_spec_v1.json")
    )
    if spec_path.resolve(strict=True) != bundled_spec.resolve(strict=True):
        raise VerificationError("source prepare spec must be the bundle-whitelisted file")
    spec = _json(spec_path, "source prepare spec")
    _self_hash(spec, "source prepare spec")
    if spec.get("ocr_policy") != "disabled_unverified":
        raise VerificationError("source prepare spec: OCR must remain disabled-unverified")
    bundle_sha, builder_sha = _validate_bundle(
        bundle_root, bundle_manifest_path, bundle_manifest_sha256, builder_revision
    )
    runtime_bindings = _runtime_bindings(runtime_dirs, spec)
    if not (len(runtime_archives) == len(runtime_archive_refs) == len(runtime_archive_sha256) == 5):
        raise VerificationError("exactly five archive paths/refs/SHAs are required")
    archive_bindings: list[dict[str, Any]] = []
    for fold, (path, reference, expected_sha) in enumerate(
        zip(runtime_archives, runtime_archive_refs, runtime_archive_sha256, strict=True)
    ):
        _hex(expected_sha, f"fold {fold} archive SHA")
        if not S3_REF.fullmatch(reference):
            raise VerificationError(f"fold {fold}: archive ref must be s3:// without query")
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha:
            raise VerificationError(f"fold {fold}: exact runtime archive SHA mismatch")
        archive_bindings.append(
            {"fold": fold, "reference": reference, "sha256": expected_sha, "size_bytes": path.stat().st_size}
        )

    inventory = _inventory(prepare_dir)
    source_rows_path = prepare_dir / "source_rows.jsonl"
    contract_path = prepare_dir / "source_contract.json"
    image_manifest_path = prepare_dir / "images" / "image_manifest.jsonl"
    contract = _json(contract_path, "source contract")
    _exact_keys(contract, SOURCE_CONTRACT_FIELDS, "source contract")
    _self_hash(contract, "source contract")
    if contract["schema_version"] != "exp689_source_contract_v2" or contract["execution_scope"] != "remote_remote_compute":
        raise VerificationError("source contract: identity mismatch")
    if contract["source_rows_sha256"] != sha256_file(source_rows_path) or contract["source_row_count"] != 300:
        raise VerificationError("source contract: source rows binding mismatch")
    if contract["runtime_sha256"] != sha256_bytes(canonical_json_bytes(runtime_bindings)):
        raise VerificationError("source contract: independently recomputed runtime binding mismatch")
    if contract["data_sha256"] != spec["input_data_sha256"] or contract["registry_sha256"] != spec["input_registry_sha256"]:
        raise VerificationError("source contract: data/registry mismatch")
    if contract["builder_revision_sha256"] != sha256_text(builder_revision):
        raise VerificationError("source contract: builder revision mismatch")
    if any(contract[field] is not False for field in ("label_fields_present", "score_fields_present")):
        raise VerificationError("source contract: supervision flags must be false")
    if any(
        contract[field] != 0
        for field in ("image_fetch_failures", "image_decode_failures", "image_hash_mismatches", "sealed_rows", "public_rows")
    ):
        raise VerificationError("source contract: failure/sealed/Public counters must be zero")
    if contract["first_image_rows_verified"] != 300:
        raise VerificationError("source contract: first-image verification count mismatch")

    rows = _jsonl(source_rows_path, "source rows")
    manifest = _jsonl(image_manifest_path, "image manifest")
    if len(rows) != 300 or len(manifest) != 300:
        raise VerificationError("source output: exact 300-row cardinality required")
    manifest_by_record: dict[str, dict[str, Any]] = {}
    stratum_counts: Counter[str] = Counter()
    fold_stratum_counts: Counter[tuple[int, str]] = Counter()
    row_tokens: set[str] = set()
    component_tokens: set[str] = set()
    source_cards: list[str] = []
    candidate_ids: list[str] = []
    for index, (row, image_item) in enumerate(zip(rows, manifest, strict=True), 1):
        context = f"source row {index}"
        _reject_forbidden(row, context)
        _exact_keys(row, SOURCE_ROW_FIELDS, context)
        _exact_keys(image_item, IMAGE_MANIFEST_FIELDS, f"image manifest {index}")
        record_id = f"G689-{index:03d}"
        if row["record_id"] != record_id or image_item["record_id"] != record_id:
            raise VerificationError(f"{context}: opaque record ordering mismatch")
        for field in ("row_token", "component_token", "family_token", "source_card_sha256"):
            _hex(row[field], f"{context}.{field}")
        if row["row_token"] in row_tokens or row["component_token"] in component_tokens:
            raise VerificationError(f"{context}: duplicate row/component token")
        row_tokens.add(row["row_token"])
        component_tokens.add(row["component_token"])
        if image_item["row_token"] != row["row_token"] or image_item["component_token"] != row["component_token"]:
            raise VerificationError(f"{context}: image manifest membership mismatch")
        if image_item["stratum"] != row["stratum"] or image_item["fold"] not in range(5):
            raise VerificationError(f"{context}: fold/stratum manifest mismatch")
        stratum_counts[row["stratum"]] += 1
        fold_stratum_counts[(image_item["fold"], row["stratum"])] += 1
        sources = row["sources"]
        if not isinstance(sources, list) or not sources:
            raise VerificationError(f"{context}: sources missing")
        for source_index, item in enumerate(sources):
            _exact_keys(item, {"source_index", "source_kind", "text", "text_sha256"}, f"{context}.source")
            if item["source_index"] != source_index or item["source_kind"] not in {"name", "description", "ocr"}:
                raise VerificationError(f"{context}: source identity mismatch")
            if not isinstance(item["text"], str) or not item["text"].strip() or sha256_text(item["text"]) != item["text_sha256"]:
                raise VerificationError(f"{context}: source text SHA mismatch")
        first_image = row["first_image"]
        _exact_keys(first_image, {"reference", "content_sha256", "decoded_rgb_sha256", "media_type", "width", "height"}, f"{context}.first_image")
        expected_reference = f"images/{record_id}.jpg"
        if first_image["reference"] != expected_reference or image_item["reference"] != expected_reference:
            raise VerificationError(f"{context}: first-image reference mismatch")
        image_path = prepare_dir / expected_reference
        if sha256_file(image_path) != first_image["content_sha256"] or image_item["transformed_jpeg_sha256"] != first_image["content_sha256"]:
            raise VerificationError(f"{context}: transformed JPEG SHA mismatch")
        try:
            with Image.open(image_path) as decoded:
                decoded.load()
                rgb = decoded.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise VerificationError(f"{context}: image decode failed: {error}") from error
        regions = _regions(rgb)
        if (
            first_image["media_type"] != "image/jpeg"
            or first_image["width"] != rgb.width
            or first_image["height"] != rgb.height
            or rgb.width * rgb.height > spec["image_transform"]["area_cap"]
            or first_image["decoded_rgb_sha256"] != regions["full"]
            or image_item["transformed_rgb_sha256"] != regions["full"]
            or image_item["region_sha256"] != regions
        ):
            raise VerificationError(f"{context}: decoded image evidence mismatch")
        _hex(image_item["original_bytes_sha256"], f"{context}.original_bytes_sha256")
        _hex(image_item["resized_rgb_sha256"], f"{context}.resized_rgb_sha256")
        card = {"sources": sources, "first_image": first_image}
        if row["source_card_sha256"] != sha256_bytes(canonical_json_bytes(card)):
            raise VerificationError(f"{context}: source-card SHA mismatch")
        candidates = row["evidence_candidates"]
        if not isinstance(candidates, list):
            raise VerificationError(f"{context}: candidates missing")
        for candidate_index, candidate in enumerate(candidates):
            _validate_candidate(candidate, candidate_index, sources, regions, f"{context}.candidate {candidate_index}")
        if {candidate["image_region"] for candidate in candidates if candidate["evidence_kind"] == "image_region"} != set(spec["image_transform"]["regions"]):
            raise VerificationError(f"{context}: full+2x2 image evidence incomplete")
        source_cards.append(row["source_card_sha256"])
        candidate_ids.extend(candidate["candidate_id"] for candidate in candidates)
        manifest_by_record[record_id] = image_item
    strata = {item["name"] for item in spec["strata"]}
    if stratum_counts != Counter({stratum: 100 for stratum in strata}):
        raise VerificationError("source rows: 100-per-stratum quota mismatch")
    if fold_stratum_counts != Counter({(fold, stratum): 20 for fold in range(5) for stratum in strata}):
        raise VerificationError("source rows: 20-per-fold-per-stratum quota mismatch")

    membership_sha = sha256_bytes(canonical_json_bytes([
        {"record_id": item["record_id"], "row_token": item["row_token"], "reference": item["reference"], "original_bytes_sha256": item["original_bytes_sha256"]}
        for item in manifest
    ]))
    transform_sha = sha256_bytes(canonical_json_bytes({
        "rules": spec["image_transform"],
        "image_manifest_sha256": sha256_file(image_manifest_path),
        "images": manifest,
    }))
    candidate_sha = sha256_bytes(canonical_json_bytes({
        "description_clause_limit": spec["description_clause_limit"],
        "ocr_policy": spec["ocr_policy"],
        "source_cards": source_cards,
        "candidate_ids": candidate_ids,
    }))
    if contract["image_membership_sha256"] != membership_sha or contract["image_transform_sha256"] != transform_sha or contract["candidate_generator_sha256"] != candidate_sha:
        raise VerificationError("source contract: independently recomputed image/candidate binding mismatch")

    if sha256_file(terminal_metadata_path) != terminal_metadata_sha256:
        raise VerificationError("terminal metadata: expected SHA mismatch")
    terminal = _json(terminal_metadata_path, "terminal metadata")
    _exact_keys(terminal, {"schema_version", "job_id", "status", "finished_at", "output_ref", "self_sha256"}, "terminal metadata")
    _self_hash(terminal, "terminal metadata")
    if terminal["schema_version"] != "exp689_source_prepare_terminal_v1" or terminal["status"] != "SUCCESS":
        raise VerificationError("terminal metadata: exact terminal SUCCESS required")
    if not all(isinstance(terminal[field], str) and terminal[field].strip() for field in ("job_id", "finished_at")):
        raise VerificationError("terminal metadata: job_id/finished_at missing")
    if terminal["output_ref"] != approved_s3_output_ref:
        raise VerificationError("terminal metadata: approved output ref mismatch")

    inventory_sha = sha256_bytes(canonical_json_bytes(inventory))
    acceptance = {
        "schema_version": "exp689_source_prepare_acceptance_v1",
        "decision": "ACCEPT",
        "builder_revision": builder_revision,
        "builder_revision_sha256": sha256_text(builder_revision),
        "builder_source_sha256": builder_sha,
        "bundle_manifest_sha256": bundle_sha,
        "source_prepare_spec_sha256": sha256_file(spec_path),
        "runtime_archives": archive_bindings,
        "runtime_bindings_sha256": sha256_bytes(canonical_json_bytes(runtime_bindings)),
        "source_contract_sha256": sha256_file(contract_path),
        "source_contract_self_sha256": contract["self_sha256"],
        "source_rows_sha256": sha256_file(source_rows_path),
        "source_row_count": 300,
        "output_inventory_sha256": inventory_sha,
        "output_inventory": inventory,
        "terminal_metadata_sha256": terminal_metadata_sha256,
        "terminal_job_id": terminal["job_id"],
        "terminal_status": terminal["status"],
        "terminal_finished_at": terminal["finished_at"],
        "approved_s3_output_ref": approved_s3_output_ref,
        "label_fields_present": False,
        "score_fields_present": False,
        "sealed_rows": 0,
        "public_rows": 0,
        "self_sha256": None,
    }
    acceptance["self_sha256"] = sha256_bytes(canonical_json_bytes(acceptance))
    acceptance_path.parent.mkdir(parents=True, exist_ok=True)
    acceptance_path.write_text(
        json.dumps(acceptance, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return acceptance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, action="append", required=True)
    parser.add_argument("--runtime-archive", type=Path, action="append", required=True)
    parser.add_argument("--runtime-archive-ref", action="append", required=True)
    parser.add_argument("--runtime-archive-sha256", action="append", required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--bundle-manifest-sha256", required=True)
    parser.add_argument("--builder-revision", required=True)
    parser.add_argument("--spec", type=Path, default=SPEC_PATH)
    parser.add_argument("--terminal-metadata", type=Path, required=True)
    parser.add_argument("--terminal-metadata-sha256", required=True)
    parser.add_argument("--approved-s3-output-ref", required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    args = parser.parse_args()
    acceptance = verify(
        prepare_dir=args.prepare_dir,
        runtime_dirs=args.runtime_dir,
        runtime_archives=args.runtime_archive,
        runtime_archive_refs=args.runtime_archive_ref,
        runtime_archive_sha256=args.runtime_archive_sha256,
        bundle_root=args.bundle_root,
        bundle_manifest_path=args.bundle_manifest,
        bundle_manifest_sha256=args.bundle_manifest_sha256,
        builder_revision=args.builder_revision,
        spec_path=args.spec,
        terminal_metadata_path=args.terminal_metadata,
        terminal_metadata_sha256=args.terminal_metadata_sha256,
        approved_s3_output_ref=args.approved_s3_output_ref,
        acceptance_path=args.acceptance,
    )
    print(json.dumps(acceptance, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
