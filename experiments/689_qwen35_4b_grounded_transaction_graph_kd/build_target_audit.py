"""Build the experiment-689 target audit from contract-bound remote inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

EXPERIMENT = Path(__file__).resolve().parent
SPEC_PATH = EXPERIMENT / "frozen_target_audit_spec.json"
SCHEMA_PATH = EXPERIMENT / "target_audit_schema_v1.json"
RUBRIC_PATH = EXPERIMENT / "review_rubric_v1.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
S3_REFERENCE = re.compile(r"^s3://[^/?#]+/[^?#]+$")
SOURCE_KINDS = ("name", "description", "ocr")


class ContractError(ValueError):
    """Raised when an immutable input violates the frozen contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def with_self_hash(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["self_sha256"] = None
    result["self_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result


def validate_self_hash(payload: dict[str, Any], context: str) -> None:
    actual = payload.get("self_sha256")
    if not isinstance(actual, str) or not HEX64.fullmatch(actual):
        raise ContractError(f"{context}: missing canonical self_sha256")
    copy = dict(payload)
    copy["self_sha256"] = None
    expected = sha256_bytes(canonical_json_bytes(copy))
    if actual != expected:
        raise ContractError(f"{context}: canonical self_sha256 mismatch")


def expect_exact_keys(value: dict[str, Any], keys: set[str], context: str) -> None:
    actual = set(value)
    if actual != keys:
        raise ContractError(
            f"{context}: exact fields required; missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )


def require_hex64(value: Any, context: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ContractError(f"{context}: expected lowercase SHA-256")
    return value


def require_remote_path(
    remote_root: Path,
    path: Path,
    *,
    context: str,
    must_exist: bool,
) -> Path:
    root = remote_root.resolve(strict=True)
    candidate = path.resolve(strict=must_exist)
    if not candidate.is_relative_to(root):
        raise ContractError(f"{context}: path must remain below --remote-root")
    return candidate


def load_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"{context}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"{context}: top level must be an object")
    return value


def read_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    raise ContractError(f"{context}: blank line at {line_number}")
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise ContractError(
                        f"{context}: invalid JSON at line {line_number}: {error}"
                    ) from error
                if not isinstance(row, dict):
                    raise ContractError(f"{context}: row {line_number} must be an object")
                rows.append(row)
    except OSError as error:
        raise ContractError(f"{context}: cannot read JSONL: {error}") from error
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(payload)


def load_spec() -> dict[str, Any]:
    spec = load_json(SPEC_PATH, "frozen spec")
    validate_self_hash(spec, "frozen spec")
    if spec.get("schema_version") != "exp689_frozen_target_audit_spec_v1":
        raise ContractError("frozen spec: schema version mismatch")
    if spec.get("execution_scope") != "remote_mlcore_only":
        raise ContractError("frozen spec: execution scope is not remote-only")
    if spec.get("target_audit_schema_sha256") != sha256_file(SCHEMA_PATH):
        raise ContractError("frozen spec: target-audit schema SHA mismatch")
    if spec.get("review_contract", {}).get("rubric_sha256") != sha256_file(RUBRIC_PATH):
        raise ContractError("frozen spec: review-rubric SHA mismatch")
    return spec


def validate_source_contract(
    contract: dict[str, Any], source_rows_path: Path, row_count: int
) -> None:
    expect_exact_keys(
        contract,
        {
            "schema_version",
            "execution_scope",
            "source_rows_sha256",
            "source_row_count",
            "runtime_sha256",
            "data_sha256",
            "registry_sha256",
            "builder_revision_sha256",
            "eligibility_universe_sha256",
            "stratum_derivation_sha256",
            "image_membership_sha256",
            "image_transform_sha256",
            "candidate_generator_sha256",
            "opaque_token_scheme",
            "label_fields_present",
            "score_fields_present",
            "first_image_rows_verified",
            "image_fetch_failures",
            "image_decode_failures",
            "image_hash_mismatches",
            "sealed_rows",
            "public_rows",
            "self_sha256",
        },
        "source contract",
    )
    validate_self_hash(contract, "source contract")
    if contract["schema_version"] != "exp689_source_contract_v2":
        raise ContractError("source contract: schema version mismatch")
    if contract["execution_scope"] != "remote_mlcore":
        raise ContractError("source contract: execution_scope must be remote_mlcore")
    if contract["source_rows_sha256"] != sha256_file(source_rows_path):
        raise ContractError("source contract: source rows SHA mismatch")
    if contract["source_row_count"] != row_count:
        raise ContractError("source contract: source row count mismatch")
    for field in (
        "runtime_sha256",
        "data_sha256",
        "registry_sha256",
        "builder_revision_sha256",
        "eligibility_universe_sha256",
        "stratum_derivation_sha256",
        "image_membership_sha256",
        "image_transform_sha256",
        "candidate_generator_sha256",
    ):
        require_hex64(contract[field], f"source contract.{field}")
    if contract["opaque_token_scheme"] != "sha256_utf8_v1":
        raise ContractError("source contract: opaque token scheme mismatch")
    if contract["label_fields_present"] is not False:
        raise ContractError("source contract: outcome labels are forbidden")
    if contract["score_fields_present"] is not False:
        raise ContractError("source contract: verdict/logit/probability/rank fields are forbidden")
    if contract["first_image_rows_verified"] != row_count:
        raise ContractError("source contract: every source row needs a verified first image")
    if any(
        contract[field] != 0
        for field in ("image_fetch_failures", "image_decode_failures", "image_hash_mismatches")
    ):
        raise ContractError("source contract: image fetch/decode/hash failures must be zero")
    if contract["sealed_rows"] != 0 or contract["public_rows"] != 0:
        raise ContractError("source contract: sealed and Public rows must both be zero")


def validate_source_prepare_acceptance(
    acceptance: dict[str, Any],
    *,
    acceptance_path: Path,
    expected_acceptance_sha256: str,
    source_rows_path: Path,
    source_contract_path: Path,
    source_contract: dict[str, Any],
    exclusion_670_sha256: str,
    exclusion_672_sha256: str,
) -> None:
    """Require an independently emitted ACCEPT and recompute its local bindings."""
    expect_exact_keys(
        acceptance,
        {
            "schema_version",
            "decision",
            "builder_revision",
            "builder_revision_sha256",
            "builder_source_sha256",
            "bundle_manifest_sha256",
            "source_prepare_spec_sha256",
            "runtime_archives",
            "runtime_bindings_sha256",
            "exclusion_670_sha256",
            "exclusion_672_sha256",
            "eligibility_universe_sha256",
            "stratum_derivation_sha256",
            "source_membership_reconstruction_sha256",
            "source_contract_sha256",
            "source_contract_self_sha256",
            "source_rows_sha256",
            "source_row_count",
            "output_inventory_sha256",
            "output_inventory",
            "terminal_metadata_sha256",
            "terminal_job_id",
            "terminal_status",
            "terminal_finished_at",
            "approved_s3_output_ref",
            "label_fields_present",
            "score_fields_present",
            "ocr_source_count",
            "sealed_rows",
            "public_rows",
            "self_sha256",
        },
        "source prepare acceptance",
    )
    require_hex64(expected_acceptance_sha256, "source prepare acceptance expected SHA")
    if sha256_file(acceptance_path) != expected_acceptance_sha256:
        raise ContractError("source prepare acceptance: exact file SHA mismatch")
    validate_self_hash(acceptance, "source prepare acceptance")
    if acceptance["schema_version"] != "exp689_source_prepare_acceptance_v1":
        raise ContractError("source prepare acceptance: schema version mismatch")
    if acceptance["decision"] != "ACCEPT":
        raise ContractError("source prepare acceptance: exact ACCEPT required")
    if not isinstance(acceptance["builder_revision"], str) or not HEX40.fullmatch(
        acceptance["builder_revision"]
    ):
        raise ContractError("source prepare acceptance: invalid builder revision")
    for field in (
        "builder_revision_sha256",
        "builder_source_sha256",
        "bundle_manifest_sha256",
        "source_prepare_spec_sha256",
        "runtime_bindings_sha256",
        "exclusion_670_sha256",
        "exclusion_672_sha256",
        "eligibility_universe_sha256",
        "stratum_derivation_sha256",
        "source_membership_reconstruction_sha256",
        "source_contract_sha256",
        "source_contract_self_sha256",
        "source_rows_sha256",
        "output_inventory_sha256",
        "terminal_metadata_sha256",
    ):
        require_hex64(acceptance[field], f"source prepare acceptance.{field}")
    if acceptance["builder_revision_sha256"] != sha256_text(
        acceptance["builder_revision"]
    ):
        raise ContractError("source prepare acceptance: builder revision SHA mismatch")
    if acceptance["builder_revision_sha256"] != source_contract["builder_revision_sha256"]:
        raise ContractError("source prepare acceptance: source-contract builder mismatch")
    if acceptance["exclusion_670_sha256"] != exclusion_670_sha256 or acceptance[
        "exclusion_672_sha256"
    ] != exclusion_672_sha256:
        raise ContractError("source prepare acceptance: exclusion input binding mismatch")
    if acceptance["eligibility_universe_sha256"] != source_contract[
        "eligibility_universe_sha256"
    ] or acceptance["stratum_derivation_sha256"] != source_contract[
        "stratum_derivation_sha256"
    ]:
        raise ContractError("source prepare acceptance: eligibility/strata binding mismatch")
    if acceptance["source_contract_sha256"] != sha256_file(source_contract_path):
        raise ContractError("source prepare acceptance: source-contract file binding mismatch")
    if acceptance["source_contract_self_sha256"] != source_contract["self_sha256"]:
        raise ContractError("source prepare acceptance: source-contract self binding mismatch")
    if acceptance["source_rows_sha256"] != sha256_file(source_rows_path):
        raise ContractError("source prepare acceptance: source-rows file binding mismatch")
    if acceptance["source_rows_sha256"] != source_contract["source_rows_sha256"]:
        raise ContractError("source prepare acceptance: source-rows contract mismatch")
    if acceptance["source_row_count"] != source_contract["source_row_count"]:
        raise ContractError("source prepare acceptance: source-row count mismatch")
    archives = acceptance["runtime_archives"]
    if not isinstance(archives, list) or len(archives) != 2:
        raise ContractError("source prepare acceptance: exactly two runtime archives required")
    covered_folds: list[int] = []
    for index, archive in enumerate(archives):
        if not isinstance(archive, dict):
            raise ContractError(f"source prepare acceptance: archive {index} must be an object")
        expect_exact_keys(
            archive,
            {"archive_id", "folds", "reference", "sha256", "size_bytes"},
            f"source prepare acceptance archive {index}",
        )
        folds = archive["folds"]
        if (
            not isinstance(archive["archive_id"], str)
            or not archive["archive_id"].strip()
            or not isinstance(folds, list)
            or any(not isinstance(fold, int) or fold not in range(5) for fold in folds)
            or folds != sorted(set(folds))
            or not isinstance(archive["size_bytes"], int)
            or archive["size_bytes"] < 0
        ):
            raise ContractError(f"source prepare acceptance: archive {index} metadata mismatch")
        covered_folds.extend(folds)
        require_hex64(archive["sha256"], f"source prepare acceptance archive {index}.sha256")
        if not isinstance(archive["reference"], str) or not S3_REFERENCE.fullmatch(
            archive["reference"]
        ):
            raise ContractError(f"source prepare acceptance: archive {index} ref is not approved S3")
    if sorted(covered_folds) != list(range(5)):
        raise ContractError("source prepare acceptance: archive fold mapping mismatch")
    inventory = acceptance["output_inventory"]
    if not isinstance(inventory, list) or len(inventory) != 303:
        raise ContractError("source prepare acceptance: exact 303-file output inventory required")
    if acceptance["output_inventory_sha256"] != sha256_bytes(
        canonical_json_bytes(inventory)
    ):
        raise ContractError("source prepare acceptance: output inventory SHA mismatch")
    inventory_by_path: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(inventory):
        if not isinstance(item, dict):
            raise ContractError(f"source prepare inventory {index}: must be an object")
        expect_exact_keys(item, {"path", "size_bytes", "sha256"}, f"source prepare inventory {index}")
        if (
            not isinstance(item["path"], str)
            or item["path"] in inventory_by_path
            or not isinstance(item["size_bytes"], int)
            or item["size_bytes"] < 0
        ):
            raise ContractError(f"source prepare inventory {index}: invalid path/size")
        require_hex64(item["sha256"], f"source prepare inventory {index}.sha256")
        inventory_by_path[item["path"]] = item
    expected_paths = {
        "source_rows.jsonl",
        "source_contract.json",
        "images/image_manifest.jsonl",
    } | {f"images/G689-{index:03d}.jpg" for index in range(1, 301)}
    if set(inventory_by_path) != expected_paths:
        raise ContractError("source prepare acceptance: output inventory paths mismatch")
    for name, actual_path in (
        ("source_rows.jsonl", source_rows_path),
        ("source_contract.json", source_contract_path),
    ):
        item = inventory_by_path[name]
        if item["size_bytes"] != actual_path.stat().st_size or item["sha256"] != sha256_file(
            actual_path
        ):
            raise ContractError(f"source prepare acceptance: recomputed {name} binding mismatch")
    if acceptance["terminal_status"] != "SUCCESS":
        raise ContractError("source prepare acceptance: terminal SUCCESS required")
    if not all(
        isinstance(acceptance[field], str) and acceptance[field].strip()
        for field in ("terminal_job_id", "terminal_finished_at")
    ):
        raise ContractError("source prepare acceptance: terminal metadata incomplete")
    if not isinstance(acceptance["approved_s3_output_ref"], str) or not S3_REFERENCE.fullmatch(
        acceptance["approved_s3_output_ref"]
    ):
        raise ContractError("source prepare acceptance: approved S3 output ref invalid")
    if acceptance["label_fields_present"] is not False or acceptance[
        "score_fields_present"
    ] is not False:
        raise ContractError("source prepare acceptance: labels/scores must be absent")
    if acceptance["ocr_source_count"] != 0:
        raise ContractError("source prepare acceptance: OCR sources must be zero")
    if acceptance["sealed_rows"] != 0 or acceptance["public_rows"] != 0:
        raise ContractError("source prepare acceptance: sealed/Public rows must be zero")


def validate_exclusion_manifest(
    manifest: dict[str, Any], expected_experiment_id: str
) -> dict[str, set[str]]:
    context = f"exclusion manifest {expected_experiment_id}"
    expect_exact_keys(
        manifest,
        {
            "schema_version",
            "experiment_id",
            "row_tokens",
            "component_tokens",
            "family_tokens",
            "self_sha256",
        },
        context,
    )
    validate_self_hash(manifest, context)
    if manifest["schema_version"] != "exp689_exclusion_manifest_v1":
        raise ContractError(f"{context}: schema version mismatch")
    if manifest["experiment_id"] != expected_experiment_id:
        raise ContractError(f"{context}: experiment lineage mismatch")
    result: dict[str, set[str]] = {}
    for field in ("row_tokens", "component_tokens", "family_tokens"):
        values = manifest[field]
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise ContractError(f"{context}: {field} must be a string list")
        if values != sorted(set(values)):
            raise ContractError(f"{context}: {field} must be sorted and unique")
        for index, value in enumerate(values):
            require_hex64(value, f"{context}.{field}[{index}]")
        result[field] = set(values)
    return result


def _json_locator(value: Any, locator: str, context: str) -> Any:
    current = value
    for part in locator.split("."):
        if not part:
            raise ContractError(f"{context}: empty JSON locator component")
        if not isinstance(current, dict) or part not in current:
            raise ContractError(f"{context}: JSON locator {locator!r} was not found")
        current = current[part]
    return current


def _normalize_component_values(
    values: list[Any], *, token_mode: str, value_field: str | None, context: str
) -> set[str]:
    tokens: list[str] = []
    for index, item in enumerate(values):
        if value_field:
            if not isinstance(item, dict) or set(item).isdisjoint({value_field}):
                raise ContractError(f"{context}: item {index} lacks {value_field!r}")
            item = item[value_field]
        if not isinstance(item, str) or not item.strip():
            raise ContractError(f"{context}: item {index} must be a nonempty string")
        if token_mode == "already_sha256":
            token = require_hex64(item, f"{context}[{index}]")
        elif token_mode == "sha256_utf8_v1":
            token = sha256_text(item)
        else:
            raise ContractError(f"{context}: unsupported token mode {token_mode!r}")
        tokens.append(token)
    if len(tokens) != len(set(tokens)):
        raise ContractError(f"{context}: component values are not unique after tokenization")
    return set(tokens)


def load_runtime_exclusion(
    *,
    path: Path,
    expected_sha256: str,
    expected_experiment_id: str,
    adapter: str,
    component_locator: str,
    component_value_field: str | None,
    token_mode: str,
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Load an approved remote exclusion without embedding its path or SHA in Git."""
    context = f"exclusion source {expected_experiment_id}"
    require_hex64(expected_sha256, f"{context}.expected_sha256")
    if sha256_file(path) != expected_sha256:
        raise ContractError(f"{context}: source SHA mismatch")
    if adapter == "canonical_json":
        manifest = load_json(path, context)
        sets = validate_exclusion_manifest(manifest, expected_experiment_id)
        binding = {
            "source_sha256": expected_sha256,
            "adapter": adapter,
            "component_locator": "component_tokens",
            "component_value_field": None,
            "token_mode": "already_sha256",
            "component_count": len(sets["component_tokens"]),
        }
        return sets, binding
    if adapter == "csv_component_column":
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if component_locator not in (reader.fieldnames or ()):
                    raise ContractError(f"{context}: CSV component column is missing")
                values = [row[component_locator] for row in reader]
        except OSError as error:
            raise ContractError(f"{context}: cannot read CSV: {error}") from error
    elif adapter == "json_component_list":
        try:
            root = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ContractError(f"{context}: cannot read JSON: {error}") from error
        values = _json_locator(root, component_locator, context)
        if not isinstance(values, list):
            raise ContractError(f"{context}: component locator must resolve to a list")
    else:
        raise ContractError(f"{context}: unsupported adapter {adapter!r}")
    components = _normalize_component_values(
        values,
        token_mode=token_mode,
        value_field=component_value_field,
        context=f"{context}.components",
    )
    sets = {"row_tokens": set(), "component_tokens": components, "family_tokens": set()}
    binding = {
        "source_sha256": expected_sha256,
        "adapter": adapter,
        "component_locator": component_locator,
        "component_value_field": component_value_field,
        "token_mode": token_mode,
        "component_count": len(components),
    }
    return sets, binding


def validate_source_row(row: dict[str, Any], spec: dict[str, Any], index: int) -> None:
    context = f"source row {index}"
    expect_exact_keys(
        row,
        {
            "schema_version",
            "record_id",
            "row_token",
            "component_token",
            "family_token",
            "stratum",
            "source_card_sha256",
            "sources",
            "first_image",
            "evidence_candidates",
        },
        context,
    )
    if row["schema_version"] != "exp689_source_row_v1":
        raise ContractError(f"{context}: schema version mismatch")
    if not isinstance(row["record_id"], str) or not row["record_id"].strip():
        raise ContractError(f"{context}: record_id must be a nonempty opaque string")
    for field in ("row_token", "component_token", "family_token", "source_card_sha256"):
        require_hex64(row[field], f"{context}.{field}")
    if row["stratum"] not in spec["audit"]["strata"]:
        raise ContractError(f"{context}: unknown stratum")

    sources = row["sources"]
    if not isinstance(sources, list) or not sources:
        raise ContractError(f"{context}: sources must be a nonempty list")
    for source_index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ContractError(f"{context}: source {source_index} must be an object")
        expect_exact_keys(
            source,
            {"source_index", "source_kind", "text", "text_sha256"},
            f"{context}.sources[{source_index}]",
        )
        if source["source_index"] != source_index:
            raise ContractError(f"{context}: source indices must be consecutive")
        if source["source_kind"] not in SOURCE_KINDS:
            raise ContractError(f"{context}: unsupported source kind")
        if not isinstance(source["text"], str) or not source["text"].strip():
            raise ContractError(f"{context}: source text must be nonempty")
        if source["text_sha256"] != sha256_text(source["text"]):
            raise ContractError(f"{context}: source text SHA mismatch")
    first_image = row["first_image"]
    if not isinstance(first_image, dict):
        raise ContractError(f"{context}: first_image must be an object")
    expect_exact_keys(
        first_image,
        {
            "reference",
            "content_sha256",
            "decoded_rgb_sha256",
            "media_type",
            "width",
            "height",
        },
        f"{context}.first_image",
    )
    if not isinstance(first_image["reference"], str) or not first_image["reference"].strip():
        raise ContractError(f"{context}: first-image reference must be nonempty")
    for field in ("content_sha256", "decoded_rgb_sha256"):
        require_hex64(first_image[field], f"{context}.first_image.{field}")
    if first_image["media_type"] not in {"image/jpeg", "image/png", "image/webp"}:
        raise ContractError(f"{context}: unsupported first-image media type")
    if (
        not isinstance(first_image["width"], int)
        or not isinstance(first_image["height"], int)
        or first_image["width"] < 2
        or first_image["height"] < 2
    ):
        raise ContractError(f"{context}: decoded first-image dimensions are invalid")
    source_identity = {"sources": sources, "first_image": first_image}
    if row["source_card_sha256"] != sha256_bytes(canonical_json_bytes(source_identity)):
        raise ContractError(f"{context}: source-card SHA mismatch")

    candidates = row["evidence_candidates"]
    if not isinstance(candidates, list):
        raise ContractError(f"{context}: evidence_candidates must be a list")
    seen_spans: set[tuple[int, int, int]] = set()
    seen_candidate_ids: set[str] = set()
    image_regions: set[str] = set()
    for candidate_index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ContractError(f"{context}: candidate {candidate_index} must be an object")
        expect_exact_keys(
            candidate,
            {
                "candidate_id",
                "candidate_index",
                "evidence_kind",
                "source_index",
                "char_start",
                "char_end",
                "image_region",
                "evidence_sha256",
            },
            f"{context}.evidence_candidates[{candidate_index}]",
        )
        if candidate["candidate_index"] != candidate_index:
            raise ContractError(f"{context}: candidate indices must be consecutive")
        require_hex64(candidate["candidate_id"], f"{context}.candidate_id")
        require_hex64(candidate["evidence_sha256"], f"{context}.evidence_sha256")
        candidate_copy = dict(candidate)
        candidate_copy["candidate_id"] = None
        if candidate["candidate_id"] != sha256_bytes(canonical_json_bytes(candidate_copy)):
            raise ContractError(f"{context}: candidate ID is not deterministic")
        if candidate["candidate_id"] in seen_candidate_ids:
            raise ContractError(f"{context}: duplicate candidate ID")
        seen_candidate_ids.add(candidate["candidate_id"])
        if candidate["evidence_kind"] == "text_span":
            source_index = candidate["source_index"]
            start = candidate["char_start"]
            end = candidate["char_end"]
            if candidate["image_region"] is not None:
                raise ContractError(f"{context}: text candidate cannot bind an image region")
            if not isinstance(source_index, int) or not 0 <= source_index < len(sources):
                raise ContractError(f"{context}: candidate source index is invalid")
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end:
                raise ContractError(f"{context}: candidate offsets are invalid")
            text = sources[source_index]["text"]
            if end > len(text) or not text[start:end].strip():
                raise ContractError(f"{context}: candidate span is empty or out of bounds")
            if candidate["evidence_sha256"] != sha256_text(text[start:end]):
                raise ContractError(f"{context}: candidate evidence SHA mismatch")
            span = (source_index, start, end)
            if span in seen_spans:
                raise ContractError(f"{context}: duplicate evidence span")
            seen_spans.add(span)
        elif candidate["evidence_kind"] == "image_region":
            if any(
                candidate[field] is not None
                for field in ("source_index", "char_start", "char_end")
            ):
                raise ContractError(f"{context}: image candidate cannot bind text offsets")
            region = candidate["image_region"]
            if region not in spec["image_contract"]["evidence_regions"]:
                raise ContractError(f"{context}: unknown image evidence region")
            if region in image_regions:
                raise ContractError(f"{context}: duplicate image evidence region")
            if region == "full" and candidate["evidence_sha256"] != first_image["decoded_rgb_sha256"]:
                raise ContractError(f"{context}: full-image evidence SHA mismatch")
            image_regions.add(region)
        else:
            raise ContractError(f"{context}: unknown evidence kind")
    if image_regions != set(spec["image_contract"]["evidence_regions"]):
        raise ContractError(f"{context}: full image and all four quadrants are required")


def request_row_hash(row: dict[str, Any]) -> str:
    copy = dict(row)
    copy["request_row_sha256"] = None
    return sha256_bytes(canonical_json_bytes(copy))


def make_teacher_request(row: dict[str, Any], audit_id: str) -> dict[str, Any]:
    request = {
        "schema_version": "exp689_teacher_request_row_v2",
        "audit_id": audit_id,
        "record_id": row["record_id"],
        "source_card_sha256": row["source_card_sha256"],
        "sources": row["sources"],
        "first_image": row["first_image"],
        "evidence_candidates": row["evidence_candidates"],
        "request_row_sha256": None,
    }
    request["request_row_sha256"] = request_row_hash(request)
    return request


def validate_teacher_request(
    row: dict[str, Any], index: int, spec: dict[str, Any] | None = None
) -> None:
    context = f"teacher request row {index}"
    expect_exact_keys(
        row,
        {
            "schema_version",
            "audit_id",
            "record_id",
            "source_card_sha256",
            "sources",
            "first_image",
            "evidence_candidates",
            "request_row_sha256",
        },
        context,
    )
    if row["schema_version"] != "exp689_teacher_request_row_v2":
        raise ContractError(f"{context}: schema version mismatch")
    if row["audit_id"] != f"G689-{index:03d}":
        raise ContractError(f"{context}: audit ID/order mismatch")
    require_hex64(row["source_card_sha256"], f"{context}.source_card_sha256")
    require_hex64(row["request_row_sha256"], f"{context}.request_row_sha256")
    if row["request_row_sha256"] != request_row_hash(row):
        raise ContractError(f"{context}: request-row SHA mismatch")
    synthetic = {
        "schema_version": "exp689_source_row_v1",
        "record_id": row["record_id"],
        "row_token": "0" * 64,
        "component_token": "1" * 64,
        "family_token": "2" * 64,
        "stratum": "direct_included_fuel",
        "source_card_sha256": row["source_card_sha256"],
        "sources": row["sources"],
        "first_image": row["first_image"],
        "evidence_candidates": row["evidence_candidates"],
    }
    validate_source_row(synthetic, spec if spec is not None else load_spec(), index)


def selection_rank(spec: dict[str, Any], row: dict[str, Any]) -> tuple[str, str]:
    payload = "\0".join(
        (
            spec["audit"]["selection_salt"],
            row["stratum"],
            row["record_id"],
            row["source_card_sha256"],
        )
    )
    return sha256_text(payload), row["record_id"]


def select_source_rows(
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
    exclusion_sets: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    excluded_rows = exclusion_sets["row_tokens"]
    excluded_components = exclusion_sets["component_tokens"]
    excluded_families = exclusion_sets["family_tokens"]
    eligible = [
        row
        for row in rows
        if row["row_token"] not in excluded_rows
        and row["component_token"] not in excluded_components
        and row["family_token"] not in excluded_families
    ]
    selected: list[dict[str, Any]] = []
    used_components: set[str] = set()
    used_source_cards: set[str] = set()
    eligible_counts: dict[str, int] = {}
    target = spec["audit"]["rows_per_stratum"]
    for stratum in spec["audit"]["strata"]:
        pool = sorted(
            (row for row in eligible if row["stratum"] == stratum),
            key=lambda row: selection_rank(spec, row),
        )
        eligible_counts[stratum] = len(pool)
        chosen: list[dict[str, Any]] = []
        for row in pool:
            if row["component_token"] in used_components:
                continue
            if row["source_card_sha256"] in used_source_cards:
                continue
            chosen.append(row)
            used_components.add(row["component_token"])
            used_source_cards.add(row["source_card_sha256"])
            if len(chosen) == target:
                break
        if len(chosen) != target:
            raise ContractError(
                f"stratum {stratum}: need {target} component- and exact-source-unique rows, "
                f"found {len(chosen)}"
            )
        selected.extend(chosen)
    if len(selected) != spec["audit"]["total_rows"]:
        raise ContractError("selection did not produce exactly 300 rows")
    return selected, eligible_counts


def make_bindings(
    selected: list[dict[str, Any]], requests: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "audit_id": request["audit_id"],
            "record_id": row["record_id"],
            "row_token": row["row_token"],
            "component_token": row["component_token"],
            "family_token": row["family_token"],
            "stratum": row["stratum"],
            "source_card_sha256": row["source_card_sha256"],
            "request_row_sha256": request["request_row_sha256"],
        }
        for row, request in zip(selected, requests, strict=True)
    ]


def _prepare_inputs(
    *,
    remote_root: Path,
    source_rows_path: Path,
    source_contract_path: Path,
    exclusion_670_path: Path,
    exclusion_672_path: Path,
    exclusion_670_sha256: str,
    exclusion_672_sha256: str,
    exclusion_670_adapter: str,
    exclusion_672_adapter: str,
    exclusion_670_component_locator: str,
    exclusion_672_component_locator: str,
    exclusion_670_component_value_field: str | None,
    exclusion_672_component_value_field: str | None,
    exclusion_670_token_mode: str,
    exclusion_672_token_mode: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, set[str]],
]:
    spec = load_spec()
    paths = {
        "source rows": source_rows_path,
        "source contract": source_contract_path,
        "exclusion 670": exclusion_670_path,
        "exclusion 672": exclusion_672_path,
    }
    resolved = {
        name: require_remote_path(remote_root, path, context=name, must_exist=True)
        for name, path in paths.items()
    }
    rows = read_jsonl(resolved["source rows"], "source rows")
    source_contract = load_json(resolved["source contract"], "source contract")
    validate_source_contract(source_contract, resolved["source rows"], len(rows))
    sets_670, binding_670 = load_runtime_exclusion(
        path=resolved["exclusion 670"],
        expected_sha256=exclusion_670_sha256,
        expected_experiment_id="670",
        adapter=exclusion_670_adapter,
        component_locator=exclusion_670_component_locator,
        component_value_field=exclusion_670_component_value_field,
        token_mode=exclusion_670_token_mode,
    )
    sets_672, binding_672 = load_runtime_exclusion(
        path=resolved["exclusion 672"],
        expected_sha256=exclusion_672_sha256,
        expected_experiment_id="672",
        adapter=exclusion_672_adapter,
        component_locator=exclusion_672_component_locator,
        component_value_field=exclusion_672_component_value_field,
        token_mode=exclusion_672_token_mode,
    )
    expected_counts = spec["exclusions"]["expected_component_counts"]
    if len(sets_670["component_tokens"]) != expected_counts["670"]:
        raise ContractError("exclusion source 670: expected exactly 300 unique components")
    if len(sets_672["component_tokens"]) != expected_counts["672"]:
        raise ContractError("exclusion source 672: expected exactly 40 unique components")
    intersection = sets_670["component_tokens"] & sets_672["component_tokens"]
    if len(intersection) != spec["exclusions"]["required_component_intersection"]:
        raise ContractError("exclusion sources 670/672: component intersection must be zero")
    union = sets_670["component_tokens"] | sets_672["component_tokens"]
    if len(union) != expected_counts["union"]:
        raise ContractError("exclusion sources 670/672: component union must equal 340")
    exclusion_sets = {
        field: sets_670[field] | sets_672[field]
        for field in ("row_tokens", "component_tokens", "family_tokens")
    }
    seen_record_ids: set[str] = set()
    seen_row_tokens: set[str] = set()
    for index, row in enumerate(rows, 1):
        validate_source_row(row, spec, index)
        if row["record_id"] in seen_record_ids:
            raise ContractError(f"source row {index}: duplicate record_id")
        if row["row_token"] in seen_row_tokens:
            raise ContractError(f"source row {index}: duplicate row_token")
        seen_record_ids.add(row["record_id"])
        seen_row_tokens.add(row["row_token"])
    return (
        spec,
        rows,
        source_contract,
        {"670": binding_670, "672": binding_672},
        exclusion_sets,
    )


def prepare(
    *,
    remote_root: Path,
    source_rows_path: Path,
    source_contract_path: Path,
    source_prepare_acceptance_path: Path,
    source_prepare_acceptance_sha256: str,
    exclusion_670_path: Path,
    exclusion_672_path: Path,
    exclusion_670_sha256: str,
    exclusion_672_sha256: str,
    exclusion_670_adapter: str = "canonical_json",
    exclusion_672_adapter: str = "canonical_json",
    exclusion_670_component_locator: str = "component_tokens",
    exclusion_672_component_locator: str = "component_tokens",
    exclusion_670_component_value_field: str | None = None,
    exclusion_672_component_value_field: str | None = None,
    exclusion_670_token_mode: str = "already_sha256",
    exclusion_672_token_mode: str = "already_sha256",
    output_dir: Path,
) -> dict[str, Any]:
    """Select exactly 3x100 rows and emit a teacher-safe request."""
    spec, rows, source_contract, exclusions, exclusion_sets = _prepare_inputs(
        remote_root=remote_root,
        source_rows_path=source_rows_path,
        source_contract_path=source_contract_path,
        exclusion_670_path=exclusion_670_path,
        exclusion_672_path=exclusion_672_path,
        exclusion_670_sha256=exclusion_670_sha256,
        exclusion_672_sha256=exclusion_672_sha256,
        exclusion_670_adapter=exclusion_670_adapter,
        exclusion_672_adapter=exclusion_672_adapter,
        exclusion_670_component_locator=exclusion_670_component_locator,
        exclusion_672_component_locator=exclusion_672_component_locator,
        exclusion_670_component_value_field=exclusion_670_component_value_field,
        exclusion_672_component_value_field=exclusion_672_component_value_field,
        exclusion_670_token_mode=exclusion_670_token_mode,
        exclusion_672_token_mode=exclusion_672_token_mode,
    )
    resolved_source_rows = require_remote_path(
        remote_root, source_rows_path, context="source rows", must_exist=True
    )
    resolved_source_contract = require_remote_path(
        remote_root, source_contract_path, context="source contract", must_exist=True
    )
    resolved_acceptance = require_remote_path(
        remote_root,
        source_prepare_acceptance_path,
        context="source prepare acceptance",
        must_exist=True,
    )
    acceptance = load_json(resolved_acceptance, "source prepare acceptance")
    validate_source_prepare_acceptance(
        acceptance,
        acceptance_path=resolved_acceptance,
        expected_acceptance_sha256=source_prepare_acceptance_sha256,
        source_rows_path=resolved_source_rows,
        source_contract_path=resolved_source_contract,
        source_contract=source_contract,
        exclusion_670_sha256=exclusion_670_sha256,
        exclusion_672_sha256=exclusion_672_sha256,
    )
    output_dir = require_remote_path(
        remote_root, output_dir, context="prepare output", must_exist=False
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output directory: {output_dir}")

    selected, eligible_counts = select_source_rows(spec, rows, exclusion_sets)
    target = spec["audit"]["rows_per_stratum"]
    requests = [
        make_teacher_request(row, f"G689-{index:03d}")
        for index, row in enumerate(selected, 1)
    ]
    bindings = make_bindings(selected, requests)
    teacher_images: list[dict[str, Any]] = []
    for request in requests:
        first_image = request["first_image"]
        relative = Path(first_image["reference"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError("teacher image reference must remain a safe relative path")
        teacher_images.append(
            {
                "schema_version": "exp689_teacher_image_manifest_row_v1",
                "audit_id": request["audit_id"],
                "reference": first_image["reference"],
                "relative_path": first_image["reference"],
                "content_sha256": first_image["content_sha256"],
                "decoded_rgb_sha256": first_image["decoded_rgb_sha256"],
                "pixel_sha256": first_image["decoded_rgb_sha256"],
                "media_type": first_image["media_type"],
                "width": first_image["width"],
                "height": first_image["height"],
            }
        )

    output_dir.mkdir(parents=True)
    request_path = output_dir / "teacher_request.jsonl"
    teacher_image_manifest_path = output_dir / "teacher_image_manifest.jsonl"
    write_jsonl(request_path, requests)
    write_jsonl(teacher_image_manifest_path, teacher_images)
    manifest = with_self_hash(
        {
            "schema_version": "exp689_selection_manifest_v1",
            "experiment_id": "689",
            "execution_scope": "remote_mlcore",
            "source_contract_self_sha256": source_contract["self_sha256"],
            "source_rows_sha256": source_contract["source_rows_sha256"],
            "source_prepare_acceptance_sha256": source_prepare_acceptance_sha256,
            "source_prepare_acceptance_self_sha256": acceptance["self_sha256"],
            "exclusion_670_binding": exclusions["670"],
            "exclusion_672_binding": exclusions["672"],
            "selection_salt": spec["audit"]["selection_salt"],
            "selection_strategy": "sha256_rank_then_component_and_exact_source_unique_v1",
            "eligible_counts_by_stratum": eligible_counts,
            "selected_counts_by_stratum": {
                stratum: target for stratum in spec["audit"]["strata"]
            },
            "selected_bindings": bindings,
            "teacher_request_sha256": sha256_file(request_path),
            "teacher_request_rows": len(requests),
            "teacher_image_manifest_sha256": sha256_file(teacher_image_manifest_path),
            "teacher_image_manifest_rows": len(teacher_images),
            "teacher_visible_family_tokens": 0,
            "teacher_visible_outcome_fields": 0,
            "student_gpu_authorized": False,
            "self_sha256": None,
        }
    )
    write_json(output_dir / "selection_manifest.json", manifest)
    return manifest


def validate_selection_manifest(manifest: dict[str, Any], spec: dict[str, Any]) -> None:
    expect_exact_keys(
        manifest,
        {
            "schema_version",
            "experiment_id",
            "execution_scope",
            "source_contract_self_sha256",
            "source_rows_sha256",
            "source_prepare_acceptance_sha256",
            "source_prepare_acceptance_self_sha256",
            "exclusion_670_binding",
            "exclusion_672_binding",
            "selection_salt",
            "selection_strategy",
            "eligible_counts_by_stratum",
            "selected_counts_by_stratum",
            "selected_bindings",
            "teacher_request_sha256",
            "teacher_request_rows",
            "teacher_image_manifest_sha256",
            "teacher_image_manifest_rows",
            "teacher_visible_family_tokens",
            "teacher_visible_outcome_fields",
            "student_gpu_authorized",
            "self_sha256",
        },
        "selection manifest",
    )
    validate_self_hash(manifest, "selection manifest")
    if manifest["schema_version"] != "exp689_selection_manifest_v1":
        raise ContractError("selection manifest: schema version mismatch")
    if manifest["experiment_id"] != "689" or manifest["execution_scope"] != "remote_mlcore":
        raise ContractError("selection manifest: experiment or execution scope mismatch")
    require_hex64(
        manifest["source_prepare_acceptance_sha256"],
        "selection manifest.source_prepare_acceptance_sha256",
    )
    require_hex64(
        manifest["source_prepare_acceptance_self_sha256"],
        "selection manifest.source_prepare_acceptance_self_sha256",
    )
    if manifest["selection_salt"] != spec["audit"]["selection_salt"]:
        raise ContractError("selection manifest: selection salt mismatch")
    if manifest["selection_strategy"] != "sha256_rank_then_component_and_exact_source_unique_v1":
        raise ContractError("selection manifest: strategy mismatch")
    expected_exclusion_binding_keys = {
        "source_sha256",
        "adapter",
        "component_locator",
        "component_value_field",
        "token_mode",
        "component_count",
    }
    for experiment_id in ("670", "672"):
        binding = manifest[f"exclusion_{experiment_id}_binding"]
        if not isinstance(binding, dict) or set(binding) != expected_exclusion_binding_keys:
            raise ContractError(
                f"selection manifest: exclusion-{experiment_id} binding fields mismatch"
            )
        require_hex64(binding["source_sha256"], f"exclusion-{experiment_id} source SHA")
        expected_count = spec["exclusions"]["expected_component_counts"][experiment_id]
        if binding["component_count"] != expected_count:
            raise ContractError(
                f"selection manifest: exclusion-{experiment_id} component count mismatch"
            )
    expected_counts = {
        stratum: spec["audit"]["rows_per_stratum"] for stratum in spec["audit"]["strata"]
    }
    if manifest["selected_counts_by_stratum"] != expected_counts:
        raise ContractError("selection manifest: stratum counts mismatch")
    if manifest["teacher_request_rows"] != spec["audit"]["total_rows"]:
        raise ContractError("selection manifest: request row count mismatch")
    require_hex64(
        manifest["teacher_image_manifest_sha256"],
        "selection manifest.teacher_image_manifest_sha256",
    )
    if manifest["teacher_image_manifest_rows"] != spec["audit"]["total_rows"]:
        raise ContractError("selection manifest: teacher image row count mismatch")
    if manifest["teacher_visible_family_tokens"] != 0:
        raise ContractError("selection manifest: family tokens leaked to teacher")
    if manifest["teacher_visible_outcome_fields"] != 0:
        raise ContractError("selection manifest: outcome fields leaked to teacher")
    if manifest["student_gpu_authorized"] is not False:
        raise ContractError("selection manifest: must not authorize student GPU")
    bindings = manifest["selected_bindings"]
    if not isinstance(bindings, list) or len(bindings) != spec["audit"]["total_rows"]:
        raise ContractError("selection manifest: bindings must contain exactly 300 rows")
    expected_binding_keys = {
        "audit_id",
        "record_id",
        "row_token",
        "component_token",
        "family_token",
        "stratum",
        "source_card_sha256",
        "request_row_sha256",
    }
    seen_components: set[str] = set()
    seen_sources: set[str] = set()
    for index, binding in enumerate(bindings, 1):
        if not isinstance(binding, dict):
            raise ContractError(f"selection binding {index}: must be an object")
        expect_exact_keys(binding, expected_binding_keys, f"selection binding {index}")
        if binding["audit_id"] != f"G689-{index:03d}":
            raise ContractError(f"selection binding {index}: audit order mismatch")
        expected_stratum = spec["audit"]["strata"][(index - 1) // 100]
        if binding["stratum"] != expected_stratum:
            raise ContractError(f"selection binding {index}: stratum order mismatch")
        for field in (
            "row_token",
            "component_token",
            "family_token",
            "source_card_sha256",
            "request_row_sha256",
        ):
            require_hex64(binding[field], f"selection binding {index}.{field}")
        if binding["component_token"] in seen_components:
            raise ContractError("selection manifest: component duplication")
        if binding["source_card_sha256"] in seen_sources:
            raise ContractError("selection manifest: exact source duplication")
        seen_components.add(binding["component_token"])
        seen_sources.add(binding["source_card_sha256"])


def load_prepared(
    *, remote_root: Path, selection_manifest_path: Path, teacher_request_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = load_spec()
    selection_manifest_path = require_remote_path(
        remote_root, selection_manifest_path, context="selection manifest", must_exist=True
    )
    teacher_request_path = require_remote_path(
        remote_root, teacher_request_path, context="teacher request", must_exist=True
    )
    manifest = load_json(selection_manifest_path, "selection manifest")
    validate_selection_manifest(manifest, spec)
    if manifest["teacher_request_sha256"] != sha256_file(teacher_request_path):
        raise ContractError("teacher request: file SHA differs from selection manifest")
    requests = read_jsonl(teacher_request_path, "teacher request")
    if len(requests) != spec["audit"]["total_rows"]:
        raise ContractError("teacher request: expected exactly 300 rows")
    for index, (request, binding) in enumerate(
        zip(requests, manifest["selected_bindings"], strict=True), 1
    ):
        validate_teacher_request(request, index, spec)
        for field in (
            "audit_id",
            "record_id",
            "source_card_sha256",
            "request_row_sha256",
        ):
            if request[field] != binding[field]:
                raise ContractError(f"teacher request row {index}: binding mismatch for {field}")
    return manifest, requests


def verify_prepared_against_sources(
    *,
    remote_root: Path,
    source_rows_path: Path,
    source_contract_path: Path,
    exclusion_670_path: Path,
    exclusion_672_path: Path,
    manifest: dict[str, Any],
    requests: list[dict[str, Any]],
) -> None:
    """Rebuild the deterministic selection in memory for final provenance validation."""
    binding_670 = manifest["exclusion_670_binding"]
    binding_672 = manifest["exclusion_672_binding"]
    spec, rows, source_contract, exclusion_bindings, exclusion_sets = _prepare_inputs(
        remote_root=remote_root,
        source_rows_path=source_rows_path,
        source_contract_path=source_contract_path,
        exclusion_670_path=exclusion_670_path,
        exclusion_672_path=exclusion_672_path,
        exclusion_670_sha256=binding_670["source_sha256"],
        exclusion_672_sha256=binding_672["source_sha256"],
        exclusion_670_adapter=binding_670["adapter"],
        exclusion_672_adapter=binding_672["adapter"],
        exclusion_670_component_locator=binding_670["component_locator"],
        exclusion_672_component_locator=binding_672["component_locator"],
        exclusion_670_component_value_field=binding_670["component_value_field"],
        exclusion_672_component_value_field=binding_672["component_value_field"],
        exclusion_670_token_mode=binding_670["token_mode"],
        exclusion_672_token_mode=binding_672["token_mode"],
    )
    if source_contract["self_sha256"] != manifest["source_contract_self_sha256"]:
        raise ContractError("selection manifest: source contract lineage mismatch")
    if source_contract["source_rows_sha256"] != manifest["source_rows_sha256"]:
        raise ContractError("selection manifest: source rows lineage mismatch")
    if exclusion_bindings != {"670": binding_670, "672": binding_672}:
        raise ContractError("selection manifest: exclusion adapter lineage mismatch")
    selected, eligible_counts = select_source_rows(spec, rows, exclusion_sets)
    rebuilt_requests = [
        make_teacher_request(row, f"G689-{index:03d}")
        for index, row in enumerate(selected, 1)
    ]
    if rebuilt_requests != requests:
        raise ContractError("teacher request: deterministic source reconstruction mismatch")
    if make_bindings(selected, rebuilt_requests) != manifest["selected_bindings"]:
        raise ContractError("selection manifest: deterministic binding reconstruction mismatch")
    if eligible_counts != manifest["eligible_counts_by_stratum"]:
        raise ContractError("selection manifest: eligible counts reconstruction mismatch")


def validate_teacher_selection_contract(
    contract: dict[str, Any], selection_rows_path: Path, manifest: dict[str, Any]
) -> None:
    expect_exact_keys(
        contract,
        {
            "schema_version",
            "execution_scope",
            "teacher_request_sha256",
            "teacher_model_id",
            "teacher_model_revision",
            "prompt_sha256",
            "decoding_sha256",
            "inference_bundle_sha256",
            "job_metadata_sha256",
            "selection_rows_sha256",
            "selection_row_count",
            "output_policy",
            "free_text_fields",
            "forbidden_fields_present",
            "self_sha256",
        },
        "teacher selection contract",
    )
    validate_self_hash(contract, "teacher selection contract")
    if contract["schema_version"] != "exp689_teacher_selection_contract_v2":
        raise ContractError("teacher selection contract: schema version mismatch")
    if contract["execution_scope"] != "remote_mlcore":
        raise ContractError("teacher selection contract: execution scope mismatch")
    if contract["teacher_request_sha256"] != manifest["teacher_request_sha256"]:
        raise ContractError("teacher selection contract: request lineage mismatch")
    for field in ("teacher_model_id", "teacher_model_revision"):
        if not isinstance(contract[field], str) or not contract[field].strip():
            raise ContractError(f"teacher selection contract: {field} must be nonempty")
    for field in (
        "prompt_sha256",
        "decoding_sha256",
        "inference_bundle_sha256",
        "job_metadata_sha256",
    ):
        require_hex64(contract[field], f"teacher selection contract.{field}")
    if contract["selection_rows_sha256"] != sha256_file(selection_rows_path):
        raise ContractError("teacher selection contract: selection rows SHA mismatch")
    if contract["selection_row_count"] != 300:
        raise ContractError("teacher selection contract: row count must equal 300")
    if contract["output_policy"] != "closed_enums_and_separate_candidate_ids_only_v2":
        raise ContractError("teacher selection contract: output policy mismatch")
    if contract["free_text_fields"] != 0 or contract["forbidden_fields_present"] is not False:
        raise ContractError("teacher selection contract: forbidden or free-text output detected")


def validate_teacher_selection(
    row: dict[str, Any], request: dict[str, Any], spec: dict[str, Any], index: int
) -> None:
    context = f"teacher selection row {index}"
    expected_fields = set(spec["teacher_output_policy"]["allowed_generated_fields"])
    expect_exact_keys(row, expected_fields, context)
    if row["schema_version"] != "exp689_teacher_selection_row_v2":
        raise ContractError(f"{context}: schema version mismatch")
    for field in ("audit_id", "record_id", "request_row_sha256"):
        if row[field] != request[field]:
            raise ContractError(f"{context}: request binding mismatch for {field}")
    for field in ("sold_object", "substance", "relation", "support_status"):
        if row[field] not in spec["enums"][field]:
            raise ContractError(f"{context}: invalid {field}")
    known_candidate_ids = {
        candidate["candidate_id"] for candidate in request["evidence_candidates"]
    }
    evidence_lists: list[list[str]] = []
    for field in (
        "object_evidence_candidate_ids",
        "substance_evidence_candidate_ids",
        "relation_evidence_candidate_ids",
    ):
        candidate_ids = row[field]
        if not isinstance(candidate_ids, list) or any(
            not isinstance(item, str) for item in candidate_ids
        ):
            raise ContractError(f"{context}: {field} must be a string list")
        if candidate_ids != sorted(set(candidate_ids)):
            raise ContractError(f"{context}: {field} must be sorted and unique")
        if any(item not in known_candidate_ids for item in candidate_ids):
            raise ContractError(f"{context}: selected a nonexistent evidence candidate ID")
        evidence_lists.append(candidate_ids)
    if not isinstance(row["supervise"], bool):
        raise ContractError(f"{context}: supervise must be boolean")
    semantic_targets = (row["sold_object"], row["substance"], row["relation"])
    if row["support_status"] == "supported":
        if "unknown" in semantic_targets:
            raise ContractError(
                f"{context}: supported rows require all three semantic targets"
            )
        if row["supervise"] is not True or any(not values for values in evidence_lists):
            raise ContractError(
                f"{context}: supported rows require supervision and evidence for every target"
            )
    else:
        if semantic_targets != ("unknown", "unknown", "unknown"):
            raise ContractError(
                f"{context}: unsupported or ambiguous rows require unknown semantic targets"
            )
        if row["supervise"] is not False or any(evidence_lists):
            raise ContractError(
                f"{context}: unsupported or ambiguous rows must abstain with no evidence IDs"
            )


def load_teacher_selections(
    *,
    remote_root: Path,
    selection_rows_path: Path,
    selection_contract_path: Path,
    manifest: dict[str, Any],
    requests: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = load_spec()
    selection_rows_path = require_remote_path(
        remote_root, selection_rows_path, context="teacher selections", must_exist=True
    )
    selection_contract_path = require_remote_path(
        remote_root,
        selection_contract_path,
        context="teacher selection contract",
        must_exist=True,
    )
    contract = load_json(selection_contract_path, "teacher selection contract")
    validate_teacher_selection_contract(contract, selection_rows_path, manifest)
    rows = read_jsonl(selection_rows_path, "teacher selections")
    if len(rows) != len(requests):
        raise ContractError("teacher selections: row count differs from request")
    for index, (row, request) in enumerate(zip(rows, requests, strict=True), 1):
        validate_teacher_selection(row, request, spec, index)
    return contract, rows


def make_target_row(
    request: dict[str, Any], binding: dict[str, Any], selection: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "exp689_target_audit_row_v2",
        "audit_id": request["audit_id"],
        "record_id": request["record_id"],
        "stratum": binding["stratum"],
        "source_card_sha256": request["source_card_sha256"],
        "request_row_sha256": request["request_row_sha256"],
        "sources": request["sources"],
        "first_image": request["first_image"],
        "evidence_candidates": request["evidence_candidates"],
        "target": {
            "sold_object": selection["sold_object"],
            "substance": selection["substance"],
            "relation": selection["relation"],
            "object_evidence_candidate_ids": selection["object_evidence_candidate_ids"],
            "substance_evidence_candidate_ids": selection[
                "substance_evidence_candidate_ids"
            ],
            "relation_evidence_candidate_ids": selection["relation_evidence_candidate_ids"],
            "support_status": selection["support_status"],
            "supervise": selection["supervise"],
        },
    }


def materialize(
    *,
    remote_root: Path,
    selection_manifest_path: Path,
    teacher_request_path: Path,
    teacher_selections_path: Path,
    teacher_selection_contract_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Bind closed teacher selections and emit the frozen human-audit packet."""
    spec = load_spec()
    manifest, requests = load_prepared(
        remote_root=remote_root,
        selection_manifest_path=selection_manifest_path,
        teacher_request_path=teacher_request_path,
    )
    selection_contract, selections = load_teacher_selections(
        remote_root=remote_root,
        selection_rows_path=teacher_selections_path,
        selection_contract_path=teacher_selection_contract_path,
        manifest=manifest,
        requests=requests,
    )
    output_dir = require_remote_path(
        remote_root, output_dir, context="materialize output", must_exist=False
    )
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite immutable output directory: {output_dir}")
    rows = [
        make_target_row(request, binding, selection)
        for request, binding, selection in zip(
            requests, manifest["selected_bindings"], selections, strict=True
        )
    ]
    output_dir.mkdir(parents=True)
    packet_path = output_dir / "target_audit.jsonl"
    write_jsonl(packet_path, rows)
    packet_contract = with_self_hash(
        {
            "schema_version": "exp689_target_audit_contract_v2",
            "experiment_id": "689",
            "execution_scope": "remote_mlcore",
            "frozen_spec_sha256": sha256_file(SPEC_PATH),
            "target_audit_schema_sha256": sha256_file(SCHEMA_PATH),
            "selection_manifest_self_sha256": manifest["self_sha256"],
            "teacher_request_sha256": manifest["teacher_request_sha256"],
            "teacher_selection_contract_self_sha256": selection_contract["self_sha256"],
            "teacher_selections_sha256": selection_contract["selection_rows_sha256"],
            "target_audit_sha256": sha256_file(packet_path),
            "target_audit_rows": len(rows),
            "selected_counts_by_stratum": manifest["selected_counts_by_stratum"],
            "gates": spec["gates"],
            "embedded_review_fields": 0,
            "student_gpu_authorized": False,
            "self_sha256": None,
        }
    )
    write_json(output_dir / "target_audit_contract.json", packet_contract)
    return packet_contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="select rows and emit teacher request")
    prepare_parser.add_argument("--remote-root", type=Path, required=True)
    prepare_parser.add_argument("--source-rows", type=Path, required=True)
    prepare_parser.add_argument("--source-contract", type=Path, required=True)
    prepare_parser.add_argument("--source-prepare-acceptance", type=Path, required=True)
    prepare_parser.add_argument("--source-prepare-acceptance-sha256", required=True)
    prepare_parser.add_argument("--exclusion-670", type=Path, required=True)
    prepare_parser.add_argument("--exclusion-672", type=Path, required=True)
    prepare_parser.add_argument("--exclusion-670-sha256", required=True)
    prepare_parser.add_argument("--exclusion-672-sha256", required=True)
    adapter_choices = ("canonical_json", "csv_component_column", "json_component_list")
    token_choices = ("already_sha256", "sha256_utf8_v1")
    prepare_parser.add_argument(
        "--exclusion-670-adapter", choices=adapter_choices, default="canonical_json"
    )
    prepare_parser.add_argument(
        "--exclusion-672-adapter", choices=adapter_choices, default="canonical_json"
    )
    prepare_parser.add_argument(
        "--exclusion-670-component-locator", default="component_tokens"
    )
    prepare_parser.add_argument(
        "--exclusion-672-component-locator", default="component_tokens"
    )
    prepare_parser.add_argument("--exclusion-670-component-value-field")
    prepare_parser.add_argument("--exclusion-672-component-value-field")
    prepare_parser.add_argument(
        "--exclusion-670-token-mode", choices=token_choices, default="already_sha256"
    )
    prepare_parser.add_argument(
        "--exclusion-672-token-mode", choices=token_choices, default="already_sha256"
    )
    prepare_parser.add_argument("--output-dir", type=Path, required=True)

    materialize_parser = subparsers.add_parser(
        "materialize", help="bind teacher selections and emit frozen audit"
    )
    materialize_parser.add_argument("--remote-root", type=Path, required=True)
    materialize_parser.add_argument("--selection-manifest", type=Path, required=True)
    materialize_parser.add_argument("--teacher-request", type=Path, required=True)
    materialize_parser.add_argument("--teacher-selections", type=Path, required=True)
    materialize_parser.add_argument("--teacher-selection-contract", type=Path, required=True)
    materialize_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        result = prepare(
            remote_root=args.remote_root,
            source_rows_path=args.source_rows,
            source_contract_path=args.source_contract,
            source_prepare_acceptance_path=args.source_prepare_acceptance,
            source_prepare_acceptance_sha256=args.source_prepare_acceptance_sha256,
            exclusion_670_path=args.exclusion_670,
            exclusion_672_path=args.exclusion_672,
            exclusion_670_sha256=args.exclusion_670_sha256,
            exclusion_672_sha256=args.exclusion_672_sha256,
            exclusion_670_adapter=args.exclusion_670_adapter,
            exclusion_672_adapter=args.exclusion_672_adapter,
            exclusion_670_component_locator=args.exclusion_670_component_locator,
            exclusion_672_component_locator=args.exclusion_672_component_locator,
            exclusion_670_component_value_field=args.exclusion_670_component_value_field,
            exclusion_672_component_value_field=args.exclusion_672_component_value_field,
            exclusion_670_token_mode=args.exclusion_670_token_mode,
            exclusion_672_token_mode=args.exclusion_672_token_mode,
            output_dir=args.output_dir,
        )
    else:
        result = materialize(
            remote_root=args.remote_root,
            selection_manifest_path=args.selection_manifest,
            teacher_request_path=args.teacher_request,
            teacher_selections_path=args.teacher_selections,
            teacher_selection_contract_path=args.teacher_selection_contract,
            output_dir=args.output_dir,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
