"""Build the label-free, image-grounded exp689 source universe on remote CPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import tempfile
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

EXPERIMENT = Path(__file__).resolve().parent
SPEC_PATH = EXPERIMENT / "source_prepare_spec_v1.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
CLAUSE_SEPARATOR = re.compile(r"(?:[.!?;]+|\n+)")

FUEL = re.compile(
    r"\b(?:газ|газов|топлив|бензин|керосин|пропан|бутан|изобутан|уголь|"
    r"розжиг|жидкост\w*\s+для\s+розжига|баллон|картридж|капсул|горюч)\w*\b",
    re.IGNORECASE,
)
DEVICE = re.compile(
    r"\b(?:горелк|плит|ламп|обогревател|грил|печ|зажигал|резак|паяльн|"
    r"генератор|котел|примус|камин|фонар)\w*\b",
    re.IGNORECASE,
)
ACCESSORY = re.compile(
    r"\b(?:адаптер|переходник|шланг|насадк|чехол|держател|креплен|"
    r"аксессуар|комплектующ|совместим)\w*\b",
    re.IGNORECASE,
)
RELATION = re.compile(
    r"\b(?:для|к)\b|\b(?:подход|совместим|использу|заправ|подключ|"
    r"работа|комплект|включен|прилага)\w*\b",
    re.IGNORECASE,
)
EXTERNAL_OR_NEGATED = re.compile(
    r"\bбез\b|\bне\s+входит\b|\bотдельно\b|\bприобретается\s+отдельно\b|"
    r"\bпуст\w*\b|\bнезаправлен\w*\b",
    re.IGNORECASE,
)


class PrepareError(ValueError):
    """Raised when a frozen PREPARE invariant is violated."""


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


def with_self_hash(payload: dict[str, Any], field: str = "self_sha256") -> dict[str, Any]:
    result = dict(payload)
    result[field] = None
    result[field] = sha256_bytes(canonical_json_bytes(result))
    return result


def validate_self_hash(payload: dict[str, Any], context: str, field: str) -> str:
    actual = payload.get(field)
    if not isinstance(actual, str) or not HEX64.fullmatch(actual):
        raise PrepareError(f"{context}: missing canonical {field}")
    copy = dict(payload)
    copy[field] = None
    expected = sha256_bytes(canonical_json_bytes(copy))
    if actual != expected:
        raise PrepareError(f"{context}: canonical {field} mismatch")
    return actual


def validate_legacy_contract(payload: dict[str, Any], context: str) -> str:
    """Validate the 641/680 audit convention (hash before adding the field)."""
    actual = payload.get("contract_sha256")
    if not isinstance(actual, str) or not HEX64.fullmatch(actual):
        raise PrepareError(f"{context}: missing canonical contract_sha256")
    copy = dict(payload)
    copy.pop("contract_sha256", None)
    if actual != sha256_bytes(canonical_json_bytes(copy)):
        raise PrepareError(f"{context}: canonical contract_sha256 mismatch")
    return actual


def _load_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PrepareError(f"{context}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise PrepareError(f"{context}: top level must be an object")
    return value


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    spec = _load_json(path, "source prepare spec")
    validate_self_hash(spec, "source prepare spec", "self_sha256")
    if spec.get("schema_version") != "exp689_source_prepare_spec_v1":
        raise PrepareError("source prepare spec: schema version mismatch")
    if spec.get("execution_scope") != "remote_cpu_only":
        raise PrepareError("source prepare spec: execution scope mismatch")
    if spec.get("ocr_policy") != "disabled_unverified":
        raise PrepareError("source prepare spec: OCR must remain disabled-unverified")
    return spec


def _read_jsonl(path: Path, context: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    raise PrepareError(f"{context}: blank line {line_number}")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise PrepareError(f"{context}: row {line_number} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise PrepareError(f"{context}: cannot read JSONL: {error}") from error
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("wb") as stream:
        for row in rows:
            stream.write(canonical_json_bytes(row) + b"\n")


def _normalise(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", str(value or "")).lower().replace("ё", "е"),
    ).strip()


def _reject_forbidden_fields(value: Any, forbidden: set[str], context: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalised = _normalise(key).replace("-", "_")
            if normalised in forbidden or any(
                normalised.startswith(f"{token}_") or normalised.endswith(f"_{token}")
                for token in forbidden
            ):
                raise PrepareError(f"{context}: forbidden field {key!r}")
            _reject_forbidden_fields(item, forbidden, context)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_fields(item, forbidden, context)


def _validate_runtime_audit(
    audit: dict[str, Any], fold: int, expected: dict[str, Any], spec: dict[str, Any]
) -> str:
    contract = validate_legacy_contract(audit, f"fold {fold} runtime audit")
    source_contract = audit.get("source_runtime_contract_sha256", contract)
    if source_contract != expected["source_runtime_contract_sha256"]:
        raise PrepareError(f"fold {fold}: frozen source runtime contract mismatch")
    if audit.get("experiment_id") != "641" or audit.get("outer_fold") != fold:
        raise PrepareError(f"fold {fold}: wrong experiment/fold runtime")
    inputs = audit.get("input_sha256")
    if not isinstance(inputs, dict):
        raise PrepareError(f"fold {fold}: runtime audit lacks input_sha256")
    if inputs.get("data") != spec["input_data_sha256"]:
        raise PrepareError(f"fold {fold}: source data SHA mismatch")
    if inputs.get("folds") != spec["input_registry_sha256"]:
        raise PrepareError(f"fold {fold}: source registry SHA mismatch")
    outputs = audit.get("output_sha256")
    if not isinstance(outputs, dict) or outputs.get("validation.jsonl") != expected["validation_sha256"]:
        raise PrepareError(f"fold {fold}: audit validation binding mismatch")
    return contract


def _load_fold(runtime_dir: Path, fold: int, spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if _normalise(runtime_dir.name) in {"train", "train.jsonl"}:
        raise PrepareError(f"fold {fold}: train paths are forbidden")
    if not runtime_dir.is_dir():
        raise PrepareError(f"fold {fold}: runtime input must be a directory")
    validation_path = runtime_dir / "validation.jsonl"
    audit_path = runtime_dir / "runtime_audit.json"
    expected = spec["folds"][str(fold)]
    audit = _load_json(audit_path, f"fold {fold} runtime audit")
    audit_contract = _validate_runtime_audit(audit, fold, expected, spec)
    validation_sha = sha256_file(validation_path)
    if validation_sha != expected["validation_sha256"]:
        raise PrepareError(f"fold {fold}: frozen validation SHA mismatch")
    rows = _read_jsonl(validation_path, f"fold {fold} validation")
    if len(rows) != expected["validation_rows"]:
        raise PrepareError(f"fold {fold}: frozen validation row count mismatch")
    forbidden = set(spec["forbidden_field_tokens"])
    allowed = set(spec["validation_fields"])
    filtered: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        _reject_forbidden_fields(row, forbidden, f"fold {fold} row {row_index}")
        if set(row) != allowed:
            raise PrepareError(f"fold {fold} row {row_index}: exact validation schema mismatch")
        if row["category"] != spec["category"]:
            continue
        if int(row["fold"]) != fold:
            raise PrepareError(f"fold {fold} row {row_index}: fold mismatch")
        if not str(row["id"]) or not str(row["semantic_component"]):
            raise PrepareError(f"fold {fold} row {row_index}: empty ID/component")
        if not str(row["image_url"]).strip():
            raise PrepareError(f"fold {fold} row {row_index}: first image is absent")
        filtered.append(row)
    if len(filtered) != len(rows):
        raise PrepareError(f"fold {fold}: frozen validation must already be flammable-only")
    return filtered, {
        "fold": fold,
        "runtime_audit_sha256": sha256_file(audit_path),
        "runtime_contract_sha256": audit_contract,
        "source_runtime_contract_sha256": expected["source_runtime_contract_sha256"],
        "validation_sha256": validation_sha,
        "validation_rows": len(rows),
    }


def _locate_json(value: Any, locator: str) -> Any:
    current = value
    for part in locator.split("."):
        if not isinstance(current, dict) or part not in current:
            raise PrepareError(f"exp672: missing JSON locator {locator}")
        current = current[part]
    return current


def _load_exclusions(
    path_670: Path,
    path_672: Path,
    expected_sha_670: str,
    expected_sha_672: str,
    spec: dict[str, Any],
) -> tuple[set[str], dict[str, Any]]:
    for name, path, expected_sha in (
        ("exp670", path_670, expected_sha_670),
        ("exp672", path_672, expected_sha_672),
    ):
        if not HEX64.fullmatch(expected_sha):
            raise PrepareError(f"{name}: expected input SHA must be lowercase SHA-256")
        if sha256_file(path) != expected_sha:
            raise PrepareError(f"{name}: frozen input SHA mismatch")
    cfg670 = spec["exclusions"]["exp670"]
    with path_670.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or cfg670["component_column"] not in reader.fieldnames:
            raise PrepareError("exp670: semantic_component column is absent")
        values_670 = [str(row[cfg670["component_column"]]).strip() for row in reader]
    cfg672 = spec["exclusions"]["exp672"]
    values_672 = _locate_json(_load_json(path_672, "exp672 exclusion"), cfg672["component_locator"])
    if not isinstance(values_672, list) or any(not isinstance(value, str) for value in values_672):
        raise PrepareError("exp672: ordered_components must be a string list")
    values_672 = [value.strip() for value in values_672]
    if any(not value for value in [*values_670, *values_672]):
        raise PrepareError("exclusions: empty component")
    sets = {"exp670": set(values_670), "exp672": set(values_672)}
    for name, values in (("exp670", values_670), ("exp672", values_672)):
        if len(values) != len(sets[name]) or len(values) != spec["exclusions"][name]["expected_count"]:
            raise PrepareError(f"{name}: exact unique component count mismatch")
    intersection = sets["exp670"] & sets["exp672"]
    union = sets["exp670"] | sets["exp672"]
    if len(intersection) != spec["exclusions"]["expected_intersection_count"]:
        raise PrepareError("exclusions: intersection count mismatch")
    if len(union) != spec["exclusions"]["expected_union_count"]:
        raise PrepareError("exclusions: union count mismatch")
    return union, {
        "exp670_sha256": expected_sha_670,
        "exp672_sha256": expected_sha_672,
        "exp670_count": len(sets["exp670"]),
        "exp672_count": len(sets["exp672"]),
        "intersection_count": len(intersection),
        "union_count": len(union),
        "union_tokens": sorted(sha256_text(value) for value in union),
    }


def _cue_flags(row: dict[str, Any]) -> dict[str, bool]:
    name = _normalise(row["name"])
    text = _normalise(f"{row['name']} {row['description']}")
    return {
        "fuel_name": bool(FUEL.search(name)),
        "fuel_any": bool(FUEL.search(text)),
        "device_any": bool(DEVICE.search(text)),
        "accessory_any": bool(ACCESSORY.search(text)),
        "relation_any": bool(RELATION.search(text)),
        "external_or_negated": bool(EXTERNAL_OR_NEGATED.search(text)),
    }


def _stable_key(namespace: str, row: dict[str, Any]) -> str:
    return sha256_text(f"{namespace}\0{row['semantic_component']}\0{row['id']}")


def _select_rows(rows: list[dict[str, Any]], exclusions: set[str], spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    component_fold: dict[str, int] = {}
    component_ids: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        row_id = str(row["id"])
        component = str(row["semantic_component"])
        fold = int(row["fold"])
        if row_id in by_id:
            raise PrepareError(f"runtime union: duplicate ID {row_id!r}")
        by_id[row_id] = row
        if component in component_fold and component_fold[component] != fold:
            raise PrepareError(f"runtime union: component {component!r} crosses folds")
        component_fold[component] = fold
        component_ids[component].add(row_id)
    representatives: list[dict[str, Any]] = []
    for component, ids in component_ids.items():
        if component in exclusions:
            continue
        candidates = [by_id[row_id] for row_id in ids]
        representative = min(candidates, key=lambda row: _stable_key("representative", row))
        copy = dict(representative)
        copy["_component_size"] = len(ids)
        copy["_cues"] = _cue_flags(copy)
        cues = copy["_cues"]
        if (cues["device_any"] or cues["accessory_any"]) and cues["fuel_any"]:
            copy["_stratum"] = "device_accessory_compatible_mention"
        elif cues["fuel_name"] and not (cues["device_any"] or cues["accessory_any"] or cues["external_or_negated"]):
            copy["_stratum"] = "direct_included_fuel"
        elif copy["_component_size"] == 1:
            copy["_stratum"] = "singleton_new_family_ambiguous"
        else:
            copy["_stratum"] = None
        representatives.append(copy)

    quota = int(spec["quota_per_fold_per_stratum"])
    cue_free_quota = int(spec["singleton_cue_free_per_fold"])
    selected: list[dict[str, Any]] = []
    ordered_strata = [item["name"] for item in spec["strata"]]
    for stratum in ordered_strata:
        for fold in range(5):
            candidates = [
                row for row in representatives
                if row["_stratum"] == stratum and int(row["fold"]) == fold
            ]
            if stratum == "singleton_new_family_ambiguous":
                cue_free = [
                    row for row in candidates
                    if not any(row["_cues"].values())
                ]
                other = [row for row in candidates if row not in cue_free]
                cue_free.sort(key=lambda row: _stable_key(f"{stratum}:cue-free", row))
                other.sort(key=lambda row: _stable_key(f"{stratum}:other", row))
                chosen = cue_free[:cue_free_quota] + other[: quota - cue_free_quota]
                if len(cue_free) < cue_free_quota or len(other) < quota - cue_free_quota:
                    raise PrepareError(f"fold {fold} {stratum}: insufficient 10+10 candidates")
            else:
                candidates.sort(key=lambda row: _stable_key(stratum, row))
                chosen = candidates[:quota]
                if len(chosen) != quota:
                    raise PrepareError(f"fold {fold} {stratum}: insufficient candidates")
            selected.extend(chosen)
    if len({str(row["semantic_component"]) for row in selected}) != len(selected):
        raise PrepareError("selected components are not disjoint")
    counts = Counter((row["_stratum"], int(row["fold"])) for row in selected)
    if set(counts.values()) != {quota} or len(counts) != 15:
        raise PrepareError("selected quota matrix is incomplete")
    report_rows = [
        {
            "row_token": sha256_text(str(row["id"])),
            "component_token": sha256_text(str(row["semantic_component"])),
            "fold": int(row["fold"]),
            "stratum": row["_stratum"],
            "component_size": row["_component_size"],
            "cue_flags": row["_cues"],
        }
        for row in selected
    ]
    return selected, {
        "union_row_count": len(rows),
        "union_component_count": len(component_ids),
        "eligible_component_count": len(representatives),
        "selected": report_rows,
    }


def _rgb_sha(image: Image.Image) -> str:
    if image.mode != "RGB":
        raise PrepareError("pixel hash requires RGB")
    return sha256_bytes(image.tobytes())


def _resize_rgb(image: Image.Image, area_cap: int) -> Image.Image:
    image = image.convert("RGB")
    if image.width * image.height <= area_cap:
        return image
    scale = math.sqrt(area_cap / (image.width * image.height))
    width = max(1, math.floor(image.width * scale))
    height = max(1, math.floor(image.height * scale))
    while width * height > area_cap:
        if width >= height:
            width -= 1
        else:
            height -= 1
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _default_fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "exp689-source-prepare/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read((32 << 20) + 1)
    if len(payload) > 32 << 20:
        raise PrepareError("first image exceeds 32 MiB")
    return payload


def _image_regions(image: Image.Image) -> dict[str, str]:
    middle_x = image.width // 2
    middle_y = image.height // 2
    boxes = {
        "full": (0, 0, image.width, image.height),
        "q00": (0, 0, max(1, middle_x), max(1, middle_y)),
        "q01": (middle_x, 0, middle_x + max(1, middle_x), max(1, middle_y)),
        "q10": (0, middle_y, max(1, middle_x), middle_y + max(1, middle_y)),
        "q11": (
            middle_x,
            middle_y,
            middle_x + max(1, middle_x),
            middle_y + max(1, middle_y),
        ),
    }
    result: dict[str, str] = {}
    for region, box in boxes.items():
        crop = image if region == "full" else image.crop(box)
        result[region] = _rgb_sha(crop)
    return result


def _description_clauses(text: str, limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in CLAUSE_SEPARATOR.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    trimmed: list[tuple[int, int]] = []
    for start, end in spans:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            trimmed.append((start, end))
    ranked = sorted(
        trimmed,
        key=lambda span: (
            0 if any(pattern.search(_normalise(text[slice(*span)])) for pattern in (FUEL, DEVICE, ACCESSORY, RELATION)) else 1,
            sha256_text(text[slice(*span)]),
            span,
        ),
    )[:limit]
    return sorted(ranked)


def _candidate(fields: dict[str, Any]) -> dict[str, Any]:
    result = {"candidate_id": None, **fields}
    result["candidate_id"] = sha256_bytes(canonical_json_bytes(result))
    return result


def _build_source_row(
    row: dict[str, Any],
    record_id: str,
    image_reference: str,
    image: Image.Image,
    transformed_sha: str,
    regions: dict[str, str],
    spec: dict[str, Any],
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    name = str(row["name"])
    description = str(row["description"])
    if not name.strip():
        raise PrepareError(f"{record_id}: name must be nonempty")
    for kind, text in (("name", name), ("description", description)):
        if text:
            sources.append(
                {
                    "source_index": len(sources),
                    "source_kind": kind,
                    "text": text,
                    "text_sha256": sha256_text(text),
                }
            )
    first_image = {
        "reference": image_reference,
        "content_sha256": transformed_sha,
        "decoded_rgb_sha256": regions["full"],
        "media_type": "image/jpeg",
        "width": image.width,
        "height": image.height,
    }
    candidates: list[dict[str, Any]] = []
    name_source = next(source for source in sources if source["source_kind"] == "name")
    candidates.append(
        _candidate(
            {
                "candidate_index": 0,
                "evidence_kind": "text_span",
                "source_index": name_source["source_index"],
                "char_start": 0,
                "char_end": len(name),
                "image_region": None,
                "evidence_sha256": sha256_text(name),
            }
        )
    )
    if description:
        description_source = next(source for source in sources if source["source_kind"] == "description")
        for start, end in _description_clauses(description, int(spec["description_clause_limit"])):
            candidates.append(
                _candidate(
                    {
                        "candidate_index": len(candidates),
                        "evidence_kind": "text_span",
                        "source_index": description_source["source_index"],
                        "char_start": start,
                        "char_end": end,
                        "image_region": None,
                        "evidence_sha256": sha256_text(description[start:end]),
                    }
                )
            )
    for region in spec["image_transform"]["regions"]:
        candidates.append(
            _candidate(
                {
                    "candidate_index": len(candidates),
                    "evidence_kind": "image_region",
                    "source_index": None,
                    "char_start": None,
                    "char_end": None,
                    "image_region": region,
                    "evidence_sha256": regions[region],
                }
            )
        )
    source_card = {"sources": sources, "first_image": first_image}
    return {
        "schema_version": "exp689_source_row_v1",
        "record_id": record_id,
        "row_token": sha256_text(str(row["id"])),
        "component_token": sha256_text(str(row["semantic_component"])),
        "family_token": sha256_text(f"component-surrogate\0{row['semantic_component']}"),
        "stratum": row["_stratum"],
        "source_card_sha256": sha256_bytes(canonical_json_bytes(source_card)),
        "sources": sources,
        "first_image": first_image,
        "evidence_candidates": candidates,
    }


def prepare(
    *,
    runtime_dirs: list[Path],
    exclusion_670_path: Path,
    exclusion_672_path: Path,
    exclusion_670_sha256: str,
    exclusion_672_sha256: str,
    builder_revision: str,
    output_dir: Path,
    image_fetcher: Callable[[str], bytes] | None = None,
    spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite an existing output path")
    if len(runtime_dirs) != 5:
        raise PrepareError("exactly five --runtime-dir inputs are required")
    if not HEX40.fullmatch(builder_revision):
        raise PrepareError("builder revision must be an exact lowercase Git commit")
    spec = load_spec() if spec is None else spec
    runtime_rows: list[dict[str, Any]] = []
    runtime_bindings: list[dict[str, Any]] = []
    seen_dirs: set[Path] = set()
    for fold, runtime_dir in enumerate(runtime_dirs):
        resolved = runtime_dir.resolve(strict=True)
        if resolved in seen_dirs:
            raise PrepareError("runtime directories must be unique")
        seen_dirs.add(resolved)
        rows, binding = _load_fold(resolved, fold, spec)
        runtime_rows.extend(rows)
        runtime_bindings.append(binding)
    exclusions, exclusion_report = _load_exclusions(
        exclusion_670_path,
        exclusion_672_path,
        exclusion_670_sha256,
        exclusion_672_sha256,
        spec,
    )
    selected, selection_report = _select_rows(runtime_rows, exclusions, spec)
    fetch = image_fetcher or _default_fetch
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        images_dir = temporary / "images"
        images_dir.mkdir()
        source_rows: list[dict[str, Any]] = []
        image_manifest: list[dict[str, Any]] = []
        for index, row in enumerate(selected, 1):
            record_id = f"G689-{index:03d}"
            url = str(row["image_url"])
            try:
                original = fetch(url)
            except Exception as error:
                raise PrepareError(f"{record_id}: first-image fetch failed: {error}") from error
            if not isinstance(original, bytes) or not original:
                raise PrepareError(f"{record_id}: first-image fetch returned no bytes")
            try:
                with Image.open(io.BytesIO(original)) as decoded:
                    decoded.load()
                    resized = _resize_rgb(decoded, int(spec["image_transform"]["area_cap"]))
            except (OSError, UnidentifiedImageError, ValueError) as error:
                raise PrepareError(f"{record_id}: first-image decode failed: {error}") from error
            pixel_sha = _rgb_sha(resized)
            encoded = io.BytesIO()
            resized.save(
                encoded,
                format="JPEG",
                quality=int(spec["image_transform"]["quality"]),
                subsampling=int(spec["image_transform"]["subsampling"]),
                optimize=bool(spec["image_transform"]["optimize"]),
                progressive=bool(spec["image_transform"]["progressive"]),
            )
            transformed = encoded.getvalue()
            transformed_sha = sha256_bytes(transformed)
            image_path = images_dir / f"{record_id}.jpg"
            image_path.write_bytes(transformed)
            with Image.open(io.BytesIO(transformed)) as decoded_transform:
                decoded_transform.load()
                teacher_rgb = decoded_transform.convert("RGB")
            regions = _image_regions(teacher_rgb)
            reference = f"images/{record_id}.jpg"
            source_rows.append(
                _build_source_row(
                    row,
                    record_id,
                    reference,
                    teacher_rgb,
                    transformed_sha,
                    regions,
                    spec,
                )
            )
            image_manifest.append(
                {
                    "record_id": record_id,
                    "row_token": sha256_text(str(row["id"])),
                    "component_token": sha256_text(str(row["semantic_component"])),
                    "fold": int(row["fold"]),
                    "stratum": row["_stratum"],
                    "reference": reference,
                    "original_bytes_sha256": sha256_bytes(original),
                    "resized_rgb_sha256": pixel_sha,
                    "transformed_jpeg_sha256": transformed_sha,
                    "transformed_rgb_sha256": regions["full"],
                    "width": teacher_rgb.width,
                    "height": teacher_rgb.height,
                    "region_sha256": regions,
                }
            )
        source_rows_path = temporary / "source_rows.jsonl"
        image_manifest_path = images_dir / "image_manifest.jsonl"
        _write_jsonl(source_rows_path, source_rows)
        _write_jsonl(image_manifest_path, image_manifest)
        runtime_sha = sha256_bytes(canonical_json_bytes(runtime_bindings))
        eligibility_sha = sha256_bytes(
            canonical_json_bytes(
                {
                    "exclusions": exclusion_report,
                    "union_row_count": selection_report["union_row_count"],
                    "union_component_count": selection_report["union_component_count"],
                    "eligible_component_count": selection_report["eligible_component_count"],
                }
            )
        )
        stratum_sha = sha256_bytes(canonical_json_bytes(selection_report["selected"]))
        membership_sha = sha256_bytes(
            canonical_json_bytes(
                [
                    {
                        "record_id": item["record_id"],
                        "row_token": item["row_token"],
                        "reference": item["reference"],
                        "original_bytes_sha256": item["original_bytes_sha256"],
                    }
                    for item in image_manifest
                ]
            )
        )
        image_transform_sha = sha256_bytes(
            canonical_json_bytes(
                {
                    "rules": spec["image_transform"],
                    "image_manifest_sha256": sha256_file(image_manifest_path),
                    "images": image_manifest,
                }
            )
        )
        candidate_sha = sha256_bytes(
            canonical_json_bytes(
                {
                    "description_clause_limit": spec["description_clause_limit"],
                    "ocr_policy": spec["ocr_policy"],
                    "source_cards": [row["source_card_sha256"] for row in source_rows],
                    "candidate_ids": [
                        candidate["candidate_id"]
                        for row in source_rows
                        for candidate in row["evidence_candidates"]
                    ],
                }
            )
        )
        contract = with_self_hash(
            {
                "schema_version": "exp689_source_contract_v2",
                "execution_scope": "remote_remote_compute",
                "source_rows_sha256": sha256_file(source_rows_path),
                "source_row_count": len(source_rows),
                "runtime_sha256": runtime_sha,
                "data_sha256": spec["input_data_sha256"],
                "registry_sha256": spec["input_registry_sha256"],
                "builder_revision_sha256": sha256_text(builder_revision),
                "eligibility_universe_sha256": eligibility_sha,
                "stratum_derivation_sha256": stratum_sha,
                "image_membership_sha256": membership_sha,
                "image_transform_sha256": image_transform_sha,
                "candidate_generator_sha256": candidate_sha,
                "opaque_token_scheme": spec["opaque_token_scheme"],
                "label_fields_present": False,
                "score_fields_present": False,
                "first_image_rows_verified": len(source_rows),
                "image_fetch_failures": 0,
                "image_decode_failures": 0,
                "image_hash_mismatches": 0,
                "sealed_rows": 0,
                "public_rows": 0,
                "self_sha256": None,
            }
        )
        _write_json(temporary / "source_contract.json", contract)
        temporary.rename(output_dir)
        return contract
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, action="append", required=True)
    parser.add_argument("--exclusion-670", type=Path, required=True)
    parser.add_argument("--exclusion-672", type=Path, required=True)
    parser.add_argument("--exclusion-670-sha256", required=True)
    parser.add_argument("--exclusion-672-sha256", required=True)
    parser.add_argument("--builder-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    contract = prepare(
        runtime_dirs=args.runtime_dir,
        exclusion_670_path=args.exclusion_670,
        exclusion_672_path=args.exclusion_672,
        exclusion_670_sha256=args.exclusion_670_sha256,
        exclusion_672_sha256=args.exclusion_672_sha256,
        builder_revision=args.builder_revision,
        output_dir=args.output_dir,
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
