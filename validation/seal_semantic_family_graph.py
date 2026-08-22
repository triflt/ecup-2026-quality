from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

GRAPH_VERSION = "semantic_family_graph_audit_draft_v3"
AUDIT_VERSION = "semantic_family_graph_manual_audit_v3"
SEALED_VERSION = "semantic_family_v3"
EXPECTED_CRITERIA = {
    "accepted_risky_precision",
    "largest_component_boundary_pass_rate",
    "all_known_false_replays_blocked",
}
EXPECTED_DRAFT_FILES = {
    "rows": "rows.csv",
    "edges": "edges.csv",
    "key_degrees": "key_degrees.csv",
    "incremental": "incremental_components.json",
    "manual_samples": "manual_audit_samples.csv",
    "fingerprints": "image_fingerprints.csv.gz",
}
EXPECTED_ROW_COLUMNS = (
    "id",
    "category",
    "label",
    "semantic_component",
    "component_size",
    "draft_partition_7",
    "draft_holdout_candidate_not_sealed",
    "draft_dev_fold_5",
)
OUTPUT_COLUMNS = (
    "id",
    "category",
    "label",
    "semantic_component",
    "component_size",
    "split",
    "development_fold",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    for attempt in range(4):
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        except PermissionError:
            if attempt == 3:
                break
            time.sleep(0.25 * (attempt + 1))
    completed = subprocess.run(
        ["shasum", "-a", "256", "--", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout
    if not output.endswith("\n") or output.count("\n") != 1:
        raise RuntimeError("invalid shasum output: expected exactly one line")
    digest, separator, reported_path = output[:-1].partition("  ")
    if separator != "  " or HEX64.fullmatch(digest.lower()) is None or reported_path != str(path):
        raise RuntimeError("invalid shasum output: digest/path mismatch")
    return digest.lower()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _strict_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid strict boolean {value!r}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_draft(draft_dir: Path, audit_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = draft_dir / "manifest.json"
    _require(manifest_path.is_file(), "draft manifest.json is missing")
    _require(audit_path.is_file(), "manual audit is missing")
    manifest = _load_json(manifest_path)
    audit = _load_json(audit_path)

    _require(manifest.get("graph_version") == GRAPH_VERSION, "draft graph version mismatch")
    _require(manifest.get("status") == "draft_requires_manual_audit", "draft status mismatch")
    _require(manifest.get("draft") is True, "source is not marked as draft")
    _require(manifest.get("requires_manual_audit") is True, "draft manual-audit gate is absent")
    _require(manifest.get("sealed") is False, "source draft is already marked sealed")
    _require(
        manifest.get("valid_for_candidate_scoring") is False,
        "source draft unexpectedly permits candidate scoring",
    )
    topology = manifest.get("topology", {})
    _require(topology.get("label_blind") is True, "draft topology is not label-blind")
    _require(topology.get("candidate_scores_used") is False, "draft topology used candidate scores")

    _require(audit.get("audit_version") == AUDIT_VERSION, "manual audit version mismatch")
    _require(audit.get("graph_candidate") == GRAPH_VERSION, "audit graph candidate mismatch")
    _require(audit.get("status") == "complete", "manual audit is incomplete")
    _require(audit.get("decision") == "GO", "manual audit decision is not GO")
    _require(audit.get("seal_candidate") is True, "manual audit did not authorize sealing")
    criteria = audit.get("criterion_checks")
    _require(isinstance(criteria, dict), "manual audit criterion_checks are missing")
    _require(set(criteria) == EXPECTED_CRITERIA, "manual audit criterion set mismatch")
    _require(all(value is True for value in criteria.values()), "not every seal criterion passed")
    protocol = audit.get("audit_protocol", {})
    _require(protocol.get("label_blind") is True, "manual audit is not label-blind")
    _require(
        protocol.get("target_annotations_inspected") is False,
        "manual audit inspected target annotations",
    )
    _require(
        protocol.get("model_outputs_inspected") is False, "manual audit inspected model outputs"
    )
    _require(
        protocol.get("data_partitions_inspected") is False,
        "manual audit inspected data partitions",
    )

    expected_hashes = manifest.get("output_sha256")
    _require(isinstance(expected_hashes, dict), "draft output SHA map is missing")
    _require(set(expected_hashes) == set(EXPECTED_DRAFT_FILES), "draft output SHA keys mismatch")
    for key, filename in EXPECTED_DRAFT_FILES.items():
        path = draft_dir / filename
        _require(path.is_file(), f"draft artifact is missing: {filename}")
        _require(
            file_sha256(path) == expected_hashes[key], f"draft artifact SHA mismatch: {filename}"
        )
    return manifest, audit


def validate_and_render_rows(
    rows_path: Path, manifest: dict[str, Any]
) -> tuple[bytes, dict[str, Any]]:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    seen_ids: set[str] = set()
    component_rows: Counter[str] = Counter()
    component_partitions: dict[str, set[int]] = defaultdict(set)
    component_splits: dict[str, set[str]] = defaultdict(set)
    component_dev_folds: dict[str, set[int]] = defaultdict(set)
    component_categories: dict[str, set[str]] = defaultdict(set)
    split_rows: Counter[str] = Counter()
    split_components: dict[str, set[str]] = defaultdict(set)
    development_fold_rows: Counter[int] = Counter()
    category_label_rows: Counter[str] = Counter()
    parsed_rows: list[tuple[str, int]] = []

    with rows_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(
            tuple(reader.fieldnames or ()) == EXPECTED_ROW_COLUMNS, "draft row columns mismatch"
        )
        for line_number, row in enumerate(reader, start=2):
            item_id = row["id"]
            _require(
                item_id != "" and item_id not in seen_ids,
                f"invalid or duplicate id at line {line_number}",
            )
            seen_ids.add(item_id)
            category = row["category"]
            _require(category in {"БАД", "Легковоспламеняющиеся"}, "unexpected category")
            try:
                label = int(row["label"])
                component_size = int(row["component_size"])
                partition = int(row["draft_partition_7"])
                development_fold = int(row["draft_dev_fold_5"])
            except ValueError as error:
                raise ValueError(f"non-integer draft row field at line {line_number}") from error
            _require(label in (0, 1), "label must be atomic 0/1")
            component = row["semantic_component"]
            _require(HEX64.fullmatch(component) is not None, "invalid semantic component id")
            _require(component_size > 0, "component size must be positive")
            _require(0 <= partition <= 6, "draft partition must be in [0, 6]")
            holdout_flag = _strict_bool(row["draft_holdout_candidate_not_sealed"])
            is_holdout = partition == 0
            _require(holdout_flag == is_holdout, "draft holdout flag disagrees with partition 0")
            if is_holdout:
                _require(development_fold == -1, "holdout row has a development fold")
                split = "sealed_holdout"
            else:
                _require(0 <= development_fold <= 4, "development row has invalid draft dev fold")
                split = "development"
                development_fold_rows[development_fold] += 1
            component_rows[component] += 1
            component_partitions[component].add(partition)
            component_splits[component].add(split)
            component_dev_folds[component].add(development_fold)
            component_categories[component].add(category)
            split_rows[split] += 1
            split_components[split].add(component)
            category_label_rows[f"{category}|{label}"] += 1
            parsed_rows.append((component, component_size))
            writer.writerow(
                {
                    "id": item_id,
                    "category": category,
                    "label": label,
                    "semantic_component": component,
                    "component_size": component_size,
                    "split": split,
                    "development_fold": development_fold,
                }
            )

    _require(len(seen_ids) == manifest.get("rows"), "draft row count differs from manifest")
    _require(
        len(component_rows) == manifest.get("components"), "component count differs from manifest"
    )
    _require(
        max(component_rows.values(), default=0) == manifest.get("largest_component"),
        "largest component differs from manifest",
    )
    _require(split_rows["sealed_holdout"] > 0, "sealed holdout is empty")
    _require(split_rows["development"] > 0, "development split is empty")
    _require(
        manifest.get("draft_assignment", {}).get("holdout_candidate_fold") == 0,
        "draft manifest holdout candidate is not partition 0",
    )
    for component, declared_size in parsed_rows:
        _require(component_rows[component] == declared_size, "declared component size mismatch")
    _require(
        all(len(values) == 1 for values in component_partitions.values()),
        "semantic component crosses draft partition",
    )
    _require(
        all(len(values) == 1 for values in component_splits.values()),
        "semantic component crosses sealed split",
    )
    _require(
        all(len(values) == 1 for values in component_dev_folds.values()),
        "semantic component crosses development fold",
    )
    _require(
        all(len(values) == 1 for values in component_categories.values()),
        "semantic component crosses category",
    )
    _require(
        set(development_fold_rows) == set(range(5)), "not all five development folds are populated"
    )
    draft_invariants = manifest.get("invariants", {})
    required_true = (
        "all_ids_unique",
        "all_rows_accounted_for",
        "draft_candidate_has_dev_fold_minus_one",
        "development_has_assigned_fold",
    )
    _require(
        all(draft_invariants.get(key) is True for key in required_true),
        "draft row invariant flag mismatch",
    )
    required_zero = (
        "component_crosses_partition_7",
        "component_crosses_draft_role",
        "component_crosses_dev_fold",
    )
    _require(
        all(draft_invariants.get(key) == 0 for key in required_zero),
        "draft component invariant flag mismatch",
    )
    _require(draft_invariants.get("topology_uses_labels") is False, "draft topology used labels")
    _require(
        draft_invariants.get("candidate_scores_used") is False,
        "draft topology used candidate scores",
    )

    counts = {
        "rows": len(seen_ids),
        "components": len(component_rows),
        "largest_component": max(component_rows.values()),
        "split_rows": dict(sorted(split_rows.items())),
        "split_components": {key: len(value) for key, value in sorted(split_components.items())},
        "development_fold_rows": {str(key): development_fold_rows[key] for key in range(5)},
        "category_label_rows": dict(sorted(category_label_rows.items())),
    }
    return output.getvalue().encode("utf-8"), counts


def seal(draft_dir: Path, audit_path: Path, output_dir: Path) -> dict[str, Any]:
    draft_dir = draft_dir.resolve()
    audit_path = audit_path.resolve()
    output_dir = output_dir.resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError(f"refusing to overwrite immutable output: {output_dir}")
    manifest, audit = verify_draft(draft_dir, audit_path)
    folds_payload, counts = validate_and_render_rows(draft_dir / "rows.csv", manifest)
    folds_sha = hashlib.sha256(folds_payload).hexdigest()
    sealed_manifest = {
        "version": SEALED_VERSION,
        "status": "sealed",
        "sealed": True,
        "immutable": True,
        "valid_for_candidate_scoring": True,
        "source_graph_version": GRAPH_VERSION,
        "audit_version": AUDIT_VERSION,
        "promotion": {
            "repartitioned": False,
            "holdout_rule": "draft_partition_7 == 0",
            "development_fold_rule": "preserve draft_dev_fold_5 exactly",
        },
        "criterion_checks": audit["criterion_checks"],
        "source_sha256": {
            "draft_manifest": file_sha256(draft_dir / "manifest.json"),
            "manual_audit": file_sha256(audit_path),
            "draft_input": manifest["input_sha256"],
            "verified_draft_outputs": manifest["output_sha256"],
        },
        "output_sha256": {"folds": folds_sha},
        "counts": counts,
        "invariants": {
            "all_ids_unique": True,
            "all_rows_accounted_for": True,
            "component_crosses_split": 0,
            "component_crosses_development_fold": 0,
            "component_crosses_category": 0,
            "holdout_development_fold_is_minus_one": True,
            "no_raw_text_or_images": True,
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.sealing-", dir=output_dir.parent))
    try:
        (temporary / "folds.csv").write_bytes(folds_payload)
        (temporary / "manifest.json").write_text(
            json.dumps(sealed_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.chmod(0o755)
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return sealed_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed immutable semantic-family v3 promotion."
    )
    parser.add_argument("--draft-dir", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = seal(args.draft_dir, args.audit, args.output_dir)
    print(
        json.dumps(
            {
                "version": manifest["version"],
                "sealed": manifest["sealed"],
                "rows": manifest["counts"]["rows"],
                "components": manifest["counts"]["components"],
                "folds_sha256": manifest["output_sha256"]["folds"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
