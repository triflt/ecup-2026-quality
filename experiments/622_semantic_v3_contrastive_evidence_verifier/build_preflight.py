#!/usr/bin/env python3
"""Build label-isolated candidate banks and a fresh private blind-audit packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

EXPERIMENT = Path(__file__).resolve().parent
REPOSITORY = EXPERIMENT.parents[1]
SPEC = json.loads((EXPERIMENT / "frozen_spec.json").read_text(encoding="utf-8"))
ONTOLOGY = json.loads((EXPERIMENT / "claim_ontology_v1.json").read_text(encoding="utf-8"))
LABEL_LIKE = {"label", "target", "gold", "verdict", "prediction", "y"}
FEATURE_COLUMNS = ("id", "category", "name", "description")
MEMBERSHIP_COLUMNS = (
    "id",
    "category",
    "semantic_component",
    "component_size",
    "split",
    "development_fold",
)
REVIEW_COLUMNS = (
    "schema_version",
    "audit_id",
    "row_id",
    "semantic_component",
    "category",
    "product_name",
    "source",
    "source_text",
    "raw_start",
    "raw_end",
    "exact_surface_span",
    "claim_concept",
    "counterclaim_concept",
    "hard_negative_mapping",
    "strict_pass",
    "critical_unsupported",
    "scope_or_negation_failure",
    "review_notes",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(*values: object) -> str:
    return _sha256_bytes("\0".join(map(str, values)).encode("utf-8"))


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _write_csv(path: Path, columns: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _require_private_output(path: Path) -> None:
    if ".local" not in path.resolve().parts:
        raise ValueError(
            "Raw candidate and human-review outputs must live below a .local directory"
        )
    if path.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite private output: {path}")


def _require_exact_columns(columns: list[str], expected: tuple[str, ...], kind: str) -> None:
    lowered = {column.strip().lower() for column in columns}
    leaked = sorted(lowered & LABEL_LIKE)
    if leaked:
        raise ValueError(f"{kind} contains forbidden label-like columns: {leaked}")
    if set(columns) != set(expected):
        raise ValueError(f"{kind} columns must be exactly {list(expected)}, got {columns}")


def _index_unique(rows: list[dict[str, str]], kind: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        row_id = row.get("id", "").strip()
        if not row_id or row_id in result:
            raise ValueError(f"{kind} has empty or duplicate id: {row_id!r}")
        result[row_id] = row
    return result


def _validate_inputs(
    features_path: Path, membership_path: Path
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    feature_columns, feature_rows = _read_csv(features_path)
    membership_columns, membership_rows = _read_csv(membership_path)
    _require_exact_columns(feature_columns, FEATURE_COLUMNS, "features")
    _require_exact_columns(membership_columns, MEMBERSHIP_COLUMNS, "membership")
    features = _index_unique(feature_rows, "features")
    membership = _index_unique(membership_rows, "membership")
    if set(features) != set(membership):
        raise ValueError("Feature and membership ID sets differ")
    components: dict[str, set[str]] = {}
    for row_id, member in membership.items():
        if member["split"] != "development":
            raise ValueError(f"Non-development/sealed row is forbidden: {row_id}")
        if member["development_fold"] not in {"0", "1", "2", "3", "4"}:
            raise ValueError(f"Invalid development fold for {row_id}")
        if features[row_id]["category"] != member["category"]:
            raise ValueError(f"Category mismatch for {row_id}")
        component = member["semantic_component"].strip()
        if not component:
            raise ValueError(f"Missing semantic component for {row_id}")
        components.setdefault(component, set()).add(member["development_fold"])
    crossing = sorted(component for component, folds in components.items() if len(folds) != 1)
    if crossing:
        raise ValueError(f"Semantic components cross development folds: {crossing[:5]}")
    return features, membership


def _load_extractor():
    path = (
        REPOSITORY
        / "experiments/490_evidence_grounded_explanations/src/evidence_grounding/extractor.py"
    )
    module_spec = importlib.util.spec_from_file_location("exp622_frozen_extractor", path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot load frozen extractor: {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    if module.VOCABULARY_VERSION != "policy_concepts_v1":
        raise ValueError("Unexpected extractor vocabulary")
    return module


def _load_donor_labels(
    path: Path, outer_fold: int, membership: dict[str, dict[str, str]]
) -> dict[str, int]:
    columns, rows = _read_csv(path)
    if set(columns) != {"id", "label"}:
        raise ValueError(f"Donor labels must contain exactly id,label: {path}")
    indexed = _index_unique(rows, f"donor labels fold {outer_fold}")
    expected = {
        row_id for row_id, row in membership.items() if int(row["development_fold"]) != outer_fold
    }
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))[:5]
        extra = sorted(set(indexed) - expected)[:5]
        raise ValueError(
            f"Donor ID contract failed for fold {outer_fold}; missing={missing}, extra={extra}"
        )
    labels: dict[str, int] = {}
    for row_id, row in indexed.items():
        if row["label"] not in {"0", "1"}:
            raise ValueError(f"Invalid donor label for {row_id}")
        labels[row_id] = int(row["label"])
    return labels


def _ontology_contract() -> tuple[dict[str, dict[str, Any]], dict[str, tuple[str, str]]]:
    claims = {row["concept"]: row for row in ONTOLOGY["claims"]}
    negatives: dict[str, tuple[str, str]] = {}
    for mapping in ONTOLOGY["hard_negative_mappings"]:
        for source, target in mapping["pairs"].items():
            if source in negatives:
                raise ValueError(f"Duplicate hard-negative mapping for {source}")
            if source not in claims or target not in claims:
                raise ValueError("Hard-negative mapping references unknown claim")
            if claims[source]["category"] != claims[target]["category"]:
                raise ValueError("Hard-negative mapping crosses categories")
            if claims[source]["supports_prediction"] == claims[target]["supports_prediction"]:
                raise ValueError("Hard-negative mapping does not invert the verdict relation")
            negatives[source] = (mapping["id"], target)
    return claims, negatives


def _candidate_from_result(
    *, result: Any, feature: dict[str, str], member: dict[str, str], outer_fold: int
) -> dict[str, Any] | None:
    if result.status != "SAFE" or result.exact_surface_span is None:
        return None
    allowed_concepts = {row["concept"] for row in ONTOLOGY["claims"]}
    if result.concept not in allowed_concepts:
        return None
    span = result.exact_surface_span
    source_text = feature[result.source]
    if result.raw_start is None or result.raw_end is None:
        raise ValueError("Exact candidate is missing raw offsets")
    surface = sys.modules[result.__class__.__module__].surface_text(source_text)
    if surface.text[result.surface_start : result.surface_end] != span:
        raise ValueError(f"Surface span mismatch for {feature['id']}")
    mapped_start, mapped_end = surface.raw_interval(result.surface_start, result.surface_end)
    if (mapped_start, mapped_end) != (result.raw_start, result.raw_end):
        raise ValueError(f"Raw offset mapping mismatch for {feature['id']}")
    if not (8 <= len(span) <= 160):
        return None
    payload = {
        "schema_version": "exp622_candidate_bank_row_v1",
        "row_id": feature["id"],
        "semantic_component": member["semantic_component"],
        "outer_fold": outer_fold,
        "role": "outer_validation"
        if int(member["development_fold"]) == outer_fold
        else "outer_train",
        "category": feature["category"],
        "supports_prediction": result.frozen_prediction,
        "source": result.source,
        "raw_start": result.raw_start,
        "raw_end": result.raw_end,
        "surface_start": result.surface_start,
        "surface_end": result.surface_end,
        "exact_surface_span": span,
        "span_sha256": _sha256_bytes(span.encode("utf-8")),
        "concept": result.concept,
        "polarity": result.polarity,
        "scope": result.scope,
        "extractor_vocabulary_version": result.vocabulary_version,
    }
    payload["candidate_id"] = _stable_hash(
        payload["row_id"],
        outer_fold,
        payload["supports_prediction"],
        payload["source"],
        payload["raw_start"],
        payload["raw_end"],
        payload["concept"],
    )
    return payload


def _build_fold(
    *,
    outer_fold: int,
    features: dict[str, dict[str, str]],
    membership: dict[str, dict[str, str]],
    labels: dict[str, int],
    extractor: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for row_id in sorted(features, key=lambda value: int(value) if value.isdigit() else value):
        feature, member = features[row_id], membership[row_id]
        for possible_prediction in (0, 1):
            result = extractor.extract_evidence(
                row_id=row_id,
                category=feature["category"],
                name=feature["name"],
                description=feature["description"],
                frozen_prediction=possible_prediction,
            )
            candidate = _candidate_from_result(
                result=result, feature=feature, member=member, outer_fold=outer_fold
            )
            if candidate is not None:
                candidates.append(candidate)
    donor_targets = [
        candidate
        for candidate in candidates
        if candidate["role"] == "outer_train"
        and candidate["supports_prediction"] == labels[candidate["row_id"]]
    ]
    donor_ids = [row["row_id"] for row in donor_targets]
    if len(donor_ids) != len(set(donor_ids)):
        raise ValueError(f"More than one positive donor target per row in fold {outer_fold}")
    return candidates, donor_targets


def _discover_legacy_ids(roots: Iterable[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    ids: set[str] = set()
    files: list[dict[str, Any]] = []
    id_keys = {"id", "row_id", "source_id", "donor_id"}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in id_keys and isinstance(child, (str, int)) and str(child).strip():
                    ids.add(str(child).strip())
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for root in roots:
        if not root.exists():
            continue
        paths = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".jsonl"}:
                continue
            if root.is_dir() and not any(
                token in part.lower()
                for part in path.relative_to(root).parts
                for token in ("audit", "review")
            ):
                continue
            before = len(ids)
            try:
                if path.suffix.lower() == ".csv":
                    _, rows = _read_csv(path)
                    visit(rows)
                elif path.suffix.lower() == ".jsonl":
                    with path.open(encoding="utf-8") as stream:
                        for line in stream:
                            if line.strip():
                                visit(json.loads(line))
                else:
                    visit(json.loads(path.read_text(encoding="utf-8")))
            except (UnicodeDecodeError, csv.Error, json.JSONDecodeError):
                continue
            display_path = (
                str(path.relative_to(REPOSITORY)) if path.is_relative_to(REPOSITORY) else path.name
            )
            files.append(
                {
                    "path": display_path,
                    "sha256": _sha256_file(path),
                    "new_ids_discovered": len(ids) - before,
                }
            )
    return ids, files


def _audit_rows(
    targets: list[dict[str, Any]],
    features: dict[str, dict[str, str]],
    legacy_ids: set[str],
    sample_size: int,
) -> list[dict[str, str]]:
    _, negatives = _ontology_contract()
    by_family: dict[str, dict[str, Any]] = {}
    for target in targets:
        if target["row_id"] in legacy_ids or target["concept"] not in negatives:
            continue
        family = target["semantic_component"]
        rank = _stable_hash(SPEC["audit"]["namespace"], target["candidate_id"])
        previous = by_family.get(family)
        if previous is None or rank < previous["_rank"]:
            by_family[family] = {**target, "_rank": rank}
    selected = sorted(by_family.values(), key=lambda row: row["_rank"])[:sample_size]
    if len(selected) != sample_size:
        raise ValueError(f"Only {len(selected)} fresh unique-family targets; need {sample_size}")
    rows: list[dict[str, str]] = []
    for index, target in enumerate(selected, 1):
        mapping_id, counterclaim = negatives[target["concept"]]
        feature = features[target["row_id"]]
        rows.append(
            {
                "schema_version": "exp622_human_audit_row_v1",
                "audit_id": f"A{index:03d}",
                "row_id": target["row_id"],
                "semantic_component": target["semantic_component"],
                "category": target["category"],
                "product_name": feature["name"],
                "source": target["source"],
                "source_text": feature[target["source"]],
                "raw_start": str(target["raw_start"]),
                "raw_end": str(target["raw_end"]),
                "exact_surface_span": target["exact_surface_span"],
                "claim_concept": target["concept"],
                "counterclaim_concept": counterclaim,
                "hard_negative_mapping": mapping_id,
                "strict_pass": "",
                "critical_unsupported": "",
                "scope_or_negation_failure": "",
                "review_notes": "",
            }
        )
    return rows


def build_preflight(
    *,
    features_path: Path,
    membership_path: Path,
    donor_label_paths: dict[int, Path],
    private_output_dir: Path,
    public_summary_path: Path,
    legacy_roots: list[Path],
) -> dict[str, Any]:
    _require_private_output(private_output_dir)
    if public_summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite public summary: {public_summary_path}")
    features, membership = _validate_inputs(features_path, membership_path)
    extractor = _load_extractor()
    claims, _ = _ontology_contract()
    legacy_ids, legacy_files = _discover_legacy_ids(legacy_roots)
    fold_outputs: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for fold in SPEC["screen_folds"]:
        labels = _load_donor_labels(donor_label_paths[fold], fold, membership)
        fold_outputs[fold] = _build_fold(
            outer_fold=fold,
            features=features,
            membership=membership,
            labels=labels,
            extractor=extractor,
        )
    all_targets = [row for _, targets in fold_outputs.values() for row in targets]
    audit_rows = _audit_rows(all_targets, features, legacy_ids, SPEC["audit"]["sample_size"])

    private_output_dir.mkdir(parents=True, exist_ok=False)
    for fold, (candidates, targets) in fold_outputs.items():
        fold_dir = private_output_dir / f"fold_{fold}"
        _write_json(fold_dir / "candidate_bank.json", candidates)
        _write_json(fold_dir / "donor_targets.json", targets)
    _write_csv(private_output_dir / "human_audit_300.csv", REVIEW_COLUMNS, audit_rows)
    private_manifest = {
        "schema_version": "exp622_private_preflight_manifest_v1",
        "human_ratings_present": False,
        "sealed_rows_written": 0,
        "legacy_ids_excluded": len(legacy_ids),
        "legacy_source_files": legacy_files,
        "files_sha256": {
            str(path.relative_to(private_output_dir)): _sha256_file(path)
            for path in sorted(private_output_dir.rglob("*"))
            if path.is_file()
        },
    }
    _write_json(private_output_dir / "private_manifest.json", private_manifest)
    summary = {
        "schema_version": "exp622_public_preflight_summary_v1",
        "experiment_id": "622",
        "status": "blocked_pre_gpu_human_audit",
        "decision": "PRE_GPU_BLOCKER",
        "screen_folds": SPEC["screen_folds"],
        "label_isolation": {
            "feature_rows": len(features),
            "feature_label_columns": 0,
            "membership_label_columns": 0,
            "validation_labels_loaded": False,
            "validation_labels_written": False,
            "sealed_rows_written": 0,
            "semantic_components_crossing_folds": 0,
        },
        "candidate_counts": {
            str(fold): {
                "all_roles": len(candidates),
                "outer_validation": sum(row["role"] == "outer_validation" for row in candidates),
                "donor_targets": len(targets),
                "by_concept": dict(sorted(Counter(row["concept"] for row in targets).items())),
            }
            for fold, (candidates, targets) in fold_outputs.items()
        },
        "ontology_claims": len(claims),
        "fresh_audit": {
            "requested_rows": SPEC["audit"]["sample_size"],
            "prepared_rows": len(audit_rows),
            "unique_row_ids": len({row["row_id"] for row in audit_rows}),
            "unique_semantic_components": len({row["semantic_component"] for row in audit_rows}),
            "overlap_with_discovered_legacy_ids": len(
                {row["row_id"] for row in audit_rows} & legacy_ids
            ),
            "human_ratings_present": False,
            "minimum_strict_pass": SPEC["audit"]["minimum_human_strict_pass"],
            "maximum_critical_unsupported": SPEC["audit"]["maximum_critical_unsupported"],
            "maximum_scope_or_negation_failure": SPEC["audit"]["maximum_scope_or_negation_failure"],
        },
        "private_manifest_sha256": _sha256_file(private_output_dir / "private_manifest.json"),
    }
    _write_json(public_summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--donor-labels-0", type=Path, required=True)
    parser.add_argument("--donor-labels-3", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    parser.add_argument("--public-summary", type=Path, required=True)
    parser.add_argument(
        "--legacy-audit-root",
        action="append",
        type=Path,
        default=None,
        help="Repeatable; defaults to experiment 490 and the frozen exp500 exclusion audit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = args.legacy_audit_root or [
        REPOSITORY / "experiments/490_evidence_grounded_explanations",
        REPOSITORY / "validation/legacy_exclusions/exp500_semantic_audit.json",
    ]
    summary = build_preflight(
        features_path=args.features,
        membership_path=args.membership,
        donor_label_paths={0: args.donor_labels_0, 3: args.donor_labels_3},
        private_output_dir=args.private_output_dir,
        public_summary_path=args.public_summary,
        legacy_roots=roots,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
