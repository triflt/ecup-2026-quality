"""Build a frozen, reviewer-blind 300-component causal target audit."""

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

HERE = Path(__file__).resolve().parent
FLAMMABLE = "Легковоспламеняющиеся"
PACKET_FIELDS = (
    "schema_version",
    "audit_id",
    "focus",
    "row_id",
    "semantic_component",
    "development_fold",
    "category",
    "name",
    "description",
    "candidate_sold_object",
    "candidate_regulated_substance",
    "candidate_relation",
    "evidence_source",
    "evidence_start",
    "evidence_end",
    "evidence_span",
    "candidate_supported",
    "reason_codes",
    "review_sold_object_correct",
    "review_regulated_substance_correct",
    "review_relation_correct",
    "review_evidence_correct",
    "review_unsupported_claim",
    "review_notes",
)


def _load_ontology():
    spec = importlib.util.spec_from_file_location("exp670_ontology", HERE / "ontology.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import ontology")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_key(seed: int, *parts: object) -> str:
    raw = "|".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _audit_ids(path: Path) -> set[str]:
    if path.suffix.lower() == ".csv":
        rows = _read_csv(path)
        for key in ("row_id", "id"):
            values = {str(row[key]) for row in rows if row.get(key) not in (None, "")}
            if values:
                return values
        raise ValueError(f"no row ID field in prior audit {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: Iterable[Any]
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("sanitized_records", payload.get("records", []))
    else:
        records = []
    values = set()
    for record in records:
        if isinstance(record, dict):
            value = record.get("row_id", record.get("id"))
            if value is not None:
                values.add(str(value))
    if not values:
        raise ValueError(f"no row IDs in prior audit {path}")
    return values


def _load_baseline(path: Path) -> dict[str, int]:
    rows = _read_csv(path)
    if not rows or set(rows[0]) != {"id", "prediction", "source"}:
        raise ValueError("baseline prediction schema mismatch")
    result = {}
    for row in rows:
        prediction = int(row["prediction"])
        if prediction not in (0, 1) or row["source"] != "exact_full140_semantic_v3":
            raise ValueError("baseline prediction contract mismatch")
        if row["id"] in result:
            raise ValueError("duplicate baseline ID")
        result[row["id"]] = prediction
    return result


def _round_robin(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Select deterministically while spreading fold/relation/substance."""

    buckets: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for row in rows:
        target = row["target"]
        key = (row["fold"], target.relation, target.regulated_substance)
        buckets.setdefault(key, []).append(row)
    for key, values in buckets.items():
        values.sort(
            key=lambda row: (
                not row["baseline_error"],
                _stable_key(seed, key, row["component"], row["id"]),
            )
        )
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets, key=lambda key: _stable_key(seed, *key))
    while len(selected) < count and keys:
        next_keys = []
        for key in keys:
            values = buckets[key]
            if values and len(selected) < count:
                selected.append(values.pop(0))
            if values:
                next_keys.append(key)
        keys = next_keys
    if len(selected) != count:
        raise ValueError(f"insufficient rows for frozen stratum: {len(selected)} < {count}")
    return selected


def build(
    *,
    data: Path,
    folds: Path,
    baseline_predictions: Path,
    prior_audit: list[Path],
    output_dir: Path,
    seed: int = 670,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    ontology = _load_ontology()
    data_rows = {row["id"]: row for row in _read_csv(data)}
    fold_rows = _read_csv(folds)
    baseline = _load_baseline(baseline_predictions)
    excluded_ids: set[str] = set()
    prior_hashes = {}
    for path in prior_audit:
        excluded_ids.update(_audit_ids(path))
        prior_hashes[str(path)] = sha256_file(path)

    candidates: list[dict[str, Any]] = []
    seen_components: set[str] = set()
    sealed_rows_read = 0
    for membership in fold_rows:
        if membership["split"] != "development":
            continue
        row_id = membership["id"]
        if row_id in excluded_ids or row_id not in data_rows:
            continue
        component = membership["semantic_component"]
        if component in seen_components:
            continue
        if row_id not in baseline:
            raise ValueError(f"missing exact-140 prediction for {row_id}")
        row = data_rows[row_id]
        target = ontology.extract_causal_target(
            category=row["category"], name=row["name"], description=row["description"]
        )
        seen_components.add(component)
        candidates.append(
            {
                "id": row_id,
                "component": component,
                "fold": int(membership["development_fold"]),
                "category": row["category"],
                "name": row["name"],
                "description": row["description"],
                "label": int(membership["label"]),
                "baseline_prediction": baseline[row_id],
                "baseline_error": int(membership["label"]) != baseline[row_id],
                "target": target,
            }
        )

    flammable = [row for row in candidates if row["category"] == FLAMMABLE]
    priority_pool = [
        row
        for row in flammable
        if row["target"].sold_object in {"device", "accessory", "kit"}
        or row["target"].relation in {"included", "compatible_external", "mentioned_only", "negated"}
    ]
    priority = _round_robin(priority_pool, 100, seed)
    used = {row["component"] for row in priority}
    direct_pool = [
        row
        for row in flammable
        if row["component"] not in used
        and row["target"].supported
        and row["target"].relation == "sold_object"
    ]
    direct = _round_robin(direct_pool, 100, seed + 1)
    used.update(row["component"] for row in direct)
    coverage_pool = [row for row in candidates if row["component"] not in used]
    coverage = _round_robin(coverage_pool, 100, seed + 2)

    selected = [
        *(dict(row, focus="transaction_scope_priority") for row in priority),
        *(dict(row, focus="direct_substance_control") for row in direct),
        *(dict(row, focus="coverage_and_abstention_control") for row in coverage),
    ]
    if len(selected) != 300 or len({row["component"] for row in selected}) != 300:
        raise AssertionError("audit must contain 300 unique semantic components")

    output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = output_dir / "human_audit_300.csv"
    with packet_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PACKET_FIELDS)
        writer.writeheader()
        for index, row in enumerate(selected, start=1):
            target = row["target"]
            writer.writerow(
                {
                    "schema_version": "exp670_causal_audit_row_v1",
                    "audit_id": f"C670-{index:03d}",
                    "focus": row["focus"],
                    "row_id": row["id"],
                    "semantic_component": row["component"],
                    "development_fold": row["fold"],
                    "category": row["category"],
                    "name": row["name"],
                    "description": row["description"],
                    "candidate_sold_object": target.sold_object,
                    "candidate_regulated_substance": target.regulated_substance,
                    "candidate_relation": target.relation,
                    "evidence_source": target.evidence_source or "",
                    "evidence_start": "" if target.evidence_start is None else target.evidence_start,
                    "evidence_end": "" if target.evidence_end is None else target.evidence_end,
                    "evidence_span": target.evidence_span or "",
                    "candidate_supported": str(target.supported).lower(),
                    "reason_codes": "|".join(target.reason_codes),
                    "review_sold_object_correct": "",
                    "review_regulated_substance_correct": "",
                    "review_relation_correct": "",
                    "review_evidence_correct": "",
                    "review_unsupported_claim": "",
                    "review_notes": "",
                }
            )

    private_manifest = {
        "schema_version": "exp670_private_manifest_v1",
        "experiment_id": "670",
        "seed": seed,
        "rows": len(selected),
        "unique_components": len({row["component"] for row in selected}),
        "sealed_rows_read": sealed_rows_read,
        "public_used": False,
        "baseline_source": "exact_full140_semantic_v3",
        "packet_sha256": sha256_file(packet_path),
        "source_sha256": {
            "data": sha256_file(data),
            "folds": sha256_file(folds),
            "baseline_predictions": sha256_file(baseline_predictions),
            "ontology": sha256_file(HERE / "ontology.py"),
            "prior_audits": prior_hashes,
        },
        "strata": dict(Counter(row["focus"] for row in selected)),
        "categories": dict(Counter(row["category"] for row in selected)),
        "folds": dict(Counter(str(row["fold"]) for row in selected)),
        "baseline_errors": sum(row["baseline_error"] for row in selected),
        "supported_targets": sum(row["target"].supported for row in selected),
        "records": [
            {
                "audit_id": f"C670-{index:03d}",
                "row_id": row["id"],
                "semantic_component": row["component"],
                "label": row["label"],
                "baseline_prediction": row["baseline_prediction"],
                "baseline_error": row["baseline_error"],
            }
            for index, row in enumerate(selected, start=1)
        ],
    }
    manifest_path = output_dir / "private_manifest.json"
    manifest_path.write_text(
        json.dumps(private_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        key: value for key, value in private_manifest.items() if key not in {"records"}
    }
    summary["private_manifest_sha256"] = sha256_file(manifest_path)
    summary_path = output_dir / "build_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--prior-audit", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=670)
    args = parser.parse_args()
    result = build(**vars(args))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
