from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from contract import (
    DEFAULT_FOLDS,
    DEFAULT_SPEC,
    EXPECTED_DEVELOPMENT_ROWS,
    PROTOCOL_VERSION,
    canonical_sha256,
    load_development_registry,
    load_json,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
EXP490_SRC = ROOT / "experiments" / "490_evidence_grounded_explanations" / "src"
if str(EXP490_SRC) not in sys.path:
    sys.path.insert(0, str(EXP490_SRC))

from evidence_grounding import extract_evidence, surface_text

TEXT_COLUMNS = ("id", "category", "name", "description")


def _safe_candidate(
    result: Any,
    *,
    expected_verdict: int,
    direct_concepts: set[str],
    forbidden_concepts: set[str],
    name: object,
    description: object,
) -> tuple[dict[str, Any] | None, str | None]:
    if result.status != "SAFE":
        return None, result.fallback_reason
    if result.frozen_prediction != expected_verdict:
        raise ValueError("extractor changed the frozen verdict")
    if result.concept in forbidden_concepts:
        return None, "forbidden_absence_concept"
    if result.concept not in direct_concepts:
        return None, "not_a_frozen_direct_concept"
    if result.source not in {"name", "description"}:
        return None, "unsupported_source"
    if result.surface_start is None or result.surface_end is None:
        return None, "missing_surface_offsets"
    source_value = name if result.source == "name" else description
    surface = surface_text(source_value).text
    exact = (
        0 <= result.surface_start < result.surface_end <= len(surface)
        and surface[result.surface_start : result.surface_end] == result.exact_surface_span
    )
    if not exact:
        return None, "exact_offset_failure"
    return (
        {
            "verdict": expected_verdict,
            "source": result.source,
            "surface_start": int(result.surface_start),
            "surface_end": int(result.surface_end),
            "exact_surface_span": result.exact_surface_span,
            "concept": result.concept,
            "polarity": result.polarity,
            "scope": result.scope,
            "vocabulary_version": result.vocabulary_version,
            "vocabulary_sha256": result.vocabulary_sha256,
        },
        None,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def build_manifest(
    *,
    development_data_path: Path,
    folds_path: Path,
    spec_path: Path,
    output_dir: Path,
    enforce_frozen: bool = True,
    expected_rows: int = EXPECTED_DEVELOPMENT_ROWS,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    registry = load_development_registry(
        folds_path,
        enforce_frozen=enforce_frozen,
        expected_rows=expected_rows,
    )
    data = pd.read_csv(
        development_data_path,
        usecols=list(TEXT_COLUMNS),
        dtype={"id": str},
    )
    if data["id"].duplicated().any():
        raise ValueError("physical development data contain duplicate ids")
    expected_ids = registry["id"].astype(str).tolist()
    actual_ids = data["id"].astype(str).tolist()
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != set(expected_ids):
        extra = sorted(set(actual_ids) - set(expected_ids))[:5]
        missing = sorted(set(expected_ids) - set(actual_ids))[:5]
        raise ValueError(
            f"physical input must contain exactly development IDs; extra={extra}, missing={missing}"
        )
    data = data.set_index("id").loc[expected_ids].reset_index()
    if not data["category"].astype(str).equals(registry["category"].astype(str)):
        raise ValueError("physical data categories differ from semantic-v3 registry")

    spec = load_json(spec_path)
    if spec.get("version") != PROTOCOL_VERSION:
        raise ValueError("frozen router spec version mismatch")
    direct_by_verdict = {
        int(verdict): set(concepts)
        for verdict, concepts in spec["direct_extractive_concepts"].items()
    }
    forbidden = set(spec["forbidden_as_flip_evidence"])
    counters: Counter[str] = Counter()
    per_category: dict[str, Counter[str]] = {
        category: Counter() for category in sorted(registry["category"].unique())
    }
    manifest_rows: list[dict[str, Any]] = []

    for data_row, registry_row in zip(
        data.to_dict("records"), registry.to_dict("records"), strict=True
    ):
        row_id = str(data_row["id"])
        category = str(data_row["category"])
        name = data_row.get("name", "")
        description = data_row.get("description", "")
        candidates: dict[str, dict[str, Any] | None] = {}
        reasons: dict[str, str | None] = {}
        for verdict in (0, 1):
            result = extract_evidence(
                row_id=row_id,
                category=category,
                name=name,
                description=description,
                frozen_prediction=verdict,
            )
            candidate, reason = _safe_candidate(
                result,
                expected_verdict=verdict,
                direct_concepts=direct_by_verdict[verdict],
                forbidden_concepts=forbidden,
                name=name,
                description=description,
            )
            candidates[str(verdict)] = candidate
            reasons[str(verdict)] = reason
            if reason == "exact_offset_failure":
                counters["exact_offset_failures"] += 1
            elif reason == "forbidden_absence_concept":
                counters["blocked_forbidden_candidates"] += 1
            elif reason == "conflicting_evidence":
                counters["extractor_conflict_abstentions"] += 1

        raw_both = candidates["0"] is not None and candidates["1"] is not None
        if raw_both:
            # Opposite direct evidence in one listing is ambiguous regardless of whether
            # the exact spans overlap. Fail closed and grant neither side a flip.
            candidates = {"0": None, "1": None}
            reasons = {"0": "opposite_direct_evidence", "1": "opposite_direct_evidence"}
            counters["resolved_opposite_evidence_conflicts"] += 1

        if candidates["0"] is not None and candidates["1"] is not None:
            counters["unresolved_conflicts"] += 1
        counters["forbidden_flip_evidence"] += sum(
            int(candidate is not None and candidate.get("concept") in forbidden)
            for candidate in candidates.values()
        )

        has_0 = candidates["0"] is not None
        has_1 = candidates["1"] is not None
        per_category[category]["rows"] += 1
        per_category[category]["direct_0"] += int(has_0)
        per_category[category]["direct_1"] += int(has_1)
        per_category[category]["direct_any"] += int(has_0 or has_1)
        per_category[category]["raw_both"] += int(raw_both)
        manifest_rows.append(
            {
                "id": row_id,
                "category": category,
                "development_fold": int(registry_row["development_fold"]),
                "semantic_component": str(registry_row["semantic_component"]),
                "candidate_for_0": candidates["0"],
                "candidate_for_1": candidates["1"],
                "abstention_reason_for_0": reasons["0"],
                "abstention_reason_for_1": reasons["1"],
                "opposite_evidence_conflict_resolved": raw_both,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "development_evidence_manifest.jsonl"
    _write_jsonl(manifest_path, manifest_rows)
    minimum_coverage = float(
        spec["preflight_gates"]["minimum_any_direct_span_coverage_per_category"]
    )
    category_audit: dict[str, dict[str, Any]] = {}
    coverage_gate = True
    for category, values in per_category.items():
        rows = int(values["rows"])
        any_coverage = float(values["direct_any"] / rows) if rows else 0.0
        coverage_gate = coverage_gate and any_coverage >= minimum_coverage
        category_audit[category] = {
            "rows": rows,
            "direct_0_rows": int(values["direct_0"]),
            "direct_1_rows": int(values["direct_1"]),
            "direct_any_rows": int(values["direct_any"]),
            "raw_both_rows": int(values["raw_both"]),
            "direct_any_coverage": any_coverage,
        }
    gates = {
        "row_count_exact": len(manifest_rows) == expected_rows,
        "exact_offsets": counters["exact_offset_failures"] == 0,
        "conflicts_fail_closed": counters["unresolved_conflicts"] == 0,
        "forbidden_absence_evidence": counters["forbidden_flip_evidence"] == 0,
        "minimum_direct_coverage_per_category": coverage_gate,
        "labels_not_loaded": True,
        "zero_sealed_rows_in_input_or_output": True,
    }
    audit = {
        "experiment_id": "620",
        "protocol_version": PROTOCOL_VERSION,
        "scope": "development_only_label_free",
        "rows": len(manifest_rows),
        "labels_loaded": False,
        "sealed_rows_in_inputs": 0,
        "sealed_rows_in_outputs": 0,
        "exact_offset_failures": int(counters["exact_offset_failures"]),
        "blocked_forbidden_candidates": int(counters["blocked_forbidden_candidates"]),
        "forbidden_flip_evidence": int(counters["forbidden_flip_evidence"]),
        "extractor_conflict_abstentions": int(counters["extractor_conflict_abstentions"]),
        "resolved_opposite_evidence_conflicts": int(
            counters["resolved_opposite_evidence_conflicts"]
        ),
        "unresolved_conflicts": int(counters["unresolved_conflicts"]),
        "per_category": category_audit,
        "gates": gates,
        "input_sha256": {
            "development_data": sha256_file(development_data_path),
            "folds": sha256_file(folds_path),
            "frozen_spec": sha256_file(spec_path),
        },
        "output_sha256": {manifest_path.name: sha256_file(manifest_path)},
        "frozen_spec_canonical_sha256": canonical_sha256(spec),
        "decision": "GO" if all(gates.values()) else "NO_GO",
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    audit_path = output_dir / "evidence_manifest_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a development-only, label-free exact evidence manifest."
    )
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = build_manifest(
        development_data_path=args.development_data.resolve(),
        folds_path=args.folds.resolve(),
        spec_path=args.spec.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if audit["decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
