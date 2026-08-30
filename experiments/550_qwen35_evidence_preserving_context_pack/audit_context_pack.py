from __future__ import annotations

"""Audit evidence retention and exact provenance without reading labels or folds."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from context_pack import (
    DESCRIPTION_BUDGET,
    baseline_positions,
    evidence,
    pack_description,
    retained_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "research/data.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "analysis"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_frame(frame: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    columns_read = ("id", "category", "name", "description")
    missing = sorted(set(columns_read) - set(frame.columns))
    if missing:
        raise ValueError(f"data lacks context-pack columns: {missing}")
    source = frame.loc[:, columns_read].copy()
    source["id"] = source.id.astype(str)
    source["category"] = source.category.astype(str)
    source["name"] = source.name.fillna("").astype(str)
    source["description"] = source.description.fillna("").astype(str)
    if source.id.duplicated().any():
        raise ValueError("context-pack data contains duplicate ids")

    records: list[dict[str, object]] = []
    deterministic = 0
    provenance_passed = 0
    budget_passed = 0
    sentence_retention_passed = 0
    for row in source.itertuples(index=False):
        surface = evidence.surface_text(row.description)
        first = pack_description(
            row_id=row.id,
            category=row.category,
            name=row.name,
            description=row.description,
        )
        second = pack_description(
            row_id=row.id,
            category=row.category,
            name=row.name,
            description=row.description,
        )
        deterministic_ok = first == second
        provenance_ok = first.packable and all(
            character == surface.text[position]
            for character, position in zip(first.text, first.source_positions)
        )
        provenance_ok = (
            provenance_ok
            and len(first.text) == len(first.source_positions)
            and len(first.text) == len(first.raw_intervals)
            and tuple(sorted(set(first.source_positions))) == first.source_positions
            and (bool(first.raw_intervals) or len(first.text) == 0)
            and first.raw_intervals
            == tuple(surface.raw_intervals[position] for position in first.source_positions)
            and all(
                0 <= raw_start < raw_end <= len(surface.raw)
                for raw_start, raw_end in first.raw_intervals
            )
        )
        budget_ok = first.packable and len(first.text) <= DESCRIPTION_BUDGET
        baseline_retained = retained_evidence(
            baseline_positions(row.description), first.evidence_spans
        )
        candidate_retained = retained_evidence(first.source_positions, first.evidence_spans)
        sentence_retention_ok = first.packable and candidate_retained == len(first.evidence_spans)
        deterministic += deterministic_ok
        provenance_passed += provenance_ok
        budget_passed += budget_ok
        sentence_retention_passed += sentence_retention_ok
        records.append(
            {
                "id": row.id,
                "category": row.category,
                "description_chars": len(surface.text),
                "long_description": len(surface.text) > DESCRIPTION_BUDGET,
                "safe_evidence_spans": len(first.evidence_spans),
                "baseline_retained_evidence": baseline_retained,
                "candidate_retained_evidence": candidate_retained,
                "strictly_improved": candidate_retained > baseline_retained,
                "regressed": candidate_retained < baseline_retained,
                "packable": first.packable,
                "blocked_reason": first.blocked_reason or "",
                "packed_chars": len(first.text),
                "text_sha256": first.text_sha256,
                "provenance_sha256": first.provenance_sha256,
                "deterministic": deterministic_ok,
                "provenance_ok": provenance_ok,
                "budget_ok": budget_ok,
                "complete_evidence_sentences": sentence_retention_ok,
            }
        )
    rows = pd.DataFrame(records)
    long_evidence = rows.long_description & (rows.safe_evidence_spans > 0)
    denominator = int(long_evidence.sum())
    improved = int((long_evidence & rows.strictly_improved).sum())
    regressed = int((long_evidence & rows.regressed).sum())
    improvement_fraction = improved / max(1, denominator)
    gates = {
        "strictly_more_evidence_on_at_least_10_percent": (
            denominator > 0 and improvement_fraction >= 0.10
        ),
        "never_less_evidence": regressed == 0,
        "all_rows_packable": bool(rows.packable.all()),
        "provenance_100_percent": provenance_passed == len(rows),
        "determinism_100_percent": deterministic == len(rows),
        "budget_100_percent": budget_passed == len(rows),
        "evidence_sentences_complete_100_percent": sentence_retention_passed == len(rows),
    }
    report: dict[str, object] = {
        "experiment_id": "550",
        "audit_version": "evidence_preserving_context_pack_v1",
        "status": "GO" if all(gates.values()) else "NO_GO",
        "label_blind": True,
        "columns_read": list(columns_read),
        "labels_read": False,
        "folds_read": False,
        "prediction_source": "constant union of frozen_prediction=0 and 1",
        "description_budget": DESCRIPTION_BUDGET,
        "rows": len(rows),
        "long_rows": int(rows.long_description.sum()),
        "long_evidence_rows": denominator,
        "long_evidence_rows_strictly_improved": improved,
        "long_evidence_rows_regressed": regressed,
        "strict_improvement_fraction": improvement_fraction,
        "safe_evidence_spans": int(rows.safe_evidence_spans.sum()),
        "baseline_retained_safe_evidence": int(rows.baseline_retained_evidence.sum()),
        "candidate_retained_safe_evidence": int(rows.candidate_retained_evidence.sum()),
        "blocked_conflict_or_ambiguity_rows": int((rows.blocked_reason != "").sum()),
        "gates": gates,
    }
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report_path = args.output_dir / "context_pack_audit.json"
    rows_path = args.output_dir / "context_pack_rows.csv.gz"
    existing = [path for path in (report_path, rows_path) if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing context-pack audit")
    frame = pd.read_csv(args.data, dtype={"id": str})
    report, rows = audit_frame(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(
        rows_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    report["input_sha256"] = {"data": sha256(args.data)}
    report["output_sha256"] = {"rows": sha256(rows_path)}
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "long_evidence_rows": report["long_evidence_rows"],
                "strict_improvement_fraction": report["strict_improvement_fraction"],
                "regressed": report["long_evidence_rows_regressed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
