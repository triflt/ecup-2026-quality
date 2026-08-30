#!/usr/bin/env python3
"""Create an 80-candidate blind review and a private unblinding manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "frozen_spec.json").read_text(encoding="utf-8"))
CANDIDATE_FIELDS = (
    "audit_id",
    "pair_id",
    "blind_slot",
    "row_id",
    "semantic_component",
    "category",
    "name",
    "description",
    "image_file",
    "frozen_prediction",
    "automatic_contract_valid",
    "candidate_verdict",
    "candidate_sold_object",
    "candidate_regulated_substance_or_marker",
    "candidate_relation",
    "candidate_evidence_source",
    "candidate_evidence_quote",
    "candidate_explanation",
    "candidate_raw_response",
)
REVIEW_FIELDS = (
    "audit_id",
    "review_verdict_consistent",
    "review_evidence_relevant",
    "review_object_relation_correct",
    "review_unsupported_fact",
    "review_visual_decisive",
    "review_notes",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build(
    *, runtime: Path, qwen35_output: Path, qwen36_output: Path, output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_dir}")
    runtime_rows = load_jsonl(runtime)
    runtime_by_id = {str(row["id"]): row for row in runtime_rows}
    if len(runtime_rows) != SPEC["rows"] or len(runtime_by_id) != SPEC["rows"]:
        raise ValueError("runtime must contain exactly 40 unique rows")
    outputs = {
        "qwen35_4b": load_jsonl(qwen35_output),
        "qwen36_27b": load_jsonl(qwen36_output),
    }
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for alias, rows in outputs.items():
        by_id = {str(row["id"]): row for row in rows}
        if len(rows) != SPEC["rows"] or set(by_id) != set(runtime_by_id):
            raise ValueError(f"{alias} output does not exactly cover frozen runtime")
        if any(row.get("candidate_alias") != alias for row in rows):
            raise ValueError(f"{alias} identity mismatch")
        indexed[alias] = by_id

    records: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, str]] = {}
    for row_id, source in runtime_by_id.items():
        pair_id = "P672-" + stable_hash(SPEC["namespace"], "pair", row_id)[:10]
        aliases = sorted(
            outputs,
            key=lambda alias: stable_hash(SPEC["namespace"], "slot", row_id, alias),
        )
        for slot_index, alias in enumerate(aliases):
            candidate = indexed[alias][row_id]
            parsed = candidate.get("parsed") if isinstance(candidate.get("parsed"), dict) else {}
            audit_id = "R672-" + stable_hash(
                SPEC["namespace"], "audit", row_id, alias
            )[:12]
            mapping[audit_id] = {"row_id": row_id, "candidate_alias": alias}
            records.append(
                {
                    "audit_id": audit_id,
                    "pair_id": pair_id,
                    "blind_slot": "A" if slot_index == 0 else "B",
                    "row_id": row_id,
                    "semantic_component": source["semantic_component"],
                    "category": source["category"],
                    "name": source["name"],
                    "description": source["description"],
                    "image_file": f"images/{row_id}.jpg",
                    "frozen_prediction": source["frozen_prediction"],
                    "automatic_contract_valid": str(bool(candidate.get("contract_valid"))).lower(),
                    "candidate_verdict": parsed.get("verdict", ""),
                    "candidate_sold_object": parsed.get("sold_object", ""),
                    "candidate_regulated_substance_or_marker": parsed.get(
                        "regulated_substance_or_marker", ""
                    ),
                    "candidate_relation": parsed.get("relation", ""),
                    "candidate_evidence_source": parsed.get("evidence_source", ""),
                    "candidate_evidence_quote": parsed.get("evidence_quote", ""),
                    "candidate_explanation": parsed.get("explanation", ""),
                    "candidate_raw_response": candidate.get("raw_response", ""),
                }
            )
    records.sort(key=lambda row: stable_hash(SPEC["namespace"], "review_order", row["audit_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "blind_candidates_80.csv"
    with candidate_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CANDIDATE_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(records)
    review_path = output_dir / "review_form_80.csv"
    with review_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS, extrasaction="raise")
        writer.writeheader()
        for row in records:
            writer.writerow({"audit_id": row["audit_id"], **{key: "" for key in REVIEW_FIELDS[1:]}})
    manifest = {
        "schema_version": "exp672_blind_manifest_v1",
        "experiment_id": "672",
        "rows": len(records),
        "pairs": len(runtime_rows),
        "runtime_sha256": sha256_file(runtime),
        "candidate_input_sha256": {
            "qwen35_4b": sha256_file(qwen35_output),
            "qwen36_27b": sha256_file(qwen36_output),
        },
        "blind_candidates_sha256": sha256_file(candidate_path),
        "review_template_sha256": sha256_file(review_path),
        "mapping": mapping,
        "labels_read": 0,
        "sealed_rows_read": 0,
        "public_used": False,
    }
    manifest_path = output_dir / "private_unblinding_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "rows": len(records),
        "pairs": len(runtime_rows),
        "blind_candidates_sha256": manifest["blind_candidates_sha256"],
        "private_manifest_sha256": sha256_file(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--qwen35-output", type=Path, required=True)
    parser.add_argument("--qwen36-output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    print(json.dumps(build(**vars(parser.parse_args())), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
