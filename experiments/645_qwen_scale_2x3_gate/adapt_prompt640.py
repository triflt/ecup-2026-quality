from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from grid_contract import (
    GRID_CONTRACT_SHA256,
    MODEL_REVISIONS,
    NO_EVIDENCE,
    PREPROCESSING_VERSION,
    PROMPT_VERSION,
    canonical_sha256,
    sha256_file,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def adapt(
    *,
    runtime_path: Path,
    score_paths: list[Path],
    report_paths: list[Path],
    model_id: str,
    model_revision: str,
    output_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    if output_path.exists() or contract_path.exists():
        raise FileExistsError("refusing to overwrite normalized prompting outputs")
    if MODEL_REVISIONS.get(model_id) != model_revision:
        raise ValueError("prompting model identity differs from the frozen grid")
    runtime = read_jsonl(runtime_path)
    if len(runtime) != 11118:
        raise ValueError("prompt runtime must contain all 11118 development rows")
    expected_keys = {
        "global_index",
        "id",
        "fold",
        "category",
        "name",
        "description",
        "image_url",
    }
    if any(set(row) != expected_keys for row in runtime):
        raise ValueError("prompt runtime schema differs from experiment 640")
    runtime.sort(key=lambda row: int(row["global_index"]))
    if [int(row["global_index"]) for row in runtime] != list(range(11118)):
        raise ValueError("prompt runtime global indices are not complete")

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    if len(reports) != len(score_paths):
        raise ValueError("every score shard needs its experiment-640 report")
    runtime_sha = sha256_file(runtime_path)
    for report, score_path in zip(reports, score_paths, strict=True):
        expected = {
            "experiment_id": "640",
            "model_id": model_id,
            "model_revision": model_revision,
            "runtime_sha256": runtime_sha,
            "output_sha256": sha256_file(score_path),
            "thinking": False,
            "threshold": 0.0,
        }
        mismatch = {
            key: {"expected": value, "actual": report.get(key)}
            for key, value in expected.items()
            if report.get(key) != value
        }
        if mismatch:
            raise ValueError(f"experiment-640 shard contract mismatch: {mismatch}")

    scores = [row for path in score_paths for row in read_jsonl(path)]
    scores.sort(key=lambda row: int(row["global_index"]))
    if len(scores) != 11118 or [int(row["global_index"]) for row in scores] != list(range(11118)):
        raise ValueError("prompting shards do not exactly cover development")
    if len({str(row["id"]) for row in scores}) != 11118:
        raise ValueError("prompting scores contain duplicate IDs")
    normalized: list[dict[str, Any]] = []
    for source, runtime_row in zip(scores, runtime, strict=True):
        for key in ("id", "fold", "category"):
            if str(source[key]) != str(runtime_row[key]):
                raise ValueError(f"prompt score {key} differs from runtime")
        score = float(source["score"])
        prediction = int(source["prediction"])
        if prediction != int(score >= 0.0):
            raise ValueError("prompt prediction differs from frozen zero threshold")
        normalized.append(
            {
                "global_index": int(source["global_index"]),
                "id": str(source["id"]),
                "fold": int(source["fold"]),
                "category": str(source["category"]),
                "score": score,
                "prediction": prediction,
                "model_id": model_id,
                "model_revision": model_revision,
                "objective": "prompting",
                "target_order": "none",
                "prompt_version": PROMPT_VERSION,
                "preprocessing_version": PREPROCESSING_VERSION,
                "raw_generation": "",
                "generated_verdict": -1,
                "format_valid": True,
                "quote": NO_EVIDENCE,
                "concept": NO_EVIDENCE,
                "grounded": True,
                "grounding_source": "none",
                "image_index": -1,
                "region_index": -1,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in normalized:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    input_view = [
        {
            key: row[key]
            for key in (
                "global_index",
                "id",
                "fold",
                "category",
                "name",
                "description",
                "image_url",
            )
        }
        for row in runtime
    ]
    contract = {
        "schema_version": 1,
        "experiment_id": "640",
        "model_id": model_id,
        "model_revision": model_revision,
        "objective": "prompting",
        "grid_contract_sha256": GRID_CONTRACT_SHA256,
        "model_input_view_sha256": canonical_sha256(input_view),
        "runtime_sha256": runtime_sha,
        "prediction_rows": len(normalized),
        "threshold": 0.0,
        "threshold_tuned": False,
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "artifacts": {output_path.name: sha256_file(output_path)},
        "decision": "GO_EVALUATE",
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize fixed experiment-640 prompting shards.")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--model-id", choices=sorted(MODEL_REVISIONS), required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    result = adapt(
        runtime_path=args.runtime,
        score_paths=args.score,
        report_paths=args.report,
        model_id=args.model_id,
        model_revision=args.model_revision,
        output_path=args.output,
        contract_path=args.contract,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
