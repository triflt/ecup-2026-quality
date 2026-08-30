from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "experiments/645_qwen_scale_2x3_gate"
STUDENT_DIR = ROOT / "experiments/698_qwen35_teacher_guided_controls"
sys.path.insert(0, str(SHARED))
sys.path.insert(0, str(STUDENT_DIR))

import grid_contract
import run_fold as student


MODEL = Path(os.environ.get("ECUP_MODEL_DIR", "models/Qwen3.5-4B"))
FLAMMABLE = "Легковоспламеняющиеся"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def binary_metrics(labels: list[int], scores: list[float]) -> dict:
    if len(labels) != len(scores) or not labels or set(labels) != {0, 1}:
        raise ValueError("binary metric input is invalid")
    positives = sum(labels)
    negatives = len(labels) - positives
    order_ascending = sorted(range(len(scores)), key=lambda index: scores[index])
    rank_sum = 0.0
    start = 0
    while start < len(order_ascending):
        end = start + 1
        while end < len(order_ascending) and scores[order_ascending[end]] == scores[order_ascending[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        rank_sum += average_rank * sum(labels[order_ascending[position]] for position in range(start, end))
        start = end
    roc_auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)

    order_descending = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    true_positives = 0
    average_precision_total = 0.0
    best = {"f1": -1.0, "threshold": None, "precision": None, "recall": None}
    for position, index in enumerate(order_descending, start=1):
        if labels[index]:
            true_positives += 1
            average_precision_total += true_positives / position
        boundary = position == len(order_descending) or scores[index] != scores[order_descending[position]]
        if boundary:
            false_positives = position - true_positives
            false_negatives = positives - true_positives
            precision = true_positives / max(1, true_positives + false_positives)
            recall = true_positives / max(1, true_positives + false_negatives)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            if f1 > best["f1"]:
                best = {
                    "f1": f1,
                    "threshold": scores[index],
                    "precision": precision,
                    "recall": recall,
                }
    return {
        "rows": len(labels),
        "positives": positives,
        "roc_auc": roc_auc,
        "average_precision": average_precision_total / positives,
        "best_f1_diagnostic": best,
    }


def main() -> None:
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-jsonl", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--category", default=FLAMMABLE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--folds-csv",
        type=Path,
        default=ROOT / "validation/grouped_text_v1/folds.csv",
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty score output")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.folds_csv.open(encoding="utf-8", newline="") as stream:
        fold_contract = {str(row["id"]): row for row in csv.DictReader(stream)}
    rows = []
    for raw in read_jsonl(args.runtime_jsonl):
        if str(raw["category"]) != args.category:
            continue
        row = dict(raw)
        expected = fold_contract.get(str(row["id"]))
        if expected is None:
            raise ValueError(f"id absent from fold contract: {row['id']}")
        if int(expected["fold"]) != int(row["fold"]) or str(expected["category"]) != str(row["category"]):
            raise ValueError(f"fold contract mismatch for id={row['id']}")
        row["label"] = int(expected["label"])
        rows.append(row)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("no rows selected")

    student.base_prompt = grid_contract.base_prompt
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1:
        raise ValueError("binary tokens are not atomic")
    base = AutoModelForMultimodalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model.eval()
    predictions: list[dict] = []
    started = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(rows), args.batch_size):
            local = rows[offset : offset + args.batch_size]
            images = [student.open_image(row) for row in local]
            try:
                values = student.scores(
                    model,
                    student.batch_inputs(
                        processor,
                        [SimpleNamespace(**row) for row in local],
                        images,
                    ),
                    zero[0],
                    one[0],
                )
            finally:
                for image in images:
                    image.close()
            predictions.extend(
                {
                    "id": str(row["id"]),
                    "fold": int(row["fold"]),
                    "category": str(row["category"]),
                    "label": int(row["label"]),
                    "score": float(score),
                }
                for row, score in zip(local, values.float().cpu(), strict=True)
            )
            processed = offset + len(local)
            if processed % 200 == 0 or processed == len(rows):
                print(json.dumps({"predicted": processed, "rows": len(rows)}), flush=True)
    prediction_path = args.output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metrics = binary_metrics(
        [int(row["label"]) for row in predictions],
        [float(row["score"]) for row in predictions],
    )
    report = {
        "schema_version": "exp706_adapter_score_v1",
        "strict_holdout": True,
        "runtime_jsonl": str(args.runtime_jsonl),
        "runtime_sha256": sha256(args.runtime_jsonl),
        "adapter": str(args.adapter),
        "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
        "category": args.category,
        "elapsed_minutes": (time.monotonic() - started) / 60,
        "predictions_sha256": sha256(prediction_path),
        "metrics": metrics,
    }
    if not all(math.isfinite(float(value)) for value in (metrics["roc_auc"], metrics["average_precision"])):
        raise ValueError("non-finite evaluation metric")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
