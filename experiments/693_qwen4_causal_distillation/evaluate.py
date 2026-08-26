from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

FLAMMABLE = "Легковоспламеняющиеся"
BAD = "БАД"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def sha256_bytes(values: list[int]) -> str:
    return hashlib.sha256(bytes(values)).hexdigest()


def tie_aware_ap(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0
    groups: dict[float, list[int]] = defaultdict(list)
    for label, score in zip(labels, scores, strict=True):
        groups[float(score)].append(int(label))
    seen = 0
    true_seen = 0
    contribution = 0.0
    for score in sorted(groups, reverse=True):
        group = groups[score]
        size = len(group)
        pos = sum(group)
        if pos:
            # Expected precision of a positive under every permutation of a tie block.
            for rank in range(1, size + 1):
                expected_before = (rank - 1) * (pos - 1) / max(1, size - 1)
                contribution += (pos / size) * (true_seen + 1 + expected_before) / (seen + rank)
        seen += size
        true_seen += pos
    return contribution / positives


def classification(labels: list[int], predictions: list[int]) -> dict[str, float | int]:
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions, strict=True))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions, strict=True))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "fp": fp, "fn": fn}


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bad_candidate = [row["candidate"] for row in rows if row["category"] == BAD]
    bad_baseline = [row["baseline"] for row in rows if row["category"] == BAD]
    if bad_candidate != bad_baseline:
        raise ValueError("BAD production replay is not byte-identical")
    categories = {}
    for category in (BAD, FLAMMABLE):
        local = [row for row in rows if row["category"] == category]
        labels = [row["label"] for row in local]
        categories[category] = {
            "control": classification(labels, [row["control"] for row in local]),
            "candidate": classification(labels, [row["candidate"] for row in local]),
        }
    control_macro = sum(categories[c]["control"]["f1"] for c in categories) / 2
    candidate_macro = sum(categories[c]["candidate"]["f1"] for c in categories) / 2
    flammable = [row for row in rows if row["category"] == FLAMMABLE]
    corrected = sum(
        row["control"] != row["label"] and row["candidate"] == row["label"] for row in rows
    )
    regressed = sum(
        row["control"] == row["label"] and row["candidate"] != row["label"] for row in rows
    )
    singleton = [row for row in rows if int(row.get("component_size", 1)) == 1]
    singleton_corrected = sum(
        row["control"] != row["label"] and row["candidate"] == row["label"] for row in singleton
    )
    singleton_regressed = sum(
        row["control"] == row["label"] and row["candidate"] != row["label"] for row in singleton
    )
    return {
        "tie_aware_ap": {
            "control": tie_aware_ap(
                [r["label"] for r in flammable], [r["control_score"] for r in flammable]
            ),
            "candidate": tie_aware_ap(
                [r["label"] for r in flammable], [r["candidate_score"] for r in flammable]
            ),
        },
        "macro": {
            "control": control_macro,
            "candidate": candidate_macro,
            "delta": candidate_macro - control_macro,
        },
        "categories": categories,
        "corrected": corrected,
        "regressed": regressed,
        "singleton_delta": singleton_corrected - singleton_regressed,
        "bad_exact": True,
        "bad_prediction_sha256": sha256_bytes(bad_candidate),
        "rows": len(rows),
    }


def aligned_rows(
    runtime_root: Path, baseline_root: Path, control_root: Path, candidate_root: Path
) -> list[dict[str, Any]]:
    output = []
    for fold in range(5):
        truth = {
            int(row["global_index"]): row
            for row in read_jsonl(runtime_root / f"fold_{fold}" / "validation.jsonl")
        }
        streams = []
        for root in (baseline_root, control_root, candidate_root):
            path = root / f"fold_{fold}" / "predictions.jsonl"
            streams.append({int(row["global_index"]): row for row in read_jsonl(path)})
        if any(set(stream) != set(truth) for stream in streams):
            raise ValueError(f"prediction alignment mismatch in fold {fold}")
        baseline, control, candidate = streams
        for key, source in truth.items():
            category = str(source["category"])
            baseline_prediction = int(baseline[key]["prediction"])
            output.append(
                {
                    "fold": fold,
                    "label": int(source["label"]),
                    "category": category,
                    "component_size": int(source.get("component_size", 1)),
                    "baseline": baseline_prediction,
                    "control": baseline_prediction
                    if category == BAD
                    else int(control[key]["prediction"]),
                    "candidate": baseline_prediction
                    if category == BAD
                    else int(candidate[key]["prediction"]),
                    "control_score": float(control[key]["score"]),
                    "candidate_score": float(candidate[key]["score"]),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite evaluation")
    rows = aligned_rows(
        args.runtime_root, args.baseline_root, args.control_root, args.candidate_root
    )
    report = {
        "folds": {
            str(fold): evaluate_rows([r for r in rows if r["fold"] == fold]) for fold in range(5)
        },
        "pooled": evaluate_rows(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
