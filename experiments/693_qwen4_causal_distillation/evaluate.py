from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

FLAMMABLE = "Легковоспламеняющиеся"
BAD = "БАД"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def sha256_bytes(values: list[int]) -> str:
    return hashlib.sha256(bytes(values)).hexdigest()


def average_precision(labels: list[int], scores: list[float]) -> float:
    """Exact exp691/sklearn threshold-tie average precision."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be nonempty and aligned")
    positives = sum(int(value) for value in labels)
    if positives == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    true_positives = 0
    total = 0.0
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        score = float(scores[order[cursor]])
        while end < len(order) and float(scores[order[end]]) == score:
            end += 1
        group_positives = sum(int(labels[order[index]]) for index in range(cursor, end))
        true_positives += group_positives
        total += (true_positives / end) * group_positives
        cursor = end
    return total / positives


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
            "control": average_precision(
                [row["label"] for row in flammable],
                [row["control_score"] for row in flammable],
            ),
            "candidate": average_precision(
                [row["label"] for row in flammable],
                [row["candidate_score"] for row in flammable],
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


def verify_paired_arms_frozen(control_root: Path, candidate_root: Path) -> None:
    """Fail before opening any label registry unless both arms are complete."""
    for fold in range(5):
        for root in (control_root, candidate_root):
            directory = root / f"fold{fold}"
            predictions = directory / "predictions.jsonl"
            contract_path = directory / "output_contract.json"
            if not predictions.is_file() or not contract_path.is_file():
                raise ValueError(f"paired arm is not frozen for fold {fold}")
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            if (
                int(contract.get("outer_fold", -1)) != fold
                or contract.get("decision") != "GO_EVALUATE"
                or contract.get("technical_smoke") is not False
            ):
                raise ValueError(f"paired arm contract is not evaluable for fold {fold}")


def build_label_registry(runtime_root: Path) -> tuple[dict[int, int], dict[int, tuple[str, str]]]:
    labels: dict[int, int] = {}
    identities: dict[int, tuple[str, str]] = {}
    for fold in range(5):
        for row in read_jsonl(runtime_root / f"fold{fold}" / "train.jsonl"):
            key = int(row["global_index"])
            label = int(row["label"])
            identity = (str(row["id"]), str(row["category"]))
            if label not in (0, 1):
                raise ValueError("cross-fold registry contains a non-binary label")
            if key in labels and (labels[key] != label or identities[key] != identity):
                raise ValueError("cross-fold label registry conflict")
            labels[key] = label
            identities[key] = identity
    return labels, identities


def aligned_rows(
    runtime_root: Path,
    baseline_root: Path,
    control_root: Path,
    candidate_root: Path,
) -> list[dict[str, Any]]:
    verify_paired_arms_frozen(control_root, candidate_root)
    # This is intentionally after the paired-artifact gate above.
    labels, identities = build_label_registry(runtime_root)
    output = []
    for fold in range(5):
        validation_rows = read_jsonl(runtime_root / f"fold{fold}" / "validation.jsonl")
        if any("label" in row for row in validation_rows):
            raise ValueError("outer validation runtime must remain label-free")
        truth = {int(row["global_index"]): row for row in validation_rows}
        paths = (
            baseline_root / f"fold{fold}" / "predictions.jsonl",
            control_root / f"fold{fold}" / "predictions.jsonl",
            candidate_root / f"fold{fold}" / "predictions.jsonl",
        )
        streams = [{int(row["global_index"]): row for row in read_jsonl(path)} for path in paths]
        if any(set(stream) != set(truth) for stream in streams):
            raise ValueError(f"prediction alignment mismatch in fold {fold}")
        baseline, control, candidate = streams
        for key, source in truth.items():
            identity = (str(source["id"]), str(source["category"]))
            if identities.get(key) != identity or key not in labels:
                raise ValueError("outer validation identity is absent from cross-fold registry")
            category = identity[1]
            baseline_prediction = int(baseline[key]["prediction"])
            output.append(
                {
                    "fold": fold,
                    "label": labels[key],
                    "category": category,
                    "component_size": int(source.get("component_size", 1)),
                    "baseline": baseline_prediction,
                    "control": (
                        baseline_prediction if category == BAD else int(control[key]["prediction"])
                    ),
                    "candidate": (
                        baseline_prediction
                        if category == BAD
                        else int(candidate[key]["prediction"])
                    ),
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
            str(fold): evaluate_rows([row for row in rows if row["fold"] == fold])
            for fold in range(5)
        },
        "pooled": evaluate_rows(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
