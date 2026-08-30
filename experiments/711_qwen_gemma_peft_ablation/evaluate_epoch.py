from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score


LORA_FUSION = {
    "БАД": (0.50, 0.25, 0.25, 0.27193570137023926),
    "Легковоспламеняющиеся": (0.15, 0.10, 0.75, 0.953912615776062),
}
PRIOR_CONFIGS = {
    "БАД": (2, 2 / 3, 2, 0.999),
    "Легковоспламеняющиеся": (1, 0.999, 999, 0.999),
}


def rank01(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float32)
    result[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return result


def summarize(labels, categories, predictions):
    result = {}
    for category in sorted(np.unique(categories)):
        mask = categories == category
        y, p = labels[mask], predictions[mask]
        result[category] = {
            "f1": float(f1_score(y, p)),
            "fp": int(((y == 0) & (p == 1)).sum()),
            "fn": int(((y == 1) & (p == 0)).sum()),
        }
    return {"macro_f1": float(np.mean([row["f1"] for row in result.values()])), "categories": result}


def normalize(value):
    return re.sub(r"\s+", " ", str(value or "").lower().replace("ё", "е")).strip()


def compose_text(name, description):
    name = normalize(name)
    return f"{name}\n{name}\n{normalize(description)}"


def fingerprint(value):
    return hashlib.sha1(normalize(value).encode()).hexdigest()


def apply_leave_one_out_prior(frame, predictions):
    result = predictions.copy()
    for category, (exact_min, exact_conf, name_min, name_conf) in PRIOR_CONFIGS.items():
        category_mask = frame["category"].astype(str).to_numpy() == category
        exact_groups = frame.groupby(["category", "text_hash"])["label"]
        name_groups = frame.groupby(["category", "normalized_name"])["label"]
        exact_count = exact_groups.transform("count").to_numpy(np.int32) - 1
        exact_sum = exact_groups.transform("sum").to_numpy(np.float32) - frame["label"].to_numpy(np.float32)
        name_count = name_groups.transform("count").to_numpy(np.int32) - 1
        name_sum = name_groups.transform("sum").to_numpy(np.float32) - frame["label"].to_numpy(np.float32)
        exact_mean = np.divide(exact_sum, exact_count, out=np.full(len(frame), np.nan), where=exact_count > 0)
        name_mean = np.divide(name_sum, name_count, out=np.full(len(frame), np.nan), where=name_count > 0)
        exact = category_mask & (exact_count >= exact_min) & (np.maximum(exact_mean, 1 - exact_mean) >= exact_conf)
        name = category_mask & ~exact & (name_count >= name_min) & (np.maximum(name_mean, 1 - name_mean) >= name_conf)
        result[exact] = (exact_mean[exact] >= 0.5).astype(np.int8)
        result[name] = (name_mean[name] >= 0.5).astype(np.int8)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--replace-leg", choices=("qwen3vl", "qwen35"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--four-head", type=Path, required=True)
    parser.add_argument("--qwen3vl", type=Path, required=True)
    parser.add_argument("--qwen35", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    candidate = pd.read_csv(args.predictions)
    four = np.load(args.four_head, allow_pickle=False)
    q3 = np.load(args.qwen3vl, allow_pickle=False)
    q35 = np.load(args.qwen35, allow_pickle=False)
    positions = np.flatnonzero(four["fold_ids"].astype(np.int8) == args.fold)
    expected_ids = four["ids"][positions].astype(str)
    if set(candidate["id"].astype(str)) != set(expected_ids):
        raise ValueError("candidate/frozen fold coverage mismatch")
    candidate = candidate.assign(_id=candidate["id"].astype(str)).set_index("_id").loc[expected_ids]
    labels = candidate["label"].to_numpy(np.int8)
    categories = candidate["category"].astype(str).to_numpy()
    scores = candidate["lora_score"].to_numpy(np.float32)
    if not np.array_equal(labels, four["labels"][positions].astype(np.int8)):
        raise ValueError("candidate labels/frozen fold mismatch")
    if not np.array_equal(categories, four["categories"][positions].astype(str)):
        raise ValueError("candidate categories/frozen fold mismatch")
    baseline = np.zeros(len(candidate), dtype=np.int8)
    before_prior = np.zeros(len(candidate), dtype=np.int8)
    for category, (w0, w1, w2, threshold) in LORA_FUSION.items():
        mask = categories == category
        local = positions[mask]
        robust = rank01(four["fused"][local].astype(np.float32))
        q3_values = rank01(q3["lora_rank"][local].astype(np.float32))
        q35_values = rank01(q35["lora_rank"][local].astype(np.float32))
        candidate_values = rank01(scores[mask])
        baseline_score = w0 * robust + w1 * q3_values + w2 * q35_values
        baseline[mask] = (baseline_score >= threshold).astype(np.int8)
        if args.replace_leg == "qwen3vl":
            candidate_score = w0 * robust + w1 * candidate_values + w2 * q35_values
        else:
            candidate_score = w0 * robust + w1 * q3_values + w2 * candidate_values
        before_prior[mask] = (candidate_score >= threshold).astype(np.int8)
    frame = pd.read_csv(args.data).fillna("")
    frame["normalized_name"] = frame["name"].map(normalize)
    frame["text_hash"] = [fingerprint(compose_text(a, b)) for a, b in zip(frame["name"], frame["description"])]
    full_baseline = np.zeros(len(frame), dtype=np.int8)
    full_candidate = np.zeros(len(frame), dtype=np.int8)
    full_baseline[positions] = baseline
    full_candidate[positions] = before_prior
    baseline = apply_leave_one_out_prior(frame, full_baseline)[positions]
    final = apply_leave_one_out_prior(frame, full_candidate)[positions]
    corrected = (baseline != labels) & (final == labels)
    regressed = (baseline == labels) & (final != labels)
    standalone = {}
    for category in sorted(np.unique(categories)):
        mask = categories == category
        ranked = rank01(scores[mask])
        standalone[category] = {
            "ap": float(average_precision_score(labels[mask], scores[mask])),
            "best_f1": max(float(f1_score(labels[mask], ranked >= threshold)) for threshold in np.unique(ranked)),
        }
    base_summary = summarize(labels, categories, baseline)
    final_summary = summarize(labels, categories, final)
    payload = {
        "schema": "exp711_peft_solution140_replay_v1",
        "fold": args.fold,
        "replace_leg": args.replace_leg,
        "rows": len(candidate),
        "standalone": standalone,
        "baseline": base_summary,
        "candidate": final_summary,
        "macro_delta": final_summary["macro_f1"] - base_summary["macro_f1"],
        "changed_decisions": int((baseline != final).sum()),
        "surviving_before_prior": int((baseline != before_prior).sum()),
        "corrections": int(corrected.sum()),
        "regressions": int(regressed.sum()),
        "public_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
