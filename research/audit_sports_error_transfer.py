from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_component_decision_survival import (
    apply_downstream_priors,
    build_neighbor_graph,
    prepare_frame,
)


ROOT = Path(__file__).resolve().parents[1]
TARGETS = (
    ROOT
    / "experiments/290_minicpm_v46_visual_screen/analysis/agent_hard_errors/hard_error_cases.csv"
)
SELECTOR = (
    ROOT
    / "experiments/320_bad_regulatory_evidence/artifacts/bad_regulatory_selector_v1.csv.gz"
)
BAD = "БАД"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--candidate-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrays = np.load(args.predictions, allow_pickle=False)
    frame = prepare_frame(args.data, arrays)
    required = {"baseline_nested_predictions", args.candidate_key}
    missing = required - set(arrays.files)
    if missing:
        raise ValueError(f"prediction archive is missing keys: {sorted(missing)}")
    neighbor_indices, neighbor_scores = build_neighbor_graph(frame)
    baseline, _ = apply_downstream_priors(
        frame,
        arrays["baseline_nested_predictions"].astype(np.int8),
        neighbor_indices,
        neighbor_scores,
    )
    candidate, _ = apply_downstream_priors(
        frame,
        arrays[args.candidate_key].astype(np.int8),
        neighbor_indices,
        neighbor_scores,
    )
    labels = arrays["labels"].astype(np.int8)
    categories = arrays["categories"].astype(str)
    ids = arrays["ids"].astype(str)
    hard = pd.read_csv(TARGETS, dtype={"id": str})
    target_ids = set(
        hard.loc[
            (hard.category == BAD)
            & (hard.product_type == "sports_nutrition")
            & (hard.baseline_after_prior != hard.label),
            "id",
        ]
    )
    if len(target_ids) != 243:
        raise ValueError(f"expected 243 frozen sports errors, got {len(target_ids)}")
    target = np.asarray([item in target_ids for item in ids])
    if not np.all(baseline[target] != labels[target]):
        raise ValueError("frozen target ids are no longer baseline errors")
    selector = pd.read_csv(SELECTOR, dtype={"id": str}).set_index("id")
    sports_ids = set(selector.index[selector.sports_nutrition.astype(bool)])
    sports = (categories == BAD) & np.asarray([item in sports_ids for item in ids])
    baseline_error = baseline != labels
    candidate_error = candidate != labels
    result = {
        "frozen_common_sports_errors": 243,
        "fixed_from_frozen_243": int((target & ~candidate_error).sum()),
        "remaining_from_frozen_243": int((target & candidate_error).sum()),
        "cue_defined_sports_rows": int(sports.sum()),
        "cue_defined_baseline_errors": int((sports & baseline_error).sum()),
        "cue_defined_candidate_errors": int((sports & candidate_error).sum()),
        "new_sports_errors_created": int(
            (sports & ~baseline_error & candidate_error).sum()
        ),
        "sports_errors_corrected": int(
            (sports & baseline_error & ~candidate_error).sum()
        ),
        "guardrail_breach": bool(
            (sports & ~baseline_error & candidate_error).sum()
            > (sports & baseline_error & ~candidate_error).sum()
        ),
        "input_sha256": {
            "predictions": sha256(args.predictions),
            "targets": sha256(TARGETS),
            "selector": sha256(SELECTOR),
            "data": sha256(args.data),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
