from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from seed_protocol import (
    ACCEPTANCE,
    EXPERIMENT_ID,
    FOLDS,
    NEW_SEEDS,
    PROTOCOL_VERSION,
    REFERENCE_SEED,
    SEEDS,
    audit_candidate,
    canonical_sha256,
    load_exp600_dependencies,
    sha256_file,
    sigmoid,
    strictly_nested_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predeclared semantic-v3 seed variance grid.")
    parser.add_argument(
        "--seed-fold-output",
        action="append",
        nargs=3,
        metavar=("SEED", "FOLD", "DIRECTORY"),
        default=[],
        help="One completed task directory; provide exactly 20 seed/fold entries.",
    )
    parser.add_argument("--folds", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _read_completed_predictions(directory: Path, *, seed: int, fold: int) -> pd.DataFrame:
    reference = seed == REFERENCE_SEED
    contract_path = directory / (
        "output_contract.runtime.json" if reference else "seed_output_contract.runtime.json"
    )
    prediction_path = directory / "lora_holdout_predictions.csv"
    if not contract_path.is_file() or not prediction_path.is_file():
        raise FileNotFoundError("completed seed task lacks its output contract or predictions")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_experiment = "600" if reference else EXPERIMENT_ID
    identity_matches = (
        contract.get("experiment_id") == expected_experiment
        and contract.get("outer_fold") == fold
        and (contract.get("component") == "original" if reference else contract.get("seed") == seed)
    )
    if not identity_matches:
        raise ValueError("completed seed task contract does not match the requested seed/fold")
    if contract.get("decision") != "GO" or contract.get("sealed_rows_in_predictions") != 0:
        raise ValueError("completed seed task contract is not an eligible development-only output")
    if contract.get("predictions_sha256") != sha256_file(prediction_path):
        raise ValueError("completed seed predictions checksum differs from contract")
    frame = pd.read_csv(prediction_path, dtype={"id": str})
    expected = {"id", "category", "label", "fold", "lora_score"}
    if set(frame.columns) != expected or not frame["fold"].astype(int).eq(fold).all():
        raise ValueError("completed seed prediction schema or fold mismatch")
    if not np.isfinite(frame["lora_score"].to_numpy(dtype=np.float64)).all():
        raise ValueError("completed seed predictions contain non-finite logits")
    return frame


def load_grid(specifications: list[list[str]], folds_path: Path) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    required = {(seed, fold) for seed in SEEDS for fold in FOLDS}
    observed: dict[tuple[int, int], Path] = {}
    for raw_seed, raw_fold, raw_directory in specifications:
        seed, fold = int(raw_seed), int(raw_fold)
        key = (seed, fold)
        if key not in required or key in observed:
            raise ValueError("seed-fold outputs must be unique members of the predeclared 4x5 grid")
        observed[key] = Path(raw_directory).resolve()
    if set(observed) != required:
        missing = sorted(required - set(observed))
        raise ValueError(f"incomplete seed grid; missing {missing}")
    shared_protocol, _ = load_exp600_dependencies()
    frozen = shared_protocol.read_folds(folds_path.resolve())
    development = frozen.loc[frozen["split"].eq("development")].copy()
    development = development.sort_values("id", kind="stable").reset_index(drop=True)
    scores: dict[int, list[pd.DataFrame]] = {seed: [] for seed in SEEDS}
    for seed, fold in sorted(required):
        frame = _read_completed_predictions(observed[(seed, fold)], seed=seed, fold=fold)
        scores[seed].append(frame)
    reference = pd.concat(scores[REFERENCE_SEED], ignore_index=True)
    reference = reference.sort_values("id", kind="stable").reset_index(drop=True)
    if reference["id"].tolist() != development["id"].tolist():
        raise ValueError("reference seed OOF ids differ from the frozen development split")
    if not np.array_equal(reference["fold"].to_numpy(np.int8), development["development_fold"].to_numpy(np.int8)):
        raise ValueError("reference seed OOF folds differ from semantic-family-v3")
    if not np.array_equal(reference["label"].to_numpy(np.int8), development["label"].to_numpy(np.int8)):
        raise ValueError("reference labels differ from semantic-family-v3")
    if not np.array_equal(reference["category"].astype(str), development["category"].astype(str)):
        raise ValueError("reference categories differ from semantic-family-v3")
    output_scores: dict[int, np.ndarray] = {}
    for seed, parts in scores.items():
        frame = pd.concat(parts, ignore_index=True).sort_values("id", kind="stable").reset_index(drop=True)
        for column in ("id", "fold", "label", "category"):
            if not np.array_equal(frame[column].astype(reference[column].dtype), reference[column]):
                raise ValueError(f"seed {seed} {column} differs from reference seed")
        output_scores[seed] = sigmoid(frame["lora_score"].to_numpy(dtype=np.float64))
    return development, output_scores


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a completed seed-variance report")
    development, scores = load_grid(args.seed_fold_output, args.folds)
    labels = development["label"].to_numpy(np.int8)
    categories = development["category"].astype(str).to_numpy()
    folds = development["development_fold"].to_numpy(np.int8)
    components = development["semantic_component"].astype(str).to_numpy()
    baseline, baseline_calibration = strictly_nested_predictions(
        probabilities=scores[REFERENCE_SEED], labels=labels, categories=categories, folds=folds
    )
    reports = {}
    arrays = {"ids": development["id"].astype(str).to_numpy(), "labels": labels, "categories": categories, "folds": folds, "baseline_seed42_predictions": baseline}
    for seed in NEW_SEEDS:
        predictions, calibration = strictly_nested_predictions(
            probabilities=scores[seed], labels=labels, categories=categories, folds=folds
        )
        reports[f"seed_{seed}"] = audit_candidate(
            name=f"independent seed {seed}", labels=labels, categories=categories, folds=folds,
            components=components, baseline=baseline, candidate=predictions, calibration=calibration,
        )
        arrays[f"seed_{seed}_probability"] = scores[seed]
        arrays[f"seed_{seed}_predictions"] = predictions
    ensemble_probability = np.mean(np.stack([scores[seed] for seed in SEEDS], axis=0), axis=0)
    ensemble, ensemble_calibration = strictly_nested_predictions(
        probabilities=ensemble_probability, labels=labels, categories=categories, folds=folds
    )
    reports["fixed_mean_all_four"] = audit_candidate(
        name="fixed arithmetic probability mean of seeds 42, 31415, 271828, 161803",
        labels=labels, categories=categories, folds=folds, components=components,
        baseline=baseline, candidate=ensemble, calibration=ensemble_calibration,
    )
    arrays["fixed_mean_all_four_probability"] = ensemble_probability
    arrays["fixed_mean_all_four_predictions"] = ensemble
    report = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "purpose": "measure independent training variance and test one predeclared equal-weight probability ensemble",
        "development_rows": len(development),
        "sealed_holdout_rows_used": 0,
        "seeds": list(SEEDS),
        "reference_seed": REFERENCE_SEED,
        "new_seed_jobs": [{"seed": seed, "fold": fold} for seed in NEW_SEEDS for fold in FOLDS],
        "ensemble": {"type": "arithmetic_probability_mean", "weights": {str(seed): 0.25 for seed in SEEDS}, "weights_tuned": False},
        "calibration": {"strictly_nested": True, "only_thresholds": True, "baseline": baseline_calibration},
        "acceptance_rules": {**ACCEPTANCE},
        "reports": reports,
        "decision": "GO" if reports["fixed_mean_all_four"]["accepted"] else "NO_GO",
    }
    report["report_sha256"] = canonical_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
