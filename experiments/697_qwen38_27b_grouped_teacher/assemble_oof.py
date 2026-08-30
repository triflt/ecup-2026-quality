from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DEFAULT_DATA = Path(os.environ.get("ECUP_DATA", "data/data.csv"))
DEFAULT_FOLDS = (
    Path(__file__).resolve().parents[2] / "validation/grouped_text_v1/folds.csv"
)
EXPECTED_DATA_SHA256 = "4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(labels) == 0 or len(np.unique(labels)) != 2:
        raise ValueError("threshold donor must contain both labels")
    order = np.argsort(scores, kind="mergesort")[::-1]
    ordered = labels[order]
    tp = np.cumsum(ordered == 1)
    fp = np.cumsum(ordered == 0)
    fn = int((labels == 1).sum()) - tp
    values = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    best = int(np.argmax(values))
    if best + 1 == len(scores):
        return float(scores[order[best]] - 1e-7)
    return float((scores[order[best]] + scores[order[best + 1]]) / 2)


def fold_category_percentile_rank(
    scores: np.ndarray, folds: np.ndarray, categories: np.ndarray
) -> np.ndarray:
    """Make scores from independently trained fold adapters comparable.

    The transform is deliberately label-blind: each validation fold/category is
    ranked using only its own scores.  Labels are used later only to fit a
    threshold on donor folds.
    """
    if not (len(scores) == len(folds) == len(categories)) or len(scores) == 0:
        raise ValueError("rank calibration inputs must be non-empty and aligned")
    frame = pd.DataFrame(
        {
            "score": np.asarray(scores, dtype=np.float64),
            "fold": np.asarray(folds),
            "category": np.asarray(categories),
        }
    )
    if not np.isfinite(frame["score"].to_numpy()).all():
        raise ValueError("rank calibration scores must be finite")
    ranks = frame.groupby(
        ["fold", "category"], sort=False, dropna=False
    )["score"].rank(method="average", pct=True)
    values = ranks.to_numpy(np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0.0) or np.any(values > 1.0):
        raise ValueError("rank calibration produced invalid values")
    return values


def load_fold_output(directory: Path, fold: int) -> tuple[list[dict], dict]:
    prediction_path = directory / "predictions.jsonl"
    contract_path = directory / "output_contract.json"
    for path in (prediction_path, contract_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing fold {fold} artifact: {path}")
    contract = json.loads(contract_path.read_text())
    expected = {
        "experiment_id": "697",
        "fold": fold,
        "technical_smoke": False,
    }
    mismatch = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise ValueError(f"fold {fold} output contract mismatch: {mismatch}")
    if contract.get("predictions_sha256") != sha256(prediction_path):
        raise ValueError(f"fold {fold} prediction checksum mismatch")
    rows = read_jsonl(prediction_path)
    if len(rows) != int(contract["validation_rows"]):
        raise ValueError(f"fold {fold} prediction row count mismatch")
    if any(set(row) != {"id", "fold", "score"} for row in rows):
        raise ValueError(f"fold {fold} prediction schema mismatch")
    if any(int(row["fold"]) != fold or not math.isfinite(float(row["score"])) for row in rows):
        raise ValueError(f"fold {fold} prediction value mismatch")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"fold {fold} contains duplicate IDs")
    return rows, contract


def assemble(
    *,
    data_path: Path,
    folds_path: Path,
    experiment_dir: Path,
    output_dir: Path,
    expected_data_sha256: str = EXPECTED_DATA_SHA256,
) -> dict:
    actual_data_sha256 = sha256(data_path)
    if actual_data_sha256 != expected_data_sha256:
        raise ValueError("canonical data checksum mismatch")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty OOF output")
    data = pd.read_csv(data_path, dtype={"id": str})
    folds = pd.read_csv(folds_path, dtype={"id": str})
    if data["id"].duplicated().any() or folds["id"].duplicated().any():
        raise ValueError("duplicate source IDs")
    if set(data["id"]) != set(folds["id"]):
        raise ValueError("data/folds ID mismatch")
    fold_map = dict(zip(folds["id"], folds["fold"].astype(int), strict=True))
    data["fold"] = [fold_map[row_id] for row_id in data["id"]]
    if set(data["fold"].unique()) != set(range(5)):
        raise ValueError("fold registry must contain exactly folds 0..4")

    score_by_id: dict[str, float] = {}
    contracts: dict[str, dict] = {}
    for fold in range(5):
        directory = experiment_dir / ".local" / f"output-f{fold}"
        rows, contract = load_fold_output(directory, fold)
        expected_ids = set(data.loc[data["fold"] == fold, "id"])
        actual_ids = {str(row["id"]) for row in rows}
        if actual_ids != expected_ids:
            raise ValueError(f"fold {fold} prediction ID coverage mismatch")
        overlap = set(score_by_id) & actual_ids
        if overlap:
            raise ValueError(f"cross-fold duplicate predictions: {sorted(overlap)[:3]}")
        score_by_id.update({str(row["id"]): float(row["score"]) for row in rows})
        contracts[str(fold)] = {
            "output_contract_sha256": sha256(directory / "output_contract.json"),
            "predictions_sha256": sha256(directory / "predictions.jsonl"),
            "validation_rows": len(rows),
        }
    if set(score_by_id) != set(data["id"]):
        raise ValueError("five-fold predictions do not cover canonical data")

    data["teacher_score"] = [score_by_id[row_id] for row_id in data["id"]]
    data["teacher_rank"] = fold_category_percentile_rank(
        data["teacher_score"].to_numpy(np.float64),
        data["fold"].to_numpy(np.int8),
        data["category"].astype(str).to_numpy(),
    )
    data["teacher_prediction"] = np.zeros(len(data), dtype=np.int8)
    nested: dict[str, dict] = {}
    for category, positions in data.groupby("category", sort=True).groups.items():
        positions = np.asarray(list(positions), dtype=np.int64)
        labels = data.loc[positions, "label"].to_numpy(np.int8)
        scores = data.loc[positions, "teacher_rank"].to_numpy(np.float64)
        local_folds = data.loc[positions, "fold"].to_numpy(np.int8)
        predictions = np.zeros(len(positions), dtype=np.int8)
        fold_reports: dict[str, dict] = {}
        for fold in range(5):
            validation = local_folds == fold
            donor = ~validation
            threshold = best_threshold(labels[donor], scores[donor])
            predictions[validation] = scores[validation] >= threshold
            fold_reports[str(fold)] = {
                "rows": int(validation.sum()),
                "threshold": threshold,
                "f1": float(f1_score(labels[validation], predictions[validation])),
            }
        data.loc[positions, "teacher_prediction"] = predictions
        nested[str(category)] = {
            "rows": len(positions),
            "f1": float(f1_score(labels, predictions)),
            "folds": fold_reports,
        }

    category_f1 = [value["f1"] for value in nested.values()]
    macro = float(np.mean(category_f1))
    output_dir.mkdir(parents=True)
    oof_path = output_dir / "teacher_oof.csv"
    columns = [
        "id",
        "fold",
        "category",
        "label",
        "teacher_score",
        "teacher_rank",
        "teacher_prediction",
    ]
    data.to_csv(oof_path, columns=columns, index=False, quoting=csv.QUOTE_MINIMAL)
    report = {
        "schema_version": "exp697_teacher_oof_v2",
        "experiment_id": "697",
        "evaluation": "nested_grouped_fold_category_percentile_rank_v2",
        "score_calibration": "label_blind_percentile_rank_within_fold_and_category_average_ties",
        "raw_teacher_score_preserved": True,
        "rows": len(data),
        "unique_ids": int(data["id"].nunique()),
        "folds": {str(key): int(value) for key, value in data["fold"].value_counts().sort_index().items()},
        "categories": nested,
        "nested_macro_f1": macro,
        "solution_140_nested_macro_f1": 0.9118425205786493,
        "standalone_delta_vs_solution_140": macro - 0.9118425205786493,
        "teacher_oof_sha256": sha256(oof_path),
        "data_sha256": actual_data_sha256,
        "folds_sha256": sha256(folds_path),
        "fold_artifacts": contracts,
        "decision": "READY_FOR_COMPONENT_AND_DISTILLATION_ANALYSIS",
    }
    report_path = output_dir / "teacher_oof_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-data-sha256",
        default=EXPECTED_DATA_SHA256,
        help="fail closed unless the input data has this SHA-256",
    )
    args = parser.parse_args()
    report = assemble(
        data_path=args.data,
        folds_path=args.folds,
        experiment_dir=args.experiment_dir,
        output_dir=args.output_dir,
        expected_data_sha256=args.expected_data_sha256,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
