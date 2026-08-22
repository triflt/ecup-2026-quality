from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_370 = ROOT / "experiments/370_flammable_metric_disagreement_replication/run.py"
spec = importlib.util.spec_from_file_location("exp370_metric_replication", SOURCE_370)
replication = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(replication)

EMBEDDINGS = ROOT / "research/all-image-artifacts/extracted/train_embeddings_fp16.npz"
QWEN35 = ROOT / "research/qwen35-hard-5fold-robust-fusion-report.npz"
QWEN3VL = ROOT / "research/lora-hard-5fold-robust-fusion-report.npz"
LOCKED = ROOT / "validation/locked_190_nested_v1/adapter_replacement_report.npz"
GUARD = ROOT / "validation/connected_family_guard_v2/rows.csv"
REPEATS = ROOT / "validation/connected_family_repeated_v1/rows.csv"
OUT = Path(__file__).resolve().parent / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    archive = np.load(EMBEDDINGS, allow_pickle=False)
    qwen35 = np.load(QWEN35, allow_pickle=True)
    qwen3vl = np.load(QWEN3VL, allow_pickle=True)
    locked = np.load(LOCKED, allow_pickle=False)
    ids = locked["ids"].astype(str)
    labels = locked["labels"].astype(np.int8)
    categories = locked["categories"].astype(str)
    for source in (archive, qwen35, qwen3vl):
        if not np.array_equal(source["ids"].astype(str), ids):
            raise ValueError("id mismatch")
        if not np.array_equal(source["labels"].astype(np.int8), labels):
            raise ValueError("label mismatch")
        if not np.array_equal(source["categories"].astype(str), categories):
            raise ValueError("category mismatch")

    embeddings = replication.metric.normalize_rows(archive["embeddings"].astype(np.float32))
    guard = pd.read_csv(GUARD, dtype={"id": str}).set_index("id").loc[ids]
    repeats = pd.read_csv(REPEATS, dtype={"id": str}).set_index("id").loc[ids]
    safe = guard.safe_for_selection.to_numpy(bool)
    components = guard.connected_component.astype(str).to_numpy()
    matrix = np.column_stack([
        qwen35["base_rank"].astype(np.float32),
        qwen3vl["lora_rank"].astype(np.float32),
        qwen35["lora_rank"].astype(np.float32),
    ])
    baseline = replication.production_baseline(matrix, categories)
    topologies: dict[str, dict[str, object]] = {}
    arrays: dict[str, np.ndarray] = {}
    for index in range(3):
        name = f"repeat_{index}"
        folds = repeats[f"repeat_{index}_fold"].to_numpy(np.int8)
        candidate, scores, reports = replication.score_repeat(
            embeddings, labels, categories, folds, safe, components, baseline
        )
        topologies[name] = replication.summarize(
            name, labels, categories, folds, safe, components, baseline,
            candidate, reports, 38042 + index,
        )
        arrays[f"{name}_folds"] = folds
        arrays[f"{name}_candidate"] = candidate
        arrays[f"{name}_scores"] = scores
        row = topologies[name]
        print(
            f"{name} delta={row['delta_macro_f1']:.9f} "
            f"corrected={row['corrected']} regressed={row['regressed']}",
            flush=True,
        )

    deltas = [topologies[f"repeat_{i}"]["delta_macro_f1"] for i in range(3)]
    total_corrected = sum(topologies[f"repeat_{i}"]["corrected"] for i in range(3))
    total_regressed = sum(topologies[f"repeat_{i}"]["regressed"] for i in range(3))
    total_wins = sum(topologies[f"repeat_{i}"]["folds_won"] for i in range(3))
    gates = {
        "all_repeat_macro_deltas_positive": all(value > 0 for value in deltas),
        "all_repeat_flammable_deltas_positive": all(
            topologies[f"repeat_{i}"]["category_delta"][replication.FLAMMABLE] > 0
            for i in range(3)
        ),
        "repeat_mean_delta_at_least_0_001": float(np.mean(deltas)) >= 0.001,
        "fold_wins_at_least_11_of_15": total_wins >= 11,
        "total_corrected_to_regressed_at_least_2": total_corrected >= 2 * max(1, total_regressed),
        "each_corrected_to_regressed_at_least_1": all(
            topologies[f"repeat_{i}"]["corrected"] >= topologies[f"repeat_{i}"]["regressed"]
            for i in range(3)
        ),
        "each_bootstrap_probability_at_least_0_80": all(
            topologies[f"repeat_{i}"]["component_bootstrap"]["probability_delta_positive"] >= 0.80
            for i in range(3)
        ),
        "structural_invariants": all(
            all(value == 0 for value in topologies[f"repeat_{i}"]["structural_invariants"].values())
            for i in range(3)
        ),
    }
    result = {
        "experiment_id": "380",
        "evaluation_version": "all_image_flammable_metric_connected_replication_v1",
        "parent_experiment": "370_flammable_metric_disagreement_replication",
        "representation_change": "first-image to all-image production-parity embeddings",
        "recipe_frozen_before_all_image_evaluation": True,
        "topologies": topologies,
        "summary": {
            "repeat_deltas": deltas,
            "mean_delta": float(np.mean(deltas)),
            "total_fold_wins": total_wins,
            "total_corrected": total_corrected,
            "total_regressed": total_regressed,
        },
        "acceptance": gates,
        "accepted_for_full_refit": all(gates.values()),
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (SOURCE_370, EMBEDDINGS, QWEN35, QWEN3VL, LOCKED, GUARD, REPEATS)
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "replication_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        OUT / "replication_predictions.npz",
        ids=ids, labels=labels, categories=categories, safe=safe, baseline=baseline, **arrays,
    )
    print(json.dumps({"accepted_for_full_refit": result["accepted_for_full_refit"], **result["summary"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
