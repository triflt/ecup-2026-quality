from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

FLAMMABLE = "Легковоспламеняющиеся"
VARIANTS = {
    "v1_positive40": ("v1", "positive_only", 40),
    "v2_positive19": ("v2", "positive_only", 19),
    "both_positive20": ("both", "positive_only", 20),
    "both_balanced80": ("both", "balanced", 80),
}


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def compose_text(row: dict[str, Any]) -> str:
    name = str(row.get("name") or "")
    return f"{name}\n{name}\n{str(row.get('description') or '')}"


def score_fold(
    train: list[dict[str, Any]], validation: list[dict[str, Any]]
) -> np.ndarray:
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    from sklearn.svm import LinearSVC

    result = np.full(len(validation), np.nan, dtype=np.float32)
    train_categories = np.asarray([str(row["category"]) for row in train])
    validation_categories = np.asarray([str(row["category"]) for row in validation])
    train_labels = np.asarray([int(row["label"]) for row in train], dtype=np.int8)
    train_text = np.asarray([compose_text(row) for row in train], dtype=object)
    validation_text = np.asarray(
        [compose_text(row) for row in validation], dtype=object
    )
    for category in sorted(set(validation_categories)):
        train_positions = np.flatnonzero(train_categories == category)
        validation_positions = np.flatnonzero(validation_categories == category)
        vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.997,
                        sublinear_tf=True,
                        max_features=180_000,
                        dtype=np.float32,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=3,
                        sublinear_tf=True,
                        max_features=220_000,
                        dtype=np.float32,
                    ),
                ),
            ]
        )
        train_matrix = vectorizer.fit_transform(train_text[train_positions])
        validation_matrix = vectorizer.transform(validation_text[validation_positions])
        if not sparse.isspmatrix_csr(train_matrix):
            train_matrix = train_matrix.tocsr()
        model = LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual="auto",
            max_iter=8000,
            random_state=42,
        )
        model.fit(train_matrix, train_labels[train_positions])
        result[validation_positions] = model.decision_function(
            validation_matrix
        ).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite TF-IDF scores")
    return result


def nested_component(evaluation, labels, categories, folds, scores):
    predictions = np.zeros(len(labels), dtype=np.int8)
    thresholds = {}
    for category in sorted(np.unique(categories)):
        positions = np.flatnonzero(categories == category)
        local, local_thresholds = evaluation.nested_component(
            labels[positions], scores[positions], folds[positions]
        )
        predictions[positions] = local
        thresholds[category] = local_thresholds
    return predictions, thresholds


def fold_deltas(
    evaluation, labels, categories, folds, candidate, baseline
) -> dict[str, float]:
    return {
        str(fold): evaluation.summarize(
            labels[folds == fold],
            categories[folds == fold],
            folds[folds == fold],
            candidate[folds == fold],
        )["macro_f1"]
        - evaluation.summarize(
            labels[folds == fold],
            categories[folds == fold],
            folds[folds == fold],
            baseline[folds == fold],
        )["macro_f1"]
        for fold in range(5)
    }


def main() -> None:
    from sklearn.metrics import average_precision_score

    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-builder", type=Path, required=True)
    parser.add_argument("--evaluation-module", type=Path, required=True)
    parser.add_argument("--parent-runtime-root", type=Path, required=True)
    parser.add_argument("--ranked-root", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--baseline-qwen3vl", type=Path, required=True)
    parser.add_argument("--baseline-qwen35", type=Path, required=True)
    parser.add_argument("--runtime-map", type=Path, required=True)
    parser.add_argument("--runtime-map-contract", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    builder = load_module("exp699_tfidf_runtime_builder", args.runtime_builder)
    evaluation = load_module("exp699_tfidf_evaluation", args.evaluation_module)
    gpu_evaluator = load_module(
        "exp699_tfidf_source_binding", Path(__file__).with_name("evaluate_gpu_screen.py")
    )
    oof = np.load(args.oof, allow_pickle=True)
    source_positions, folds, global_indices = gpu_evaluator.load_source_split(
        args.runtime_map,
        args.runtime_map_contract,
        oof["ids"].astype(str),
        oof["categories"].astype(str),
    )
    ids = oof["ids"].astype(str)[source_positions]
    labels = oof["labels"].astype(np.int8)[source_positions]
    categories = oof["categories"].astype(str)[source_positions]
    scores = {
        "real": np.full(len(ids), np.nan, dtype=np.float32),
        **{
            name: np.full(len(ids), np.nan, dtype=np.float32)
            for name in VARIANTS
        },
    }
    source_keys = {
        (str(ids[index]), int(folds[index]), str(categories[index])): index
        for index in range(len(ids))
    }
    if len(source_keys) != len(ids):
        raise ValueError("source binding key is not unique")
    args.work_root.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        parent = args.parent_runtime_root / f"fold{fold}"
        real_train = read_jsonl(parent / "train.jsonl")
        validation = read_jsonl(parent / "validation.jsonl")
        positions = np.asarray(
            [
                source_keys[(str(row["id"]), fold, str(row["category"]))]
                for row in validation
            ],
            dtype=np.int64,
        )
        if not np.array_equal(
            global_indices[positions],
            np.asarray([int(row["global_index"]) for row in validation]),
        ):
            raise ValueError(f"fold{fold} validation global-index mismatch")
        scores["real"][positions] = score_fold(real_train, validation)
        for name, (source, mode, cap) in VARIANTS.items():
            with tempfile.TemporaryDirectory(
                prefix=f"tfidf-{name}-f{fold}-", dir=args.work_root
            ) as temporary:
                runtime = Path(temporary) / "runtime"
                builder.build_runtime(
                    parent_dir=parent,
                    ranked_path=args.ranked_root / f"fold{fold}_ranked.jsonl",
                    output_dir=runtime,
                    fold=fold,
                    source=source,
                    mode=mode,
                    cap=cap,
                )
                candidate_train = read_jsonl(runtime / "train.jsonl")
                candidate_validation = read_jsonl(runtime / "validation.jsonl")
                if candidate_validation != validation:
                    raise ValueError(f"fold{fold} validation changed")
                scores[name][positions] = score_fold(
                    candidate_train, candidate_validation
                )
        print(json.dumps({"phase": "fold_complete", "fold": fold}), flush=True)
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise ValueError("TF-IDF OOF coverage mismatch")

    q3 = np.load(args.baseline_qwen3vl, allow_pickle=True)
    q35 = np.load(args.baseline_qwen35, allow_pickle=True)
    for source in (q3, q35):
        if (
            not np.array_equal(source["ids"].astype(str), oof["ids"].astype(str))
            or not np.array_equal(
                source["labels"].astype(np.int8), oof["labels"].astype(np.int8)
            )
        ):
            raise ValueError("baseline array binding mismatch")
    q3_rank = q3["lora_rank"].astype(np.float32)[source_positions]
    q35_rank = q35["lora_rank"].astype(np.float32)[source_positions]
    robust_production = q3["base_rank"].astype(np.float32)[source_positions]
    production_predictions, production_selection = evaluation.nested_fusion(
        labels,
        categories,
        folds,
        {
            "robust_base": robust_production,
            "qwen3vl": q3_rank,
            "qwen35": q35_rank,
        },
    )
    production_summary = evaluation.summarize(
        labels, categories, folds, production_predictions
    )
    flammable = categories == FLAMMABLE
    raw_fused = oof["fused"].astype(np.float32)[source_positions]
    raw_text = oof["text"].astype(np.float32)[source_positions]

    variants: dict[str, Any] = {}
    routed_predictions: dict[str, np.ndarray] = {}
    component_predictions: dict[str, np.ndarray] = {}
    for name, values in scores.items():
        component, thresholds = nested_component(
            evaluation, labels, categories, folds, values
        )
        component_predictions[name] = component
        robust = raw_fused.copy()
        robust[flammable] += 0.65 * (
            evaluation.rank01(values[flammable]) - raw_text[flammable]
        )
        robust = evaluation.fold_category_ranks(robust, folds, categories)
        routed, selection = evaluation.nested_fusion(
            labels,
            categories,
            folds,
            {"robust_base": robust, "qwen3vl": q3_rank, "qwen35": q35_rank},
        )
        routed_predictions[name] = routed
        variants[name] = {
            "component": {
                **evaluation.summarize(
                    labels, categories, folds, component, values
                ),
                "thresholds": thresholds,
            },
            "routed": {
                **evaluation.summarize(labels, categories, folds, routed),
                "selection": selection,
            },
            "flammable_ap": float(
                average_precision_score(labels[flammable], values[flammable])
            ),
        }
    real_component = variants["real"]["component"]
    real_routed = variants["real"]["routed"]
    baseline_fn = real_routed["categories"][FLAMMABLE]["confusion"]["fn"]
    for name in VARIANTS:
        component_comparison = evaluation.compare(
            component_predictions[name],
            component_predictions["real"],
            labels,
            categories,
        )
        routed_comparison = evaluation.compare(
            routed_predictions[name], routed_predictions["real"], labels, categories
        )
        production_comparison = evaluation.compare(
            routed_predictions[name], production_predictions, labels, categories
        )
        fold_delta = fold_deltas(
            evaluation,
            labels,
            categories,
            folds,
            routed_predictions[name],
            routed_predictions["real"],
        )
        macro_delta = variants[name]["routed"]["macro_f1"] - real_routed["macro_f1"]
        production_delta = (
            variants[name]["routed"]["macro_f1"] - production_summary["macro_f1"]
        )
        flammable_f1_delta = (
            variants[name]["routed"]["categories"][FLAMMABLE]["f1"]
            - real_routed["categories"][FLAMMABLE]["f1"]
        )
        fn = variants[name]["routed"]["categories"][FLAMMABLE]["confusion"]["fn"]
        variants[name]["vs_real"] = {
            "component_macro_delta": variants[name]["component"]["macro_f1"]
            - real_component["macro_f1"],
            "component": component_comparison,
            "routed_macro_delta": macro_delta,
            "routed_flammable_f1_delta": flammable_f1_delta,
            "routed_fold_macro_delta": fold_delta,
            "routed_fold_wins": sum(value > 0 for value in fold_delta.values()),
            "routed": routed_comparison,
            "flammable_ap_delta": variants[name]["flammable_ap"]
            - variants["real"]["flammable_ap"],
            "flammable_fn_delta": fn - baseline_fn,
        }
        variants[name]["vs_production"] = {
            "macro_delta": production_delta,
            "comparison": production_comparison,
        }
        variants[name]["full5_pass"] = bool(
            macro_delta > 0
            and sum(value > 0 for value in fold_delta.values()) >= 4
            and flammable_f1_delta > 0
            and fn <= baseline_fn
            and routed_comparison["corrections"]
            > routed_comparison["regressions"]
            and variants[name]["vs_real"]["flammable_ap_delta"] > 0
            and production_delta >= 0
        )
    report: dict[str, Any] = {
        "schema": "exp699_filtered_tfidf_full5_v1",
        "experiment_id": "699",
        "folds": [0, 1, 2, 3, 4],
        "variants": variants,
        "production_baseline": {
            **production_summary,
            "selection": production_selection,
        },
        "validation_labels_read_by_training": 0,
        "sealed_rows_used": 0,
        "public_rows_used": 0,
        "runtime_minutes": (time.monotonic() - started) / 60,
    }
    report["self_sha256"] = canonical_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "EXP699_TFIDF="
        + json.dumps(
            {
                name: {
                    "full5_pass": item["full5_pass"],
                    "vs_real": item["vs_real"],
                    "vs_production": item["vs_production"],
                }
                for name, item in variants.items()
                if name != "real"
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
