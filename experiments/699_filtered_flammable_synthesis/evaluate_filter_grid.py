from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

FLAMMABLE = "Легковоспламеняющиеся"
CAPS = (20, 40, 80)
SOURCES = ("v1", "v2", "both")
MODES = ("balanced", "positive_only", "negative_only", "negative_2x")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rows(payload: bytes) -> list[dict]:
    return [json.loads(line) for line in payload.decode("utf-8").splitlines()]


def select_rows(rows: list[dict], source: str, mode: str, cap: int) -> list[dict]:
    eligible = rows if source == "both" else [row for row in rows if row["source"] == source]
    by_label = {
        label: sorted(
            (row for row in eligible if int(row["label"]) == label),
            key=lambda row: (int(row["label_rank"]), row["candidate_id"]),
        )
        for label in (0, 1)
    }
    limits = {
        "balanced": {0: cap, 1: cap},
        "positive_only": {0: 0, 1: cap},
        "negative_only": {0: cap, 1: 0},
        "negative_2x": {0: 2 * cap, 1: cap},
    }[mode]
    return by_label[0][: limits[0]] + by_label[1][: limits[1]]


def make_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.997,
                    sublinear_tf=True,
                    max_features=140_000,
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
                    max_features=180_000,
                    dtype=np.float32,
                ),
            ),
        ]
    )


def row_text(row: dict) -> str:
    name = str(row.get("name", ""))
    return f"{name}\n{name}\n{row.get('description', '')}"


def train_scores(
    real_rows: list[dict], fold_rows: dict[int, list[dict]]
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(
        [index for index, row in enumerate(real_rows) if row["category"] == FLAMMABLE],
        dtype=np.int64,
    )
    labels = np.asarray([real_rows[index]["label"] for index in positions], dtype=np.int8)
    folds = np.asarray([real_rows[index]["fold"] for index in positions], dtype=np.int8)
    ids = np.asarray([real_rows[index]["id"] for index in positions], dtype=str)
    variants = ["real"] + [
        f"{source}_{mode}_cap{cap}"
        for source in SOURCES
        for mode in MODES
        for cap in CAPS
    ]
    scores = {name: np.full(len(positions), np.nan, dtype=np.float32) for name in variants}
    real_texts = np.asarray([row_text(real_rows[index]) for index in positions], dtype=object)

    for fold in range(5):
        train_pos = np.flatnonzero(folds != fold)
        valid_pos = np.flatnonzero(folds == fold)
        pool = fold_rows[fold]
        pool_texts = [row_text(row) for row in pool]
        vectorizer = make_vectorizer()
        vectorizer.fit(real_texts[train_pos].tolist() + pool_texts)
        train_x = vectorizer.transform(real_texts[train_pos].tolist())
        valid_x = vectorizer.transform(real_texts[valid_pos].tolist())
        pool_x = vectorizer.transform(pool_texts)
        train_y = labels[train_pos]

        baseline = LinearSVC(
            C=1.0, class_weight="balanced", dual="auto", max_iter=8000, random_state=42
        )
        baseline.fit(train_x, train_y)
        scores["real"][valid_pos] = baseline.decision_function(valid_x).astype(np.float32)

        pool_index = {row["candidate_id"]: index for index, row in enumerate(pool)}
        for source in SOURCES:
            for mode in MODES:
                for cap in CAPS:
                    name = f"{source}_{mode}_cap{cap}"
                    selected = select_rows(pool, source, mode, cap)
                    selected_indices = [pool_index[row["candidate_id"]] for row in selected]
                    selected_y = np.asarray([row["label"] for row in selected], dtype=np.int8)
                    fit_x = train_x
                    fit_y = train_y
                    if selected_indices:
                        fit_x = sparse.vstack([train_x, pool_x[selected_indices]], format="csr")
                        fit_y = np.concatenate([train_y, selected_y])
                    model = LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        dual="auto",
                        max_iter=8000,
                        random_state=42,
                    )
                    model.fit(fit_x, fit_y)
                    scores[name][valid_pos] = model.decision_function(valid_x).astype(np.float32)
    for name, values in scores.items():
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite scores: {name}")
    return scores, positions, ids, labels, folds


def build_report(
    *,
    filter_module_path: Path,
    evaluation_module_path: Path,
    runtime_path: Path,
    labels_path: Path,
    synth_paths: list[Path],
    four_head_path: Path,
    qwen3vl_path: Path,
    qwen35_path: Path,
) -> dict:
    filter_module = load_module("exp699_filter", filter_module_path)
    evaluation = load_module("exp698_evaluation", evaluation_module_path)
    filter_report, payloads = filter_module.build_filter_payloads(
        runtime_path, labels_path, synth_paths
    )
    fold_rows = {fold: load_rows(payloads[f"fold{fold}_ranked.jsonl"]) for fold in range(5)}
    real_rows = filter_module.load_real(runtime_path, labels_path)
    scores, positions, ids, labels, folds = train_scores(real_rows, fold_rows)

    four_head = np.load(four_head_path, allow_pickle=True)
    qwen3vl = np.load(qwen3vl_path, allow_pickle=True)
    qwen35 = np.load(qwen35_path, allow_pickle=True)
    full_ids = np.asarray([row["id"] for row in real_rows], dtype=str)
    full_labels = np.asarray([row["label"] for row in real_rows], dtype=np.int8)
    full_folds = np.asarray([row["fold"] for row in real_rows], dtype=np.int8)
    categories = np.asarray([row["category"] for row in real_rows], dtype=str)
    for source in (four_head, qwen3vl, qwen35):
        if not np.array_equal(source["ids"].astype(str), full_ids):
            raise ValueError("baseline ID binding mismatch")
    if not np.array_equal(four_head["labels"].astype(np.int8), full_labels):
        raise ValueError("baseline label binding mismatch")
    if not np.array_equal(four_head["fold_ids"].astype(np.int8), full_folds):
        raise ValueError("baseline fold binding mismatch")
    if not np.array_equal(ids, full_ids[positions]):
        raise ValueError("flammable position binding mismatch")

    variants = {}
    baseline_component_predictions = None
    baseline_ensemble_predictions = None
    for name, values in scores.items():
        component_predictions, thresholds = evaluation.nested_component(labels, values, folds)
        component_summary = evaluation.summarize(
            labels,
            np.asarray([FLAMMABLE] * len(labels)),
            folds,
            component_predictions,
            values,
        )
        component_summary["thresholds"] = thresholds
        component_summary["ap"] = float(average_precision_score(labels, values))

        robust = np.asarray(four_head["fused"], dtype=np.float32).copy()
        old_text = np.asarray(four_head["text"], dtype=np.float32)[positions]
        robust[positions] += 0.65 * (evaluation.rank01(values) - old_text)
        robust_rank = evaluation.fold_category_ranks(robust, full_folds, categories)
        ensemble_predictions, selection = evaluation.nested_fusion(
            full_labels,
            categories,
            full_folds,
            {
                "robust_base": robust_rank,
                "qwen3vl": qwen3vl["lora_rank"].astype(np.float32),
                "qwen35": qwen35["lora_rank"].astype(np.float32),
            },
        )
        ensemble_summary = evaluation.summarize(
            full_labels, categories, full_folds, ensemble_predictions
        )
        ensemble_summary["selection"] = selection
        if name == "real":
            baseline_component_predictions = component_predictions
            baseline_ensemble_predictions = ensemble_predictions
        variants[name] = {
            "component": component_summary,
            "ensemble": ensemble_summary,
        }

    if baseline_component_predictions is None or baseline_ensemble_predictions is None:
        raise RuntimeError("missing baseline")
    baseline_macro = variants["real"]["ensemble"]["macro_f1"]
    baseline_ap = variants["real"]["component"]["ap"]
    baseline_fn = variants["real"]["ensemble"]["categories"][FLAMMABLE]["confusion"]["fn"]
    leaderboard = []
    for name, value in variants.items():
        component_predictions, _ = evaluation.nested_component(labels, scores[name], folds)
        robust = value["ensemble"]
        row = {
            "variant": name,
            "component_ap": value["component"]["ap"],
            "component_ap_delta": value["component"]["ap"] - baseline_ap,
            "component_f1": value["component"]["categories"][FLAMMABLE]["f1"],
            "component_fold_f1": value["component"]["categories"][FLAMMABLE]["fold_f1"],
            "ensemble_macro_f1": robust["macro_f1"],
            "ensemble_macro_delta": robust["macro_f1"] - baseline_macro,
            "ensemble_flammable_f1": robust["categories"][FLAMMABLE]["f1"],
            "ensemble_flammable_confusion": robust["categories"][FLAMMABLE]["confusion"],
            "component_corrections_regressions": evaluation.compare(
                component_predictions,
                baseline_component_predictions,
                labels,
                np.asarray([FLAMMABLE] * len(labels)),
            ),
        }
        row["screen_pass"] = bool(
            name != "real"
            and row["component_ap_delta"] > 0
            and row["ensemble_macro_delta"] > 0
            and row["ensemble_flammable_confusion"]["fn"] <= baseline_fn
            and all(
                row["component_fold_f1"][str(fold)]
                >= variants["real"]["component"]["categories"][FLAMMABLE]["fold_f1"][str(fold)]
                for fold in (0, 3)
            )
        )
        leaderboard.append(row)
    leaderboard.sort(
        key=lambda row: (
            row["screen_pass"],
            row["ensemble_macro_delta"],
            row["component_ap_delta"],
        ),
        reverse=True,
    )
    report = {
        "schema": "exp699_tfidf_filter_grid_v1",
        "filter_report_self_sha256": filter_report["self_sha256"],
        "filter_code_sha256": sha256(filter_module_path),
        "evaluation_code_sha256": sha256(Path(__file__)),
        "runtime_sha256": sha256(runtime_path),
        "labels_sha256": sha256(labels_path),
        "four_head_sha256": sha256(four_head_path),
        "qwen3vl_sha256": sha256(qwen3vl_path),
        "qwen35_sha256": sha256(qwen35_path),
        "variants": len(leaderboard),
        "baseline": leaderboard[[row["variant"] for row in leaderboard].index("real")],
        "leaderboard": leaderboard,
        "public_used": False,
        "sealed_rows": 0,
    }
    report["self_sha256"] = hashlib.sha256(canonical_bytes(report)).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter-module", type=Path, required=True)
    parser.add_argument("--evaluation-module", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--synth", type=Path, action="append", required=True)
    parser.add_argument("--four-head", type=Path, required=True)
    parser.add_argument("--qwen3vl", type=Path, required=True)
    parser.add_argument("--qwen35", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        filter_module_path=args.filter_module,
        evaluation_module_path=args.evaluation_module,
        runtime_path=args.runtime,
        labels_path=args.labels,
        synth_paths=args.synth,
        four_head_path=args.four_head,
        qwen3vl_path=args.qwen3vl,
        qwen35_path=args.qwen35,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
