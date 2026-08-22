from __future__ import annotations

import gzip
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("research")
DATA = ROOT / "data.csv"
BASE = ROOT / "oof-cache-extracted/oof_scores.npz"
LORA = ROOT / "lora-hard-5fold-robust-fusion-report.npz"
MANIFEST = ROOT / "multi_image_manifest.tsv.gz"
REPORT = ROOT / "nested_meta_fusion_report.json"
MODEL = ROOT / "nested_meta_fusion_model.json"

PATTERNS = {
    "bad_explicit": re.compile(r"(?iu)(?:\bбад\b|биологически\s+активн\w*\s+добавк|dietary\s+supplement)"),
    "sports": re.compile(r"(?iu)(?:спортивн\w*\s+питан|спортпит|\bbcaa\b|\bпротеин|гейнер|креатин|карнитин|аминокислот)"),
    "flame_source": re.compile(r"(?iu)(?:зажигалк|спич(?:к|еч)|сух\w*\s+горюч|жидкост\w*\s+для\s+розжиг|газов\w*\s+(?:баллон|картридж)|пропан|бутан|керосин|бензин|топлив)"),
    "equipment": re.compile(r"(?iu)(?:мангал|грил|газов\w*\s+плит|горелк|печ(?:ь|ка)|насадк\w*\s+(?:на|для)\s+баллон)"),
    "kit": re.compile(r"(?iu)(?:в\s+комплект|комплект\w*\s+с|входит\s+в\s+комплект)"),
}


def f1(labels, predictions):
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def rank01(values):
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float32)
    result[order] = np.linspace(0, 1, len(values), dtype=np.float32)
    return result


def best_threshold(labels, scores):
    order = np.argsort(scores)[::-1]
    ordered = labels[order]
    tp = np.cumsum(ordered == 1)
    fp = np.cumsum(ordered == 0)
    fn = int((labels == 1).sum()) - tp
    values = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    best = int(np.argmax(values))
    threshold = float(scores[order[best]])
    return float(values[best]), threshold


class LogisticModel:
    def __init__(self, c_value, class_weight):
        self.c_value = c_value
        self.class_weight = class_weight

    def fit(self, frame, labels):
        values = np.asarray(frame, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        self.mean = values.mean(axis=0)
        self.scale = values.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        values = (values - self.mean) / self.scale
        values = np.column_stack([np.ones(len(values)), values])
        sample_weight = np.ones(len(labels), dtype=np.float64)
        if self.class_weight == "balanced":
            positive = max(1, int(labels.sum()))
            negative = max(1, len(labels) - positive)
            sample_weight[labels == 1] = len(labels) / (2 * positive)
            sample_weight[labels == 0] = len(labels) / (2 * negative)
        normalizer = sample_weight.sum()
        penalty = 1.0 / self.c_value
        coefficient = np.zeros(values.shape[1], dtype=np.float64)
        for _ in range(25):
            linear = np.clip(values @ coefficient, -35, 35)
            probability = 1 / (1 + np.exp(-linear))
            gradient = values.T @ ((probability - labels) * sample_weight) / normalizer
            gradient[1:] += penalty * coefficient[1:] / len(labels)
            curvature = probability * (1 - probability) * sample_weight / normalizer
            hessian = values.T @ (values * curvature[:, None])
            hessian[1:, 1:] += np.eye(values.shape[1] - 1) * penalty / len(labels)
            step = np.linalg.solve(hessian + np.eye(values.shape[1]) * 1e-8, gradient)
            coefficient -= step
            if np.max(np.abs(step)) < 1e-7:
                break
        self.coefficient = coefficient
        return self

    def predict_proba(self, frame):
        values = (np.asarray(frame, dtype=np.float64) - self.mean) / self.scale
        values = np.column_stack([np.ones(len(values)), values])
        linear = np.clip(values @ self.coefficient, -35, 35)
        positive = 1 / (1 + np.exp(-linear))
        return np.column_stack([1 - positive, positive])

    def serializable(self):
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficient": self.coefficient.tolist(),
        }


def build_model(c_value, class_weight):
    return LogisticModel(c_value, class_weight)


def prepare():
    frame = pd.read_csv(DATA)
    frame["name"] = frame.name.fillna("").astype(str)
    frame["description"] = frame.description.fillna("").astype(str)
    base = np.load(BASE, allow_pickle=True)
    lora = np.load(LORA, allow_pickle=True)
    frame["fold"] = lora["folds"]
    frame["base_rank"] = lora["base_rank"]
    frame["lora_rank"] = lora["lora_rank"]
    text_rank = np.zeros(len(frame), dtype=np.float32)
    mm_rank = np.zeros(len(frame), dtype=np.float32)
    for category in frame.category.unique():
        for fold in sorted(frame.fold.unique()):
            mask = (frame.category.to_numpy() == category) & (frame.fold.to_numpy() == fold)
            text_rank[mask] = rank01(base["text_scores"][mask])
            mm_rank[mask] = rank01(base["mm_scores"][mask])
    frame["text_rank"] = text_rank
    frame["mm_rank"] = mm_rank
    manifest = pd.read_csv(MANIFEST, sep="\t", compression="gzip")
    manifest["id"] = manifest.id.astype(str)
    manifest["image_count"] = manifest.image_urls.map(lambda value: len(json.loads(value)))
    frame["id"] = frame.id.astype(str)
    frame = frame.merge(manifest[["id", "image_count"]], on="id", how="left", validate="one_to_one")
    text = (frame.name + "\n" + frame.description).map(lambda value: html.unescape(value))
    frame["description_log"] = np.log1p(frame.description.str.len())
    frame["rank_gap"] = frame.lora_rank - frame.base_rank
    frame["rank_abs_gap"] = frame.rank_gap.abs()
    frame["lora_x_length"] = frame.lora_rank * frame.description_log
    frame["base_x_length"] = frame.base_rank * frame.description_log
    for name, pattern in PATTERNS.items():
        frame[name] = text.map(lambda value, p=pattern: float(bool(p.search(value))))
    return frame


def main():
    frame = prepare()
    feature_sets = {
        "scores": ["base_rank", "lora_rank", "text_rank", "mm_rank", "rank_gap", "rank_abs_gap"],
        "scores_meta": [
            "base_rank", "lora_rank", "text_rank", "mm_rank", "rank_gap", "rank_abs_gap",
            "description_log", "image_count", "lora_x_length", "base_x_length",
            "bad_explicit", "sports", "flame_source", "equipment", "kit",
        ],
    }
    candidates = [
        (features, c_value, class_weight)
        for features in feature_sets
        for c_value in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
        for class_weight in [None, "balanced"]
    ]
    report = {"categories": {}}
    saved = {}
    macro = []
    for category in sorted(frame.category.unique()):
        category_mask = frame.category.to_numpy() == category
        outer_predictions = np.zeros(category_mask.sum(), dtype=np.int8)
        category_positions = np.flatnonzero(category_mask)
        fold_reports = []
        for outer_fold in sorted(frame.fold.unique()):
            train = category_mask & (frame.fold.to_numpy() != outer_fold)
            valid = category_mask & (frame.fold.to_numpy() == outer_fold)
            best = None
            for feature_name, c_value, class_weight in candidates:
                columns = feature_sets[feature_name]
                inner_scores = np.zeros(train.sum(), dtype=np.float64)
                train_positions = np.flatnonzero(train)
                for inner_fold in sorted(frame.loc[train, "fold"].unique()):
                    inner_train = train & (frame.fold.to_numpy() != inner_fold)
                    inner_valid = train & (frame.fold.to_numpy() == inner_fold)
                    model = build_model(c_value, class_weight)
                    model.fit(frame.loc[inner_train, columns], frame.loc[inner_train, "label"])
                    destinations = np.searchsorted(train_positions, np.flatnonzero(inner_valid))
                    inner_scores[destinations] = model.predict_proba(frame.loc[inner_valid, columns])[:, 1]
                inner_f1, threshold = best_threshold(frame.loc[train, "label"].to_numpy(), inner_scores)
                preference = (inner_f1, feature_name == "scores", class_weight is None, -c_value)
                if best is None or preference > best[0]:
                    best = (preference, feature_name, c_value, class_weight, threshold)
            _, feature_name, c_value, class_weight, threshold = best
            columns = feature_sets[feature_name]
            model = build_model(c_value, class_weight)
            model.fit(frame.loc[train, columns], frame.loc[train, "label"])
            scores = model.predict_proba(frame.loc[valid, columns])[:, 1]
            predictions = (scores >= threshold).astype(np.int8)
            destinations = np.searchsorted(category_positions, np.flatnonzero(valid))
            outer_predictions[destinations] = predictions
            fold_reports.append({
                "fold": int(outer_fold), "features": feature_name, "C": c_value,
                "class_weight": class_weight, "threshold": threshold,
                "validation_f1": f1(frame.loc[valid, "label"], predictions),
            })
        labels = frame.loc[category_mask, "label"].to_numpy()
        nested_f1 = f1(labels, outer_predictions)
        report["categories"][category] = {"nested_f1": nested_f1, "folds": fold_reports}
        macro.append(nested_f1)

        # Choose the most frequent nested configuration, refit on all OOF rows, and
        # calibrate its final threshold with five-fold meta OOF predictions.
        choices = [(row["features"], row["C"], row["class_weight"]) for row in fold_reports]
        selected = max(set(choices), key=lambda value: (choices.count(value), value[0] == "scores", value[2] is None, -value[1]))
        feature_name, c_value, class_weight = selected
        columns = feature_sets[feature_name]
        cv_scores = np.zeros(category_mask.sum(), dtype=np.float64)
        category_positions = np.flatnonzero(category_mask)
        for fold in sorted(frame.fold.unique()):
            train = category_mask & (frame.fold.to_numpy() != fold)
            valid = category_mask & (frame.fold.to_numpy() == fold)
            model = build_model(c_value, class_weight)
            model.fit(frame.loc[train, columns], frame.loc[train, "label"])
            destinations = np.searchsorted(category_positions, np.flatnonzero(valid))
            cv_scores[destinations] = model.predict_proba(frame.loc[valid, columns])[:, 1]
        cv_f1, threshold = best_threshold(labels, cv_scores)
        final_model = build_model(c_value, class_weight)
        final_model.fit(frame.loc[category_mask, columns], labels)
        saved[category] = {
            "model": final_model, "features": columns, "threshold": threshold,
            "cv_f1": cv_f1, "configuration": selected,
        }
        report["categories"][category]["final"] = {
            "features": feature_name, "C": c_value, "class_weight": class_weight,
            "threshold": threshold, "five_fold_meta_f1": cv_f1,
        }
    report["nested_macro_f1"] = float(np.mean(macro))
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        category: {
            "features": item["features"],
            "threshold": item["threshold"],
            "cv_f1": item["cv_f1"],
            "configuration": list(item["configuration"]),
            "model": item["model"].serializable(),
        }
        for category, item in saved.items()
    }
    MODEL.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
