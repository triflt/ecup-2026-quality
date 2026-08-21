from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


DATA = Path("/work/input/data.csv")
OOF = Path("/work/input/four_head_oof.npz")
OUTPUT = Path("/work/output/rule_feature_report.json")

PATTERNS = [
    r"\bбад\b",
    r"биологически\s+активн\w*\s+добавк",
    r"dietary\s+supplement",
    r"food\s+supplement",
    r"nahrungserg[aä]nzung",
    r"не\s+явля\w*\s+(?:бад|биологически)",
    r"не\s+лекарствен",
    r"спортивн\w*\s+питан|спортпит",
    r"\bbcaas?\b|\bпротеин|гейнер|креатин|карнитин|carnitine|аминокислот",
    r"капсул|таблет|дозиров|мг\b|mg\b",
    r"зажигалк|спич(?:к|еч)|огнив|розжиг",
    r"сух\w*\s+горюч|жидкост\w*\s+для\s+розжиг|топлив\w*\s+(?:таблет|брикет)",
    r"газов\w*\s+(?:баллон|картридж)|пропан|бутан|керосин|бензин",
    r"легковоспламен|горюч\w*\s+(?:веществ|газ)",
    r"дым(?:овая|овой)\s+шаш|пиротех|фаер|бенгальск",
    r"мангал|грил|газов\w*\s+плит|горелк|печ(?:ь|ка)",
    r"без\s+(?:газа|топлива|баллона)|пуст\w*\s+баллон|не\s+заправлен",
    r"встроен\w*\s+(?:зажиг|поджиг)|пьезо(?:поджиг|элемент)",
    r"в\s+комплект\w*\s+(?:входит|включен|есть)|комплект\w*\s+с\s+",
    r"уголь\w*\s+(?:для\s+рисован|в\s+фильтр)|активированн\w*\s+угол",
]
REGEXES = [re.compile(pattern, re.I | re.U) for pattern in PATTERNS]


def rank01(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def best_threshold(labels, scores):
    best = (-1.0, 0.0)
    for threshold in np.unique(np.quantile(scores, np.linspace(0.002, 0.998, 700))):
        value = f1_score(labels, scores >= threshold)
        if value > best[0]:
            best = (float(value), float(threshold))
    return best


def engineered(frame):
    names = frame["name"].fillna("").astype(str).str.lower().tolist()
    descriptions = frame["description"].fillna("").astype(str).str.lower().tolist()
    rows = []
    for name, description in zip(names, descriptions):
        text = name + "\n" + description
        values = [min(3, len(regex.findall(text))) for regex in REGEXES]
        name_values = [int(bool(regex.search(name))) for regex in REGEXES]
        explicit_bad = int(any(values[index] for index in (0, 1, 2, 3, 4)))
        sport = int(any(values[index] for index in (7, 8)))
        equipment = int(bool(values[15]))
        included = int(bool(values[18]))
        fuel = int(any(values[index] for index in (10, 11, 12, 13, 14)))
        values.extend([
            explicit_bad * sport,
            sport * (1 - explicit_bad),
            equipment * included,
            equipment * (1 - included),
            fuel * included,
            math_log_length(text),
        ])
        rows.append(values + name_values)
    matrix = np.asarray(rows, dtype=np.float32)
    matrix -= matrix.mean(axis=0, keepdims=True)
    matrix /= np.maximum(matrix.std(axis=0, keepdims=True), 1e-4)
    return matrix


def math_log_length(value):
    return float(np.log1p(len(value)) / 10.0)


def oof_scores(features, labels, folds, c_value, class_weight="balanced"):
    scores = np.empty(len(labels), dtype=np.float32)
    for fold in range(5):
        train = folds != fold
        valid = folds == fold
        model = LinearSVC(
            C=c_value, class_weight=class_weight, dual="auto",
            max_iter=10000, random_state=42 + fold,
        )
        model.fit(features[train], labels[train])
        scores[valid] = model.decision_function(features[valid])
    value, threshold = best_threshold(labels, scores)
    return value, threshold, scores


def best_four(labels, text, all_images, first, trees):
    heads = [rank01(item) for item in (text, all_images, first, trees)]
    best = None
    for text_steps in range(11):
        for all_steps in range(11 - text_steps):
            for first_steps in range(11 - text_steps - all_steps):
                tree_steps = 10 - text_steps - all_steps - first_steps
                weights = np.asarray([text_steps, all_steps, first_steps, tree_steps]) / 10
                scores = sum(weight * head for weight, head in zip(weights, heads))
                value, threshold = best_threshold(labels, scores)
                item = {
                    "f1": value,
                    "threshold": threshold,
                    "weight_text": float(weights[0]),
                    "weight_all_images": float(weights[1]),
                    "weight_first_image": float(weights[2]),
                    "weight_extra_trees": float(weights[3]),
                }
                if best is None or value > best["f1"]:
                    best = item
    return best


def main():
    started = time.monotonic()
    if not DATA.exists():
        urllib.request.urlretrieve(os.environ["DATA_URL"], DATA)
    frame = pd.read_csv(DATA)
    oof = np.load(OOF, allow_pickle=True)
    ids = frame["id"].astype(str).to_numpy()
    if not np.array_equal(ids, oof["ids"].astype(str)):
        raise ValueError("OOF id mismatch")
    names = frame["name"].fillna("").astype(str)
    descriptions = frame["description"].fillna("").astype(str)
    texts = names + "\n" + names + "\n" + descriptions
    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.997,
            sublinear_tf=True, max_features=160_000, dtype=np.float32,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=3,
            sublinear_tf=True, max_features=200_000, dtype=np.float32,
        )),
    ])
    text_matrix = vectorizer.fit_transform(texts)
    rules = engineered(frame)
    categories = frame["category"].astype(str).to_numpy()
    labels_all = frame["label"].to_numpy(dtype=np.int8)
    folds_all = oof["fold_ids"].astype(np.int8)
    report = {"validation": "same grouped OOF folds", "categories": {}}
    macro_text, macro_four = [], []
    for category in sorted(np.unique(categories)):
        positions = np.flatnonzero(categories == category)
        labels = labels_all[positions]
        folds = folds_all[positions]
        best = None
        for scale in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            features = sparse.hstack([
                text_matrix[positions],
                sparse.csr_matrix(rules[positions] * scale),
            ], format="csr")
            for c_value in (0.3, 1.0, 3.0):
                value, threshold, scores = oof_scores(features, labels, folds, c_value)
                item = {
                    "f1": value,
                    "threshold": threshold,
                    "C": c_value,
                    "rule_scale": scale,
                    "scores": scores,
                }
                if best is None or value > best["f1"]:
                    best = item
        four = best_four(
            labels,
            best["scores"],
            oof["all_images"][positions].astype(np.float32),
            oof["first_image"][positions].astype(np.float32),
            oof["extra_trees"][positions].astype(np.float32),
        )
        report["categories"][category] = {
            "rule_text": {key: value for key, value in best.items() if key != "scores"},
            "four_head": four,
        }
        macro_text.append(best["f1"])
        macro_four.append(four["f1"])
        print(category, json.dumps(report["categories"][category], ensure_ascii=False), flush=True)
    report["macro_rule_text"] = float(np.mean(macro_text))
    report["macro_four_head"] = float(np.mean(macro_four))
    report["runtime_minutes"] = (time.monotonic() - started) / 60
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
