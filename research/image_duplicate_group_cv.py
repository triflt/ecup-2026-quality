from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


DATA = Path(os.environ.get("ECUP_DATA", "/work/input/data.csv"))
DATA_PARTS = Path(os.environ.get("ECUP_DATA_PARTS", "/work/input/data_parts"))
ALL_IMAGES = Path(os.environ.get("ECUP_ALL_IMAGE_EMBEDDINGS", "/work/input/all/train_embeddings_fp16.npz"))
FIRST_IMAGE = Path(os.environ.get("ECUP_FIRST_IMAGE_EMBEDDINGS", "/work/input/first/train_embeddings_fp16.npz"))
OUTPUT = Path(os.environ.get("ECUP_REPORT", "/work/output/image_duplicate_group_report.json"))


def normalize(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", str(value).lower())
    value = re.sub(r"[^a-zа-яё0-9]+", " ", value)
    return " ".join(value.split())


class DSU:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int32)
        self.size = np.ones(size, dtype=np.int32)

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def union_by_key(dsu: DSU, keys) -> None:
    first = {}
    for index, key in enumerate(keys):
        if key in first:
            dsu.union(index, first[key])
        else:
            first[key] = index


def best_threshold(labels, scores):
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(np.int64)
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    fn = int(sorted_labels.sum()) - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    boundary = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    candidates = np.flatnonzero(boundary)
    index = int(candidates[np.argmax(f1[candidates])])
    return float(f1[index]), float(sorted_scores[index])


def rank01(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def oof_svc(features, labels, folds, c_value):
    scores = np.empty(len(labels), dtype=np.float32)
    for fold, (train_idx, valid_idx) in enumerate(folds):
        model = LinearSVC(
            C=c_value,
            class_weight="balanced",
            dual="auto",
            max_iter=10000,
            random_state=42 + fold,
        )
        model.fit(features[train_idx], labels[train_idx])
        scores[valid_idx] = model.decision_function(features[valid_idx])
    f1, threshold = best_threshold(labels, scores)
    return {"f1": f1, "threshold": threshold, "scores": scores}


def oof_trees(features, labels, folds):
    scores = np.empty(len(labels), dtype=np.float32)
    for fold, (train_idx, valid_idx) in enumerate(folds):
        model = ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=8,
            random_state=42 + fold,
        )
        model.fit(features[train_idx], labels[train_idx])
        scores[valid_idx] = model.predict_proba(features[valid_idx])[:, 1]
    f1, threshold = best_threshold(labels, scores)
    return {"f1": f1, "threshold": threshold, "scores": scores}


def best_triple(labels, text_scores, all_scores, first_scores):
    ranks = [rank01(text_scores), rank01(all_scores), rank01(first_scores)]
    best = None
    for text_steps in range(21):
        for all_steps in range(21 - text_steps):
            first_steps = 20 - text_steps - all_steps
            weights = np.array([text_steps, all_steps, first_steps], dtype=float) / 20
            scores = weights[0] * ranks[0] + weights[1] * ranks[1] + weights[2] * ranks[2]
            f1, threshold = best_threshold(labels, scores)
            item = {
                "f1": f1,
                "threshold": threshold,
                "weight_text": float(weights[0]),
                "weight_all_images": float(weights[1]),
                "weight_first_image": float(weights[2]),
            }
            if best is None or item["f1"] > best["f1"]:
                best = item
    return best


def best_quad(labels, text_scores, all_scores, first_scores, tree_scores):
    heads = [rank01(item) for item in (text_scores, all_scores, first_scores, tree_scores)]
    best = None
    for text_steps in range(21):
        for all_steps in range(21 - text_steps):
            for first_steps in range(21 - text_steps - all_steps):
                tree_steps = 20 - text_steps - all_steps - first_steps
                weights = np.array(
                    [text_steps, all_steps, first_steps, tree_steps], dtype=float
                ) / 20
                scores = sum(weight * head for weight, head in zip(weights, heads))
                f1, threshold = best_threshold(labels, scores)
                item = {
                    "f1": f1,
                    "threshold": threshold,
                    "weight_text": float(weights[0]),
                    "weight_all_images": float(weights[1]),
                    "weight_first_image": float(weights[2]),
                    "weight_extra_trees": float(weights[3]),
                }
                if best is None or item["f1"] > best["f1"]:
                    best = item
    return best


def main():
    started = time.monotonic()
    if not DATA.exists():
        parts = sorted(DATA_PARTS.glob("data.csv.gz.part-*"))
        if parts:
            with DATA.open("wb") as output:
                for part in parts:
                    output.write(part.read_bytes())
        elif os.environ.get("DATA_URL"):
            urllib.request.urlretrieve(os.environ["DATA_URL"], DATA)
        else:
            raise FileNotFoundError("missing compressed data parts and DATA_URL")
    frame = pd.read_csv(DATA)
    all_archive = np.load(ALL_IMAGES, allow_pickle=True)
    first_archive = np.load(FIRST_IMAGE, allow_pickle=True)
    ids = frame["id"].astype(str).to_numpy()
    for name, archive in (("all", all_archive), ("first", first_archive)):
        if not np.array_equal(ids, archive["ids"].astype(str)):
            raise ValueError(f"{name} id mismatch")

    names = frame["name"].fillna("").astype(str)
    descriptions = frame["description"].fillna("").astype(str)
    texts = names + "\n" + names + "\n" + descriptions
    normalized_text = (names + " " + descriptions).map(normalize).to_numpy()
    categories = frame["category"].astype(str).to_numpy()
    labels_all = frame["label"].to_numpy(dtype=np.int8)
    first_fp16 = first_archive["embeddings"]

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
    report = {
        "validation": "5-fold StratifiedGroupKFold; union of exact normalized text and exact first-image embedding duplicates",
        "categories": {},
    }
    macro = {
        "text": [], "all_images": [], "first_image": [],
        "extra_trees": [], "tri_fusion": [], "four_head_fusion": [],
    }
    for category in sorted(np.unique(categories)):
        positions = np.flatnonzero(categories == category)
        labels = labels_all[positions]
        dsu = DSU(len(positions))
        union_by_key(dsu, normalized_text[positions])
        # Exact fp16 equality means the visual encoder received the same first
        # image representation. This catches copied product photos without
        # subjective similarity thresholds or label information.
        image_keys = [first_fp16[index].tobytes() for index in positions]
        union_by_key(dsu, image_keys)
        groups = np.array([dsu.find(i) for i in range(len(positions))], dtype=np.int32)
        counts = np.unique(groups, return_counts=True)[1]
        folds = list(StratifiedGroupKFold(
            5, shuffle=True, random_state=42
        ).split(np.zeros(len(positions)), labels, groups))
        text = oof_svc(text_matrix[positions], labels, folds, 1.0)
        all_images = oof_svc(
            all_archive["embeddings"][positions].astype(np.float32), labels, folds, 3.0
        )
        first_image = oof_svc(
            first_archive["embeddings"][positions].astype(np.float32), labels, folds, 10.0
        )
        extra_trees = oof_trees(
            all_archive["embeddings"][positions].astype(np.float32), labels, folds
        )
        triple = best_triple(labels, text["scores"], all_images["scores"], first_image["scores"])
        quad = best_quad(
            labels, text["scores"], all_images["scores"],
            first_image["scores"], extra_trees["scores"],
        )
        entry = {
            "rows": int(len(positions)),
            "groups": int(len(counts)),
            "duplicate_groups": int(np.sum(counts > 1)),
            "rows_in_duplicate_groups": int(np.sum(counts[counts > 1])),
            "largest_group": int(counts.max()),
            "text": {k: v for k, v in text.items() if k != "scores"},
            "all_images": {k: v for k, v in all_images.items() if k != "scores"},
            "first_image": {k: v for k, v in first_image.items() if k != "scores"},
            "extra_trees": {k: v for k, v in extra_trees.items() if k != "scores"},
            "tri_fusion": triple,
            "four_head_fusion": quad,
        }
        report["categories"][category] = entry
        for key in macro:
            macro[key].append(entry[key]["f1"])
        print(category, json.dumps(entry, ensure_ascii=False), flush=True)

    report["macro_f1"] = {key: float(np.mean(values)) for key, values in macro.items()}
    report["runtime_minutes"] = (time.monotonic() - started) / 60
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["macro_f1"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
