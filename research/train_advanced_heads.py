from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC


DATA = Path(os.environ.get("ECUP_DATA", "/work/input/data.csv"))
ALL_IMAGES = Path(os.environ.get("ECUP_ALL_IMAGE_EMBEDDINGS", "/work/input/all/train_embeddings_fp16.npz"))
FIRST_IMAGE = Path(os.environ.get("ECUP_FIRST_IMAGE_EMBEDDINGS", "/work/input/first/train_embeddings_fp16.npz"))
TEXT_ONLY = Path(os.environ.get("ECUP_TEXT_EMBEDDINGS", "/work/input/text/train_embeddings_fp16.npz"))
OUTPUT = Path(os.environ.get("ECUP_OUTPUT_DIR", "/work/output"))


def normalized(features):
    features = features.astype(np.float32, copy=False)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)


def interaction(text, image):
    return np.concatenate([
        text,
        image,
        np.abs(text - image),
        text * image,
    ], axis=1).astype(np.float32, copy=False)


def visual_duplicate_prototypes(features_fp16, labels, categories):
    result = {}
    for category in sorted(np.unique(categories)):
        positions = np.flatnonzero(categories == category)
        groups = {}
        for index in positions:
            groups.setdefault(features_fp16[index].tobytes(), []).append(index)
        vectors, targets, sizes = [], [], []
        conflicts = 0
        for members in groups.values():
            if len(members) < 2:
                continue
            values = np.unique(labels[members])
            if len(values) != 1:
                conflicts += 1
                continue
            vectors.append(features_fp16[members[0]])
            targets.append(values[0])
            sizes.append(len(members))
        result[category] = {
            "vectors_fp16": np.asarray(vectors, dtype=np.float16),
            "labels": np.asarray(targets, dtype=np.int8),
            "group_sizes": np.asarray(sizes, dtype=np.int16),
            "conflicting_groups_excluded": conflicts,
        }
    return result


def main():
    started = time.monotonic()
    if not DATA.exists():
        urllib.request.urlretrieve(os.environ["DATA_URL"], DATA)
    frame = pd.read_csv(DATA)
    archives = {
        "all": np.load(ALL_IMAGES, allow_pickle=True),
        "first": np.load(FIRST_IMAGE, allow_pickle=True),
        "text": np.load(TEXT_ONLY, allow_pickle=True),
    }
    ids = frame["id"].astype(str).to_numpy()
    for name, archive in archives.items():
        if not np.array_equal(ids, archive["ids"].astype(str)):
            raise ValueError(f"{name} id mismatch")
    labels = frame["label"].to_numpy(dtype=np.int8)
    categories = frame["category"].astype(str).to_numpy()
    all_features = archives["all"]["embeddings"].astype(np.float32)
    first_fp16 = archives["first"]["embeddings"]
    first_features = normalized(first_fp16)
    text_features = normalized(archives["text"]["embeddings"])
    interaction_features = interaction(text_features, first_features)

    first_models, tree_models, interaction_models = {}, {}, {}
    metrics = {"categories": {}}
    for category in sorted(np.unique(categories)):
        mask = categories == category
        first_model = LinearSVC(
            C=10.0, class_weight="balanced", dual="auto",
            max_iter=10000, random_state=42,
        ).fit(first_features[mask], labels[mask])
        tree_model = ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=8,
            random_state=42,
        ).fit(all_features[mask], labels[mask])
        interaction_model = LinearSVC(
            C=0.3, class_weight="balanced", dual="auto",
            max_iter=10000, random_state=42,
        ).fit(interaction_features[mask], labels[mask])
        first_models[category] = first_model
        tree_models[category] = tree_model
        interaction_models[category] = interaction_model
        metrics["categories"][category] = {
            "rows": int(mask.sum()),
            "positive": int(labels[mask].sum()),
            "first_C": 10.0,
            "trees": 300,
            "interaction_C": 0.3,
        }
        print(category, json.dumps(metrics["categories"][category], ensure_ascii=False), flush=True)

    duplicate_prototypes = visual_duplicate_prototypes(first_fp16, labels, categories)
    bundle = {
        "embedding_dim": 2048,
        "first_models": first_models,
        "extra_trees_models": tree_models,
        "interaction_models": interaction_models,
        "visual_duplicate_prototypes": duplicate_prototypes,
        "visual_duplicate_cosine_threshold": 0.99999,
    }
    metrics["visual_duplicate_prototypes"] = {
        category: {
            "consistent_groups": int(len(value["labels"])),
            "rows_represented": int(value["group_sizes"].sum()),
            "conflicting_groups_excluded": int(value["conflicting_groups_excluded"]),
        }
        for category, value in duplicate_prototypes.items()
    }
    metrics["runtime_seconds"] = time.monotonic() - started
    OUTPUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUTPUT / "advanced_heads.joblib", compress=3)
    (OUTPUT / "advanced_heads_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
