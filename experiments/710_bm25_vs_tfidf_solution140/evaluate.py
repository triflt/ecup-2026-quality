from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import average_precision_score, f1_score


FUSION_ALPHA = {"БАД": 0.85, "Легковоспламеняющиеся": 0.65}
LORA_FUSION = {
    "БАД": (0.50, 0.25, 0.25, 0.27193570137023926),
    "Легковоспламеняющиеся": (0.15, 0.10, 0.75, 0.953912615776062),
}
TOP_K = 30


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def fold_category_ranks(values, folds, categories):
    result = np.zeros(len(values), dtype=np.float32)
    for fold in sorted(np.unique(folds)):
        for category in sorted(np.unique(categories)):
            mask = (folds == fold) & (categories == category)
            result[mask] = rank01(values[mask])
    return result


def bm25_documents(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    counts = counts.astype(np.float32).tocsr()
    rows, _ = counts.shape
    document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log((rows - document_frequency + 0.5) / (document_frequency + 0.5) + 1.0)
    lengths = np.asarray(counts.sum(axis=1)).ravel()
    average_length = max(float(lengths.mean()), 1.0)
    k1, b = 1.5, 0.75
    row_scale = k1 * (1.0 - b + b * lengths / average_length)
    result = counts.copy()
    for row in range(rows):
        start, end = result.indptr[row : row + 2]
        values = result.data[start:end]
        result.data[start:end] = values * (k1 + 1.0) / (values + row_scale[row])
    return result.multiply(idf).tocsr()


def bm25_oof(texts, labels, folds, categories):
    result = np.zeros(len(texts), dtype=np.float32)
    for fold in sorted(np.unique(folds)):
        for category in sorted(np.unique(categories)):
            donors = np.flatnonzero((folds != fold) & (categories == category))
            queries = np.flatnonzero((folds == fold) & (categories == category))
            vectorizer = CountVectorizer(
                ngram_range=(1, 2), min_df=2, max_features=200_000
            )
            donor_counts = vectorizer.fit_transform(texts[donors]).tocsr()
            query_counts = vectorizer.transform(texts[queries]).sign().astype(np.float32)
            similarities = (query_counts @ bm25_documents(donor_counts).T).tocsr()
            prior = float(labels[donors].mean())
            for local, query in enumerate(queries):
                row = similarities.getrow(local)
                if row.nnz == 0:
                    result[query] = prior
                    continue
                order = np.argsort(-row.data, kind="mergesort")[:TOP_K]
                weights = np.maximum(row.data[order], 0.0)
                selected = row.indices[order]
                if float(weights.sum()) <= 0:
                    result[query] = prior
                else:
                    result[query] = float(np.average(labels[donors[selected]], weights=weights))
    return result


def summarize(labels, categories, predictions):
    per_category = {}
    for category in sorted(np.unique(categories)):
        mask = categories == category
        y, p = labels[mask], predictions[mask]
        per_category[category] = {
            "f1": float(f1_score(y, p)),
            "fp": int(((y == 0) & (p == 1)).sum()),
            "fn": int(((y == 1) & (p == 0)).sum()),
        }
    return {
        "macro_f1": float(np.mean([v["f1"] for v in per_category.values()])),
        "categories": per_category,
    }


def final_predictions(robust_rank, q3_rank, q35_rank, folds, categories):
    result = np.zeros(len(categories), dtype=np.int8)
    for fold in sorted(np.unique(folds)):
        for category, (w0, w1, w2, threshold) in LORA_FUSION.items():
            mask = (folds == fold) & (categories == category)
            score = w0 * rank01(robust_rank[mask]) + w1 * rank01(q3_rank[mask]) + w2 * rank01(q35_rank[mask])
            result[mask] = (score >= threshold).astype(np.int8)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--four-head", type=Path, required=True)
    parser.add_argument("--qwen3vl", type=Path, required=True)
    parser.add_argument("--qwen35", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    frame = pd.read_csv(args.data).fillna("")
    four = np.load(args.four_head, allow_pickle=False)
    q3 = np.load(args.qwen3vl, allow_pickle=False)
    q35 = np.load(args.qwen35, allow_pickle=False)
    ids = frame["id"].astype(str).to_numpy()
    if not np.array_equal(ids, four["ids"].astype(str)):
        raise ValueError("data/OOF ID mismatch")
    labels = four["labels"].astype(np.int8)
    folds = four["fold_ids"].astype(np.int8)
    categories = four["categories"].astype(str)
    texts = (frame["name"].astype(str) + " \n " + frame["description"].astype(str)).to_numpy()
    bm25 = bm25_oof(texts, labels, folds, categories)
    bm25_rank = fold_category_ranks(bm25, folds, categories)
    original_robust = four["fused"].astype(np.float32)
    candidate_robust = original_robust.copy()
    for category, alpha in FUSION_ALPHA.items():
        mask = categories == category
        candidate_robust[mask] += alpha * (bm25_rank[mask] - four["text"][mask])

    original = final_predictions(
        original_robust, q3["lora_rank"], q35["lora_rank"], folds, categories
    )
    candidate = final_predictions(
        candidate_robust, q3["lora_rank"], q35["lora_rank"], folds, categories
    )
    corrected = (original != labels) & (candidate == labels)
    regressed = (original == labels) & (candidate != labels)
    original_summary = summarize(labels, categories, original)
    candidate_summary = summarize(labels, categories, candidate)
    payload = {
        "schema": "exp710_bm25_vs_tfidf_solution140_v1",
        "changed_factor": "tfidf_text_rank_to_bm25_retrieval_score",
        "bm25": {"top_k": TOP_K, "word_ngram_range": [1, 2], "min_df": 2},
        "component": {
            category: {
                "tfidf_ap": float(average_precision_score(labels[categories == category], four["text"][categories == category])),
                "bm25_ap": float(average_precision_score(labels[categories == category], bm25[categories == category])),
            }
            for category in sorted(np.unique(categories))
        },
        "solution140_tfidf": original_summary,
        "solution140_bm25": candidate_summary,
        "macro_delta": candidate_summary["macro_f1"] - original_summary["macro_f1"],
        "changed_decisions": int((original != candidate).sum()),
        "corrections": int(corrected.sum()),
        "regressions": int(regressed.sum()),
        "inputs_sha256": {
            "data": sha256_file(args.data),
            "four_head": sha256_file(args.four_head),
            "qwen3vl": sha256_file(args.qwen3vl),
            "qwen35": sha256_file(args.qwen35),
        },
        "public_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
