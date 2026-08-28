from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

FLAMMABLE = "Легковоспламеняющиеся"
FORBIDDEN_TEXT = (
    "синтетический пример",
    "метка класса",
    "ответ модели",
    "```",
    "<assistant",
    "<system",
)


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


def normalize(value: object) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value).lower())
    text = re.sub(r"[^a-zа-яё0-9]+", " ", text)
    return " ".join(text.split())


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise TypeError(f"non-object at {path}:{line_number}")
            rows.append(row)
    return rows


def load_real(runtime_path: Path, labels_path: Path) -> list[dict]:
    runtime = load_jsonl(runtime_path)
    labels = load_jsonl(labels_path)
    if len(runtime) != len(labels):
        raise ValueError("runtime/labels row count mismatch")
    result = []
    for row, target in zip(runtime, labels, strict=True):
        key = (row["global_index"], row["id"], row["fold"], row["category"])
        target_key = (
            target["global_index"],
            target["id"],
            target["fold"],
            target["category"],
        )
        if key != target_key:
            raise ValueError("runtime/labels ordered binding mismatch")
        result.append({**row, "label": int(target["label"])})
    return result


def structural_filter(paths: list[Path]) -> tuple[list[dict], dict]:
    accepted = []
    counts: defaultdict[str, int] = defaultdict(int)
    exact_seen: set[str] = set()
    for path in paths:
        source = "v1" if "v1" in path.name else "v2"
        for source_index, row in enumerate(load_jsonl(path)):
            counts["input"] += 1
            required = {"name", "description", "label", "category", "subtype", "why_unambiguous"}
            if not required <= set(row):
                counts["reject_missing_fields"] += 1
                continue
            if row["category"] != FLAMMABLE or type(row["label"]) is not int or row["label"] not in (0, 1):
                counts["reject_contract"] += 1
                continue
            name = str(row["name"]).strip()
            description = str(row["description"]).strip()
            why = str(row["why_unambiguous"]).strip()
            combined = f"{name}\n{description}"
            if not (4 <= len(name) <= 240 and 40 <= len(description) <= 1800 and len(why) >= 20):
                counts["reject_length_or_missing_reason"] += 1
                continue
            lowered = combined.lower()
            if any(marker in lowered for marker in FORBIDDEN_TEXT):
                counts["reject_meta_text"] += 1
                continue
            normalized = normalize(combined)
            if normalized in exact_seen:
                counts["reject_exact_duplicate"] += 1
                continue
            exact_seen.add(normalized)
            accepted.append(
                {
                    "candidate_id": hashlib.sha256(
                        canonical_bytes([source, source_index, normalized])
                    ).hexdigest()[:20],
                    "source": source,
                    "source_index": source_index,
                    "name": name,
                    "description": description,
                    "label": row["label"],
                    "category": FLAMMABLE,
                    "subtype": str(row["subtype"]).strip() or "unknown",
                    "why_unambiguous": why,
                    "prompt_id": str(row.get("prompt_id", "")),
                    "synthetic": True,
                    "normalized_text": normalized,
                }
            )
    counts["structurally_accepted"] = len(accepted)
    return accepted, dict(counts)


def similarity_filter(candidates: list[dict], real_rows: list[dict]) -> tuple[list[dict], dict]:
    real_text = [normalize(f"{row.get('name', '')}\n{row.get('description', '')}") for row in real_rows]
    synth_text = [row["normalized_text"] for row in candidates]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)
    matrix = vectorizer.fit_transform(real_text + synth_text)
    real_matrix = matrix[: len(real_text)]
    synth_matrix = matrix[len(real_text) :]
    max_real = np.asarray((synth_matrix @ real_matrix.T).max(axis=1).toarray()).ravel()

    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]["source"] != "v1",
            candidates[index]["candidate_id"],
        ),
    )
    kept_indices: list[int] = []
    rejected_real = 0
    rejected_synth = 0
    for index in order:
        if max_real[index] >= 0.72:
            rejected_real += 1
            continue
        if kept_indices:
            similarities = synth_matrix[index] @ synth_matrix[kept_indices].T
            if float(similarities.max()) >= 0.86:
                rejected_synth += 1
                continue
        kept_indices.append(index)
    kept = []
    for index in kept_indices:
        row = dict(candidates[index])
        row["max_real_char_similarity"] = float(max_real[index])
        kept.append(row)
    return kept, {
        "near_duplicate_real_threshold": 0.72,
        "near_duplicate_synth_threshold": 0.86,
        "reject_near_real": rejected_real,
        "reject_near_synth": rejected_synth,
        "similarity_accepted": len(kept),
    }


def consensus_scores(train_rows: list[dict], candidates: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    train_text = [normalize(f"{row.get('name', '')}\n{row.get('description', '')}") for row in train_rows]
    synth_text = [row["normalized_text"] for row in candidates]
    labels = np.asarray([row["label"] for row in train_rows], dtype=np.int8)
    all_text = train_text + synth_text

    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=160_000, sublinear_tf=True)
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200_000, sublinear_tf=True
    )
    word_x = word.fit_transform(all_text)
    char_x = char.fit_transform(all_text)
    train_count = len(train_rows)
    matrices = (word_x, char_x, sparse.hstack([word_x, char_x], format="csr"))
    decisions = []
    models = (
        LinearSVC(C=1.0, class_weight="balanced", dual="auto", random_state=42),
        LinearSVC(C=0.7, class_weight="balanced", dual="auto", random_state=43),
        LinearSVC(C=0.35, class_weight="balanced", dual="auto", random_state=44),
    )
    for model, matrix in zip(models, matrices, strict=True):
        model.fit(matrix[:train_count], labels)
        decision = model.decision_function(matrix[train_count:])
        scale = max(float(np.median(np.abs(decision))), 1e-6)
        decisions.append(np.asarray(decision, dtype=np.float64) / scale)
    values = np.stack(decisions, axis=1)
    target_sign = 2 * np.asarray([row["label"] for row in candidates], dtype=np.int8) - 1
    signed = values * target_sign[:, None]
    agreement = np.sum(signed > 0, axis=1)
    confidence = np.mean(signed, axis=1)
    return agreement, confidence


def diversified_order(rows: list[dict]) -> list[dict]:
    buckets: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["source"], normalize(row["subtype"]) or "unknown")].append(row)
    for bucket in buckets.values():
        bucket.sort(
            key=lambda row: (
                -row["selector_agreement"],
                -row["selector_confidence"],
                -row["novelty"],
                row["candidate_id"],
            )
        )
    ordered = []
    keys = sorted(buckets)
    while keys:
        next_keys = []
        for key in keys:
            bucket = buckets[key]
            if bucket:
                ordered.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return ordered


def build_filter_payloads(
    runtime_path: Path, labels_path: Path, synth_paths: list[Path]
) -> tuple[dict, dict[str, bytes]]:
    real_rows = load_real(runtime_path, labels_path)
    candidates, structural = structural_filter(synth_paths)
    candidates, similarity = similarity_filter(candidates, real_rows)

    fold_reports = {}
    payloads: dict[str, bytes] = {}
    for fold in range(5):
        train = [
            row
            for row in real_rows
            if row["category"] == FLAMMABLE and int(row["fold"]) != fold
        ]
        agreement, confidence = consensus_scores(train, candidates)
        eligible = []
        for row, agree, conf in zip(candidates, agreement, confidence, strict=True):
            item = dict(row)
            item["selector_agreement"] = int(agree)
            item["selector_confidence"] = float(conf)
            item["novelty"] = 1.0 - float(item["max_real_char_similarity"])
            item["selector_fold"] = fold
            if agree >= 2 and conf >= 0.0:
                eligible.append(item)

        selected = []
        for label in (0, 1):
            label_rows = diversified_order([row for row in eligible if row["label"] == label])
            for rank, row in enumerate(label_rows[:320], 1):
                row["label_rank"] = rank
                selected.append(row)
        selected.sort(key=lambda row: (row["label"], row["label_rank"], row["candidate_id"]))
        payload = b"".join(
            canonical_bytes(
                {key: value for key, value in row.items() if key != "normalized_text"}
            )
            + b"\n"
            for row in selected
        )
        filename = f"fold{fold}_ranked.jsonl"
        payloads[filename] = payload
        counts = {
            str(label): sum(row["label"] == label for row in selected) for label in (0, 1)
        }
        fold_reports[str(fold)] = {
            "real_train_rows": len(train),
            "real_train_positive": sum(row["label"] == 1 for row in train),
            "eligible_before_cap": len(eligible),
            "selected_by_label": counts,
            "selected_sha256": hashlib.sha256(payload).hexdigest(),
        }

    report = {
        "schema": "exp699_synth_filter_v1",
        "runtime_sha256": sha256(runtime_path),
        "labels_sha256": sha256(labels_path),
        "synth_inputs": {path.name: sha256(path) for path in synth_paths},
        "structural": structural,
        "similarity": similarity,
        "selection": {
            "outer_fold_safe": True,
            "validation_labels_read_by_selector": 0,
            "minimum_model_agreement": 2,
            "minimum_mean_signed_margin": 0.0,
            "maximum_rows_per_label": 320,
            "recommended_caps_per_label": [40, 80, 160],
        },
        "folds": fold_reports,
    }
    report["self_sha256"] = hashlib.sha256(canonical_bytes(report)).hexdigest()
    payloads["filter_report.json"] = (
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    return report, payloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--synth", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report, payloads = build_filter_payloads(args.runtime, args.labels, args.synth)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for filename, payload in payloads.items():
        (args.output_dir / filename).write_bytes(payload)
    report_path = args.output_dir / "filter_report.json"
    if sha256(report_path) != hashlib.sha256(payloads["filter_report.json"]).hexdigest():
        raise ValueError("written report hash mismatch")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
