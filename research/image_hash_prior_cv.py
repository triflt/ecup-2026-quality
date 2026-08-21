from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


DATA = Path(os.environ.get("ECUP_DATA", "/work/input/data.csv"))
MANIFEST = Path(os.environ.get("ECUP_MANIFEST", "/work/input/multi_image_manifest.tsv.gz"))
LORA = Path(os.environ.get("ECUP_FUSION", "/work/input/lora_fusion.npz"))
OUTPUT = Path(os.environ.get("ECUP_OUTPUT_DIR", "/work/output"))


def f1(labels, predictions):
    labels = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / max(1, 2 * tp + fp + fn)


def normalize(value):
    import re
    return re.sub(r"\s+", " ", str(value or "").lower().replace("ё", "е")).strip()


def compose_text(name, description):
    name = normalize(name)
    return f"{name}\n{name}\n{normalize(description)}"


def text_fingerprint(name, description):
    return hashlib.sha1(compose_text(name, description).encode("utf-8")).hexdigest()


def group_loo(frame, key):
    counts = frame.groupby(["category", key]).label.transform("count").to_numpy(np.int32)
    sums = frame.groupby(["category", key]).label.transform("sum").to_numpy(np.float32)
    other_count = counts - 1
    other_mean = np.divide(
        sums - frame.label.to_numpy(np.float32), other_count,
        out=np.full(len(frame), np.nan, np.float32), where=other_count > 0,
    )
    return other_count, np.maximum(other_mean, 1 - other_mean), (other_mean >= 0.5).astype(np.int8)


def apply_text_prior(base, frame):
    exact = group_loo(frame, "text_hash")
    name = group_loo(frame, "normalized_name")
    configs = {
        "БАД": (2, 2 / 3, 2, 0.999),
        "Легковоспламеняющиеся": (1, 0.999, 999, 0.999),
    }
    result = base.copy()
    changed = np.zeros(len(frame), dtype=bool)
    categories = frame.category.to_numpy(str)
    for category, (exact_min, exact_conf, name_min, name_conf) in configs.items():
        local = categories == category
        use_exact = local & (exact[0] >= exact_min) & (exact[1] >= exact_conf)
        use_name = local & ~use_exact & (name[0] >= name_min) & (name[1] >= name_conf)
        result[use_exact] = exact[2][use_exact]
        result[use_name] = name[2][use_name]
        changed |= use_exact | use_name
    return result, changed


def perceptual_key(image):
    gray = image.convert("L")
    dh = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    ah = np.asarray(gray.resize((8, 8), Image.Resampling.LANCZOS), dtype=np.float32)
    d_bits = (dh[:, 1:] > dh[:, :-1]).reshape(-1)
    a_bits = (ah >= ah.mean()).reshape(-1)
    d_value = sum(int(bit) << index for index, bit in enumerate(d_bits))
    a_value = sum(int(bit) << index for index, bit in enumerate(a_bits))
    return f"{d_value:016x}{a_value:016x}"


def hash_url(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        pkey = perceptual_key(image)
    return hashlib.sha1(payload).hexdigest(), pkey


def load_manifest():
    rows = []
    with gzip.open(MANIFEST, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            for url in json.loads(row["image_urls"]):
                rows.append((str(row["id"]), url))
    return rows


def download_hashes(items):
    by_id = defaultdict(lambda: {"exact": set(), "perceptual": set()})
    failures = []
    with ThreadPoolExecutor(max_workers=48) as pool:
        futures = {pool.submit(hash_url, url): (item_id, url) for item_id, url in items}
        for index, future in enumerate(as_completed(futures), 1):
            item_id, url = futures[future]
            try:
                exact, perceptual = future.result()
                by_id[item_id]["exact"].add(exact)
                by_id[item_id]["perceptual"].add(perceptual)
            except Exception as error:
                failures.append({"id": item_id, "url": url, "error": str(error)})
            if index % 1000 == 0 or index == len(futures):
                print(f"hashed={index}/{len(futures)} failures={len(failures)}", flush=True)
    return by_id, failures


def token_statistics(frame, row_tokens, kind):
    stats = defaultdict(Counter)
    categories = frame.category.to_numpy(str)
    labels = frame.label.to_numpy(np.int8)
    for index, tokens in enumerate(row_tokens):
        for token in tokens[kind]:
            stats[(categories[index], token)][int(labels[index])] += 1
    return stats


def best_loo_candidate(frame, row_tokens, kind, stats):
    categories = frame.category.to_numpy(str)
    labels = frame.label.to_numpy(np.int8)
    counts = np.zeros(len(frame), np.int32)
    confidence = np.zeros(len(frame), np.float32)
    predictions = np.zeros(len(frame), np.int8)
    for index, tokens in enumerate(row_tokens):
        best = None
        for token in tokens[kind]:
            counter = stats[(categories[index], token)].copy()
            counter[int(labels[index])] -= 1
            total = counter[0] + counter[1]
            if total <= 0:
                continue
            positive_rate = counter[1] / total
            conf = max(positive_rate, 1 - positive_rate)
            candidate = (conf, total, int(positive_rate >= 0.5))
            if best is None or candidate > best:
                best = candidate
        if best is not None:
            confidence[index], counts[index], predictions[index] = best
    return counts, confidence, predictions


def apply_image_config(base, text_changed, arrays, config, positions):
    exact_min, exact_conf, perceptual_min, perceptual_conf = config
    exact, perceptual = arrays
    result = base[positions].copy()
    available = ~text_changed[positions]
    use_exact = available & (exact[0][positions] >= exact_min) & (exact[1][positions] >= exact_conf)
    use_perceptual = available & ~use_exact & (perceptual[0][positions] >= perceptual_min) & (perceptual[1][positions] >= perceptual_conf)
    result[use_exact] = exact[2][positions][use_exact]
    result[use_perceptual] = perceptual[2][positions][use_perceptual]
    return result, int(use_exact.sum()), int(use_perceptual.sum())


def build_mapping(frame, row_tokens, configs):
    labels = frame.label.to_numpy(np.int8)
    categories = frame.category.to_numpy(str)
    mapping = {"exact": {}, "perceptual": {}, "configs": configs}
    for kind in ("exact", "perceptual"):
        stats = token_statistics(frame, row_tokens, kind)
        min_support_index = 0 if kind == "exact" else 2
        min_conf_index = 1 if kind == "exact" else 3
        for (category, token), counter in stats.items():
            total = counter[0] + counter[1]
            rate = counter[1] / total
            confidence = max(rate, 1 - rate)
            config = configs[category]
            if total >= config[min_support_index] and confidence >= config[min_conf_index] and rate != 0.5:
                mapping[kind][f"{category}\t{token}"] = int(rate >= 0.5)
    return mapping


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        urllib.request.urlretrieve(os.environ["DATA_URL"], DATA)
    frame = pd.read_csv(DATA)
    frame["name"] = frame.name.fillna("").astype(str)
    frame["description"] = frame.description.fillna("").astype(str)
    frame["category"] = frame.category.astype(str)
    frame["normalized_name"] = frame.name.map(normalize)
    frame["text_hash"] = [text_fingerprint(a, b) for a, b in zip(frame.name, frame.description)]
    lora = np.load(LORA, allow_pickle=True)
    if not np.array_equal(frame.id.astype(str).to_numpy(), lora["ids"].astype(str)):
        raise ValueError("id mismatch")
    base = np.zeros(len(frame), np.int8)
    for category, (base_weight, lora_weight, threshold) in {
        "БАД": (0.40, 0.60, 0.25814030990600584),
        "Легковоспламеняющиеся": (0.75, 0.25, 0.9654546632766724),
    }.items():
        mask = frame.category.to_numpy(str) == category
        base[mask] = base_weight * lora["base_rank"][mask] + lora_weight * lora["lora_rank"][mask] >= threshold
    text_prior, text_changed = apply_text_prior(base, frame)
    items = load_manifest()
    by_id, failures = download_hashes(items)
    row_tokens = [by_id[str(item_id)] for item_id in frame.id]
    exact_stats = token_statistics(frame, row_tokens, "exact")
    perceptual_stats = token_statistics(frame, row_tokens, "perceptual")
    arrays = (
        best_loo_candidate(frame, row_tokens, "exact", exact_stats),
        best_loo_candidate(frame, row_tokens, "perceptual", perceptual_stats),
    )
    grid = list(product(
        [1, 2, 3, 4, 999], [2 / 3, 0.75, 0.8, 0.9, 0.999],
        [2, 3, 4, 5, 8, 999], [0.75, 0.8, 0.9, 0.999],
    ))
    labels = frame.label.to_numpy(np.int8)
    categories = frame.category.to_numpy(str)
    folds = lora["folds"].astype(np.int8)
    report = {"images": len(items), "download_failures": len(failures), "categories": {}}
    chosen_configs = {}
    macro = []
    for category in sorted(frame.category.unique()):
        category_mask = categories == category
        nested = text_prior.copy()
        choices, fold_rows = [], []
        for fold in sorted(np.unique(folds)):
            train = np.flatnonzero(category_mask & (folds != fold))
            valid = np.flatnonzero(category_mask & (folds == fold))
            best = None
            for config in grid:
                predictions, exact_count, perceptual_count = apply_image_config(text_prior, text_changed, arrays, config, train)
                value = f1(labels[train], predictions)
                candidate = (value, -(exact_count + perceptual_count), config)
                if best is None or candidate > best:
                    best = candidate
            config = best[2]
            predictions, exact_count, perceptual_count = apply_image_config(text_prior, text_changed, arrays, config, valid)
            nested[valid] = predictions
            choices.append(config)
            fold_rows.append({
                "fold": int(fold), "config": list(config),
                "validation_f1": f1(labels[valid], predictions),
                "exact_overrides": exact_count, "perceptual_overrides": perceptual_count,
            })
        chosen = Counter(choices).most_common(1)[0][0]
        chosen_configs[category] = chosen
        local = np.flatnonzero(category_mask)
        full_predictions, exact_count, perceptual_count = apply_image_config(text_prior, text_changed, arrays, chosen, local)
        nested_f1 = f1(labels[category_mask], nested[category_mask])
        report["categories"][category] = {
            "base_lora_f1": f1(labels[category_mask], base[category_mask]),
            "text_prior_f1": f1(labels[category_mask], text_prior[category_mask]),
            "nested_image_prior_f1": nested_f1,
            "chosen_config": list(chosen),
            "chosen_loo_f1": f1(labels[category_mask], full_predictions),
            "exact_overrides": exact_count,
            "perceptual_overrides": perceptual_count,
            "folds": fold_rows,
        }
        macro.append(nested_f1)
    report["nested_macro_f1"] = float(np.mean(macro))
    mapping = build_mapping(frame, row_tokens, chosen_configs)
    (OUTPUT / "image_hash_prior_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with gzip.open(OUTPUT / "image_hash_prior.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(mapping, stream, ensure_ascii=False, separators=(",", ":"))
    (OUTPUT / "image_hash_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print({"mapping_exact": len(mapping["exact"]), "mapping_perceptual": len(mapping["perceptual"])} , flush=True)


if __name__ == "__main__":
    main()
