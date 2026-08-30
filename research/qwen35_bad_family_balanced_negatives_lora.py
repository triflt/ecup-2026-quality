from __future__ import annotations

import csv
import gzip
import html
import io
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from transformers import AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoProcessor


MODEL = Path(os.environ.get("ECUP_MODEL_ROOT", "/hf_models"))
DATA = Path(os.environ.get("ECUP_DATA", "/work/input/data.csv"))
DATA_PARTS = Path(os.environ.get("ECUP_DATA_PARTS", "/work/input/data_parts"))
MANIFEST = Path(os.environ.get("ECUP_MANIFEST", "/work/input/lora_image_manifest.tsv.gz"))
OOF = Path(os.environ.get("ECUP_OOF", "/work/input/four_head_oof.npz"))
IMAGE_DIR = Path(os.environ.get("ECUP_IMAGES", "/work/images"))
OUTPUT = Path(os.environ.get("ECUP_OUTPUT_DIR", "/work/output"))
VENDOR = Path(os.environ.get("ECUP_VENDOR", "/work/vendor"))
SEED = int(os.environ.get("SEED", "42"))
HOLDOUT_FOLD = int(os.environ.get("HOLDOUT_FOLD", "4"))
MAX_LENGTH = 1536
BATCH_SIZE = 4
GRAD_ACCUM = 4
EPOCHS = 1
TRAINING_MODE = os.environ.get("TRAINING_MODE", "hard").strip().lower()
MODEL_CLASS = os.environ.get("MODEL_CLASS", "image_text").strip().lower()
USE_CHAT_BATCH = MODEL_CLASS == "multimodal" or os.environ.get("USE_CHAT_BATCH") == "1"
FULL_TRAIN = os.environ.get("FULL_TRAIN") == "1"
DESCRIPTION_LIMIT = int(os.environ.get("DESCRIPTION_LIMIT", "1800"))
FAMILY_BALANCE_FLAMMABLE = os.environ.get("FAMILY_BALANCE_FLAMMABLE", "1") == "1"
FAMILY_DIVERSE_BAD_POSITIVES = (
    os.environ.get("FAMILY_DIVERSE_BAD_POSITIVES", "1") == "1"
)
FAMILY_BALANCE_BAD_NEGATIVES = (
    os.environ.get("FAMILY_BALANCE_BAD_NEGATIVES", "1") == "1"
)
FAMILY_DIVERSE_FLAMMABLE_NEGATIVES = (
    os.environ.get("FAMILY_DIVERSE_FLAMMABLE_NEGATIVES", "0") == "1"
)


RULES = {
    "БАД": (
        "Метка 1 только если в описании или на упаковке есть прямое указание БАД "
        "или dietary supplement. Спортивное питание без такой маркировки, явное "
        "отрицание или отсутствие маркировки — метка 0."
    ),
    "Легковоспламеняющиеся": (
        "Метка 1 для самостоятельного источника огня, горючего вещества или газа, "
        "либо если такой товар входит в комплект. Пустое оборудование, встроенный "
        "источник, горючий материал только как компонент или предмет не в комплекте — 0."
    ),
}


def install_peft():
    VENDOR.mkdir(parents=True, exist_ok=True)
    if not (VENDOR / "peft").is_dir():
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "--target", str(VENDOR), "--no-cache-dir", "--no-deps", "peft==0.20.0",
        ], check=True)
    sys.path.insert(0, str(VENDOR))


def rank01(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


def fused_oof_scores(oof):
    key = "fused_scores" if "fused_scores" in oof.files else "fused"
    return oof[key].astype(np.float32)


def oof_thresholds(oof, categories):
    values = (
        {"БАД": 0.24864045896205267, "Легковоспламеняющиеся": 0.9591804083988902}
        if "fused_scores" in oof.files
        else {"БАД": 0.251901438832283, "Легковоспламеняющиеся": 0.9540719747543336}
    )
    return np.asarray([values[category] for category in categories], dtype=np.float32)


def best_threshold(labels, scores):
    candidates = np.unique(np.quantile(scores, np.linspace(0.002, 0.998, 700)))
    best = (-1.0, 0.0)
    for threshold in candidates:
        value = f1_score(labels, scores >= threshold)
        if value > best[0]:
            best = (float(value), float(threshold))
    return best


def load_urls():
    with gzip.open(MANIFEST, "rt", encoding="utf-8", newline="") as stream:
        return {row["id"]: row["image_url"] for row in csv.DictReader(stream, delimiter="\t")}


def download_one(item_id, url):
    destination = IMAGE_DIR / f"{item_id}.jpg"
    if destination.is_file() and destination.stat().st_size > 0:
        return item_id, True, "cached"
    last_error = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            image = Image.open(io.BytesIO(payload)).convert("RGB")
            image.thumbnail((448, 448), Image.Resampling.LANCZOS)
            image.save(destination, format="JPEG", quality=92)
            return item_id, True, ""
        except Exception as error:
            last_error = error
    Image.new("RGB", (32, 32), "white").save(destination, format="JPEG")
    return item_id, False, str(last_error)


def predownload(ids, urls):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(download_one, item_id, urls[item_id]) for item_id in ids]
        for index, future in enumerate(as_completed(futures), 1):
            item_id, ok, error = future.result()
            if not ok:
                failures.append((item_id, error))
            if index % 500 == 0 or index == len(futures):
                print(
                    f"downloaded={index}/{len(futures)} failures={len(failures)} "
                    f"elapsed_min={(time.monotonic()-started)/60:.1f}", flush=True,
                )
    return failures


def load_training_frame():
    if DATA.exists():
        return pd.read_csv(DATA)
    parts = sorted(DATA_PARTS.glob("data.csv.gz.part-*"))
    if parts:
        payload = b"".join(part.read_bytes() for part in parts)
        return pd.read_csv(io.BytesIO(gzip.decompress(payload)))
    urllib.request.urlretrieve(os.environ["DATA_URL"], DATA)
    return pd.read_csv(DATA)


def hard_random(indices, scores, count, rng):
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= count:
        return indices.tolist()
    hard_count = count // 2
    hard = indices[np.argsort(scores[indices])[:hard_count]]
    remaining = np.setdiff1d(indices, hard, assume_unique=False)
    random_part = rng.choice(remaining, size=count - hard_count, replace=False)
    return np.concatenate([hard, random_part]).tolist()


def normalize_group_text(value):
    value = "" if pd.isna(value) else html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    value = unicodedata.normalize("NFKC", value).lower().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def family_key(row):
    name = normalize_group_text(row["name"])
    description = normalize_group_text(row["description"])
    return f"{name}\n{name}\n{description}"


def family_balanced_records(frame, indices, total_count, rng):
    groups = {}
    for index in np.asarray(indices, dtype=np.int64):
        groups.setdefault(family_key(frame.iloc[index]), []).append(int(index))
    keys = list(groups)
    rng.shuffle(keys)
    base, remainder = divmod(total_count, len(keys))
    records = []
    exposures = []
    for position, key in enumerate(keys):
        count = base + int(position < remainder)
        records.extend(rng.choice(groups[key], size=count, replace=True).tolist())
        exposures.append(count)
    return records, {
        "positive_rows": int(len(indices)),
        "positive_families": int(len(keys)),
        "total_exposures": int(total_count),
        "min_family_exposures": int(min(exposures)),
        "max_family_exposures": int(max(exposures)),
    }


def family_diverse_hard_random(frame, indices, scores, count, rng):
    """Select one row per family, half hard and half random.

    Family hardness is the uncertainty of its most ambiguous row. The random
    half is sampled uniformly over the remaining families rather than rows, so
    large duplicate families cannot occupy multiple training slots.
    """
    groups = {}
    for index in np.asarray(indices, dtype=np.int64):
        groups.setdefault(family_key(frame.iloc[index]), []).append(int(index))
    if len(groups) < count:
        raise ValueError(
            f"need {count} distinct negative families, found only {len(groups)}"
        )
    representatives = {
        key: min(local, key=lambda index: float(scores[index]))
        for key, local in groups.items()
    }
    hard_count = count // 2
    ordered = sorted(representatives, key=lambda key: float(scores[representatives[key]]))
    hard_keys = ordered[:hard_count]
    remaining_keys = np.asarray(ordered[hard_count:], dtype=object)
    random_keys = rng.choice(
        remaining_keys, size=count - hard_count, replace=False
    ).tolist()
    selected = [representatives[key] for key in hard_keys]
    selected.extend(
        int(rng.choice(groups[key])) for key in random_keys
    )
    return selected, {
        "eligible_rows": int(len(indices)),
        "eligible_families": int(len(groups)),
        "selected_rows": int(len(selected)),
        "selected_families": int(len({family_key(frame.iloc[index]) for index in selected})),
        "hard_families": int(len(hard_keys)),
        "random_families": int(len(random_keys)),
        "max_family_exposures": 1,
    }


def family_cover_hard_records(frame, indices, scores, count):
    """Cover every family once, then give a second slot to the hardest ones.

    The first representative is the most uncertain row of each family. If the
    requested count exceeds the number of families, additional slots go to the
    hardest families. A different row is used where possible; otherwise the
    first representative is repeated. This keeps the original class exposure
    count while preventing a large duplicated family from occupying many slots.
    """
    groups = {}
    for index in np.asarray(indices, dtype=np.int64):
        groups.setdefault(family_key(frame.iloc[index]), []).append(int(index))
    if len(groups) > count:
        raise ValueError(
            f"cannot cover {len(groups)} negative families with {count} records"
        )
    ordered_groups = {
        key: sorted(local, key=lambda index: float(scores[index]))
        for key, local in groups.items()
    }
    keys = sorted(
        ordered_groups,
        key=lambda key: float(scores[ordered_groups[key][0]]),
    )
    selected = [ordered_groups[key][0] for key in keys]
    extra_count = count - len(selected)
    if extra_count > len(keys):
        raise ValueError(
            f"family coverage needs {extra_count} extra slots for only {len(keys)} families"
        )
    for key in keys[:extra_count]:
        local = ordered_groups[key]
        selected.append(local[1] if len(local) > 1 else local[0])
    family_exposures = {}
    for index in selected:
        key = family_key(frame.iloc[index])
        family_exposures[key] = family_exposures.get(key, 0) + 1
    return selected, {
        "eligible_rows": int(len(indices)),
        "eligible_families": int(len(groups)),
        "selected_rows": int(len(selected)),
        "selected_families": int(len(family_exposures)),
        "duplicate_slots": int(len(selected) - len(family_exposures)),
        "max_family_exposures": int(max(family_exposures.values())),
        "all_families_covered": len(family_exposures) == len(groups),
    }


def select_training(frame, oof):
    rng = np.random.default_rng(SEED)
    categories = frame["category"].astype(str).to_numpy()
    labels = frame["label"].to_numpy(dtype=np.int8)
    folds = oof["fold_ids"].astype(np.int8)
    # Smaller absolute distance means a harder example. All selection is within
    # the outer training folds, so holdout labels never affect adaptation.
    thresholds = oof_thresholds(oof, categories)
    fused_scores = fused_oof_scores(oof)
    uncertainty = np.abs(fused_scores - thresholds)
    train_mask = np.ones(len(folds), dtype=bool) if FULL_TRAIN else folds != HOLDOUT_FOLD
    if TRAINING_MODE == "clean":
        # Rakuten's winning solution removed the most likely label errors
        # based on cross-validated predictions. Reproduce that idea without
        # touching the outer holdout: rank suspicious contradictions inside
        # each category and remove the top 10% from training eligibility.
        threshold_map = (
            {"БАД": 0.24864045896205267, "Легковоспламеняющиеся": 0.9591804083988902}
            if "fused_scores" in oof.files
            else {"БАД": 0.251901438832283, "Легковоспламеняющиеся": 0.9540719747543336}
        )
        for category, threshold in threshold_map.items():
            local = np.flatnonzero(train_mask & (categories == category))
            score = fused_scores[local]
            suspicious = np.where(
                labels[local] == 1,
                np.maximum(threshold - score, 0.0),
                np.maximum(score - threshold, 0.0),
            )
            remove_count = int(round(0.10 * len(local)))
            if remove_count:
                remove = local[np.argsort(suspicious)[-remove_count:]]
                train_mask[remove] = False
    records = []

    bad = np.flatnonzero(train_mask & (categories == "БАД"))
    bad_pos = bad[labels[bad] == 1]
    bad_neg = bad[labels[bad] == 0]
    bad_limit = 1900 if FULL_TRAIN else 1500
    bad_count = min(bad_limit, len(bad_neg), len(bad_pos))
    if TRAINING_MODE == "clean":
        selected_bad_positives = rng.choice(
            bad_pos, size=bad_count, replace=False
        ).tolist()
        bad_positive_selection_audit = {
            "family_diverse": False,
            "selected_rows": int(bad_count),
        }
        records.extend(selected_bad_positives)
        selected_bad_negatives = rng.choice(
            bad_neg, size=bad_count, replace=False
        ).tolist()
        records.extend(selected_bad_negatives)
        bad_negative_selection_audit = {
            "family_balanced": False,
            "reason": "clean mode uses its frozen random-row control",
            "selected_rows": int(len(selected_bad_negatives)),
        }
    else:
        # Always consume the control selector first. This keeps the RNG state
        # for BAD negatives and the flammable strata byte-for-byte identical
        # to the parent training run.
        control_bad_positives = hard_random(
            bad_pos, uncertainty, bad_count, rng
        )
        if FAMILY_DIVERSE_BAD_POSITIVES:
            family_rng = np.random.default_rng(SEED + 26001)
            selected_bad_positives, bad_positive_selection_audit = family_diverse_hard_random(
                frame, bad_pos, uncertainty, bad_count, family_rng
            )
            bad_positive_selection_audit["family_diverse"] = True
            bad_positive_selection_audit["control_rng_consumed"] = True
        else:
            selected_bad_positives = control_bad_positives
            bad_positive_selection_audit = {
                "family_diverse": False,
                "selected_rows": int(bad_count),
            }
        records.extend(selected_bad_positives)
        # Consume the unchanged row-level selector first so the random state of
        # all later flammable sampling is identical to the parent experiment.
        control_bad_negatives = hard_random(
            bad_neg, uncertainty, bad_count, rng
        )
        if FAMILY_BALANCE_BAD_NEGATIVES:
            selected_bad_negatives, bad_negative_selection_audit = (
                family_cover_hard_records(
                    frame, bad_neg, uncertainty, bad_count
                )
            )
            bad_negative_selection_audit["family_balanced"] = True
            bad_negative_selection_audit["control_rng_consumed"] = True
        else:
            selected_bad_negatives = control_bad_negatives
            keys = [family_key(frame.iloc[index]) for index in selected_bad_negatives]
            bad_negative_selection_audit = {
                "family_balanced": False,
                "selected_rows": int(len(selected_bad_negatives)),
                "selected_families": int(len(set(keys))),
                "duplicate_slots": int(len(keys) - len(set(keys))),
            }
        records.extend(selected_bad_negatives)

    flam = np.flatnonzero(train_mask & (categories == "Легковоспламеняющиеся"))
    flam_pos = flam[labels[flam] == 1]
    flam_neg = flam[labels[flam] == 0]
    if FAMILY_BALANCE_FLAMMABLE:
        family_rng = np.random.default_rng(SEED + 24001)
        flammable_positive_records, selection_audit = family_balanced_records(
            frame, flam_pos, len(flam_pos) * 5, family_rng
        )
        records.extend(flammable_positive_records)
    else:
        records.extend(np.repeat(flam_pos, 5).tolist())
        selection_audit = {
            "positive_rows": int(len(flam_pos)),
            "positive_families": None,
            "total_exposures": int(len(flam_pos) * 5),
            "min_family_exposures": None,
            "max_family_exposures": None,
        }
    flam_neg_limit = 2000 if FULL_TRAIN else 1600
    if TRAINING_MODE == "clean":
        count = min(flam_neg_limit, len(flam_neg))
        records.extend(rng.choice(flam_neg, size=count, replace=False).tolist())
        negative_selection_audit = {
            "family_diverse": False,
            "reason": "clean mode uses its frozen random-row control",
            "selected_rows": int(count),
        }
    elif FAMILY_DIVERSE_FLAMMABLE_NEGATIVES:
        selected_negatives, negative_selection_audit = family_diverse_hard_random(
            frame,
            flam_neg,
            uncertainty,
            min(flam_neg_limit, len(flam_neg)),
            rng,
        )
        negative_selection_audit["family_diverse"] = True
        records.extend(selected_negatives)
    else:
        records.extend(hard_random(flam_neg, uncertainty, min(flam_neg_limit, len(flam_neg)), rng))
        negative_selection_audit = {
            "family_diverse": False,
            "selected_rows": int(min(flam_neg_limit, len(flam_neg))),
        }
    random.Random(SEED).shuffle(records)
    selection_audit["family_balanced"] = FAMILY_BALANCE_FLAMMABLE
    selection_audit["bad_positive_selection"] = bad_positive_selection_audit
    selection_audit["bad_negative_selection"] = bad_negative_selection_audit
    selection_audit["negative_selection"] = negative_selection_audit
    return records, selection_audit


def compact_text(value, limit=1800):
    """Keep the decision-bearing ends while bounding multimodal token length."""
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    head = int(limit * 0.7)
    return value[:head].rstrip() + " … " + value[-(limit - head):].lstrip()


def user_text(row):
    return (
        f"Категория: {row.category}\n"
        f"Название: {compact_text(row.name, 320)}\n"
        f"Описание: {compact_text(row.description, DESCRIPTION_LIMIT)}\n"
        f"Правило: {RULES[row.category]}\n"
        "Определи правильность категории. Ответь только одной цифрой: 1 или 0."
    )


def messages(row, with_answer=False, image=None):
    value = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image if image is not None else str(IMAGE_DIR / f"{row.id}.jpg"),
            },
            {"type": "text", "text": user_text(row)},
        ],
    }]
    if with_answer:
        value.append({"role": "assistant", "content": [{"type": "text", "text": str(int(row.label))}]})
    return value


def open_images(rows):
    return [Image.open(IMAGE_DIR / f"{row.id}.jpg").convert("RGB") for row in rows]


def chat_batch(processor, conversations, add_generation_prompt):
    kwargs = dict(
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    try:
        return processor.apply_chat_template(
            conversations, enable_thinking=False, **kwargs
        )
    except TypeError:
        return processor.apply_chat_template(conversations, **kwargs)


def training_batch(processor, rows):
    if USE_CHAT_BATCH:
        images = open_images(rows)
        batch = chat_batch(
            processor,
            [messages(row, True, image) for row, image in zip(rows, images)],
            False,
        )
        prompt_batch = chat_batch(
            processor,
            [messages(row, False, image) for row, image in zip(rows, images)],
            True,
        )
        for image in images:
            image.close()
    else:
        full_prompts = [
            processor.apply_chat_template(messages(row, True), tokenize=False, add_generation_prompt=False)
            for row in rows
        ]
        generation_prompts = [
            processor.apply_chat_template(messages(row, False), tokenize=False, add_generation_prompt=True)
            for row in rows
        ]
        images = open_images(rows)
        batch = processor(
            text=full_prompts,
            images=images,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        prompt_batch = processor(
            text=generation_prompts,
            images=images,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        for image in images:
            image.close()
    labels = torch.full_like(batch["input_ids"], -100)
    for row_index, row in enumerate(rows):
        full_positions = torch.nonzero(batch["attention_mask"][row_index]).flatten()
        prompt_positions = torch.nonzero(prompt_batch["attention_mask"][row_index]).flatten()
        full_ids = batch["input_ids"][row_index, full_positions]
        prompt_ids = prompt_batch["input_ids"][row_index, prompt_positions]
        limit = min(len(full_ids), len(prompt_ids))
        mismatch = torch.nonzero(full_ids[:limit] != prompt_ids[:limit]).flatten()
        common = int(mismatch[0]) if len(mismatch) else limit
        if common < len(prompt_ids) - 2 or common >= len(full_ids):
            raise ValueError(
                f"chat suffix alignment failed for id={row.id}: "
                f"common={common} prompt={len(prompt_ids)} full={len(full_ids)}"
            )
        answer_positions = full_positions[common:]
        labels[row_index, answer_positions] = batch["input_ids"][row_index, answer_positions]
    batch["labels"] = labels
    return batch


@torch.inference_mode()
def validation_scores(model, processor, frame, positions, token_zero, token_one):
    model.eval()
    scores = []
    for start in range(0, len(positions), 8):
        local = positions[start:start + 8]
        rows = [frame.iloc[index] for index in local]
        if USE_CHAT_BATCH:
            images = open_images(rows)
            batch = chat_batch(
                processor,
                [messages(row, False, image) for row, image in zip(rows, images)],
                True,
            )
            for image in images:
                image.close()
        else:
            prompts = [
                processor.apply_chat_template(messages(row, False), tokenize=False, add_generation_prompt=True)
                for row in rows
            ]
            images = open_images(rows)
            batch = processor(
                text=prompts,
                images=images,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            for image in images:
                image.close()
        batch = {key: value.to("cuda") for key, value in batch.items()}
        outputs = model(**batch)
        sequence = torch.arange(batch["attention_mask"].shape[1], device="cuda")[None, :]
        last = torch.where(batch["attention_mask"].bool(), sequence, -1).max(dim=1).values
        logits = outputs.logits[torch.arange(len(rows), device="cuda"), last]
        values = logits[:, token_one] - logits[:, token_zero]
        scores.extend(values.float().cpu().numpy().tolist())
        if len(scores) % 400 < len(rows) or len(scores) == len(positions):
            print(f"validated={len(scores)}/{len(positions)}", flush=True)
    return np.asarray(scores, dtype=np.float32)


def save_adapter(model):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT / "adapter")
    shutil.make_archive(str(OUTPUT / "adapter"), "zip", OUTPUT / "adapter")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    install_peft()
    from peft import LoraConfig, TaskType, get_peft_model

    frame = load_training_frame()
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["category"] = frame["category"].astype(str)
    oof = np.load(OOF, allow_pickle=True)
    ids = frame["id"].astype(str).to_numpy()
    if not np.array_equal(ids, oof["ids"].astype(str)):
        raise ValueError("OOF id mismatch")
    urls = load_urls()
    train_records, selection_audit = select_training(frame, oof)
    val_positions = (
        np.asarray([], dtype=np.int64)
        if FULL_TRAIN
        else np.flatnonzero(oof["fold_ids"].astype(np.int8) == HOLDOUT_FOLD)
    )
    needed = sorted(set(ids[train_records]) | set(ids[val_positions]))
    failures = predownload(needed, urls)
    print(json.dumps({
        "training_mode": TRAINING_MODE,
        "train_records": len(train_records),
        "train_unique": len(set(train_records)),
        "validation": len(val_positions),
        "images": len(needed),
        "download_failures": len(failures),
        "flammable_selection": selection_audit,
    }), flush=True)

    processor_kwargs = {"local_files_only": True, "trust_remote_code": True}
    if MODEL_CLASS == "image_text":
        processor_kwargs.update(min_pixels=4 * 28 * 28, max_pixels=262144)
    processor = AutoProcessor.from_pretrained(MODEL, **processor_kwargs)
    processor.tokenizer.padding_side = "left"
    token_zero_ids = processor.tokenizer.encode("0", add_special_tokens=False)
    token_one_ids = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(token_zero_ids) != 1 or len(token_one_ids) != 1:
        raise ValueError(f"digit tokens are not atomic: {token_zero_ids}, {token_one_ids}")
    token_zero, token_one = token_zero_ids[0], token_one_ids[0]
    print(json.dumps({"token_zero": token_zero, "token_one": token_one}), flush=True)

    loader = AutoModelForMultimodalLM if MODEL_CLASS == "multimodal" else AutoModelForImageTextToText
    model = loader.from_pretrained(
        MODEL, dtype=torch.bfloat16, local_files_only=True,
        trust_remote_code=True, attn_implementation="eager",
    ).to("cuda")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if os.environ.get("LINEAR_ONLY_TARGETS") == "1":
        suffixes = tuple(target_modules)
        target_modules = [
            name for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and name.endswith(suffixes)
        ]
        if not target_modules:
            raise ValueError("no supported linear attention projections found")
        print(json.dumps({
            "linear_lora_targets": len(target_modules),
            "target_examples": target_modules[:8],
        }), flush=True)
    model = get_peft_model(model, LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=True,
    ))
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.01)
    batches_per_epoch = math.ceil(len(train_records) / BATCH_SIZE)
    optimizer_steps = math.ceil(batches_per_epoch * EPOCHS / GRAD_ACCUM)
    warmup = max(1, int(optimizer_steps * 0.05))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, optimizer_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    optimizer.zero_grad(set_to_none=True)
    step, update, running_loss = 0, 0, 0.0
    started = time.monotonic()
    for epoch in range(EPOCHS):
        random.Random(SEED + epoch).shuffle(train_records)
        for start in range(0, len(train_records), BATCH_SIZE):
            positions = train_records[start:start + BATCH_SIZE]
            rows = [frame.iloc[index] for index in positions]
            batch = training_batch(processor, rows)
            batch = {key: value.to("cuda") for key, value in batch.items()}
            loss = model(**batch).loss / GRAD_ACCUM
            loss.backward()
            running_loss += float(loss.detach().cpu()) * GRAD_ACCUM
            step += 1
            if step % GRAD_ACCUM == 0 or start + BATCH_SIZE >= len(train_records):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if update % 25 == 0 or update == optimizer_steps:
                    print(json.dumps({
                        "epoch": epoch,
                        "update": update,
                        "updates": optimizer_steps,
                        "loss": running_loss / step,
                        "lr": scheduler.get_last_lr()[0],
                        "elapsed_min": (time.monotonic() - started) / 60,
                        "max_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    }), flush=True)

    if FULL_TRAIN:
        save_adapter(model)
        report = {
            "full_train": True,
            "training_mode": TRAINING_MODE,
            "train_records": len(train_records),
            "train_unique": len(set(train_records)),
            "download_failures": len(failures),
            "optimizer_updates": optimizer_steps,
            "flammable_selection": selection_audit,
            "runtime_minutes": (time.monotonic() - started) / 60,
        }
        (OUTPUT / "full_train_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return

    scores = validation_scores(
        model, processor, frame, val_positions, token_zero, token_one
    )
    labels_all = frame["label"].to_numpy(dtype=np.int8)
    categories_all = frame["category"].astype(str).to_numpy()
    report = {
        "holdout_fold": HOLDOUT_FOLD,
        "train_records": len(train_records),
        "download_failures": len(failures),
        "flammable_selection": selection_audit,
        "categories": {},
    }
    macro_lora, macro_fused = [], []
    for category in sorted(frame["category"].unique()):
        local_mask = categories_all[val_positions] == category
        local_positions = val_positions[local_mask]
        labels = labels_all[local_positions]
        local_scores = scores[local_mask]
        lora_f1, lora_threshold = best_threshold(labels, local_scores)
        base_rank = rank01(fused_oof_scores(oof)[local_positions])
        lora_rank = rank01(local_scores)
        best_fusion = None
        for base_weight in np.linspace(0.0, 1.0, 21):
            fused = base_weight * base_rank + (1 - base_weight) * lora_rank
            value, threshold = best_threshold(labels, fused)
            item = {
                "f1": value,
                "threshold": threshold,
                "weight_base": float(base_weight),
                "weight_lora": float(1 - base_weight),
            }
            if best_fusion is None or item["f1"] > best_fusion["f1"]:
                best_fusion = item
        report["categories"][category] = {
            "rows": int(local_mask.sum()),
            "positive": int(labels.sum()),
            "lora_f1": lora_f1,
            "lora_threshold": lora_threshold,
            "best_fusion": best_fusion,
        }
        macro_lora.append(lora_f1)
        macro_fused.append(best_fusion["f1"])
        print(category, json.dumps(report["categories"][category], ensure_ascii=False), flush=True)
    report["macro_lora"] = float(np.mean(macro_lora))
    report["macro_fused"] = float(np.mean(macro_fused))
    report["runtime_minutes"] = (time.monotonic() - started) / 60
    save_adapter(model)
    pd.DataFrame({
        "id": ids[val_positions],
        "category": categories_all[val_positions],
        "label": labels_all[val_positions],
        "fold": HOLDOUT_FOLD,
        "lora_score": scores,
    }).to_csv(OUTPUT / "lora_holdout_predictions.csv", index=False)
    (OUTPUT / "lora_holdout_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
