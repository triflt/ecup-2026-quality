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
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
DOWNSAMPLE_MODE = os.environ.get("DOWNSAMPLE_MODE", "").strip()
MAX_SLICE_NUMS = int(os.environ.get("MAX_SLICE_NUMS", "0"))
SOFT_TARGETS = Path(os.environ["SOFT_TARGETS"]) if os.environ.get("SOFT_TARGETS") else None
LAST_LOGIT_ONLY = os.environ.get("LAST_LOGIT_ONLY") == "1"
FIRST_IMAGE_MAX_EDGE = int(os.environ.get("QWEN3VL_FIRST_IMAGE_MAX_EDGE", "448"))
FIRST_IMAGE_MAX_PIXELS = int(os.environ.get("QWEN3VL_FIRST_IMAGE_MAX_PIXELS", "262144"))
CHECKPOINT_EACH_EPOCH = os.environ.get("CHECKPOINT_EACH_EPOCH") == "1"


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
    last_error = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            image = Image.open(io.BytesIO(payload)).convert("RGB")
            image.thumbnail(
                (FIRST_IMAGE_MAX_EDGE, FIRST_IMAGE_MAX_EDGE), Image.Resampling.LANCZOS
            )
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
        records.extend(rng.choice(bad_pos, size=bad_count, replace=False).tolist())
        records.extend(rng.choice(bad_neg, size=bad_count, replace=False).tolist())
    else:
        records.extend(hard_random(bad_pos, uncertainty, bad_count, rng))
        records.extend(hard_random(bad_neg, uncertainty, bad_count, rng))

    flam = np.flatnonzero(train_mask & (categories == "Легковоспламеняющиеся"))
    flam_pos = flam[labels[flam] == 1]
    flam_neg = flam[labels[flam] == 0]
    records.extend(np.repeat(flam_pos, 5).tolist())
    flam_neg_limit = 2000 if FULL_TRAIN else 1600
    if TRAINING_MODE == "clean":
        count = min(flam_neg_limit, len(flam_neg))
        records.extend(rng.choice(flam_neg, size=count, replace=False).tolist())
    else:
        records.extend(hard_random(flam_neg, uncertainty, min(flam_neg_limit, len(flam_neg)), rng))
    random.Random(SEED).shuffle(records)
    return records


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
    if DOWNSAMPLE_MODE:
        kwargs["downsample_mode"] = DOWNSAMPLE_MODE
    if MAX_SLICE_NUMS:
        kwargs["max_slice_nums"] = MAX_SLICE_NUMS
    try:
        return processor.apply_chat_template(
            conversations, enable_thinking=False, **kwargs
        )
    except TypeError:
        return processor.apply_chat_template(conversations, **kwargs)


def training_batch(processor, rows):
    if SOFT_TARGETS is not None:
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
                processor.apply_chat_template(
                    messages(row, False), tokenize=False, add_generation_prompt=True
                )
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
        targets = torch.tensor(
            [float(row.soft_target) for row in rows], dtype=torch.float32
        )
        return batch, targets
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
        forward_kwargs = {"downsample_mode": DOWNSAMPLE_MODE} if DOWNSAMPLE_MODE else {}
        if LAST_LOGIT_ONLY:
            forward_kwargs["logits_to_keep"] = 1
        outputs = model(**batch, **forward_kwargs)
        if LAST_LOGIT_ONLY:
            logits = outputs.logits[:, -1]
        else:
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
    if SOFT_TARGETS is not None:
        soft = pd.read_csv(SOFT_TARGETS, dtype={"id": str}).set_index("id")
        missing = sorted(set(ids) - set(soft.index))
        if missing:
            raise ValueError(f"missing soft targets for {len(missing)} ids")
        aligned = soft.loc[ids]
        targets = aligned["soft_target"].to_numpy(np.float32)
        if not np.isfinite(targets).all() or np.any((targets < 0) | (targets > 1)):
            raise ValueError("soft targets must be finite probabilities in [0, 1]")
        frame["soft_target"] = targets
    urls = load_urls()
    train_records = select_training(frame, oof)
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
        "soft_targets": SOFT_TARGETS is not None,
        "soft_target_mean": float(frame.loc[train_records, "soft_target"].mean())
        if SOFT_TARGETS is not None else None,
        "first_image_max_edge": FIRST_IMAGE_MAX_EDGE,
        "first_image_max_pixels": FIRST_IMAGE_MAX_PIXELS,
    }), flush=True)

    processor_kwargs = {"local_files_only": True, "trust_remote_code": True}
    if MODEL_CLASS == "image_text":
        processor_kwargs.update(
            min_pixels=4 * 28 * 28,
            max_pixels=FIRST_IMAGE_MAX_PIXELS,
        )
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

    def evaluate_epoch_checkpoint(epoch_number):
        epoch_output = OUTPUT / f"epoch_{epoch_number}"
        scores = validation_scores(
            model, processor, frame, val_positions, token_zero, token_one
        )
        labels_all = frame["label"].to_numpy(dtype=np.int8)
        categories_all = frame["category"].astype(str).to_numpy()
        report = {
            "holdout_fold": HOLDOUT_FOLD,
            "epoch": epoch_number,
            "train_records": len(train_records),
            "optimizer_updates_completed": update,
            "download_failures": len(failures),
            "soft_targets": SOFT_TARGETS is not None,
            "last_logit_only": LAST_LOGIT_ONLY,
            "first_image_max_edge": FIRST_IMAGE_MAX_EDGE,
            "first_image_max_pixels": FIRST_IMAGE_MAX_PIXELS,
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
        report["macro_lora"] = float(np.mean(macro_lora))
        report["macro_fused"] = float(np.mean(macro_fused))
        report["runtime_minutes"] = (time.monotonic() - started) / 60
        epoch_output.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(epoch_output / "adapter")
        pd.DataFrame({
            "id": ids[val_positions],
            "category": categories_all[val_positions],
            "label": labels_all[val_positions],
            "fold": HOLDOUT_FOLD,
            "lora_score": scores,
        }).to_csv(epoch_output / "lora_holdout_predictions.csv", index=False)
        (epoch_output / "lora_holdout_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"epoch_checkpoint": report}, ensure_ascii=False), flush=True)
        model.train()

    for epoch in range(EPOCHS):
        random.Random(SEED + epoch).shuffle(train_records)
        for start in range(0, len(train_records), BATCH_SIZE):
            positions = train_records[start:start + BATCH_SIZE]
            rows = [frame.iloc[index] for index in positions]
            training_data = training_batch(processor, rows)
            if SOFT_TARGETS is not None:
                batch, soft_targets = training_data
                soft_targets = soft_targets.to("cuda")
            else:
                batch = training_data
            batch = {key: value.to("cuda") for key, value in batch.items()}
            forward_kwargs = {"downsample_mode": DOWNSAMPLE_MODE} if DOWNSAMPLE_MODE else {}
            if SOFT_TARGETS is not None and LAST_LOGIT_ONLY:
                forward_kwargs["logits_to_keep"] = 1
            if SOFT_TARGETS is not None:
                outputs = model(**batch, **forward_kwargs)
                if LAST_LOGIT_ONLY:
                    logits = outputs.logits[:, -1]
                else:
                    sequence = torch.arange(
                        batch["attention_mask"].shape[1], device="cuda"
                    )[None, :]
                    last = torch.where(
                        batch["attention_mask"].bool(), sequence, -1
                    ).max(dim=1).values
                    logits = outputs.logits[torch.arange(len(rows), device="cuda"), last]
                binary_logits = logits[:, token_one].float() - logits[:, token_zero].float()
                loss = F.binary_cross_entropy_with_logits(
                    binary_logits, soft_targets
                ) / GRAD_ACCUM
            else:
                loss = model(**batch, **forward_kwargs).loss / GRAD_ACCUM
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
        if CHECKPOINT_EACH_EPOCH and not FULL_TRAIN:
            evaluate_epoch_checkpoint(epoch + 1)

    if CHECKPOINT_EACH_EPOCH and not FULL_TRAIN:
        return

    if FULL_TRAIN:
        save_adapter(model)
        report = {
            "full_train": True,
            "training_mode": TRAINING_MODE,
            "train_records": len(train_records),
            "train_unique": len(set(train_records)),
            "download_failures": len(failures),
            "optimizer_updates": optimizer_steps,
            "runtime_minutes": (time.monotonic() - started) / 60,
            "soft_targets": SOFT_TARGETS is not None,
            "last_logit_only": LAST_LOGIT_ONLY,
            "first_image_max_edge": FIRST_IMAGE_MAX_EDGE,
            "first_image_max_pixels": FIRST_IMAGE_MAX_PIXELS,
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
        "soft_targets": SOFT_TARGETS is not None,
        "last_logit_only": LAST_LOGIT_ONLY,
        "first_image_max_edge": FIRST_IMAGE_MAX_EDGE,
        "first_image_max_pixels": FIRST_IMAGE_MAX_PIXELS,
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
