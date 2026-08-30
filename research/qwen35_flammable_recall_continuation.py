from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoProcessor

import qwen35_bad_family_diverse_positives_lora as base


STAGE = os.environ["STAGE"]
TRAINING_MANIFEST = Path(os.environ.get("RECALL_TRAINING_MANIFEST", "/work/input/training_manifest.json"))
PARENT_ADAPTER = Path(os.environ.get("PARENT_ADAPTER", "/work/parent_adapter"))
LEARNING_RATE = float(os.environ.get("CONTINUATION_LR", "2e-5"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(frame: pd.DataFrame) -> tuple[list[int], dict[str, object]]:
    payload = json.loads(TRAINING_MANIFEST.read_text(encoding="utf-8"))
    if payload["experiment_id"] != "420" or not payload["immutable"]:
        raise ValueError("unexpected or mutable continuation manifest")
    stage = payload["stages"][STAGE]
    ids = frame["id"].astype(str).to_numpy()
    position_by_id = {item_id: index for index, item_id in enumerate(ids)}
    if len(position_by_id) != len(ids):
        raise ValueError("dataset ids are not unique")
    ordered_ids = [str(item_id) for item_id in stage["ordered_record_ids"]]
    try:
        records = [position_by_id[item_id] for item_id in ordered_ids]
    except KeyError as error:
        raise ValueError(f"manifest id absent from data: {error}") from error
    digest = hashlib.sha256("\n".join(ordered_ids).encode("utf-8")).hexdigest()
    if digest != stage["ordered_records_sha256"]:
        raise ValueError("ordered training record checksum mismatch")
    local = frame.iloc[records]
    if set(local["category"].astype(str)) != {"Легковоспламеняющиеся"}:
        raise ValueError("continuation manifest contains a non-flammable row")
    counts = local["label"].astype(int).value_counts().to_dict()
    if counts.get(0, 0) != counts.get(1, 0):
        raise ValueError(f"continuation records are not balanced: {counts}")
    return records, stage


def save_adapter(model) -> None:
    base.OUTPUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(base.OUTPUT / "adapter")
    shutil.make_archive(str(base.OUTPUT / "adapter"), "zip", base.OUTPUT / "adapter")


def main() -> None:
    torch.manual_seed(base.SEED)
    np.random.seed(base.SEED)
    base.install_peft()
    from peft import PeftModel

    frame = base.load_training_frame()
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["category"] = frame["category"].astype(str)
    oof = np.load(base.OOF, allow_pickle=True)
    ids = frame["id"].astype(str).to_numpy()
    if not np.array_equal(ids, oof["ids"].astype(str)):
        raise ValueError("OOF id mismatch")
    train_records, selection_audit = load_records(frame)

    if STAGE == "full":
        if not base.FULL_TRAIN:
            raise ValueError("full stage requires FULL_TRAIN=1")
        val_positions = np.asarray([], dtype=np.int64)
    else:
        expected = f"fold_{base.HOLDOUT_FOLD}"
        if base.FULL_TRAIN or STAGE != expected:
            raise ValueError(f"stage/fold mismatch: stage={STAGE} expected={expected}")
        fold_ids = oof["fold_ids"].astype(np.int8)
        categories = frame["category"].astype(str).to_numpy()
        val_positions = np.flatnonzero(
            (fold_ids == base.HOLDOUT_FOLD) & (categories == "Легковоспламеняющиеся")
        )

    urls = base.load_urls()
    needed = sorted(set(ids[train_records]) | set(ids[val_positions]))
    failures = base.predownload(needed, urls)
    print(json.dumps({
        "experiment": 420,
        "stage": STAGE,
        "parent": "260 matching fold/full adapter",
        "train_records": len(train_records),
        "train_unique": len(set(train_records)),
        "validation_flammable": len(val_positions),
        "images": len(needed),
        "download_failures": len(failures),
        "selection": selection_audit,
    }, ensure_ascii=False), flush=True)

    processor_kwargs = {"local_files_only": True, "trust_remote_code": True}
    if base.MODEL_CLASS == "image_text":
        processor_kwargs.update(min_pixels=4 * 28 * 28, max_pixels=262144)
    processor = AutoProcessor.from_pretrained(base.MODEL, **processor_kwargs)
    processor.tokenizer.padding_side = "left"
    token_zero_ids = processor.tokenizer.encode("0", add_special_tokens=False)
    token_one_ids = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(token_zero_ids) != 1 or len(token_one_ids) != 1:
        raise ValueError(f"digit tokens are not atomic: {token_zero_ids}, {token_one_ids}")
    token_zero, token_one = token_zero_ids[0], token_one_ids[0]

    loader = AutoModelForMultimodalLM if base.MODEL_CLASS == "multimodal" else AutoModelForImageTextToText
    model = loader.from_pretrained(
        base.MODEL,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
    if not (PARENT_ADAPTER / "adapter_config.json").is_file():
        raise FileNotFoundError(f"parent adapter is incomplete: {PARENT_ADAPTER}")
    model = PeftModel.from_pretrained(model, PARENT_ADAPTER, is_trainable=True)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("continued adapter has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
    batches_per_epoch = math.ceil(len(train_records) / base.BATCH_SIZE)
    optimizer_steps = math.ceil(batches_per_epoch / base.GRAD_ACCUM)
    warmup = max(1, int(optimizer_steps * 0.05))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, optimizer_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    optimizer.zero_grad(set_to_none=True)
    step = update = 0
    running_loss = 0.0
    started = time.monotonic()
    # The immutable manifest already stores the stable, pre-prediction order.
    records = list(train_records)
    for start in range(0, len(records), base.BATCH_SIZE):
        positions = records[start:start + base.BATCH_SIZE]
        rows = [frame.iloc[index] for index in positions]
        batch = base.training_batch(processor, rows)
        batch = {key: value.to("cuda") for key, value in batch.items()}
        loss = model(**batch).loss / base.GRAD_ACCUM
        loss.backward()
        running_loss += float(loss.detach().cpu()) * base.GRAD_ACCUM
        step += 1
        if step % base.GRAD_ACCUM == 0 or start + base.BATCH_SIZE >= len(records):
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            if update % 25 == 0 or update == optimizer_steps:
                print(json.dumps({
                    "stage": STAGE,
                    "update": update,
                    "updates": optimizer_steps,
                    "loss": running_loss / step,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_min": (time.monotonic() - started) / 60,
                    "max_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3,
                }), flush=True)

    save_adapter(model)
    common_report = {
        "experiment_id": "420",
        "stage": STAGE,
        "parent_experiment": "260",
        "continuation_lr": LEARNING_RATE,
        "train_records": len(records),
        "train_unique": len(set(records)),
        "optimizer_updates": optimizer_steps,
        "download_failures": len(failures),
        "selection": selection_audit,
        "training_manifest_sha256": sha256_file(TRAINING_MANIFEST),
        "parent_adapter_config_sha256": sha256_file(PARENT_ADAPTER / "adapter_config.json"),
        "training_runtime_minutes": (time.monotonic() - started) / 60,
    }
    if base.FULL_TRAIN:
        common_report["total_runtime_minutes"] = (time.monotonic() - started) / 60
        (base.OUTPUT / "full_train_report.json").write_text(
            json.dumps(common_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(common_report, ensure_ascii=False, indent=2), flush=True)
        return

    scores = base.validation_scores(model, processor, frame, val_positions, token_zero, token_one)
    labels = frame["label"].to_numpy(dtype=np.int8)[val_positions]
    diagnostic_f1, diagnostic_threshold = base.best_threshold(labels, scores)
    report = {
        **common_report,
        "holdout_fold": base.HOLDOUT_FOLD,
        "validation_rows": len(val_positions),
        "positive": int(labels.sum()),
        "standalone_flammable_f1": diagnostic_f1,
        "standalone_flammable_threshold": diagnostic_threshold,
        "diagnostic_only": True,
        "total_runtime_minutes": (time.monotonic() - started) / 60,
    }
    pd.DataFrame({
        "id": ids[val_positions],
        "category": frame["category"].astype(str).to_numpy()[val_positions],
        "label": labels,
        "fold": base.HOLDOUT_FOLD,
        "lora_score": scores,
    }).to_csv(base.OUTPUT / "lora_holdout_predictions.csv", index=False)
    (base.OUTPUT / "lora_holdout_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
