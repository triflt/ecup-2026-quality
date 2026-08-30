from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import qwen3vl_lora_holdout as base
import torch
import torch.nn.functional as F

RDROP_ALPHA = float(os.environ.get("RDROP_ALPHA", "1.0"))
TRAINING_MANIFEST = Path(
    os.environ.get("RDROP_TRAINING_MANIFEST", "/work/input/rdrop_training_manifest.json")
)
SELECTOR_BUNDLE = Path(
    os.environ.get("RDROP_SELECTOR_BUNDLE", "/work/input/qwen_rdrop_selector.zip")
)
OBJECTIVE = "mean assistant-suffix CE plus alpha times symmetric binary class-token KL"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_record_sha256(ids: np.ndarray, records: list[int]) -> str:
    payload = ("\n".join(ids[records].astype(str).tolist()) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_frozen_recipe(
    *, frame: pd.DataFrame, oof: np.lib.npyio.NpzFile
) -> tuple[list[int], dict[str, object]]:
    if not TRAINING_MANIFEST.is_file():
        raise FileNotFoundError(f"missing frozen training manifest: {TRAINING_MANIFEST}")
    manifest = json.loads(TRAINING_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "440":
        raise ValueError("training manifest is not for experiment 440")
    if manifest.get("rdrop_alpha") != 1.0 or RDROP_ALPHA != 1.0:
        raise ValueError("experiment 440 freezes RDROP_ALPHA=1.0")
    if base.FULL_TRAIN:
        key = "full"
    else:
        key = f"fold_{base.HOLDOUT_FOLD}"
    expected = manifest.get("stages", {}).get(key)
    if expected is None:
        raise ValueError(f"stage {key} is not authorized by the frozen manifest")
    if not SELECTOR_BUNDLE.is_file():
        raise FileNotFoundError(f"missing frozen selector bundle: {SELECTOR_BUNDLE}")
    wanted_bundle_sha = manifest.get("selector_bundle", {}).get("sha256")
    actual_bundle_sha = sha256(SELECTOR_BUNDLE)
    if actual_bundle_sha != wanted_bundle_sha:
        raise ValueError(
            f"selector bundle checksum mismatch: {actual_bundle_sha} != {wanted_bundle_sha}"
        )
    with zipfile.ZipFile(SELECTOR_BUNDLE) as archive:
        names = archive.namelist()
        if names != ["selector_records.json"]:
            raise ValueError(f"unexpected selector bundle members: {names}")
        selector = json.loads(archive.read(names[0]))
    if selector.get("experiment_id") != "440":
        raise ValueError("selector bundle is not for experiment 440")
    frozen_stage = selector.get("stages", {}).get(key)
    if frozen_stage is None:
        raise ValueError(f"selector bundle lacks stage {key}")
    records = [int(value) for value in frozen_stage.get("record_indices", [])]
    if any(value < 0 or value >= len(frame) for value in records):
        raise ValueError(f"selector bundle contains an out-of-range row for {key}")
    ids = frame["id"].astype(str).to_numpy()
    observed = {
        "records": len(records),
        "unique_rows": len(set(records)),
        "ordered_id_sha256": ordered_record_sha256(ids, records),
    }
    for field, value in observed.items():
        if expected.get(field) != value:
            raise ValueError(
                f"frozen recipe mismatch for {key}.{field}: {value} != {expected.get(field)}"
            )
        if frozen_stage.get(field) != value:
            raise ValueError(
                f"selector bundle mismatch for {key}.{field}: "
                f"{value} != {frozen_stage.get(field)}"
            )
    source_paths = {
        "data": base.DATA,
        "oof": base.OOF,
        "image_manifest": base.MANIFEST,
    }
    for name, path in source_paths.items():
        wanted = manifest.get("sources", {}).get(name, {}).get("sha256")
        if wanted is None:
            raise ValueError(f"manifest lacks source checksum: {name}")
        actual = sha256(path)
        if actual != wanted:
            raise ValueError(f"source checksum mismatch for {name}: {actual} != {wanted}")
    return records, {
        "stage": key,
        **observed,
        "selector_bundle_sha256": actual_bundle_sha,
    }


def binary_class_logits(
    logits: torch.Tensor, labels: torch.Tensor, token_zero: int, token_one: int
) -> torch.Tensor:
    supervised = labels.ne(-100)
    if not bool(supervised.any(dim=1).all()):
        raise ValueError("each row must contain an assistant suffix")
    first_answer = supervised.to(torch.int64).argmax(dim=1)
    if int(first_answer.min()) < 1:
        raise ValueError("assistant suffix has no preceding prediction position")
    row = torch.arange(labels.shape[0], device=labels.device)
    prediction_position = first_answer - 1
    selected = logits[row, prediction_position]
    return torch.stack(
        [selected[:, token_zero], selected[:, token_one]], dim=-1
    ).float()


def symmetric_kl(first_logits: torch.Tensor, second_logits: torch.Tensor) -> torch.Tensor:
    first_logp = F.log_softmax(first_logits, dim=-1)
    second_logp = F.log_softmax(second_logits, dim=-1)
    first_p = first_logp.exp()
    second_p = second_logp.exp()
    first_to_second = (first_p * (first_logp - second_logp)).sum(dim=-1)
    second_to_first = (second_p * (second_logp - first_logp)).sum(dim=-1)
    return 0.5 * (first_to_second + second_to_first).mean()


def main() -> None:
    if base.SOFT_TARGETS is not None:
        raise ValueError("experiment 440 preserves experiment-110 hard targets")
    if RDROP_ALPHA != 1.0:
        raise ValueError("experiment 440 freezes RDROP_ALPHA=1.0")
    torch.manual_seed(base.SEED)
    np.random.seed(base.SEED)
    frame = base.load_training_frame()
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["category"] = frame["category"].astype(str)
    oof = np.load(base.OOF, allow_pickle=True)
    ids = frame["id"].astype(str).to_numpy()
    if not np.array_equal(ids, oof["ids"].astype(str)):
        raise ValueError("OOF id mismatch")
    train_records, frozen = verify_frozen_recipe(frame=frame, oof=oof)
    base.install_peft()
    from peft import LoraConfig, TaskType, get_peft_model

    val_positions = (
        np.asarray([], dtype=np.int64)
        if base.FULL_TRAIN
        else np.flatnonzero(oof["fold_ids"].astype(np.int8) == base.HOLDOUT_FOLD)
    )
    urls = base.load_urls()
    needed = sorted(set(ids[train_records]) | set(ids[val_positions]))
    failures = base.predownload(needed, urls)
    print(
        json.dumps(
            {
                "objective": OBJECTIVE,
                "rdrop_alpha": RDROP_ALPHA,
                "frozen_recipe": frozen,
                "training_mode": base.TRAINING_MODE,
                "train_records": len(train_records),
                "train_unique": len(set(train_records)),
                "validation": len(val_positions),
                "images": len(needed),
                "download_failures": len(failures),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    processor_kwargs = {"local_files_only": True, "trust_remote_code": True}
    if base.MODEL_CLASS == "image_text":
        processor_kwargs.update(min_pixels=4 * 28 * 28, max_pixels=262144)
    processor = base.AutoProcessor.from_pretrained(base.MODEL, **processor_kwargs)
    processor.tokenizer.padding_side = "left"
    token_zero_ids = processor.tokenizer.encode("0", add_special_tokens=False)
    token_one_ids = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(token_zero_ids) != 1 or len(token_one_ids) != 1:
        raise ValueError(f"digit tokens are not atomic: {token_zero_ids}, {token_one_ids}")
    token_zero, token_one = token_zero_ids[0], token_one_ids[0]

    loader = (
        base.AutoModelForMultimodalLM
        if base.MODEL_CLASS == "multimodal"
        else base.AutoModelForImageTextToText
    )
    model = loader.from_pretrained(
        base.MODEL,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if os.environ.get("LINEAR_ONLY_TARGETS") == "1":
        suffixes = tuple(target_modules)
        target_modules = [
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and name.endswith(suffixes)
        ]
        if not target_modules:
            raise ValueError("no supported linear attention projections found")
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_rslora=True,
        ),
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.01)
    batches_per_epoch = math.ceil(len(train_records) / base.BATCH_SIZE)
    optimizer_steps = math.ceil(batches_per_epoch * base.EPOCHS / base.GRAD_ACCUM)
    warmup = max(1, int(optimizer_steps * 0.05))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, optimizer_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    optimizer.zero_grad(set_to_none=True)
    step = update = 0
    running_ce = running_kl = running_total = 0.0
    started = time.monotonic()
    for epoch in range(base.EPOCHS):
        random.Random(base.SEED + epoch).shuffle(train_records)
        for start in range(0, len(train_records), base.BATCH_SIZE):
            positions = train_records[start : start + base.BATCH_SIZE]
            rows = [frame.iloc[index] for index in positions]
            batch = base.training_batch(processor, rows)
            batch = {key: value.to("cuda") for key, value in batch.items()}
            forward_kwargs = (
                {"downsample_mode": base.DOWNSAMPLE_MODE} if base.DOWNSAMPLE_MODE else {}
            )
            first = model(**batch, **forward_kwargs)
            second = model(**batch, **forward_kwargs)
            ce = 0.5 * (first.loss.float() + second.loss.float())
            first_binary = binary_class_logits(
                first.logits, batch["labels"], token_zero, token_one
            )
            second_binary = binary_class_logits(
                second.logits, batch["labels"], token_zero, token_one
            )
            kl = symmetric_kl(first_binary, second_binary)
            total = ce + RDROP_ALPHA * kl
            loss = total / base.GRAD_ACCUM
            loss.backward()
            running_ce += float(ce.detach().cpu())
            running_kl += float(kl.detach().cpu())
            running_total += float(total.detach().cpu())
            step += 1
            if step % base.GRAD_ACCUM == 0 or start + base.BATCH_SIZE >= len(train_records):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                if update % 25 == 0 or update == optimizer_steps:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "update": update,
                                "updates": optimizer_steps,
                                "mean_ce": running_ce / step,
                                "mean_symmetric_kl": running_kl / step,
                                "mean_total": running_total / step,
                                "lr": scheduler.get_last_lr()[0],
                                "elapsed_min": (time.monotonic() - started) / 60,
                                "max_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3,
                            }
                        ),
                        flush=True,
                    )

    common_report = {
        "objective": OBJECTIVE,
        "rdrop_alpha": RDROP_ALPHA,
        "training_mode": base.TRAINING_MODE,
        "frozen_recipe": frozen,
        "train_records": len(train_records),
        "train_unique": len(set(train_records)),
        "download_failures": len(failures),
        "optimizer_updates": optimizer_steps,
        "mean_ce": running_ce / max(1, step),
        "mean_symmetric_kl": running_kl / max(1, step),
        "mean_total": running_total / max(1, step),
    }
    if base.FULL_TRAIN:
        base.save_adapter(model)
        report = {
            **common_report,
            "full_train": True,
            "runtime_minutes": (time.monotonic() - started) / 60,
        }
        (base.OUTPUT / "full_train_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return

    scores = base.validation_scores(
        model, processor, frame, val_positions, token_zero, token_one
    )
    labels_all = frame["label"].to_numpy(dtype=np.int8)
    categories_all = frame["category"].astype(str).to_numpy()
    report: dict[str, object] = {
        **common_report,
        "holdout_fold": base.HOLDOUT_FOLD,
        "categories": {},
    }
    macro_lora: list[float] = []
    macro_fused: list[float] = []
    for category in sorted(frame["category"].unique()):
        local_mask = categories_all[val_positions] == category
        local_positions = val_positions[local_mask]
        labels = labels_all[local_positions]
        local_scores = scores[local_mask]
        lora_f1, lora_threshold = base.best_threshold(labels, local_scores)
        base_rank = base.rank01(base.fused_oof_scores(oof)[local_positions])
        lora_rank = base.rank01(local_scores)
        best_fusion = None
        for base_weight in np.linspace(0.0, 1.0, 21):
            fused = base_weight * base_rank + (1 - base_weight) * lora_rank
            value, threshold = base.best_threshold(labels, fused)
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
    base.save_adapter(model)
    pd.DataFrame(
        {
            "id": ids[val_positions],
            "category": categories_all[val_positions],
            "label": labels_all[val_positions],
            "fold": base.HOLDOUT_FOLD,
            "lora_score": scores,
        }
    ).to_csv(base.OUTPUT / "lora_holdout_predictions.csv", index=False)
    (base.OUTPUT / "lora_holdout_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
