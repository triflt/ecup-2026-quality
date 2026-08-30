from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from alignment import align_parent_prompt, predicted_token_span_to_canonical_offsets
from protocol import (
    CONCEPTS,
    FROZEN_INPUT_SHA256,
    read_jsonl,
    sha256_file,
    validate_label_free_predictions,
)
from renderer import render_explanation

PARENT_PATH = ROOT / "research/qwen3vl_lora_holdout.py"
PARENT_SHA256 = "c30e690ad260af72fcc625c8d3e6d9ab9c5a096d8443d6d9f5f7adbcaa52123c"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPERIMENT_ID = "623"
SEED = 42
BATCH_SIZE = 4
GRAD_ACCUM = 4
EPOCHS = 1
LEARNING_RATE = 2e-4


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_parent(*, runtime_dir: Path, images: Path, model_root: Path, vendor: Path, fold: int):
    if sha256_file(PARENT_PATH) != PARENT_SHA256:
        raise ValueError("exact experiment-600 parent checksum mismatch")
    locked = {
        "ECUP_MODEL_ROOT": str(model_root),
        "ECUP_MANIFEST": str(runtime_dir / "development_image_manifest.tsv.gz"),
        "ECUP_OOF": str(runtime_dir / "development_selector_oof.npz"),
        "ECUP_IMAGES": str(images),
        "ECUP_VENDOR": str(vendor),
        "SEED": str(SEED),
        "HOLDOUT_FOLD": str(fold),
        "FULL_TRAIN": "0",
        "TRAINING_MODE": "hard",
        "MODEL_CLASS": "multimodal",
        "USE_CHAT_BATCH": "1",
        "DESCRIPTION_LIMIT": "1800",
    }
    forbidden = ("SOFT_TARGETS", "DOWNSAMPLE_MODE", "MAX_SLICE_NUMS", "LAST_LOGIT_ONLY", "LINEAR_ONLY_TARGETS")
    for key in forbidden:
        if os.environ.get(key) not in (None, "", "0", "false"):
            raise ValueError(f"non-parent switch is forbidden: {key}")
    os.environ.update(locked)
    spec = importlib.util.spec_from_file_location("_exp623_exact_parent", PARENT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load exact parent")
    parent = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = parent
    spec.loader.exec_module(parent)
    expected = {
        "SEED": SEED,
        "HOLDOUT_FOLD": fold,
        "FULL_TRAIN": False,
        "TRAINING_MODE": "hard",
        "MODEL_CLASS": "multimodal",
        "USE_CHAT_BATCH": True,
        "DESCRIPTION_LIMIT": 1800,
        "BATCH_SIZE": BATCH_SIZE,
        "GRAD_ACCUM": GRAD_ACCUM,
        "EPOCHS": EPOCHS,
    }
    mismatch = {key: [value, getattr(parent, key, None)] for key, value in expected.items() if getattr(parent, key, None) != value}
    if mismatch:
        raise ValueError(f"exact parent recipe mismatch: {mismatch}")
    return parent


def _read_runtime(runtime_dir: Path, fold: int) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], list[str], dict[str, Any]]:
    required = [
        runtime_dir / "train.jsonl",
        runtime_dir / "validation.jsonl",
        runtime_dir / "development_selector_oof.npz",
        runtime_dir / "development_image_manifest.tsv.gz",
        runtime_dir / "runtime_audit.json",
    ]
    missing = [path.name for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError(f"runtime contract incomplete: {missing}")
    audit = json.loads((runtime_dir / "runtime_audit.json").read_text(encoding="utf-8"))
    if audit.get("experiment_id") != "623" or audit.get("outer_fold") != fold or audit.get("decision") != "GO":
        raise ValueError("runtime audit identity/decision mismatch")
    if audit.get("frozen_input_hashes_enforced") is not True:
        raise ValueError("runtime was not built with frozen input enforcement")
    if audit.get("input_sha256") != FROZEN_INPUT_SHA256:
        raise ValueError("runtime audit input hashes differ from the frozen parent contract")
    expected_hashes = audit.get("output_sha256", {})
    for path in required[:-1]:
        if expected_hashes.get(path.name) != sha256_file(path):
            raise ValueError(f"runtime checksum mismatch: {path.name}")
    train = read_jsonl(runtime_dir / "train.jsonl")
    validation = read_jsonl(runtime_dir / "validation.jsonl")
    if any("label" in row or "rationale" in row for row in validation):
        raise ValueError("validation runtime contains supervision")
    all_rows = sorted(train + validation, key=lambda row: int(row["row_index"]))
    if [int(row["row_index"]) for row in all_rows] != list(range(len(all_rows))):
        raise ValueError("runtime row indices are not a complete ordered development scope")
    frame = pd.DataFrame([{key: row[key] for key in ("id", "category", "name", "description")} for row in all_rows])
    train_by_id = {str(row["id"]): row for row in train}
    validation_ids = [str(row["id"]) for row in validation]
    if set(train_by_id) & set(validation_ids):
        raise ValueError("training and validation IDs overlap")
    return frame, train_by_id, validation_ids, audit


def _batch(parent: Any, processor: Any, rows: list[SimpleNamespace], *, with_rationale: bool):
    import torch
    from PIL import Image

    images = [Image.open(parent.IMAGE_DIR / f"{row.id}.jpg").convert("RGB") for row in rows]
    try:
        batch = parent.chat_batch(
            processor,
            [parent.messages(row, False, image) for row, image in zip(rows, images, strict=True)],
            True,
        )
    finally:
        for image in images:
            image.close()
    alignments = []
    for row_index, row in enumerate(rows):
        active = torch.nonzero(batch["attention_mask"][row_index]).flatten().tolist()
        active_ids = batch["input_ids"][row_index, active].tolist()
        local = align_parent_prompt(
            tokenizer=processor.tokenizer,
            full_input_ids=active_ids,
            parent_user_text=parent.user_text(row),
            rationale=row.rationale if with_rationale else None,
        )
        # Convert unpadded active positions back into the padded tensor positions.
        mapping = {local_index: int(full_index) for local_index, full_index in enumerate(active)}
        alignments.append(
            type(local)(
                [bool(index in {mapping[i] for i, flag in enumerate(local.text_token_mask) if flag}) for index in range(batch["input_ids"].shape[1])],
                mapping.get(local.start_target, batch["input_ids"].shape[1]),
                mapping.get(local.end_target, batch["input_ids"].shape[1]),
                local.concept_target,
                local.quality_weight,
                {mapping[index]: offset for index, offset in local.full_to_prompt_offset.items()},
            )
        )
    return batch, alignments


def _to_rows(items: list[dict[str, Any]]) -> list[SimpleNamespace]:
    return [SimpleNamespace(**item) for item in items]


def _predict(model, parent, processor, rows: list[dict[str, Any]], fold: int) -> list[dict[str, Any]]:
    import torch

    model.eval()
    output: list[dict[str, Any]] = []
    for start in range(0, len(rows), 8):
        records = _to_rows(rows[start : start + 8])
        batch, alignments = _batch(parent, processor, records, with_rationale=False)
        device_batch = {key: value.to("cuda") for key, value in batch.items()}
        text_mask = torch.tensor([item.text_token_mask for item in alignments], dtype=torch.bool, device="cuda")
        with torch.inference_mode():
            values = model(text_token_mask=text_mask, **device_batch)
        raw_scores = values["verdict_logits"].float().cpu()
        probabilities = raw_scores.sigmoid()
        starts = values["start_logits"].float().argmax(dim=1).cpu().tolist()
        ends = values["end_logits"].float().argmax(dim=1).cpu().tolist()
        concepts = values["concept_logits"].float().argmax(dim=1).cpu().tolist()
        for index, row in enumerate(records):
            char_start, char_end = predicted_token_span_to_canonical_offsets(
                start_token=starts[index],
                end_token=ends[index],
                alignment=alignments[index],
                parent_user_text=parent.user_text(row),
                name=row.name,
                description=row.description,
            )
            rendered = render_explanation(
                name=row.name,
                description=row.description,
                char_start=char_start,
                char_end=char_end,
                concept=CONCEPTS[concepts[index]] if char_start is not None else None,
            )
            output.append({
                "id": str(row.id),
                "category": str(row.category),
                "fold": fold,
                "lora_score": float(raw_scores[index]),
                "verdict_probability": float(probabilities[index]),
                "evidence": rendered["evidence"],
                "concept": rendered["concept"] or "NO_EVIDENCE",
                "explanation": rendered["explanation"],
                "char_start": -1 if char_start is None else char_start,
                "char_end": -1 if char_end is None else char_end,
            })
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from model import Qwen35VerdictSpanModel
    from multitask_head import multitask_loss
    from safetensors.torch import save_file
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if args.model_revision != MODEL_REVISION:
        raise ValueError("exact experiment-600 model revision mismatch")
    output_dir = args.output_dir.resolve()
    images = args.images.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty output directory")
    if images.exists() and any(images.iterdir()):
        raise FileExistsError("refusing to reuse a nonempty image directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    runtime_dir = args.runtime_dir.resolve()
    frame, train_by_id, validation_ids, _runtime_audit = _read_runtime(runtime_dir, args.fold)
    parent = _load_parent(runtime_dir=runtime_dir, images=images, model_root=args.model_root.resolve(), vendor=args.vendor.resolve(), fold=args.fold)
    selector = np.load(runtime_dir / "development_selector_oof.npz", allow_pickle=False)
    selection_frame = frame.copy()
    selection_frame["label"] = [int(train_by_id[item_id]["label"]) if item_id in train_by_id else 0 for item_id in selection_frame["id"].astype(str)]
    selected = [int(value) for value in parent.select_training(selection_frame, selector)]
    selected_ids = selection_frame.iloc[selected]["id"].astype(str).tolist()
    if any(item_id in set(validation_ids) for item_id in selected_ids):
        raise ValueError("exact parent selector admitted outer validation")
    if any(item_id not in train_by_id for item_id in selected_ids):
        raise ValueError("exact parent selector admitted a row without donor supervision")
    needed = sorted(set(selected_ids) | set(validation_ids))
    failures = parent.predownload(needed, parent.load_urls())
    if failures:
        raise RuntimeError(f"strict image download failed for {len(failures)} rows")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    vendor = args.vendor.resolve()
    if not vendor.is_dir():
        raise FileNotFoundError("vendored PEFT directory is missing")
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    import peft

    if getattr(peft, "__version__", None) != "0.20.0":
        raise ValueError("exact vendored PEFT 0.20.0 is required")
    from peft import LoraConfig, TaskType, get_peft_model

    processor = AutoProcessor.from_pretrained(args.model_root.resolve(), local_files_only=True, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1:
        raise ValueError("parent digit tokens are not atomic")
    backbone = AutoModelForMultimodalLM.from_pretrained(
        args.model_root.resolve(), dtype=torch.bfloat16, local_files_only=True,
        trust_remote_code=True, attn_implementation="eager",
    ).to("cuda")
    backbone = get_peft_model(backbone, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM, use_rslora=True,
    ))
    backbone.config.use_cache = False
    backbone.enable_input_require_grads()
    backbone.gradient_checkpointing_enable()
    hidden_size = int(backbone.config.text_config.hidden_size)
    model = Qwen35VerdictSpanModel(
        backbone, hidden_size=hidden_size, concept_count=len(CONCEPTS),
        token_zero=zero[0], token_one=one[0],
    ).to("cuda")
    model.auxiliary_head.to(dtype=torch.bfloat16)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
    batches = math.ceil(len(selected_ids) / BATCH_SIZE)
    updates = math.ceil(batches * EPOCHS / GRAD_ACCUM)
    warmup = max(1, int(updates * 0.05))
    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, updates - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    optimizer.zero_grad(set_to_none=True)
    step = update = 0
    masked_alignment_rows = 0
    started = time.monotonic()
    records = list(selected_ids)
    for epoch in range(EPOCHS):
        random.Random(SEED + epoch).shuffle(records)
        for offset in range(0, len(records), BATCH_SIZE):
            rows = _to_rows([train_by_id[item_id] for item_id in records[offset : offset + BATCH_SIZE]])
            batch, alignments = _batch(parent, processor, rows, with_rationale=True)
            masked_alignment_rows += sum(item.quality_weight <= 0 and row.rationale.get("has_evidence") for item, row in zip(alignments, rows, strict=True))
            device_batch = {key: value.to("cuda") for key, value in batch.items()}
            text_mask = torch.tensor([item.text_token_mask for item in alignments], dtype=torch.bool, device="cuda")
            result = model(text_token_mask=text_mask, **device_batch)
            loss, _ = multitask_loss(
                verdict_logits=result["verdict_logits"], auxiliary=result,
                verdict_targets=torch.tensor([row.label for row in rows], dtype=torch.float32, device="cuda"),
                start_targets=torch.tensor([item.start_target for item in alignments], device="cuda"),
                end_targets=torch.tensor([item.end_target for item in alignments], device="cuda"),
                concept_targets=torch.tensor([item.concept_target for item in alignments], device="cuda"),
                quality_weights=torch.tensor([item.quality_weight for item in alignments], device="cuda"),
            )
            (loss / GRAD_ACCUM).backward()
            step += 1
            if step % GRAD_ACCUM == 0 or offset + BATCH_SIZE >= len(records):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                update += 1

    validation_rows = [
        {**frame.loc[frame["id"].astype(str).eq(item_id)].iloc[0].to_dict(), "row_index": int(frame.index[frame["id"].astype(str).eq(item_id)][0])}
        for item_id in validation_ids
    ]
    predictions = _predict(model, parent, processor, validation_rows, args.fold)
    prediction_path = output_dir / "validation_predictions.csv"
    pd.DataFrame(predictions).to_csv(prediction_path, index=False)
    prediction_audit = validate_label_free_predictions(
        prediction_path=prediction_path,
        validation_path=runtime_dir / "validation.jsonl",
        outer_fold=args.fold,
    )
    adapter_dir = output_dir / "adapter"
    model.backbone.save_pretrained(adapter_dir)
    shutil.make_archive(str(output_dir / "adapter"), "zip", adapter_dir)
    aux_path = output_dir / "auxiliary_head.safetensors"
    save_file({key: value.detach().float().cpu().contiguous() for key, value in model.auxiliary_head.state_dict().items()}, str(aux_path))
    aux_config_path = output_dir / "auxiliary_head_config.json"
    aux_config_path.write_text(
        json.dumps(
            {
                "schema_version": "exp623_auxiliary_head_v1",
                "hidden_size": hidden_size,
                "concepts": list(CONCEPTS),
                "no_evidence_boundary_class": "sequence_length",
                "dtype_on_disk": "float32",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    selection_audit = {
        "parent_sha256": PARENT_SHA256,
        "model_revision": MODEL_REVISION,
        "prediction_audit": prediction_audit,
        "outer_fold": args.fold,
        "training_records": len(selected_ids),
        "training_unique_rows": len(set(selected_ids)),
        "selected_id_multiset_sha256": canonical_sha256(sorted(Counter(selected_ids).items())),
        "outer_validation_occurrences": 0,
        "runtime_audit_sha256": sha256_file(runtime_dir / "runtime_audit.json"),
        "decision": "GO",
    }
    selection_path = output_dir / "selection_audit.json"
    selection_path.write_text(json.dumps(selection_audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "experiment_id": EXPERIMENT_ID, "outer_fold": args.fold, "seed": SEED,
        "training_records": len(selected_ids), "training_unique_rows": len(set(selected_ids)),
        "validation_rows": len(predictions), "download_failures": 0,
        "optimizer_updates": updates, "alignment_masked_safe_candidates": masked_alignment_rows,
        "runtime_minutes": (time.monotonic() - started) / 60,
        "parent_sha256": PARENT_SHA256,
        "artifacts": {
            "adapter.zip": sha256_file(output_dir / "adapter.zip"),
            "auxiliary_head.safetensors": sha256_file(aux_path),
            "auxiliary_head_config.json": sha256_file(aux_config_path),
            "validation_predictions.csv": sha256_file(prediction_path),
            "selection_audit.json": sha256_file(selection_path),
        },
        "validation_labels_written": 0, "sealed_rows_used": 0, "decision": "GO",
    }
    report["contract_sha256"] = canonical_sha256(report)
    (output_dir / "output_contract.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one label-isolated experiment-623 fold.")
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--vendor", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
