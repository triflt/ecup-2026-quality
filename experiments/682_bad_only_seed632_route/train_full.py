"""Exact development-only FULL_TRAIN=1 seed-632 multitask refit worker."""

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

PARENT_SHA256 = "c30e690ad260af72fcc625c8d3e6d9ab9c5a096d8443d6d9f5f7adbcaa52123c"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SEED = 31415
BATCH_SIZE = 4
GRAD_ACCUM = 4
EPOCHS = 1
LEARNING_RATE = 2e-4
UPSTREAM_SHA256 = {
    "run_fold.py": "24410ad5267d3c2586280a48241c20dc5e982dc9df15657b1ce8ed33091c6ebe",
    "model.py": "20d2d1a95e9d711f33eb356e67ba2037525c1e8c842185e03c39c7adcece2300",
    "multitask_head.py": "75014ed23d1aaa67823329ce14e7faf5a0a3d8ff6e8130759a1470edf1ca16b1",
    "alignment.py": "ae092f047cad23fa42b820d6f3fb209f464bfebf1a9e22fd2c8fa44b5d39c926",
    "protocol.py": "abe01ed810399369e82651aacaac779a858b8a7e31c60106c59751fcb36edef0",
    "renderer.py": "072138d1c4acf9ce411a44f8d49aaf0457c130f1015b21a213a6f072f2ac2a63",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_upstream(upstream_dir: Path):
    for name, expected in UPSTREAM_SHA256.items():
        path = upstream_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen experiment-623 source mismatch: {name}")
    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))
    path = upstream_dir / "run_fold.py"
    spec = importlib.util.spec_from_file_location("_exp682_exact_623", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_parent_full(
    *, parent_path: Path, runtime_dir: Path, images: Path, model_root: Path, vendor: Path
):
    if sha256_file(parent_path) != PARENT_SHA256:
        raise ValueError("exact experiment-600 parent checksum mismatch")
    locked = {
        "ECUP_MODEL_ROOT": str(model_root),
        "ECUP_MANIFEST": str(runtime_dir / "development_image_manifest.tsv.gz"),
        "ECUP_OOF": str(runtime_dir / "development_selector_oof.npz"),
        "ECUP_IMAGES": str(images),
        "ECUP_VENDOR": str(vendor),
        "SEED": str(SEED),
        "HOLDOUT_FOLD": "-1",
        "FULL_TRAIN": "1",
        "TRAINING_MODE": "hard",
        "MODEL_CLASS": "multimodal",
        "USE_CHAT_BATCH": "1",
        "DESCRIPTION_LIMIT": "1800",
    }
    forbidden = (
        "SOFT_TARGETS",
        "DOWNSAMPLE_MODE",
        "MAX_SLICE_NUMS",
        "LAST_LOGIT_ONLY",
        "LINEAR_ONLY_TARGETS",
    )
    for key in forbidden:
        if os.environ.get(key) not in (None, "", "0", "false"):
            raise ValueError(f"non-parent switch is forbidden: {key}")
    os.environ.update(locked)
    spec = importlib.util.spec_from_file_location("_exp682_exact_full_parent", parent_path)
    if spec is None or spec.loader is None:
        raise ImportError(parent_path)
    parent = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = parent
    spec.loader.exec_module(parent)
    expected = {
        "SEED": SEED,
        "HOLDOUT_FOLD": -1,
        "FULL_TRAIN": True,
        "TRAINING_MODE": "hard",
        "MODEL_CLASS": "multimodal",
        "USE_CHAT_BATCH": True,
        "DESCRIPTION_LIMIT": 1800,
        "BATCH_SIZE": BATCH_SIZE,
        "GRAD_ACCUM": GRAD_ACCUM,
        "EPOCHS": EPOCHS,
    }
    mismatch = {
        key: {"expected": value, "actual": getattr(parent, key, None)}
        for key, value in expected.items()
        if getattr(parent, key, None) != value
    }
    if mismatch:
        raise ValueError(f"exact parent full recipe mismatch: {mismatch}")
    return parent


def _read_runtime(runtime_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    required = (
        "train.jsonl",
        "development_selector_oof.npz",
        "development_image_manifest.tsv.gz",
        "selected_indices.npy",
        "runtime_audit.json",
    )
    missing = [name for name in required if not (runtime_dir / name).is_file()]
    if missing:
        raise ValueError(f"full runtime incomplete: {missing}")
    audit = json.loads((runtime_dir / "runtime_audit.json").read_text(encoding="utf-8"))
    observed_self = audit.get("runtime_sha256")
    payload = {key: value for key, value in audit.items() if key != "runtime_sha256"}
    if observed_self != canonical_sha256(payload):
        raise ValueError("full runtime self hash mismatch")
    expected = {
        "schema_version": "exp682_full_runtime_v1",
        "experiment_id": "682",
        "mode": "development_only_FULL_TRAIN_1",
        "decision": "GO",
        "development_rows": 11118,
        "sealed_rows_read": 0,
        "sealed_labels_read": 0,
        "sealed_rows_written": 0,
        "public_feedback_used": False,
        "selected_records": 6116,
        "selected_unique_rows": 5436,
        "selected_id_multiset_sha256": "f4734df62119ca67d86e22681fb09544b8c56761c6ffaf7617875cccdb983e9f",
        "optimizer_updates": 383,
    }
    mismatch = {
        key: {"expected": value, "actual": audit.get(key)}
        for key, value in expected.items()
        if audit.get(key) != value
    }
    if mismatch:
        raise ValueError(f"full runtime contract mismatch: {mismatch}")
    for name in required[:-1]:
        if audit.get("output_sha256", {}).get(name) != sha256_file(runtime_dir / name):
            raise ValueError(f"full runtime output checksum mismatch: {name}")
    with (runtime_dir / "train.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != 11118 or any(
        int(row.get("label", -1)) not in (0, 1) or not isinstance(row.get("rationale"), dict)
        for row in rows
    ):
        raise ValueError("full runtime supervised rows mismatch")
    selected = np.load(runtime_dir / "selected_indices.npy", allow_pickle=False)
    if selected.dtype != np.int64 or selected.ndim != 1 or len(selected) != 6116:
        raise ValueError("selected index array mismatch")
    return rows, selected, audit


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if args.model_revision != MODEL_REVISION:
        raise ValueError("exact Qwen3.5 model revision mismatch")
    output = args.output_dir.resolve()
    images = args.images.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty output directory")
    if images.exists() and any(images.iterdir()):
        raise FileExistsError("refusing to reuse nonempty image directory")
    output.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    upstream = _load_upstream(args.upstream_dir.resolve())
    rows, frozen_selected, runtime_audit = _read_runtime(args.runtime_dir.resolve())
    frame = pd.DataFrame(
        [
            {key: row[key] for key in ("id", "category", "name", "description", "label")}
            for row in rows
        ]
    )
    train_by_id = {str(row["id"]): row for row in rows}
    if len(train_by_id) != 11118:
        raise ValueError("development IDs are not unique")
    parent = _load_parent_full(
        parent_path=args.parent.resolve(),
        runtime_dir=args.runtime_dir.resolve(),
        images=images,
        model_root=args.model_root.resolve(),
        vendor=args.vendor.resolve(),
    )
    selector = np.load(
        args.runtime_dir.resolve() / "development_selector_oof.npz", allow_pickle=False
    )
    selected = np.asarray(parent.select_training(frame, selector), dtype=np.int64)
    if not np.array_equal(selected, frozen_selected):
        raise ValueError("GPU worker selector differs from CPU-frozen selected indices")
    selected_ids = frame.iloc[selected]["id"].astype(str).tolist()
    if canonical_sha256(sorted(Counter(selected_ids).items())) != runtime_audit[
        "selected_id_multiset_sha256"
    ]:
        raise ValueError("GPU worker selected-ID multiset mismatch")
    failures = parent.predownload(sorted(set(selected_ids)), parent.load_urls())
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
    from model import Qwen35VerdictSpanModel
    from multitask_head import multitask_loss

    processor = AutoProcessor.from_pretrained(
        args.model_root.resolve(), local_files_only=True, trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1:
        raise ValueError("parent digit tokens are not atomic")
    backbone = AutoModelForMultimodalLM.from_pretrained(
        args.model_root.resolve(),
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
    backbone = get_peft_model(
        backbone,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_rslora=True,
        ),
    )
    backbone.config.use_cache = False
    backbone.enable_input_require_grads()
    backbone.gradient_checkpointing_enable()
    hidden_size = int(backbone.config.text_config.hidden_size)
    model = Qwen35VerdictSpanModel(
        backbone,
        hidden_size=hidden_size,
        concept_count=len(upstream.CONCEPTS),
        token_zero=zero[0],
        token_one=one[0],
    ).to("cuda")
    model.auxiliary_head.to(dtype=torch.bfloat16)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
    batches = math.ceil(len(selected_ids) / BATCH_SIZE)
    updates = math.ceil(batches * EPOCHS / GRAD_ACCUM)
    if updates != runtime_audit["optimizer_updates"]:
        raise ValueError("optimizer update count differs from CPU-frozen runtime")
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
    records = list(selected_ids)
    started = time.monotonic()
    for epoch in range(EPOCHS):
        random.Random(SEED + epoch).shuffle(records)
        for offset in range(0, len(records), BATCH_SIZE):
            batch_rows = [
                SimpleNamespace(**train_by_id[item_id])
                for item_id in records[offset : offset + BATCH_SIZE]
            ]
            batch, alignments = upstream._batch(
                parent, processor, batch_rows, with_rationale=True
            )
            masked_alignment_rows += sum(
                item.quality_weight <= 0 and row.rationale.get("has_evidence")
                for item, row in zip(alignments, batch_rows, strict=True)
            )
            device_batch = {key: value.to("cuda") for key, value in batch.items()}
            text_mask = torch.tensor(
                [item.text_token_mask for item in alignments], dtype=torch.bool, device="cuda"
            )
            result = model(text_token_mask=text_mask, **device_batch)
            loss, _ = multitask_loss(
                verdict_logits=result["verdict_logits"],
                auxiliary=result,
                verdict_targets=torch.tensor(
                    [row.label for row in batch_rows], dtype=torch.float32, device="cuda"
                ),
                start_targets=torch.tensor(
                    [item.start_target for item in alignments], device="cuda"
                ),
                end_targets=torch.tensor([item.end_target for item in alignments], device="cuda"),
                concept_targets=torch.tensor(
                    [item.concept_target for item in alignments], device="cuda"
                ),
                quality_weights=torch.tensor(
                    [item.quality_weight for item in alignments], device="cuda"
                ),
            )
            (loss / GRAD_ACCUM).backward()
            step += 1
            if step % GRAD_ACCUM == 0 or offset + BATCH_SIZE >= len(records):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
    if update != updates:
        raise AssertionError("realized optimizer updates differ from frozen count")

    adapter_dir = output / "adapter"
    model.backbone.save_pretrained(adapter_dir)
    shutil.make_archive(str(output / "adapter"), "zip", adapter_dir)
    auxiliary_path = output / "auxiliary_head.safetensors"
    save_file(
        {
            key: value.detach().float().cpu().contiguous()
            for key, value in model.auxiliary_head.state_dict().items()
        },
        str(auxiliary_path),
    )
    report: dict[str, Any] = {
        "schema_version": "exp682_full_adapter_contract_v1",
        "experiment_id": "682",
        "seed": SEED,
        "full_train": True,
        "training_scope": "development_only_11118",
        "training_records": len(selected_ids),
        "training_unique_rows": len(set(selected_ids)),
        "selected_id_multiset_sha256": canonical_sha256(
            sorted(Counter(selected_ids).items())
        ),
        "optimizer_updates": update,
        "warmup_updates": warmup,
        "masked_alignment_rows": masked_alignment_rows,
        "runtime_minutes": (time.monotonic() - started) / 60,
        "model_revision": MODEL_REVISION,
        "parent_runner_sha256": PARENT_SHA256,
        "full_runtime_audit_sha256": sha256_file(
            args.runtime_dir.resolve() / "runtime_audit.json"
        ),
        "adapter_zip_sha256": sha256_file(output / "adapter.zip"),
        "auxiliary_head_sha256": sha256_file(auxiliary_path),
        "auxiliary_head_used_for_verdict": False,
        "auxiliary_head_packaged": False,
        "verdict_inference": "token-1 minus token-0 logit",
        "sealed_training_rows": 0,
        "public_feedback_used": False,
        "decision": "GO",
    }
    report["contract_sha256"] = canonical_sha256(report)
    (output / "full_train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
