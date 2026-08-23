from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import io
import json
import math
import os
import random
import shutil
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from grid_contract import (
    CELL_SPECS,
    GRID_CONTRACT_SHA256,
    NO_EVIDENCE,
    PREPROCESSING_VERSION,
    PROMPT_VERSION,
    base_prompt,
    canonical_sha256,
    conditioned_verdict_prompt,
    evidence_prompt,
    parse_structured_generation,
    resolve_grounding,
    sha256_file,
    structured_target,
)

MAX_LENGTH = 1536
SEED = 42
EPOCHS = 1
LEARNING_RATE = 2e-4


def verify_fast_linear_attention_dependencies() -> dict[str, str]:
    """Fail before model loading if Qwen's memory-safe training path is unavailable."""
    # `flash-linear-attention` is only the high-level distribution as of 0.5.x;
    # the actual `fla.ops` implementation lives in `fla-core`.  Pin and verify
    # both so an older preinstalled core cannot silently select the torch path.
    required = {
        "fla.ops.gated_delta_rule": (
            "chunk_gated_delta_rule",
            "fused_recurrent_gated_delta_rule",
        ),
        "causal_conv1d": ("causal_conv1d_fn", "causal_conv1d_update"),
    }
    for module_name, attributes in required.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            raise RuntimeError(
                f"required fast-path module is unavailable: {module_name}"
            ) from error
        missing = [name for name in attributes if not callable(getattr(module, name, None))]
        if missing:
            raise RuntimeError(
                f"required fast-path symbols are unavailable in {module_name}: {missing}"
            )
    import torch
    import triton

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the required Qwen fast path")
    try:
        triton_backend = triton.runtime.driver.active.get_current_target().backend
    except Exception as error:
        raise RuntimeError("Triton cannot initialize its active CUDA target") from error
    if triton_backend != "cuda":
        raise RuntimeError(f"unexpected Triton backend for Qwen fast path: {triton_backend}")
    versions = {
        "transformers": importlib.metadata.version("transformers"),
        "triton": importlib.metadata.version("triton"),
        "einops": importlib.metadata.version("einops"),
        "fla_core": importlib.metadata.version("fla-core"),
        "flash_linear_attention": importlib.metadata.version("flash-linear-attention"),
        "causal_conv1d": importlib.metadata.version("causal-conv1d"),
    }
    if versions != {
        "transformers": "5.15.1",
        "triton": "3.6.0",
        "einops": "0.8.2",
        "fla_core": "0.5.2",
        "flash_linear_attention": "0.5.2",
        "causal_conv1d": "1.6.2.post1",
    }:
        raise RuntimeError(f"unexpected fast-path package versions: {versions}")

    # Transformers 5.15.1 asks the FLA namespace for
    # `recurrent_gated_delta_rule`, while FLA 0.5.2 exports the same supported
    # kernel as `fused_recurrent_gated_delta_rule`.  Install the compatibility
    # alias before the lazy Qwen module is imported.
    gated = importlib.import_module("fla.ops.gated_delta_rule")
    if not hasattr(gated, "recurrent_gated_delta_rule"):
        gated.recurrent_gated_delta_rule = gated.fused_recurrent_gated_delta_rule

    from transformers.utils import (
        is_causal_conv1d_available,
        is_flash_linear_attention_available,
    )

    if not is_flash_linear_attention_available() or not is_causal_conv1d_available():
        raise RuntimeError("Transformers does not recognize the installed fast-path packages")
    return versions


def verify_qwen35_fast_path_binding() -> dict[str, str]:
    """Prove that Qwen wrappers captured compiled functions, not torch fallbacks."""
    modeling_qwen3_5 = importlib.import_module(
        "transformers.models.qwen3_5.modeling_qwen3_5"
    )

    targets = {
        "chunk_gated_delta_rule": (
            modeling_qwen3_5.torch_chunk_gated_delta_rule,
            "fla.",
        ),
        "recurrent_gated_delta_rule": (
            modeling_qwen3_5.torch_recurrent_gated_delta_rule,
            "fla.",
        ),
        "causal_conv1d_fn": (modeling_qwen3_5.causal_conv1d_fn, "causal_conv1d"),
        "causal_conv1d_update": (
            modeling_qwen3_5.causal_conv1d_update,
            "causal_conv1d",
        ),
    }
    bound: dict[str, str] = {}
    for name, (wrapper, expected_prefix) in targets.items():
        implementation = inspect.getclosurevars(wrapper).nonlocals.get("implementation")
        module_name = str(getattr(implementation, "__module__", ""))
        if not callable(implementation) or not module_name.startswith(expected_prefix):
            raise RuntimeError(
                f"Qwen fast-path binding failed for {name}: {module_name or 'torch fallback'}"
            )
        bound[name] = module_name
    return bound


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def load_runtime(
    runtime_dir: Path, spec_id: str, fold: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_path = runtime_dir / "train.jsonl"
    validation_path = runtime_dir / "validation.jsonl"
    audit_path = runtime_dir / "runtime_audit.json"
    for path in (train_path, validation_path, audit_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"runtime file missing or empty: {path.name}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    payload = dict(audit)
    digest = payload.pop("contract_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("runtime audit self-hash mismatch")
    expected = {
        "experiment_id": spec_id,
        "objective": CELL_SPECS[spec_id].objective,
        "outer_fold": fold,
        "grid_contract_sha256": GRID_CONTRACT_SHA256,
        "sealed_rows_written": 0,
        "validation_labels_written": 0,
        "outer_validation_occurrences": 0,
        "decision": "GO",
    }
    mismatch = {
        key: {"expected": value, "actual": audit.get(key)}
        for key, value in expected.items()
        if audit.get(key) != value
    }
    if mismatch:
        raise ValueError(f"runtime audit mismatch: {mismatch}")
    if audit["output_sha256"] != {
        "train.jsonl": sha256_file(train_path),
        "validation.jsonl": sha256_file(validation_path),
    }:
        raise ValueError("runtime payload checksum mismatch")
    train = read_jsonl(train_path)
    validation = read_jsonl(validation_path)
    if any("label" in row or "evidence_target" in row for row in validation):
        raise ValueError("validation supervision is forbidden")
    if any(int(row["fold"]) == fold for row in train):
        raise ValueError("outer fold entered training")
    if any(int(row["fold"]) != fold for row in validation):
        raise ValueError("validation fold mismatch")
    return train, validation, audit


def _cache_path(cache: Path, row_id: str) -> Path:
    return cache / f"{hashlib.sha256(row_id.encode()).hexdigest()}.img"


def ensure_image(cache: Path, row: dict[str, Any]) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache, str(row["id"]))
    if path.is_file() and path.stat().st_size > 0:
        return path
    if path.exists():
        raise ValueError("refusing to replace an invalid image-cache entry")
    request = urllib.request.Request(str(row["image_url"]), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    if not payload:
        raise ValueError("downloaded image is empty")
    # Validate before committing the cache entry. The original bytes are retained.
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
    path.write_bytes(payload)
    return path


def open_image(cache: Path, row: dict[str, Any]):
    from PIL import Image

    path = ensure_image(cache, row)
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if width * height > 262144:
        scale = math.sqrt(262144 / (width * height))
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        resized = image.resize(
            (max(28, int(width * scale)), max(28, int(height * scale))),
            resampling,
        )
        image.close()
        image = resized
    return image


def messages(
    row: SimpleNamespace, image: Any, *, prompt_text: str, answer: str | None = None
) -> list[dict[str, Any]]:
    conversation: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    if answer is not None:
        conversation.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return conversation


def _processor_batch(
    processor: Any, conversations: list[list[dict[str, Any]]], *, add_generation_prompt: bool
):
    return processor.apply_chat_template(
        conversations,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        enable_thinking=False,
    )


def _last_logits(model: Any, batch: dict[str, Any], zero_token: int, one_token: int):
    import torch

    outputs = model(**batch, use_cache=False)
    positions = torch.arange(batch["attention_mask"].shape[1], device=model.device)[None, :]
    last = torch.where(batch["attention_mask"].bool(), positions, -1).max(dim=1).values
    rows = torch.arange(batch["attention_mask"].shape[0], device=model.device)
    logits = outputs.logits[rows, last]
    return logits[:, one_token] - logits[:, zero_token]


def primary_loss(
    model: Any,
    processor: Any,
    rows: list[SimpleNamespace],
    images: list[Any],
    zero_token: int,
    one_token: int,
):
    import torch
    from torch.nn import functional

    conversations = [
        messages(row, image, prompt_text=base_prompt(row))
        for row, image in zip(rows, images, strict=True)
    ]
    batch = _processor_batch(processor, conversations, add_generation_prompt=True).to(model.device)
    scores = _last_logits(model, batch, zero_token, one_token)
    labels = torch.tensor(
        [int(row.label) for row in rows], dtype=torch.float32, device=model.device
    )
    return functional.binary_cross_entropy_with_logits(scores.float(), labels)


def _locate_last(sequence: list[int], subsequence: list[int]) -> int:
    hits = [
        index
        for index in range(len(sequence) - len(subsequence) + 1)
        if sequence[index : index + len(subsequence)] == subsequence
    ]
    if not hits:
        raise ValueError("assistant target token sequence is absent after chat templating")
    return hits[-1]


def auxiliary_loss(model: Any, processor: Any, rows: list[SimpleNamespace], images: list[Any]):
    import torch

    order_losses = []
    # Keep each order in a separate forward pass. This preserves the frozen
    # 50/50 objective without doubling the 27B microbatch peak memory.
    for order in ("class_first", "evidence_first"):
        answers = [
            structured_target(
                verdict=int(row.label),
                quote=str(row.evidence_target["quote"]),
                concept=str(row.evidence_target["concept"]),
                order=order,
            )
            for row in rows
        ]
        conversations = [
            messages(row, image, prompt_text=evidence_prompt(row, order), answer=answer)
            for row, image, answer in zip(rows, images, answers, strict=True)
        ]
        batch = _processor_batch(processor, conversations, add_generation_prompt=False)
        labels = torch.full_like(batch["input_ids"], -100)
        for row_index, answer in enumerate(answers):
            active_positions = torch.nonzero(batch["attention_mask"][row_index]).flatten().tolist()
            active_ids = batch["input_ids"][row_index, active_positions].tolist()
            target_ids = processor.tokenizer.encode(answer, add_special_tokens=False)
            start = _locate_last(active_ids, target_ids)
            for local_index, token_id in enumerate(target_ids):
                full_index = active_positions[start + local_index]
                labels[row_index, full_index] = int(token_id)
        device_batch = {key: value.to(model.device) for key, value in batch.items()}
        order_losses.append(
            model(**device_batch, labels=labels.to(model.device), use_cache=False).loss
        )
    return sum(order_losses) / len(order_losses)


def _single_score(
    model: Any,
    processor: Any,
    row: SimpleNamespace,
    image: Any,
    prompt_text: str,
    zero_token: int,
    one_token: int,
) -> float:
    import torch

    batch = _processor_batch(
        processor,
        [messages(row, image, prompt_text=prompt_text)],
        add_generation_prompt=True,
    ).to(model.device)
    with torch.inference_mode():
        score = _last_logits(model, batch, zero_token, one_token)
    return float(score[0].float().cpu())


def _generate(model: Any, processor: Any, row: SimpleNamespace, image: Any, order: str) -> str:
    import torch

    batch = _processor_batch(
        processor,
        [messages(row, image, prompt_text=evidence_prompt(row, order))],
        add_generation_prompt=True,
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **batch,
            max_new_tokens=150,
            do_sample=False,
            use_cache=True,
        )
    prompt_length = batch["input_ids"].shape[1]
    return processor.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()


def _base_prediction_record(
    row: SimpleNamespace, *, spec: Any, score: float, order: str
) -> dict[str, Any]:
    return {
        "global_index": int(row.global_index),
        "id": str(row.id),
        "fold": int(row.fold),
        "category": str(row.category),
        "score": score,
        "prediction": int(score >= 0.0),
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "objective": spec.objective,
        "target_order": order,
        "prompt_version": PROMPT_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
    }


def predict_class_only(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    cache: Path,
    spec: Any,
    zero_token: int,
    one_token: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in rows:
        row = SimpleNamespace(**item)
        image = open_image(cache, item)
        try:
            score = _single_score(
                model, processor, row, image, base_prompt(row), zero_token, one_token
            )
        finally:
            image.close()
        record = _base_prediction_record(row, spec=spec, score=score, order="none")
        record.update(
            {
                "raw_generation": "",
                "generated_verdict": -1,
                "format_valid": True,
                "quote": NO_EVIDENCE,
                "concept": NO_EVIDENCE,
                "grounded": True,
                "grounding_source": "none",
                "image_index": -1,
                "region_index": -1,
            }
        )
        output.append(record)
    return output


def predict_evidence_order(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    cache: Path,
    spec: Any,
    zero_token: int,
    one_token: int,
    *,
    order: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in rows:
        row = SimpleNamespace(**item)
        image = open_image(cache, item)
        try:
            raw = _generate(model, processor, row, image, order)
            parsed = parse_structured_generation(raw, expected_order=order)
            grounding = resolve_grounding(item, str(parsed["quote"]))
            verified_evidence = (
                {"quote": parsed["quote"], "concept": parsed["concept"]}
                if parsed["format_valid"] and grounding["grounded"]
                else {"quote": NO_EVIDENCE, "concept": NO_EVIDENCE}
            )
            if order == "class_first":
                score_prompt = base_prompt(row)
            else:
                score_prompt = conditioned_verdict_prompt(row, verified_evidence)
            score = _single_score(model, processor, row, image, score_prompt, zero_token, one_token)
        finally:
            image.close()
        record = _base_prediction_record(row, spec=spec, score=score, order=order)
        record.update(
            {
                "raw_generation": raw,
                "generated_verdict": (
                    int(parsed["generated_verdict"])
                    if parsed["generated_verdict"] is not None
                    else -1
                ),
                "format_valid": bool(parsed["format_valid"]),
                "quote": str(parsed["quote"]),
                "concept": str(parsed["concept"]),
                "grounded": bool(grounding["grounded"]),
                "grounding_source": str(grounding["source"]),
                "image_index": int(grounding["image_index"]),
                "region_index": int(grounding["region_index"]),
            }
        )
        output.append(record)
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run(spec_id: str, args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    spec = CELL_SPECS[spec_id]
    # The Hub fused GDN kernel is hardware-gated and is not the H100 training
    # path.  Disable Hub substitution and require the official FLA backend.
    os.environ["USE_HUB_KERNELS"] = "NO"
    fast_path_packages = verify_fast_linear_attention_dependencies()
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    fast_path_bindings = verify_qwen35_fast_path_binding()
    if args.model_revision != spec.model_revision:
        raise ValueError("model revision differs from frozen cell contract")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, validation_rows, runtime_audit = load_runtime(args.runtime_dir, spec_id, args.fold)
    if args.technical_smoke:
        train_rows = train_rows[: max(spec.micro_batch_size, 8)]
        validation_rows = validation_rows[:2]
    if spec.objective == "grounded_evidence" and any(
        "evidence_target" not in row for row in train_rows
    ):
        raise ValueError("grounded-evidence runtime lacks exact evidence targets")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if not args.vendor.is_dir():
        raise FileNotFoundError("vendored PEFT 0.20.0 directory is missing")
    sys.path.insert(0, str(args.vendor.resolve()))
    import peft

    if peft.__version__ != "0.20.0":
        raise ValueError("exact vendored PEFT 0.20.0 is required")
    from peft import LoraConfig, TaskType, get_peft_model

    processor = AutoProcessor.from_pretrained(
        args.model_root.resolve(), local_files_only=True, trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1 or zero == one:
        raise ValueError("0/1 must be distinct atomic tokens")
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_root.resolve(),
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
        use_kernels=False,
    ).to("cuda")
    model = get_peft_model(
        model,
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
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
    batches = math.ceil(len(train_rows) / spec.micro_batch_size)
    updates = math.ceil(batches * EPOCHS / spec.gradient_accumulation)
    warmup = max(1, int(updates * 0.05))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, updates - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    optimizer.zero_grad(set_to_none=True)
    step = 0
    started = time.monotonic()
    indices = list(range(len(train_rows)))
    for epoch in range(EPOCHS):
        random.Random(SEED + epoch).shuffle(indices)
        for offset in range(0, len(indices), spec.micro_batch_size):
            local = [
                train_rows[index] for index in indices[offset : offset + spec.micro_batch_size]
            ]
            rows = [SimpleNamespace(**item) for item in local]
            images = [open_image(args.images, item) for item in local]
            try:
                loss = primary_loss(model, processor, rows, images, zero[0], one[0])
                if spec.objective == "grounded_evidence":
                    loss = loss + spec.evidence_auxiliary_weight * auxiliary_loss(
                        model, processor, rows, images
                    )
                (loss / spec.gradient_accumulation).backward()
            finally:
                for image in images:
                    image.close()
            step += 1
            if step % spec.gradient_accumulation == 0 or offset + spec.micro_batch_size >= len(
                indices
            ):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

    model.eval()
    model.config.use_cache = True
    outputs: dict[str, Path] = {}
    if spec.objective == "class_only":
        records = predict_class_only(
            model, processor, validation_rows, args.images, spec, zero[0], one[0]
        )
        path = args.output_dir / "predictions.jsonl"
        write_jsonl(path, records)
        outputs["predictions.jsonl"] = path
    else:
        for order in ("class_first", "evidence_first"):
            records = predict_evidence_order(
                model,
                processor,
                validation_rows,
                args.images,
                spec,
                zero[0],
                one[0],
                order=order,
            )
            path = args.output_dir / f"predictions_{order}.jsonl"
            write_jsonl(path, records)
            outputs[path.name] = path

    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    shutil.make_archive(str(args.output_dir / "adapter"), "zip", adapter_dir)
    outputs["adapter.zip"] = args.output_dir / "adapter.zip"
    report = {
        "schema_version": 1,
        "experiment_id": spec_id,
        "outer_fold": args.fold,
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "objective": spec.objective,
        "grid_contract_sha256": GRID_CONTRACT_SHA256,
        "runtime_contract_sha256": runtime_audit["contract_sha256"],
        "model_input_view_sha256": runtime_audit["model_input_view_sha256"],
        "seed": SEED,
        "epochs": EPOCHS,
        "effective_batch_size": spec.micro_batch_size * spec.gradient_accumulation,
        "evidence_auxiliary_weight": spec.evidence_auxiliary_weight,
        "optimized_training_kernels": True,
        "fast_path_packages": fast_path_packages,
        "fast_path_bindings": fast_path_bindings,
        "target_orders": (
            ["class_first", "evidence_first"] if spec.objective == "grounded_evidence" else []
        ),
        "train_occurrences": len(train_rows),
        "validation_rows": len(validation_rows),
        "technical_smoke": bool(args.technical_smoke),
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "threshold": 0.0,
        "threshold_tuned": False,
        "runtime_minutes": (time.monotonic() - started) / 60,
        "artifacts": {name: sha256_file(path) for name, path in outputs.items()},
        "decision": "TECHNICAL_SMOKE_ONLY" if args.technical_smoke else "GO_EVALUATE",
    }
    report["contract_sha256"] = canonical_sha256(report)
    (args.output_dir / "output_contract.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parser_for(spec_id: str) -> argparse.ArgumentParser:
    spec = CELL_SPECS[spec_id]
    parser = argparse.ArgumentParser(
        description=f"Train label-isolated experiment {spec_id}: {spec.objective}."
    )
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--vendor", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--technical-smoke",
        action="store_true",
        help="Use eight train occurrences and two validation rows; never accepted by evaluator.",
    )
    return parser
