from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_device(value: Any) -> str:
    if isinstance(value, int):
        return f"cuda:{value}"
    text = str(value).lower()
    if text.isdigit():
        return f"cuda:{text}"
    if text == "cuda":
        return "cuda:0"
    return text


def summarize_device_map(device_map: dict[str, Any]) -> dict[str, int]:
    return dict(sorted(Counter(normalize_device(value) for value in device_map.values()).items()))


def validate_device_map(device_map: dict[str, Any], *, minimum_cuda_devices: int) -> dict[str, int]:
    if not device_map:
        raise RuntimeError("model-parallel load did not publish hf_device_map")
    summary = summarize_device_map(device_map)
    forbidden = [device for device in summary if device in {"cpu", "disk", "meta"}]
    if forbidden:
        raise RuntimeError(f"model was offloaded outside CUDA: {forbidden}")
    cuda_devices = {device for device in summary if device.startswith("cuda:")}
    if len(cuda_devices) < minimum_cuda_devices:
        raise RuntimeError(
            f"model uses {len(cuda_devices)} CUDA devices, expected at least {minimum_cuda_devices}"
        )
    return summary


def adapter_manifest(adapter_dir: Path) -> dict[str, str]:
    files = sorted(path for path in adapter_dir.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError("adapter directory is empty")
    return {str(path.relative_to(adapter_dir)): sha256_file(path) for path in files}


def first_parameter_device(model: Any):
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    raise RuntimeError("model has no materialized parameters")


def prompt(name: str, description: str) -> str:
    return (
        "Категория: БАД\n"
        f"Название: {name}\n"
        f"Описание: {description}\n"
        "Правило: метка 1 только при прямом указании БАД или dietary supplement; "
        "явное отрицание или отсутствие маркировки означает 0.\n"
        "Ответь только одной цифрой: 1 или 0."
    )


def processor_batch(processor: Any, image: Any, text: str):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]
    return processor.apply_chat_template(
        [conversation],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        enable_thinking=False,
    )


def score(model: Any, batch: Any, *, zero_token: int, one_token: int, input_device: Any):
    import torch

    device_batch = {key: value.to(input_device) for key, value in batch.items()}
    outputs = model(**device_batch, use_cache=False)
    mask = device_batch["attention_mask"]
    positions = torch.arange(mask.shape[1], device=mask.device)[None, :]
    last = torch.where(mask.bool(), positions, -1).max(dim=1).values
    rows = torch.arange(mask.shape[0], device=outputs.logits.device)
    logits = outputs.logits[rows, last.to(outputs.logits.device)]
    return logits[:, one_token] - logits[:, zero_token]


def write_zip(output_dir: Path, report_path: Path, adapter_dir: Path) -> Path:
    archive = output_dir / "qwen36_27b_lora_smoke.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(report_path, "report.json")
        for path in sorted(adapter_dir.rglob("*")):
            if path.is_file():
                bundle.write(path, str(Path("adapter") / path.relative_to(adapter_dir)))
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt ZIP member: {bad}")
    return archive


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.model_revision != MODEL_REVISION:
        raise ValueError("model revision differs from the frozen experiment contract")
    if args.expected_cuda_devices < 2:
        raise ValueError("large-model smoke requires at least two CUDA devices")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = args.output_dir / "adapter"

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.expected_cuda_devices:
        raise RuntimeError("requested CUDA devices are unavailable")
    if not args.vendor.is_dir():
        raise FileNotFoundError("vendored PEFT directory is missing")
    import sys

    sys.path.insert(0, str(args.vendor.resolve()))
    import peft

    if peft.__version__ != "0.20.0":
        raise RuntimeError("exact vendored PEFT 0.20.0 is required")
    from peft import LoraConfig, TaskType, get_peft_model

    processor = AutoProcessor.from_pretrained(args.model_root, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1 or zero == one:
        raise RuntimeError("0 and 1 are not distinct atomic tokens")

    started = time.monotonic()
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_root,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map="balanced",
        attn_implementation="eager",
    )
    device_summary = validate_device_map(
        getattr(model, "hf_device_map", {}),
        minimum_cuda_devices=args.expected_cuda_devices,
    )
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
    model.train()
    input_device = first_parameter_device(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.device.type != "cuda" for parameter in trainable):
        raise RuntimeError("trainable adapter parameters are not fully resident on CUDA")
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.01)
    optimizer.zero_grad(set_to_none=True)

    examples = [
        ("Витаминный комплекс", "Биологически активная добавка к пище.", 1, (230, 245, 255)),
        ("Футляр", "Не является БАД. Чехол для хранения.", 0, (245, 240, 225)),
    ]
    losses: list[float] = []
    gradients_finite = True
    for name, description, label, color in examples:
        image = Image.new("RGB", (64, 64), color=color)
        batch = processor_batch(processor, image, prompt(name, description))
        logits = score(
            model,
            batch,
            zero_token=zero[0],
            one_token=one[0],
            input_device=input_device,
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.float(), torch.tensor([label], dtype=torch.float32, device=logits.device)
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite training loss")
        (loss / len(examples)).backward()
        losses.append(float(loss.detach().cpu()))
        image.close()
    for parameter in trainable:
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            gradients_finite = False
            break
    if not gradients_finite:
        raise RuntimeError("non-finite adapter gradients")
    torch.nn.utils.clip_grad_norm_(trainable, 1.0, foreach=False)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    model.eval()
    model.save_pretrained(adapter_dir)
    manifest = adapter_manifest(adapter_dir)
    model.load_adapter(str(adapter_dir), adapter_name="roundtrip", is_trainable=False)
    model.set_adapter("roundtrip")
    reload_image = Image.new("RGB", (64, 64), color=(230, 245, 255))
    reload_batch = processor_batch(
        processor,
        reload_image,
        prompt("Витаминный комплекс", "Биологически активная добавка к пище."),
    )
    with torch.inference_mode():
        reload_score = score(
            model,
            reload_batch,
            zero_token=zero[0],
            one_token=one[0],
            input_device=input_device,
        )
    reload_image.close()
    reload_value = float(reload_score.float().cpu()[0])
    if not math.isfinite(reload_value):
        raise RuntimeError("non-finite score after adapter reload")

    report = {
        "schema_version": 1,
        "experiment_id": "653",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "synthetic_rows": len(examples),
        "competition_rows": 0,
        "sealed_rows": 0,
        "seed": SEED,
        "cuda_device_count": torch.cuda.device_count(),
        "device_map_summary": device_summary,
        "cpu_or_disk_offload": False,
        "losses": losses,
        "gradients_finite": gradients_finite,
        "optimizer_steps": 1,
        "adapter_reloaded": True,
        "reload_score": reload_value,
        "adapter_manifest": manifest,
        "packages": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "peft": peft.__version__,
        },
        "elapsed_seconds": time.monotonic() - started,
        "decision": "TECHNICAL_GO",
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    archive = write_zip(args.output_dir, report_path, adapter_dir)
    delivery = {
        "schema_version": 1,
        "experiment_id": "653",
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "report_sha256": sha256_file(report_path),
    }
    (args.output_dir / "delivery.json").write_text(
        json.dumps(delivery, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["delivery"] = delivery
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--model-root", type=Path, required=True)
    result.add_argument("--model-revision", default=MODEL_REVISION)
    result.add_argument("--vendor", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--expected-cuda-devices", type=int, default=4)
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
