from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoProcessor


MODEL = Path(os.environ.get("MODEL_PATH", "/hf_models"))
MODEL_CLASS = os.environ.get("MODEL_CLASS", "image_text")
IMAGE = Path("/work/input/0.jpg")
VENDOR = Path("/work/vendor")


def install_peft():
    VENDOR.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--target", str(VENDOR),
        "--no-cache-dir", "--no-deps", "peft==0.20.0",
    ], check=True)
    sys.path.insert(0, str(VENDOR))


def encode(processor, messages, add_generation_prompt):
    kwargs = dict(
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def main():
    install_peft()
    from peft import LoraConfig, TaskType, get_peft_model

    processor_kwargs = {"local_files_only": True, "trust_remote_code": True}
    if MODEL_CLASS == "image_text":
        processor_kwargs.update(min_pixels=4 * 28 * 28, max_pixels=262144)
    processor = AutoProcessor.from_pretrained(MODEL, **processor_kwargs)
    loader = AutoModelForMultimodalLM if MODEL_CLASS == "multimodal" else AutoModelForImageTextToText
    started = time.monotonic()
    model = loader.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
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
    print(json.dumps({
        "model_class": model.__class__.__name__,
        "loaded_sec": time.monotonic() - started,
        "trainable": sum(p.numel() for p in trainable),
        "total": sum(p.numel() for p in model.parameters()),
    }), flush=True)

    image = Image.open(IMAGE).convert("RGB")
    image.thumbnail((448, 448), Image.Resampling.LANCZOS)
    prompt_messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Определи корректность категории товара. Ответь только цифрой 0 или 1."},
        ],
    }]
    full_messages = prompt_messages + [{
        "role": "assistant",
        "content": [{"type": "text", "text": "0"}],
    }]
    full = encode(processor, full_messages, False)
    prompt = encode(processor, prompt_messages, True)
    full_ids = full["input_ids"][0, full["attention_mask"][0].bool()]
    prompt_ids = prompt["input_ids"][0, prompt["attention_mask"][0].bool()]
    limit = min(len(full_ids), len(prompt_ids))
    mismatch = torch.nonzero(full_ids[:limit] != prompt_ids[:limit]).flatten()
    common = int(mismatch[0]) if len(mismatch) else limit
    if common < len(prompt_ids) - 2 or common >= len(full_ids):
        raise ValueError(
            f"assistant suffix alignment failed: common={common} "
            f"prompt={len(prompt_ids)} full={len(full_ids)}"
        )
    labels = torch.full_like(full["input_ids"], -100)
    positions = torch.nonzero(full["attention_mask"][0]).flatten()
    labels[0, positions[common:]] = full["input_ids"][0, positions[common:]]
    full["labels"] = labels
    full = {key: value.to("cuda") if hasattr(value, "to") else value for key, value in full.items()}
    output = model(**full)
    output.loss.backward()
    gradients = [p.grad for p in trainable if p.grad is not None]
    print(json.dumps({
        "prompt_tokens": len(prompt_ids),
        "full_tokens": len(full_ids),
        "answer_decoded": processor.tokenizer.decode(full_ids[common:].tolist()),
        "loss": float(output.loss.detach().cpu()),
        "grad_tensors": len(gradients),
        "grad_norm": float(torch.sqrt(sum((g.float() ** 2).sum() for g in gradients)).cpu()),
        "max_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
