from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL = Path("/hf_models")
IMAGE = Path("/work/input/0.jpg")
VENDOR = Path("/work/vendor")


def install_peft():
    VENDOR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(VENDOR),
            "--no-cache-dir",
            "--no-deps",
            "peft==0.20.0",
        ],
        check=True,
    )
    sys.path.insert(0, str(VENDOR))


def main():
    install_peft()
    import peft
    from peft import LoraConfig, TaskType, get_peft_model

    print(json.dumps({
        "peft": peft.__version__,
        "torch": torch.__version__,
    }), flush=True)
    processor = AutoProcessor.from_pretrained(
        MODEL,
        local_files_only=True,
        min_pixels=4 * 28 * 28,
        max_pixels=262144,
    )
    started = time.monotonic()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    ).to("cuda")
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=True,
    )
    model = get_peft_model(model, config)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(json.dumps({
        "loaded_sec": time.monotonic() - started,
        "model_class": model.__class__.__name__,
        "trainable": trainable,
        "total": total,
        "trainable_pct": 100 * trainable / total,
    }), flush=True)

    prompt_messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(IMAGE)},
            {"type": "text", "text": "Определи метку товара. Ответь только 0 или 1."},
        ],
    }]
    messages = prompt_messages + [{
        "role": "assistant",
        "content": [{"type": "text", "text": "0"}],
    }]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    generation_prompt = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    image = Image.open(IMAGE).convert("RGB")
    image.thumbnail((448, 448), Image.Resampling.LANCZOS)
    inputs = processor(
        text=[prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    prompt_inputs = processor(
        text=[generation_prompt],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    full_ids = inputs["input_ids"][0, inputs["attention_mask"][0].bool()]
    prompt_ids = prompt_inputs["input_ids"][0, prompt_inputs["attention_mask"][0].bool()].to("cuda")
    limit = min(len(full_ids), len(prompt_ids))
    mismatch = torch.nonzero(full_ids[:limit] != prompt_ids[:limit]).flatten()
    common = int(mismatch[0]) if len(mismatch) else limit
    labels = torch.full_like(inputs["input_ids"], -100)
    full_positions = torch.nonzero(inputs["attention_mask"][0]).flatten()
    answer_positions = full_positions[common:]
    labels[0, answer_positions] = inputs["input_ids"][0, answer_positions]
    print(json.dumps({
        "common": common,
        "prompt_tokens": len(prompt_ids),
        "full_tokens": len(full_ids),
        "answer_tokens": len(answer_positions),
        "answer_decoded": processor.tokenizer.decode(full_ids[common:].tolist()),
    }, ensure_ascii=False), flush=True)
    outputs = model(**inputs, labels=labels)
    outputs.loss.backward()
    grad_tensors = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    print(json.dumps({
        "loss": float(outputs.loss.detach().cpu()),
        "grad_tensors": len(grad_tensors),
        "grad_norm": float(torch.sqrt(sum((g.float() ** 2).sum() for g in grad_tensors)).cpu()),
        "max_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }), flush=True)


if __name__ == "__main__":
    main()
