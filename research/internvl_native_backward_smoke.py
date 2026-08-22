from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


MODEL = Path(os.environ.get("MODEL_PATH", "/hf_models"))
IMAGE = Path("/work/input/0.jpg")
VENDOR = Path("/work/vendor/peft")
IMAGE_SIZE = 448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def install_peft() -> None:
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
            "einops==0.8.1",
            "timm==1.0.22",
        ],
        check=True,
    )
    sys.path.insert(0, str(VENDOR))


def image_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform(image).unsqueeze(0)


def query(model, tokenizer, question: str, answer: str | None) -> str:
    template = copy.deepcopy(model.conv_template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], answer)
    value = template.get_prompt()
    image_tokens = (
        "<img>" + "<IMG_CONTEXT>" * model.num_image_token + "</img>"
    )
    return value.replace("<image>", image_tokens, 1)


def main() -> None:
    install_peft()
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    started = time.monotonic()
    model = AutoModel.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
        use_flash_attn=False,
    ).to("cuda")
    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj")
    target_modules = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith("language_model.")
        and name.endswith(suffixes)
    ]
    if not target_modules:
        raise ValueError("no language-model attention projections found")
    print(
        json.dumps(
            {
                "linear_lora_targets": len(target_modules),
                "target_examples": target_modules[:8],
            }
        ),
        flush=True,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
            use_rslora=True,
        ),
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    print(
        json.dumps(
            {
                "model_class": model.__class__.__name__,
                "loaded_sec": time.monotonic() - started,
                "trainable": sum(parameter.numel() for parameter in trainable),
                "total": sum(parameter.numel() for parameter in model.parameters()),
            }
        ),
        flush=True,
    )

    question = (
        "<image>\nОпредели корректность категории товара. "
        "Ответь только цифрой 0 или 1."
    )
    prompt_text = query(model.base_model.model, tokenizer, question, None)
    full_text = query(model.base_model.model, tokenizer, question, "0")
    prompt = tokenizer(prompt_text, return_tensors="pt")
    full = tokenizer(full_text, return_tensors="pt")
    prompt_ids = prompt["input_ids"][0]
    full_ids = full["input_ids"][0]
    limit = min(len(prompt_ids), len(full_ids))
    mismatch = torch.nonzero(prompt_ids[:limit] != full_ids[:limit]).flatten()
    common = int(mismatch[0]) if len(mismatch) else limit
    if common < len(prompt_ids) - 2 or common >= len(full_ids):
        raise ValueError(
            f"assistant suffix alignment failed: common={common} "
            f"prompt={len(prompt_ids)} full={len(full_ids)}"
        )
    labels = torch.full_like(full["input_ids"], -100)
    labels[:, common:] = full["input_ids"][:, common:]
    pixel_values = image_tensor(IMAGE).to(device="cuda", dtype=torch.bfloat16)
    model.base_model.model.img_context_token_id = tokenizer.convert_tokens_to_ids(
        "<IMG_CONTEXT>"
    )
    batch = {key: value.to("cuda") for key, value in full.items()}
    output = model(
        pixel_values=pixel_values,
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        image_flags=torch.ones((1, 1), dtype=torch.long, device="cuda"),
        labels=labels.to("cuda"),
        return_dict=True,
    )
    loss = output.loss
    loss.backward()
    gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
    print(
        json.dumps(
            {
                "prompt_tokens": len(prompt_ids),
                "full_tokens": len(full_ids),
                "answer_decoded": tokenizer.decode(full_ids[common:].tolist()),
                "image_tiles": 1,
                "image_tokens_per_tile": model.base_model.model.num_image_token,
                "loss": float(loss.detach().cpu()),
                "grad_tensors": len(gradients),
                "grad_norm": float(
                    torch.sqrt(sum((gradient.float() ** 2).sum() for gradient in gradients)).cpu()
                ),
                "max_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
