from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL = Path("/shared_models/PaddlePaddle/PaddleOCR-VL-1.5")
IMAGES = Path("/work/input/images")


def load_model():
    processor = AutoProcessor.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    ).to("cuda").eval()
    dynamic_module = importlib.import_module(model.__class__.__module__)
    original_create_causal_mask = dynamic_module.create_causal_mask

    def compatible_create_causal_mask(*args, **kwargs):
        if "inputs_embeds" in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return original_create_causal_mask(*args, **kwargs)

    dynamic_module.create_causal_mask = compatible_create_causal_mask
    return processor, model


def messages_for(image: Image.Image):
    return [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "OCR:"},
    ]}]


def main():
    print(json.dumps({
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }), flush=True)
    started = time.monotonic()
    processor, model = load_model()
    loaded = time.monotonic()
    paths = sorted(IMAGES.rglob("*.jpg"))[:4]
    images = [Image.open(path).convert("RGB") for path in paths]
    conversations = [messages_for(image) for image in images]
    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        images_kwargs={"size": {
            "shortest_edge": processor.image_processor.min_pixels,
            "longest_edge": 1280 * 28 * 28,
        }},
    ).to(model.device)
    prepared = time.monotonic()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    generated = time.monotonic()
    prompt_length = inputs["input_ids"].shape[-1]
    results = [
        processor.decode(output[prompt_length:-1])
        for output in outputs
    ]
    print(json.dumps({
        "paths": [str(path) for path in paths],
        "ocr": results,
        "load_sec": loaded - started,
        "prepare_sec": prepared - loaded,
        "generate_sec": generated - prepared,
        "total_sec": generated - started,
        "per_image_generate_sec": (generated - prepared) / len(paths),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
