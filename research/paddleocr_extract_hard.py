from __future__ import annotations

import csv
import importlib
import io
import json
import os
import time
import urllib.request
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL = Path(os.environ.get("ECUP_MODEL_ROOT", "/shared_models/PaddlePaddle/PaddleOCR-VL-1.5"))
MANIFEST = Path(os.environ.get("ECUP_MANIFEST", "/work/input/ocr_manifest.jsonl"))
OUTPUT = Path(os.environ.get("ECUP_REPORT", "/work/output/ocr_hard_cases.csv"))
BATCH_SIZE = 1
MAX_NEW_TOKENS = 64


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


def download_image(url: str) -> Image.Image:
    last_error = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return Image.open(io.BytesIO(response.read())).convert("RGB")
        except Exception as error:
            last_error = error
            time.sleep(1)
    raise RuntimeError(f"image download failed: {last_error}")


def messages_for(image: Image.Image):
    return [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "OCR:"},
    ]}]


def infer_batch(processor, model, records):
    images = [download_image(record["image_url"]) for record in records]
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
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
    prompt_length = inputs["input_ids"].shape[-1]
    return [
        processor.decode(output[prompt_length:], skip_special_tokens=True).strip()
        for output in outputs
    ]


def main():
    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError("empty OCR manifest")
    started = time.monotonic()
    processor, model = load_model()
    loaded = time.monotonic()
    rows = []
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        texts = infer_batch(processor, model, batch)
        for record, text in zip(batch, texts):
            clean = {key: value for key, value in record.items() if key != "image_url"}
            clean["ocr"] = text
            rows.append(clean)
        if len(rows) % 20 == 0 or len(rows) == len(records):
            elapsed = time.monotonic() - loaded
            print(
                f"ocr={len(rows)}/{len(records)} elapsed_min={elapsed/60:.1f} "
                f"sec_per_image={elapsed/len(rows):.2f}",
                flush=True,
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "rows": len(rows),
        "load_sec": loaded - started,
        "inference_sec": time.monotonic() - loaded,
        "max_new_tokens": MAX_NEW_TOKENS,
        "batch_size": BATCH_SIZE,
    }
    (OUTPUT.parent / "ocr_timing.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
