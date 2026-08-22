from __future__ import annotations

import json
import importlib
import time
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


MODEL = Path("/shared_models/PaddlePaddle/PaddleOCR-VL-1.5")
IMAGE = Path("/work/input/0.jpg")


def main():
    print(json.dumps({
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }), flush=True)
    started = time.monotonic()
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True
    ).to("cuda").eval()
    dynamic_module = importlib.import_module(model.__class__.__module__)
    original_create_causal_mask = dynamic_module.create_causal_mask
    def compatible_create_causal_mask(*args, **kwargs):
        if "inputs_embeds" in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return original_create_causal_mask(*args, **kwargs)
    dynamic_module.create_causal_mask = compatible_create_causal_mask
    print(f"loaded_sec={time.monotonic()-started:.2f}", flush=True)
    image = Image.open(IMAGE).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "OCR:"},
    ]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images_kwargs={"size": {
            "shortest_edge": processor.image_processor.min_pixels,
            "longest_edge": 1280 * 28 * 28,
        }},
    ).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    result = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])
    print(json.dumps({"ocr": result, "total_sec": time.monotonic()-started}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
