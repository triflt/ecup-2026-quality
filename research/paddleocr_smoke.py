from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


MODEL = Path("/shared_models/PaddlePaddle/PaddleOCR-VL-1.5")
IMAGE = Path("/work/input/0.jpg")


def main():
    print(json.dumps({
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "model_files": sorted(p.name for p in MODEL.iterdir()),
    }), flush=True)
    started = time.monotonic()
    # PaddleOCR-VL custom code targets Transformers 4.x, while the competition image
    # contains Transformers 5.x where the default RoPE function was removed from this map.
    if "default" not in ROPE_INIT_FUNCTIONS:
        def default_rope(config, device=None, seq_len=None, **kwargs):
            del seq_len, kwargs
            dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            base = getattr(config, "rope_theta", 10_000.0)
            inv_freq = 1.0 / (base ** (
                torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim
            ))
            return inv_freq, 1.0
        ROPE_INIT_FUNCTIONS["default"] = default_rope
    print("rope_types", sorted(ROPE_INIT_FUNCTIONS), flush=True)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    config_class = get_class_from_dynamic_module(
        "configuration_paddleocr_vl.PaddleOCRVLConfig", MODEL, local_files_only=True
    )
    model_class = get_class_from_dynamic_module(
        "modeling_paddleocr_vl.PaddleOCRVLForConditionalGeneration", MODEL, local_files_only=True
    )
    # Transformers 5 tries to reinitialize non-persistent RoPE buffers that the
    # Transformers-4-targeted custom class already creates in its constructor.
    model_class._initialize_missing_keys = lambda self, is_quantized: None
    def compat_prepare_inputs(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
        cache_position=None, position_ids=None, use_cache=True, **kwargs,
    ):
        del self, position_ids
        if past_key_values is not None and cache_position is not None:
            input_ids = input_ids[:, cache_position]
        result = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "position_ids": None,
            "use_cache": use_cache,
        }
        if inputs_embeds is not None and past_key_values is None:
            result["inputs_embeds"] = inputs_embeds
            result.pop("input_ids", None)
        for key in (
            "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw",
            "second_per_grid_ts",
        ):
            if key in kwargs:
                result[key] = kwargs[key]
        if cache_position is not None and cache_position[0] != 0:
            result["pixel_values"] = None
            result["pixel_values_videos"] = None
        return result
    model_class.prepare_inputs_for_generation = compat_prepare_inputs
    config = config_class.from_pretrained(MODEL, local_files_only=True)
    config.use_flash_attention = False
    model = model_class.from_pretrained(
        MODEL, config=config, torch_dtype=torch.bfloat16, local_files_only=True
    )
    model = model.to("cuda").eval()
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
