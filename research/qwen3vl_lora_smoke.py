from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL = Path("/hf_models")


def main():
    try:
        import peft
        peft_version = peft.__version__
    except Exception as error:
        peft_version = f"unavailable: {error}"
    print(json.dumps({
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft_version,
        "cuda": torch.cuda.is_available(),
    }), flush=True)
    started = time.monotonic()
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    ).to("cuda").eval()
    print(json.dumps({
        "loaded_sec": time.monotonic() - started,
        "model_class": model.__class__.__name__,
        "processor_class": processor.__class__.__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }), flush=True)


if __name__ == "__main__":
    main()
