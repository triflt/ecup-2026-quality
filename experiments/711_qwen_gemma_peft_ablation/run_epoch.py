from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


ARCHITECTURES = ("qwen35_4b", "gemma4_e4b")
METHODS = ("dora", "rsdora", "rspissa", "rsloraplus")


def load(path: Path):
    spec = importlib.util.spec_from_file_location("exp711_parent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cache_filename(item_id: str) -> str:
    return hashlib.sha256(item_id.encode()).hexdigest() + ".img"


def configure_lora_kwargs(method: str, kwargs: dict) -> dict:
    if kwargs.get("r") != 16 or kwargs.get("lora_alpha") != 32:
        raise ValueError("parent LoRA contract drifted")
    result = dict(kwargs)
    if result.get("use_rslora") is not True:
        raise ValueError("accepted rsLoRA control is required")
    if method == "dora":
        result["use_rslora"] = False
        result["use_dora"] = True
    elif method == "rsdora":
        result["use_dora"] = True
    elif method == "rspissa":
        result["init_lora_weights"] = "pissa_niter_4"
    elif method == "rsloraplus":
        pass
    else:
        raise ValueError(f"unknown PEFT method: {method}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--epochs", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()

    parent = load(args.parent)
    parent.EPOCHS = args.epochs
    parent.IMAGE_DIR = args.cache
    technical_smoke = os.environ.get("TECHNICAL_SMOKE") == "1"
    if technical_smoke:
        parent.EPOCHS = 1
        parent.FULL_TRAIN = True
        parent.CHECKPOINT_EACH_EPOCH = False
        original_select_training = parent.select_training

        def smoke_training(frame, oof):
            selected = original_select_training(frame, oof)
            if len(selected) < parent.BATCH_SIZE:
                raise ValueError("not enough records for technical smoke")
            return selected[: parent.BATCH_SIZE]

        parent.select_training = smoke_training

    model_holder: dict[str, object] = {}

    def use_accepted_vendor():
        vendor = Path(os.environ["ECUP_VENDOR"])
        sys.path.insert(0, str(vendor))
        import peft
        import torch

        if peft.__version__ != "0.20.0":
            raise ValueError("accepted PEFT 0.20.0 required")
        original_config = peft.LoraConfig
        original_get_peft_model = peft.get_peft_model

        def bound_lora_config(*values, **kwargs):
            return original_config(*values, **configure_lora_kwargs(args.method, kwargs))

        def bound_get_peft_model(*values, **kwargs):
            model = original_get_peft_model(*values, **kwargs)
            model_holder["model"] = model
            return model

        peft.LoraConfig = bound_lora_config
        peft.get_peft_model = bound_get_peft_model

        if args.method == "rsloraplus":
            from peft.optimizers import create_loraplus_optimizer

            original_adamw = torch.optim.AdamW

            def loraplus_adamw(_parameters, *, lr, weight_decay, **kwargs):
                if "model" not in model_holder:
                    raise RuntimeError("LoRA+ optimizer requested before PEFT model creation")
                return create_loraplus_optimizer(
                    model_holder["model"],
                    original_adamw,
                    lr=lr,
                    loraplus_lr_ratio=16.0,
                    weight_decay=weight_decay,
                    **kwargs,
                )

            torch.optim.AdamW = loraplus_adamw

    parent.install_peft = use_accepted_vendor

    def accepted_cache(ids, _urls):
        def source_for(item_id):
            direct = args.cache / f"{item_id}.jpg"
            if direct.exists():
                return direct
            return args.cache / cache_filename(str(item_id))

        missing = [str(item_id) for item_id in ids if not source_for(item_id).exists()]
        if missing:
            raise FileNotFoundError(
                f"accepted cache missing {len(missing)} rows; first={missing[:3]}"
            )
        view = Path(os.environ["EXP711_IMAGE_VIEW"])
        view.mkdir(parents=True, exist_ok=True)
        for item_id in ids:
            source = source_for(item_id)
            target = view / f"{item_id}.jpg"
            if not target.exists():
                target.symlink_to(source)
        parent.IMAGE_DIR = view
        return []

    parent.predownload = accepted_cache
    contract = {
        "schema": "exp711_qwen_gemma_peft_ablation_v1",
        "experiment": 711,
        "architecture": args.architecture,
        "method": args.method,
        "epochs": parent.EPOCHS,
        "lora_r": 16,
        "lora_alpha": 32,
        "loraplus_lr_ratio": 16.0 if args.method == "rsloraplus" else None,
        "fold": int(os.environ.get("HOLDOUT_FOLD", "0")),
        "technical_smoke": technical_smoke,
        "public_used": False,
    }
    output = Path(os.environ["ECUP_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "method_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(contract, ensure_ascii=False), flush=True)
    parent.main()


if __name__ == "__main__":
    main()
