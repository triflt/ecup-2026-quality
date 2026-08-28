from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


def load(path: Path):
    spec = importlib.util.spec_from_file_location("exp701_parent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cache_filename(item_id: str, architecture: str) -> str:
    suffix = ".jpg" if architecture == "qwen3vl_2b" else ".img"
    return hashlib.sha256(item_id.encode()).hexdigest() + suffix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--architecture", choices=("qwen35_4b", "qwen3vl_2b"), required=True)
    parser.add_argument("--epochs", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    parent = load(args.parent)
    parent.EPOCHS = args.epochs
    parent.IMAGE_DIR = args.cache

    def use_accepted_vendor():
        vendor = Path(os.environ["ECUP_VENDOR"])
        sys.path.insert(0, str(vendor))
        import peft
        if peft.__version__ != "0.20.0":
            raise ValueError("accepted PEFT 0.20.0 required")

    parent.install_peft = use_accepted_vendor

    def accepted_cache(ids, _urls):
        def source_for(item_id):
            direct = args.cache / f"{item_id}.jpg"
            if direct.exists():
                return direct
            return args.cache / cache_filename(str(item_id), args.architecture)

        missing = [str(item_id) for item_id in ids if not source_for(item_id).exists()]
        if missing:
            raise FileNotFoundError(f"accepted cache missing {len(missing)} rows")
        # The parent expects id.jpg; expose deterministic links in a per-arm view.
        view = Path(os.environ["EXP701_IMAGE_VIEW"])
        view.mkdir(parents=True, exist_ok=True)
        for item_id in ids:
            source = source_for(item_id)
            target = view / f"{item_id}.jpg"
            if not target.exists():
                target.symlink_to(source)
        parent.IMAGE_DIR = view
        return []

    parent.predownload = accepted_cache
    print(json.dumps({"experiment":701,"architecture":args.architecture,"epochs":args.epochs,"fold":int(os.environ.get("HOLDOUT_FOLD","0")),"public_used":False}), flush=True)
    parent.main()


if __name__ == "__main__":
    main()
