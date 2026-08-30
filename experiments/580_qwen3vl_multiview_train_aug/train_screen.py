from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from collections.abc import MutableMapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gallery_augmentation import OccurrencePlanner, load_gallery_urls, parent_448_jpeg
from multiview_contract import (
    FIRST_IMAGE_MAX_EDGE,
    FIRST_IMAGE_MAX_PIXELS,
    INFERENCE_VIEW_POLICY,
    SCREEN_FOLDS,
    SEED,
    TRAINING_VIEW_POLICY,
)
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PARENT_RUNNER = ROOT / "research/qwen3vl_lora_holdout.py"
DEFAULT_GALLERY_MANIFEST = Path(
    os.environ.get("QWEN3VL_GALLERY_MANIFEST", "/work/input/multi_image_manifest.tsv.gz")
)

LOCKED_ENVIRONMENT = {
    "SEED": "42",
    "TRAINING_MODE": "hard",
    "MODEL_CLASS": "image_text",
    "DESCRIPTION_LIMIT": "1800",
    "QWEN3VL_FIRST_IMAGE_MAX_EDGE": str(FIRST_IMAGE_MAX_EDGE),
    "QWEN3VL_FIRST_IMAGE_MAX_PIXELS": str(FIRST_IMAGE_MAX_PIXELS),
}
LOCKED_DISABLED_ENVIRONMENT = {
    "FULL_TRAIN": ("", "0", "false"),
    "SOFT_TARGETS": ("",),
    "DOWNSAMPLE_MODE": ("",),
    "MAX_SLICE_NUMS": ("", "0"),
    "LAST_LOGIT_ONLY": ("", "0", "false"),
    "USE_CHAT_BATCH": ("", "0", "false"),
    "LINEAR_ONLY_TARGETS": ("", "0", "false"),
}


def configure_environment(
    fold: int, environment: MutableMapping[str, str]
) -> MutableMapping[str, str]:
    if fold not in SCREEN_FOLDS:
        raise ValueError(f"fold {fold} is not in the predeclared screen {SCREEN_FOLDS}")
    existing_fold = environment.get("HOLDOUT_FOLD")
    if existing_fold not in (None, "", str(fold)):
        raise ValueError("HOLDOUT_FOLD conflicts with the explicit locked fold")
    for key, expected in LOCKED_ENVIRONMENT.items():
        current = environment.get(key)
        if current not in (None, "", expected):
            raise ValueError(f"{key} must remain at the parent value {expected}")
        environment[key] = expected
    enabled = [
        key
        for key, disabled_values in LOCKED_DISABLED_ENVIRONMENT.items()
        if environment.get(key, "").strip().lower() not in disabled_values
    ]
    if enabled:
        raise ValueError(f"non-parent switches are forbidden: {sorted(enabled)}")
    environment["HOLDOUT_FOLD"] = str(fold)
    return environment


def _load_parent():
    spec = importlib.util.spec_from_file_location("_exp580_parent_qwen3vl", PARENT_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load parent runner: {PARENT_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GalleryRuntime:
    def __init__(self, parent, galleries: dict[str, tuple[str, ...]]) -> None:
        self.parent = parent
        self.galleries = galleries
        self.planner = OccurrencePlanner(
            {item_id: len(urls) for item_id, urls in galleries.items()}, seed=SEED
        )
        self.training_ids: set[str] = set()
        self.expected_training_occurrences = 0
        self.downloaded_images = 0
        self.download_failures = 0

    def select_training(self, frame, oof):
        records = self.parent_select_training(frame, oof)
        ids = frame["id"].astype(str).to_numpy()
        self.training_ids = {str(ids[position]) for position in set(records)}
        self.expected_training_occurrences = len(records)
        return records

    def _download_one(self, item_id: str, image_index: int, url: str):
        destination = self.parent.IMAGE_DIR / f"{item_id}_{image_index}.jpg"
        last_error: Exception | None = None
        for _ in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = response.read()
                rendered = parent_448_jpeg(payload)
                with Image.open(io.BytesIO(rendered)) as check:
                    check.load()
                destination.write_bytes(rendered)
                return item_id, image_index, True, ""
            except Exception as error:  # noqa: BLE001 - fail closed on every transfer/decode error
                last_error = error
        return item_id, image_index, False, type(last_error).__name__

    def predownload(self, ids, urls):
        self.parent.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        tasks = [
            (str(item_id), image_index, url)
            for item_id in ids
            for image_index, url in enumerate(urls[str(item_id)])
            if str(item_id) in self.training_ids or image_index == 0
        ]
        failures: list[tuple[str, int, str]] = []
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = [pool.submit(self._download_one, *task) for task in tasks]
            for index, future in enumerate(as_completed(futures), 1):
                item_id, image_index, ok, error_type = future.result()
                if not ok:
                    failures.append((item_id, image_index, error_type))
                if index % 500 == 0 or index == len(futures):
                    print(
                        json.dumps(
                            {
                                "downloaded": index,
                                "expected": len(futures),
                                "failures": len(failures),
                                "elapsed_minutes": (time.monotonic() - started) / 60,
                            }
                        ),
                        flush=True,
                    )
        self.downloaded_images = len(tasks) - len(failures)
        self.download_failures = len(failures)
        if failures:
            sanitized = [
                {"id": item_id, "image_index": index, "error_type": error_type}
                for item_id, index, error_type in failures[:20]
            ]
            raise RuntimeError(
                "gallery download/decode failed closed: "
                + json.dumps({"count": len(failures), "examples": sanitized})
            )
        return []

    def open_images(self, rows):
        images = []
        for row in rows:
            item_id = str(row.id)
            image_index = self.planner.choose(item_id)
            images.append(
                Image.open(self.parent.IMAGE_DIR / f"{item_id}_{image_index}.jpg").convert("RGB")
            )
        return images

    def validation_scores(self, *args, **kwargs):
        self.planner.training = False
        return self.parent_validation_scores(*args, **kwargs)

    def report(self) -> dict[str, object]:
        plan = self.planner.report()
        if plan["training_occurrences"] != self.expected_training_occurrences:
            raise RuntimeError("training occurrence plan did not match the parent steps")
        counts = Counter(len(self.galleries[item_id]) for item_id in self.training_ids)
        return {
            "training_view_policy": TRAINING_VIEW_POLICY,
            "gallery_seed": SEED,
            "gallery_epoch": 0,
            "training_images_per_occurrence": 1,
            "inference_view_policy": INFERENCE_VIEW_POLICY,
            "inference_image_index": 0,
            "inference_image_count": 1,
            "inference_passes": 1,
            "first_image_max_edge": FIRST_IMAGE_MAX_EDGE,
            "first_image_max_pixels": FIRST_IMAGE_MAX_PIXELS,
            "gallery_training_unique_rows": len(self.training_ids),
            "gallery_training_size_histogram": dict(sorted(counts.items())),
            "downloaded_gallery_images": self.downloaded_images,
            "download_failures": self.download_failures,
            "gallery_plan": plan,
        }


def install_hooks(parent, galleries: dict[str, tuple[str, ...]]) -> GalleryRuntime:
    runtime = GalleryRuntime(parent, galleries)
    runtime.parent_select_training = parent.select_training
    runtime.parent_validation_scores = parent.validation_scores
    parent.load_urls = lambda: galleries
    parent.select_training = runtime.select_training
    parent.predownload = runtime.predownload
    parent.open_images = runtime.open_images
    parent.validation_scores = runtime.validation_scores
    return runtime


def annotate_candidate_report(parent, runtime: GalleryRuntime, *, fold: int) -> Path:
    path = parent.OUTPUT / "lora_holdout_report.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("holdout_fold") != fold:
        raise ValueError("parent report fold mismatch")
    raw.update(runtime.report())
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=SCREEN_FOLDS)
    parser.add_argument("--gallery-manifest", type=Path, default=DEFAULT_GALLERY_MANIFEST)
    args = parser.parse_args(argv)
    configure_environment(args.fold, os.environ)
    if not PARENT_RUNNER.is_file():
        raise FileNotFoundError(PARENT_RUNNER)
    if not args.gallery_manifest.is_file():
        raise FileNotFoundError(args.gallery_manifest)
    galleries = load_gallery_urls(args.gallery_manifest)
    parent = _load_parent()
    forbidden_existing = [
        parent.OUTPUT / "lora_holdout_predictions.csv",
        parent.OUTPUT / "lora_holdout_report.json",
        parent.OUTPUT / "adapter.zip",
    ]
    existing = [path for path in forbidden_existing if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing training outputs")
    runtime = install_hooks(parent, galleries)
    parent.main()
    report = annotate_candidate_report(parent, runtime, fold=args.fold)
    print(json.dumps({"candidate_report": str(report), **runtime.report()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
