from __future__ import annotations

import csv
import gzip
import html
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

MODEL = Path(os.environ.get("MODEL_PATH", "/hf_models"))
DATA = Path(os.environ.get("DATA_PATH", "/work/input/data.csv"))
MANIFEST = Path(os.environ.get("IMAGE_MANIFEST", "/work/input/lora_image_manifest.tsv.gz"))
IMAGE_DIR = Path(os.environ.get("IMAGE_DIR", "/work/images"))
OUTPUT = Path(os.environ.get("OUTPUT_DIR", "/work/output"))
VENDOR = Path(os.environ.get("VENDOR_PATH", "/work/vendor"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
DESCRIPTION_LIMIT = 1600
INSTRUCTION = (
    "Represent this marketplace product for deciding whether the sold item itself "
    "or an included item is a flammable substance, gas, fuel, ignition source, "
    "or pyrotechnic product."
)


def install_runtime() -> None:
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
            "sentence-transformers==5.4.0",
            "qwen-vl-utils==0.0.14",
        ],
        check=True,
    )
    sys.path.insert(0, str(VENDOR))


def load_data() -> pd.DataFrame:
    if not DATA.exists():
        urllib.request.urlretrieve(os.environ["DATA_URL"], DATA)
    frame = pd.read_csv(DATA, dtype={"id": str})
    required = {"id", "name", "description", "category", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"data columns missing: {sorted(required - set(frame.columns))}")
    if frame["id"].duplicated().any():
        raise ValueError("duplicate ids")
    return frame


def load_urls() -> dict[str, str]:
    with gzip.open(MANIFEST, "rt", encoding="utf-8", newline="") as stream:
        return {
            str(row["id"]): row["image_url"]
            for row in csv.DictReader(stream, delimiter="\t")
        }


def download_one(item_id: str, url: str) -> tuple[str, bool, str]:
    destination = IMAGE_DIR / f"{item_id}.jpg"
    last_error = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            image = Image.open(io.BytesIO(payload)).convert("RGB")
            image.thumbnail((448, 448), Image.Resampling.LANCZOS)
            image.save(destination, format="JPEG", quality=92)
            return item_id, True, ""
        except Exception as error:  # noqa: BLE001
            last_error = error
    Image.new("RGB", (32, 32), "white").save(destination, format="JPEG")
    return item_id, False, str(last_error)


def download_images(ids: np.ndarray, urls: dict[str, str]) -> list[dict[str, str]]:
    missing = sorted(set(ids) - set(urls))
    if missing:
        raise ValueError(f"image manifest misses {len(missing)} ids, examples={missing[:5]}")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(download_one, item_id, urls[item_id]) for item_id in ids]
        for index, future in enumerate(as_completed(futures), 1):
            item_id, ok, error = future.result()
            if not ok:
                failures.append({"id": item_id, "error": error})
            if index % 500 == 0 or index == len(futures):
                print(
                    f"downloaded={index}/{len(futures)} failures={len(failures)} "
                    f"elapsed_min={(time.monotonic() - started) / 60:.2f}",
                    flush=True,
                )
    return failures


def clean_text(value: str) -> str:
    return " ".join(html.unescape(str(value)).replace("<br/>", " ").split())


def main() -> None:
    started = time.monotonic()
    install_runtime()
    from sentence_transformers import SentenceTransformer

    frame = load_data()
    ids = frame["id"].astype(str).to_numpy()
    failures = download_images(ids, load_urls())
    load_started = time.monotonic()
    model = SentenceTransformer(
        str(MODEL),
        device="cuda",
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    load_seconds = time.monotonic() - load_started
    embeddings = np.empty((len(frame), 2048), dtype=np.float16)
    inference_started = time.monotonic()
    for start in range(0, len(frame), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(frame))
        items = []
        for row in frame.iloc[start:stop].itertuples(index=False):
            title = clean_text(row.name)
            description = clean_text(row.description)[:DESCRIPTION_LIMIT]
            items.append(
                {
                    "text": f"{title}\n{title}\n{description}",
                    "image": str(IMAGE_DIR / f"{row.id}.jpg"),
                }
            )
        batch = model.encode(
            items,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            prompt=INSTRUCTION,
        ).astype(np.float32)
        batch /= np.maximum(np.linalg.norm(batch, axis=1, keepdims=True), 1e-12)
        if batch.shape != (stop - start, 2048) or not np.isfinite(batch).all():
            raise ValueError(f"invalid embedding batch {start}:{stop}: {batch.shape}")
        embeddings[start:stop] = batch.astype(np.float16)
        if stop % 256 == 0 or stop == len(frame):
            print(
                f"embedded={stop}/{len(frame)} "
                f"elapsed_min={(time.monotonic() - inference_started) / 60:.2f} "
                f"peak_gib={torch.cuda.max_memory_allocated() / 1024**3:.2f}",
                flush=True,
            )

    stored_norms = np.linalg.norm(embeddings.astype(np.float32), axis=1)
    report = {
        "experiment_id": "460",
        "model": "Qwen/Qwen3-VL-Embedding-2B",
        "rows": len(frame),
        "dimensions": 2048,
        "batch_size": BATCH_SIZE,
        "description_limit": DESCRIPTION_LIMIT,
        "image_size": "within 448x448",
        "download_failures": len(failures),
        "failure_examples": failures[:20],
        "stored_norm_min": float(stored_norms.min()),
        "stored_norm_max": float(stored_norms.max()),
        "load_seconds": load_seconds,
        "inference_minutes": (time.monotonic() - inference_started) / 60,
        "runtime_minutes": (time.monotonic() - started) / 60,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "passed": bool(len(failures) == 0 and np.isfinite(embeddings).all()),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "qwen3vl_embedding_train_fp16.npz",
        ids=ids,
        labels=frame["label"].to_numpy(np.int8),
        categories=frame["category"].astype(str).to_numpy(dtype=str),
        embeddings=embeddings,
    )
    (OUTPUT / "extraction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
