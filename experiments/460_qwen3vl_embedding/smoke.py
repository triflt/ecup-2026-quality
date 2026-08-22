from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

MODEL = Path(os.environ.get("MODEL_PATH", "/hf_models"))
IMAGE = Path(os.environ.get("IMAGE_PATH", "/work/input/0.jpg"))
OUTPUT = Path(os.environ.get("OUTPUT_DIR", "/work/output"))
VENDOR = Path(os.environ.get("VENDOR_PATH", "/work/vendor"))


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


def main() -> None:
    started = time.monotonic()
    install_runtime()
    from sentence_transformers import SentenceTransformer

    load_started = time.monotonic()
    model = SentenceTransformer(
        str(MODEL),
        device="cuda",
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    load_seconds = time.monotonic() - load_started
    instruction = (
        "Represent this marketplace product for deciding whether the sold item "
        "itself or an included item is flammable."
    )
    item = {
        "text": "Газовая горелка для туризма. Баллон в комплект не входит.",
        "image": str(IMAGE),
    }
    altered = {
        "text": item["text"],
    }
    inference_started = time.monotonic()
    embeddings = model.encode(
        [item, item, altered],
        batch_size=3,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        prompt=instruction,
    ).astype(np.float32)
    inference_seconds = time.monotonic() - inference_started
    raw_norms = np.linalg.norm(embeddings, axis=1)
    embeddings /= np.maximum(raw_norms[:, None], 1e-12)
    norms = np.linalg.norm(embeddings, axis=1)
    repeat_cosine = float(
        np.dot(embeddings[0], embeddings[1]) / (norms[0] * norms[1])
    )
    image_ablation_cosine = float(
        np.dot(embeddings[0], embeddings[2]) / (norms[0] * norms[2])
    )
    checks = {
        "shape_is_3_by_2048": embeddings.shape == (3, 2048),
        "finite": bool(np.isfinite(embeddings).all()),
        "unit_norm": bool(np.max(np.abs(norms - 1.0)) < 1e-3),
        "repeat_cosine_at_least_0_9999": repeat_cosine >= 0.9999,
        "image_changes_embedding": image_ablation_cosine < 0.9999,
    }
    report = {
        "experiment_id": "460",
        "model": "Qwen/Qwen3-VL-Embedding-2B",
        "model_path": str(MODEL),
        "embedding_shape": list(embeddings.shape),
        "raw_bfloat16_norms": raw_norms.tolist(),
        "norms": norms.tolist(),
        "repeat_cosine": repeat_cosine,
        "image_ablation_cosine": image_ablation_cosine,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "checks": checks,
        "passed": all(checks.values()),
        "runtime_seconds": time.monotonic() - started,
        "versions": {
            "torch": torch.__version__,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
