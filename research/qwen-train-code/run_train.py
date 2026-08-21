from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import logging
import os
import platform
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from qwen_vl_utils.vision_process import process_vision_info
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC


LOGGER = logging.getLogger("ecup_multimodal")
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_ONLY = os.environ.get("QWEN_IMAGE_ONLY", "0") == "1"
INSTRUCTION = (
    "Represent the visible marketplace product and packaging for verifying whether it "
    "genuinely belongs to its declared moderation category. Focus on packaging text, "
    "product contents, purpose, and included items."
    if IMAGE_ONLY
    else
    "Represent this marketplace product card for verifying whether it genuinely belongs "
    "to its declared moderation category. Use the description, packaging text, product "
    "contents, purpose, and every supplied image."
)
CATEGORY_HINTS = {
    "БАД": (
        "Rule: confirmation requires an explicit БАД or dietary supplement marking in the "
        "description or image. Sports nutrition or absence of that marking means the declared "
        "category is not confirmed."
    ),
    "Легковоспламеняющиеся": (
        "Rule: confirm a standalone ignition source, combustible substance or gas, or a "
        "flammable item included in the kit. Empty equipment, an integrated ignition source, "
        "or combustible material used only as a component does not confirm the category."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/s3_data/data.csv")
    parser.add_argument("--images-zip", default="/s3_images/images.zip")
    parser.add_argument("--model", default="/hf_models")
    parser.add_argument("--work-dir", default="/work/ecup")
    parser.add_argument("--artifacts", default="/work/artifacts")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-pixels", type=int, default=261120)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--omit-product-text", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_from_env(variable: str, destination: Path) -> None:
    url = os.environ.get(variable)
    if not url:
        raise RuntimeError(f"missing required environment variable {variable}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    LOGGER.info("downloading %s to %s", variable, destination)
    started = time.monotonic()
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        while chunk := response.read(16 * 1024 * 1024):
            output.write(chunk)
    temporary.replace(destination)
    LOGGER.info(
        "downloaded %s: %.2f GiB in %.1f min",
        variable,
        destination.stat().st_size / 1024**3,
        (time.monotonic() - started) / 60,
    )


def safe_extract(zip_path: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        for info in members:
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe ZIP path: {info.filename}")
        LOGGER.info("extracting %d ZIP members to %s", len(members), destination)
        archive.extractall(destination)
    return {
        "members": len(members),
        "uncompressed_bytes": int(sum(item.file_size for item in members)),
    }


def locate_images_root(extracted: Path, product_ids: set[str]) -> Path:
    candidates = [extracted, extracted / "images"]
    candidates.extend(path for path in extracted.iterdir() if path.is_dir())
    scored = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        score = sum((candidate / product_id).is_dir() for product_id in list(product_ids)[:500])
        scored.append((score, candidate))
    if not scored or max(scored, key=lambda pair: pair[0])[0] == 0:
        raise ValueError("could not locate product-id image directories after ZIP extraction")
    score, root = max(scored, key=lambda pair: pair[0])
    LOGGER.info("selected images root %s (sample matches=%d)", root, score)
    return root


def normalize_id(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def prepare_frame(
    data_path: Path,
    images_root: Path,
    max_images: int = 0,
    omit_product_text: bool = False,
) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    required = {"id", "name", "description", "category", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing CSV columns: {sorted(required - set(frame.columns))}")
    if frame["id"].duplicated().any():
        raise ValueError("training CSV contains duplicate ids")
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["category"] = frame["category"].astype(str)
    frame["label"] = frame["label"].astype(np.int8)
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("label must be binary")

    image_paths: list[list[str]] = []
    texts: list[str] = []
    for row in frame.itertuples(index=False):
        product_id = normalize_id(row.id)
        folder = images_root / product_id
        paths = []
        if folder.is_dir():
            paths = [
                str(path)
                for path in sorted(folder.iterdir())
                if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
            ]
        if max_images < 0:
            paths = []
        elif max_images > 0:
            paths = paths[:max_images]
        image_paths.append(paths)
        hint = CATEGORY_HINTS.get(str(row.category), "")
        if omit_product_text:
            texts.append(f"Declared category: {row.category}\n{hint}")
        else:
            texts.append(
                f"Declared category: {row.category}\n"
                f"Product name: {row.name}\n"
                f"Description: {row.description}\n"
                f"{hint}"
            )
    frame["text"] = texts
    frame["image_paths"] = image_paths
    return frame


def load_official_embedder(model_path: Path, max_pixels: int):
    script_path = model_path / "scripts" / "qwen3_vl_embedding.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"official Qwen embedding script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("qwen3_vl_embedding_official", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kwargs = {"torch_dtype": torch.bfloat16, "default_instruction": INSTRUCTION, "max_pixels": max_pixels}
    try:
        embedder = module.Qwen3VLEmbedder(
            model_name_or_path=str(model_path), attn_implementation="flash_attention_2", **kwargs
        )
        attention = "flash_attention_2"
    except Exception as error:
        LOGGER.warning("flash_attention_2 unavailable (%s); falling back to eager attention", error)
        torch.cuda.empty_cache()
        embedder = module.Qwen3VLEmbedder(model_name_or_path=str(model_path), **kwargs)
        attention = "eager"
    return embedder, attention


def conversation(text: str, image_paths: list[str], max_pixels: int) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for path in image_paths:
        content.append(
            {
                "type": "image",
                "image": "file://" + str(Path(path).resolve()),
                "min_pixels": 4096,
                "max_pixels": max_pixels,
            }
        )
    content.append({"type": "text", "text": text or "NULL"})
    return [
        {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
        {"role": "user", "content": content},
    ]


def preprocess(embedder, conversations: list[list[dict[str, Any]]]) -> dict[str, torch.Tensor]:
    prompt_texts = embedder.processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False
    )
    images, video_inputs, video_kwargs = process_vision_info(
        conversations,
        image_patch_size=16,
        return_video_metadata=True,
        return_video_kwargs=True,
    )
    if video_inputs is not None:
        videos, video_metadata = zip(*video_inputs)
        videos, video_metadata = list(videos), list(video_metadata)
    else:
        videos, video_metadata = None, None
    inputs = embedder.processor(
        text=prompt_texts,
        images=images,
        videos=videos,
        video_metadata=video_metadata,
        truncation=True,
        max_length=8192,
        padding=True,
        do_resize=False,
        return_tensors="pt",
        **video_kwargs,
    )
    return {key: value.to(embedder.model.device) for key, value in inputs.items()}


@torch.inference_mode()
def embed_conversations(embedder, conversations: list[list[dict[str, Any]]]) -> np.ndarray:
    inputs = preprocess(embedder, conversations)
    outputs = embedder.forward(inputs)
    embeddings = embedder._pooling_last(outputs["last_hidden_state"], outputs["attention_mask"])
    embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
    return embeddings.cpu().numpy().astype(np.float32, copy=False)


def compute_embeddings(embedder, frame: pd.DataFrame, batch_size: int, max_pixels: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    start = 0
    active_batch = batch_size
    started = time.monotonic()
    while start < len(frame):
        end = min(start + active_batch, len(frame))
        batch = frame.iloc[start:end]
        conversations = [
            conversation(row.text, row.image_paths, max_pixels)
            for row in batch.itertuples(index=False)
        ]
        try:
            result = embed_conversations(embedder, conversations)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if active_batch == 1:
                LOGGER.warning("single multimodal sample OOM at row %d; using text only", start)
                row = batch.iloc[0]
                result = embed_conversations(embedder, [conversation(row.text, [], max_pixels)])
            else:
                active_batch = max(1, active_batch // 2)
                LOGGER.warning("OOM; retrying row %d with batch_size=%d", start, active_batch)
                continue
        rows.append(result)
        start = end
        if active_batch < batch_size:
            active_batch = min(batch_size, active_batch * 2)
        if start % 240 < len(result) or start == len(frame):
            elapsed = time.monotonic() - started
            LOGGER.info("embedded %d/%d rows in %.1f min", start, len(frame), elapsed / 60)
    return np.concatenate(rows, axis=0)


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.quantile(scores, np.linspace(0.005, 0.995, 500)))
    best = (-1.0, 0.0)
    for threshold in candidates:
        value = f1_score(labels, scores >= threshold)
        if value > best[0]:
            best = (float(value), float(threshold))
    return best[1], best[0]


def train_classifier(frame: pd.DataFrame, embeddings: np.ndarray, seed: int):
    models: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    oof_scores = np.full(len(frame), np.nan, dtype=np.float32)
    oof_predictions = np.zeros(len(frame), dtype=np.int8)
    c_values = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)

    for category in sorted(frame["category"].unique()):
        positions = np.flatnonzero(frame["category"].to_numpy() == category)
        labels = frame.iloc[positions]["label"].to_numpy(dtype=np.int8)
        features = embeddings[positions].astype(np.float32, copy=False)
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        best: tuple[float, float, float, np.ndarray] | None = None
        for c_value in c_values:
            scores = np.empty(len(positions), dtype=np.float32)
            for train_idx, valid_idx in folds.split(features, labels):
                model = LinearSVC(
                    C=c_value,
                    class_weight="balanced",
                    dual=False,
                    max_iter=6000,
                    random_state=seed,
                )
                model.fit(features[train_idx], labels[train_idx])
                scores[valid_idx] = model.decision_function(features[valid_idx])
            threshold, score = best_threshold(labels, scores)
            if best is None or score > best[0]:
                best = (score, threshold, c_value, scores.copy())
        assert best is not None
        score, threshold, c_value, scores = best
        final_model = LinearSVC(
            C=c_value,
            class_weight="balanced",
            dual=False,
            max_iter=6000,
            random_state=seed,
        )
        final_model.fit(features, labels)
        models[category] = final_model
        oof_scores[positions] = scores
        oof_predictions[positions] = (scores >= threshold).astype(np.int8)
        metrics[category] = {
            "rows": int(len(positions)),
            "positive_rate": float(labels.mean()),
            "C": c_value,
            "threshold": threshold,
            "oof_f1": score,
        }
        LOGGER.info("category=%s C=%s threshold=%.6f OOF_F1=%.6f", category, c_value, threshold, score)

    macro = float(np.mean([entry["oof_f1"] for entry in metrics.values()]))
    bundle = {
        "version": 1,
        "instruction": INSTRUCTION,
        "category_hints": CATEGORY_HINTS,
        "models": models,
        "thresholds": {category: entry["threshold"] for category, entry in metrics.items()},
        "embedding_dim": int(embeddings.shape[1]),
        "normalization": "l2",
    }
    return bundle, metrics, macro, oof_scores, oof_predictions


def artifact_size(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the embedding run")
    data_path = Path(args.data)
    zip_path = Path(args.images_zip)
    model_path = Path(args.model)
    work_dir = Path(args.work_dir)
    artifacts = Path(args.artifacts)
    extracted = work_dir / "extracted"
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)

    if not data_path.is_file():
        download_from_env("DATA_URL", data_path)
    if args.max_images >= 0 and not zip_path.is_file():
        download_from_env("IMAGES_URL", zip_path)
    expected_data_bytes = int(os.environ.get("EXPECTED_DATA_BYTES", "0"))
    expected_zip_bytes = int(os.environ.get("EXPECTED_IMAGES_BYTES", "0"))
    if expected_data_bytes and data_path.stat().st_size != expected_data_bytes:
        raise ValueError(f"data size mismatch: {data_path.stat().st_size} != {expected_data_bytes}")
    if args.max_images >= 0 and expected_zip_bytes and zip_path.stat().st_size != expected_zip_bytes:
        raise ValueError(f"ZIP size mismatch: {zip_path.stat().st_size} != {expected_zip_bytes}")

    if args.max_images < 0:
        zip_report = {"members": 0, "uncompressed_bytes": 0, "skipped": True}
        images_root = work_dir / "no_images"
        images_root.mkdir(parents=True, exist_ok=True)
    else:
        raw_frame = pd.read_csv(data_path, usecols=["id"])
        product_ids = {normalize_id(value) for value in raw_frame["id"]}
        zip_report = safe_extract(zip_path, extracted)
        images_root = locate_images_root(extracted, product_ids)
    frame = prepare_frame(
        data_path,
        images_root,
        max_images=args.max_images,
        omit_product_text=args.omit_product_text,
    )
    image_counts = frame["image_paths"].map(len)
    LOGGER.info(
        "prepared rows=%d products_with_images=%d image_files=%d",
        len(frame), int((image_counts > 0).sum()), int(image_counts.sum()),
    )

    embedder, attention = load_official_embedder(model_path, args.max_pixels)
    embeddings = compute_embeddings(embedder, frame, args.batch_size, args.max_pixels)
    if embeddings.shape[0] != len(frame) or not np.isfinite(embeddings).all():
        raise ValueError(f"invalid embeddings: shape={embeddings.shape}")
    del embedder
    gc.collect()
    torch.cuda.empty_cache()

    bundle, category_metrics, macro_f1, oof_scores, oof_predictions = train_classifier(
        frame, embeddings, args.seed
    )
    joblib.dump(bundle, artifacts / "multimodal_classifier.joblib", compress=3)
    np.savez_compressed(
        artifacts / "train_embeddings_fp16.npz",
        ids=frame["id"].to_numpy(),
        embeddings=embeddings.astype(np.float16),
        labels=frame["label"].to_numpy(dtype=np.int8),
        categories=frame["category"].to_numpy(dtype=str),
    )
    pd.DataFrame(
        {
            "id": frame["id"],
            "category": frame["category"],
            "label": frame["label"],
            "image_count": image_counts,
            "oof_score": oof_scores,
            "oof_prediction": oof_predictions,
        }
    ).to_csv(artifacts / "oof_predictions.csv", index=False)

    report = {
        "status": "succeeded",
        "macro_category_oof_f1": macro_f1,
        "categories": category_metrics,
        "inputs": {
            "data_bytes": data_path.stat().st_size,
            "data_sha256": sha256(data_path),
            "images_zip_bytes": zip_path.stat().st_size if zip_path.is_file() else 0,
            "zip": zip_report,
            "rows": len(frame),
            "products_with_images": int((image_counts > 0).sum()),
            "image_files": int(image_counts.sum()),
        },
        "embedding": {
            "model_path": str(model_path),
            "shape": list(embeddings.shape),
            "dtype_saved": "float16",
            "instruction": INSTRUCTION,
            "max_pixels_per_image": args.max_pixels,
            "requested_batch_size": args.batch_size,
            "attention": attention,
            "max_images": args.max_images,
            "omit_product_text": args.omit_product_text,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    (artifacts / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = artifact_size(artifacts)
    LOGGER.info("artifact bytes=%d (%.2f MiB)", total, total / 1024**2)
    if total > 95 * 1024**2:
        raise RuntimeError(f"artifacts exceed safe private compute platform limit: {total} bytes")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
