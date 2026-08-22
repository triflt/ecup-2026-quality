from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from qwen_vl_utils.vision_process import process_vision_info


ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.environ.get(
        "QWEN_EMBED_MODEL_PATH",
        os.path.join(
            os.environ.get("SHARED_MODELS_PATH", "/shared_models"),
            "Qwen/Qwen3-VL-Embedding-2B",
        ),
    )
)
CLASSIFIER_PATH = ROOT / "multimodal_classifier.joblib"
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_PIXELS = int(os.environ.get("QWEN_MAX_PIXELS", "261120"))
BATCH_SIZE = int(os.environ.get("QWEN_EMBED_BATCH_SIZE", "24"))
INSTRUCTION = (
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
OUTPUT_RE = re.compile(r"^<комментарий>(.{50,300})<вердикт>(бан|не бан)$", re.S)


def normalize_id(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def prepare_frame(data_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    required = {"id", "name", "description", "category"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing CSV columns: {sorted(required - set(frame.columns))}")
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["category"] = frame["category"].astype(str)
    images_root = data_path.parent / "images"
    image_paths, texts = [], []
    for row in frame.itertuples(index=False):
        folder = images_root / normalize_id(row.id)
        paths = []
        if folder.is_dir():
            paths = [
                str(path)
                for path in sorted(folder.iterdir())
                if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
            ]
        image_paths.append(paths)
        texts.append(
            f"Declared category: {row.category}\n"
            f"Product name: {row.name}\n"
            f"Description: {row.description}\n"
            f"{CATEGORY_HINTS.get(str(row.category), '')}"
        )
    frame["text"] = texts
    frame["image_paths"] = image_paths
    return frame


def load_embedder(model_path: Path):
    script_path = model_path / "scripts" / "qwen3_vl_embedding.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"official Qwen embedding script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("qwen3_vl_embedding_official", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "default_instruction": INSTRUCTION,
        "max_pixels": MAX_PIXELS,
    }
    try:
        return module.Qwen3VLEmbedder(
            model_name_or_path=str(model_path),
            attn_implementation="flash_attention_2",
            **kwargs,
        )
    except Exception as error:
        print(f"flash_attention_2 unavailable ({error}); using eager attention", flush=True)
        torch.cuda.empty_cache()
        return module.Qwen3VLEmbedder(model_name_or_path=str(model_path), **kwargs)


def conversation(text: str, image_paths: list[str]) -> list[dict[str, Any]]:
    content = [
        {
            "type": "image",
            "image": "file://" + str(Path(path).resolve()),
            "min_pixels": 4096,
            "max_pixels": MAX_PIXELS,
        }
        for path in image_paths
    ]
    content.append({"type": "text", "text": text or "NULL"})
    return [
        {"role": "system", "content": [{"type": "text", "text": INSTRUCTION}]},
        {"role": "user", "content": content},
    ]


def preprocess(embedder, conversations):
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
def embed_conversations(embedder, conversations) -> np.ndarray:
    inputs = preprocess(embedder, conversations)
    outputs = embedder.forward(inputs)
    embeddings = embedder._pooling_last(outputs["last_hidden_state"], outputs["attention_mask"])
    embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
    return embeddings.cpu().numpy().astype(np.float32, copy=False)


def compute_embeddings(embedder, frame: pd.DataFrame) -> np.ndarray:
    chunks, start, active_batch = [], 0, BATCH_SIZE
    started = time.monotonic()
    while start < len(frame):
        end = min(start + active_batch, len(frame))
        batch = frame.iloc[start:end]
        conversations = [conversation(row.text, row.image_paths) for row in batch.itertuples(index=False)]
        try:
            result = embed_conversations(embedder, conversations)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if active_batch == 1:
                row = batch.iloc[0]
                print(f"single sample OOM at row={start}; using text only", flush=True)
                result = embed_conversations(embedder, [conversation(row.text, [])])
            else:
                active_batch = max(1, active_batch // 2)
                print(f"OOM at row={start}; retry batch_size={active_batch}", flush=True)
                continue
        chunks.append(result)
        start = end
        if active_batch < BATCH_SIZE:
            active_batch = min(BATCH_SIZE, active_batch * 2)
        if start % 240 < len(result) or start == len(frame):
            print(f"embedded={start}/{len(frame)} elapsed_min={(time.monotonic()-started)/60:.1f}", flush=True)
    return np.concatenate(chunks, axis=0)


def explain(category: str, prediction: int) -> str:
    if category == "БАД":
        return (
            "Текст и изображения подтверждают маркировку товара как биологически активной добавки."
            if prediction
            else "Текст и изображения не подтверждают обязательную маркировку товара как биологически активной добавки."
        )
    return (
        "Текст и изображения подтверждают наличие самостоятельного горючего товара или источника воспламенения."
        if prediction
        else "Текст и изображения не подтверждают наличие самостоятельного горючего товара или источника воспламенения."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--test_data_path", "--test-data-path", dest="input", required=True)
    parser.add_argument("-o", "--output_path", "--output-path", dest="output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    frame = prepare_frame(Path(args.input))
    print(
        f"prepared rows={len(frame)} products_with_images={int(frame.image_paths.map(bool).sum())} "
        f"image_files={int(frame.image_paths.map(len).sum())}",
        flush=True,
    )
    bundle = joblib.load(CLASSIFIER_PATH)
    if bundle.get("instruction") != INSTRUCTION or bundle.get("embedding_dim") != 2048:
        raise ValueError("classifier and embedding configuration mismatch")
    embedder = load_embedder(MODEL_PATH)
    embeddings = compute_embeddings(embedder, frame)
    if embeddings.shape != (len(frame), 2048) or not np.isfinite(embeddings).all():
        raise ValueError(f"invalid embeddings: {embeddings.shape}")
    predictions = np.zeros(len(frame), dtype=np.int8)
    for category, model in bundle["models"].items():
        mask = frame["category"].to_numpy() == category
        scores = model.decision_function(embeddings[mask])
        predictions[mask] = (scores >= bundle["thresholds"][category]).astype(np.int8)
    results = []
    for row, prediction in zip(frame.itertuples(index=False), predictions):
        verdict = "не бан" if prediction else "бан"
        result = f"<комментарий>{explain(row.category, int(prediction))}<вердикт>{verdict}"
        if OUTPUT_RE.fullmatch(result) is None:
            raise ValueError(f"invalid output: {result!r}")
        results.append(result)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": frame["id"], "result": results}).to_csv(output_path, index=False)
    print(f"saved rows={len(results)} path={output_path}", flush=True)


if __name__ == "__main__":
    main()
