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
from src.model import compose_text, fingerprint, normalize


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
CLASSIFIER_PATH = Path(os.environ.get("MM_CLASSIFIER_PATH", ROOT / "multimodal_classifier.joblib"))
TEXT_CLASSIFIER_PATH = Path(os.environ.get("TEXT_CLASSIFIER_PATH", ROOT / "strong_text_full.joblib"))
ADVANCED_HEADS_PATH = Path(os.environ.get("ADVANCED_HEADS_PATH", ROOT / "advanced_heads.joblib"))
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_PIXELS_ALL = int(os.environ.get("QWEN_MAX_PIXELS", "261120"))
MAX_PIXELS_FIRST = int(os.environ.get("QWEN_FIRST_MAX_PIXELS", "589824"))
BATCH_SIZE_ALL = int(os.environ.get("QWEN_EMBED_BATCH_SIZE", "24"))
BATCH_SIZE_FIRST = int(os.environ.get("QWEN_FIRST_BATCH_SIZE", "16"))
BATCH_SIZE_TEXT = int(os.environ.get("QWEN_TEXT_BATCH_SIZE", "32"))
GENERIC_INSTRUCTION = (
    "Represent this marketplace product card for verifying whether it genuinely belongs "
    "to its declared moderation category. Use the description, packaging text, product "
    "contents, purpose, and every supplied image."
)
IMAGE_INSTRUCTION = (
    "Represent the visible marketplace product and packaging for verifying whether it "
    "genuinely belongs to its declared moderation category. Focus on packaging text, "
    "product contents, purpose, and included items."
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
    image_paths, texts, first_texts = [], [], []
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
        first_texts.append(
            f"Declared category: {row.category}\n"
            f"{CATEGORY_HINTS.get(str(row.category), '')}"
        )
    frame["text"] = texts
    frame["first_text"] = first_texts
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
        "default_instruction": GENERIC_INSTRUCTION,
        "max_pixels": MAX_PIXELS_FIRST,
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


def conversation(
    text: str,
    image_paths: list[str],
    instruction: str,
    max_pixels: int,
) -> list[dict[str, Any]]:
    content = [
        {
            "type": "image",
            "image": "file://" + str(Path(path).resolve()),
            "min_pixels": 4096,
            "max_pixels": max_pixels,
        }
        for path in image_paths
    ]
    content.append({"type": "text", "text": text or "NULL"})
    return [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
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


def compute_embeddings(
    embedder,
    texts: list[str],
    image_paths: list[list[str]],
    batch_size: int,
    max_pixels: int,
    instruction: str,
    view_name: str,
) -> np.ndarray:
    chunks, start, active_batch = [], 0, batch_size
    started = time.monotonic()
    while start < len(texts):
        end = min(start + active_batch, len(texts))
        conversations = [
            conversation(text, paths, instruction, max_pixels)
            for text, paths in zip(texts[start:end], image_paths[start:end])
        ]
        try:
            result = embed_conversations(embedder, conversations)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if active_batch == 1:
                print(f"single sample OOM at view={view_name} row={start}; using text only", flush=True)
                result = embed_conversations(
                    embedder,
                    [conversation(texts[start], [], instruction, max_pixels)],
                )
            else:
                active_batch = max(1, active_batch // 2)
                print(f"OOM at row={start}; retry batch_size={active_batch}", flush=True)
                continue
        chunks.append(result)
        start = end
        if active_batch < batch_size:
            active_batch = min(batch_size, active_batch * 2)
        if start % 240 < len(result) or start == len(texts):
            print(
                f"view={view_name} embedded={start}/{len(texts)} "
                f"elapsed_min={(time.monotonic()-started)/60:.1f}", flush=True,
            )
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


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


FUSION = {
    "БАД": {
        "threshold": 0.23807610301950974,
        "text": 0.70,
        "interaction": 0.30,
    },
    "Легковоспламеняющиеся": {
        "threshold": 0.9540719747543336,
        "text": 0.40,
        "all_images": 0.20,
        "first_image": 0.05,
        "extra_trees": 0.35,
    },
}


def normalized(features: np.ndarray) -> np.ndarray:
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)


def interaction_features(text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
    text_features = normalized(text_features.astype(np.float32, copy=False))
    image_features = normalized(image_features.astype(np.float32, copy=False))
    return np.concatenate([
        text_features,
        image_features,
        np.abs(text_features - image_features),
        text_features * image_features,
    ], axis=1).astype(np.float32, copy=False)


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
    text_bundle = joblib.load(TEXT_CLASSIFIER_PATH)
    advanced = joblib.load(ADVANCED_HEADS_PATH)
    if bundle.get("instruction") != GENERIC_INSTRUCTION or bundle.get("embedding_dim") != 2048:
        raise ValueError("classifier and embedding configuration mismatch")
    if advanced.get("embedding_dim") != 2048:
        raise ValueError("advanced head configuration mismatch")
    embedder = load_embedder(MODEL_PATH)
    categories = frame["category"].astype(str).to_numpy()
    bad_positions = np.flatnonzero(categories == "БАД")
    flammable_positions = np.flatnonzero(categories == "Легковоспламеняющиеся")

    first_embeddings = compute_embeddings(
        embedder,
        frame["first_text"].tolist(),
        [paths[:1] for paths in frame["image_paths"]],
        BATCH_SIZE_FIRST,
        MAX_PIXELS_FIRST,
        IMAGE_INSTRUCTION,
        "first-image",
    )
    text_embeddings_bad = compute_embeddings(
        embedder,
        frame.iloc[bad_positions]["text"].tolist(),
        [[] for _ in bad_positions],
        BATCH_SIZE_TEXT,
        MAX_PIXELS_ALL,
        GENERIC_INSTRUCTION,
        "text-only-bad",
    ) if len(bad_positions) else np.empty((0, 2048), dtype=np.float32)
    all_embeddings_flammable = compute_embeddings(
        embedder,
        frame.iloc[flammable_positions]["text"].tolist(),
        frame.iloc[flammable_positions]["image_paths"].tolist(),
        BATCH_SIZE_ALL,
        MAX_PIXELS_ALL,
        GENERIC_INSTRUCTION,
        "all-images-flammable",
    ) if len(flammable_positions) else np.empty((0, 2048), dtype=np.float32)
    for name, values, expected in (
        ("first", first_embeddings, len(frame)),
        ("text_bad", text_embeddings_bad, len(bad_positions)),
        ("all_flammable", all_embeddings_flammable, len(flammable_positions)),
    ):
        if values.shape != (expected, 2048) or not np.isfinite(values).all():
            raise ValueError(f"invalid {name} embeddings: {values.shape}")
    text_inputs = [compose_text(name, description) for name, description in zip(
        frame["name"], frame["description"]
    )]
    text_matrix = text_bundle.vectorizer.transform(text_inputs)
    predictions = np.zeros(len(frame), dtype=np.int8)
    if len(bad_positions):
        category = "БАД"
        text_scores = text_bundle.category_models[category].decision_function(text_matrix[bad_positions])
        interaction_scores = advanced["interaction_models"][category].decision_function(
            interaction_features(text_embeddings_bad, first_embeddings[bad_positions])
        )
        config = FUSION[category]
        fused = config["text"] * rank01(text_scores) + config["interaction"] * rank01(interaction_scores)
        predictions[bad_positions] = (fused >= config["threshold"]).astype(np.int8)
        print(
            f"category={category!r} rows={len(bad_positions)} "
            f"predicted={int(predictions[bad_positions].sum())} weights=70/30-text-interaction",
            flush=True,
        )
    if len(flammable_positions):
        category = "Легковоспламеняющиеся"
        config = FUSION[category]
        text_scores = text_bundle.category_models[category].decision_function(text_matrix[flammable_positions])
        all_scores = bundle["models"][category].decision_function(all_embeddings_flammable)
        first_scores = advanced["first_models"][category].decision_function(
            first_embeddings[flammable_positions]
        )
        tree_scores = advanced["extra_trees_models"][category].predict_proba(
            all_embeddings_flammable
        )[:, 1]
        fused = (
            config["text"] * rank01(text_scores)
            + config["all_images"] * rank01(all_scores)
            + config["first_image"] * rank01(first_scores)
            + config["extra_trees"] * rank01(tree_scores)
        )
        predictions[flammable_positions] = (fused >= config["threshold"]).astype(np.int8)
        print(
            f"category={category!r} rows={len(flammable_positions)} "
            f"predicted={int(predictions[flammable_positions].sum())} weights=40/20/5/35",
            flush=True,
        )

    # Exact repeated products with fully consistent train labels remain the strongest evidence.
    for index, (name, text, category) in enumerate(zip(frame["name"], text_inputs, categories)):
        exact_key = (category, fingerprint(text))
        name_key = (category, normalize(name))
        if exact_key in text_bundle.exact_lookup:
            predictions[index] = text_bundle.exact_lookup[exact_key]
        elif name_key in text_bundle.name_lookup:
            predictions[index] = text_bundle.name_lookup[name_key]
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
