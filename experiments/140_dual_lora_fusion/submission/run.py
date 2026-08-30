from __future__ import annotations

import argparse
import gc
import gzip
import html
import importlib.util
import json
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
from explanation_contract import format_result
from explanation_runtime import attach_explanation_adapter, generate_explanations
from PIL import Image
from qwen_vl_utils.vision_process import process_vision_info
from src.model import compose_text, fingerprint, normalize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
MODEL_PATH = Path(
    os.environ.get(
        "QWEN_EMBED_MODEL_PATH",
        os.path.join(
            os.environ.get("SHARED_MODELS_PATH", "/shared_models"),
            "Qwen/Qwen3-VL-Embedding-2B",
        ),
    )
)
INSTRUCT_MODEL_PATH = Path(
    os.environ.get(
        "QWEN_INSTRUCT_MODEL_PATH",
        os.path.join(
            os.environ.get("SHARED_MODELS_PATH", "/shared_models"),
            "Qwen/Qwen3-VL-2B-Instruct",
        ),
    )
)
QWEN35_MODEL_PATH = Path(
    os.environ.get(
        "QWEN35_MODEL_PATH",
        os.path.join(
            os.environ.get("SHARED_MODELS_PATH", "/shared_models"),
            "Qwen/Qwen3.5-4B",
        ),
    )
)
QWEN3VL_ADAPTER_PATH = Path(
    os.environ.get("QWEN3VL_LORA_ADAPTER_PATH", ROOT / "adapter_qwen3vl")
)
QWEN35_ADAPTER_PATH = Path(
    os.environ.get("QWEN35_LORA_ADAPTER_PATH", ROOT / "adapter_qwen35")
)
EXPLANATION_ADAPTER_PATH = Path(
    os.environ.get("QWEN35_EXPLANATION_ADAPTER_PATH", ROOT / "adapter_reasoner")
)
ANNOTATOR_PRIOR_PATH = Path(
    os.environ.get("ANNOTATOR_PRIOR_PATH", ROOT / "annotator_prior.json.gz")
)
CLASSIFIER_PATH = Path(os.environ.get("MM_CLASSIFIER_PATH", ROOT / "multimodal_classifier.joblib"))
TEXT_CLASSIFIER_PATH = Path(os.environ.get("TEXT_CLASSIFIER_PATH", ROOT / "strong_text_full.joblib"))
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MAX_PIXELS = int(os.environ.get("QWEN_MAX_PIXELS", "261120"))
BATCH_SIZE = int(os.environ.get("QWEN_EMBED_BATCH_SIZE", "24"))
LORA_BATCH_SIZE = int(os.environ.get("QWEN_LORA_BATCH_SIZE", "8"))
EXPLANATION_BATCH_SIZE = int(os.environ.get("QWEN_EXPLANATION_BATCH_SIZE", "2"))
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
LORA_RULES = {
    "БАД": (
        "Метка 1 только если в описании или на упаковке есть прямое указание БАД "
        "или dietary supplement. Спортивное питание без такой маркировки, явное "
        "отрицание или отсутствие маркировки — метка 0."
    ),
    "Легковоспламеняющиеся": (
        "Метка 1 для самостоятельного источника огня, горючего вещества или газа, "
        "либо если такой товар входит в комплект. Пустое оборудование, встроенный "
        "источник, горючий материал только как компонент или предмет не в комплекте — 0."
    ),
}
OUTPUT_RE = re.compile(r"^<комментарий>(.{50,300})<вердикт>(бан|не бан)$", re.DOTALL)


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
    except Exception as error:  # noqa: BLE001 - optional backend failures vary by stack
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


def compact_text(value: Any, limit: int = 1800) -> str:
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    head = int(limit * 0.7)
    return value[:head].rstrip() + " … " + value[-(limit - head):].lstrip()


def lora_user_text(row: Any) -> str:
    return (
        f"Категория: {row.category}\n"
        f"Название: {compact_text(row.name, 320)}\n"
        f"Описание: {compact_text(row.description)}\n"
        f"Правило: {LORA_RULES[row.category]}\n"
        "Определи правильность категории. Ответь только одной цифрой: 1 или 0."
    )


def load_lora_model(model_path: Path, adapter_path: Path, model_class: str):
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoProcessor

    if not adapter_path.is_dir():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
    processor_kwargs = {"local_files_only": True, "trust_remote_code": True}
    if model_class == "image_text":
        processor_kwargs.update(min_pixels=4 * 28 * 28, max_pixels=262144)
    processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
    processor.tokenizer.padding_side = "left"
    loader = AutoModelForMultimodalLM if model_class == "multimodal" else AutoModelForImageTextToText
    base = loader.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    return model, processor


def lora_messages(row: Any, image_path: str | None) -> list[dict[str, Any]]:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_path or "missing.jpg"},
            {"type": "text", "text": lora_user_text(row)},
        ],
    }]


def open_lora_image(path: str | None) -> Image.Image:
    image = Image.open(path).convert("RGB") if path else Image.new("RGB", (32, 32), "white")
    # Training normalized every first image to this bound before both Qwen
    # processors. It also keeps Qwen3.5 visual tokens inside the 1,536-token
    # context instead of truncating multimodal special tokens.
    image.thumbnail((448, 448), Image.Resampling.LANCZOS)
    return image


@torch.inference_mode()
def compute_lora_scores(
    model, processor, frame: pd.DataFrame, *, use_chat_batch: bool = False
) -> np.ndarray:
    zero_ids = processor.tokenizer.encode("0", add_special_tokens=False)
    one_ids = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero_ids) != 1 or len(one_ids) != 1:
        raise ValueError(f"digit tokens are not atomic: {zero_ids}, {one_ids}")
    token_zero, token_one = zero_ids[0], one_ids[0]
    scores: list[float] = []
    started = time.monotonic()
    for start in range(0, len(frame), LORA_BATCH_SIZE):
        rows = list(frame.iloc[start:start + LORA_BATCH_SIZE].itertuples(index=False))
        paths = [row.image_paths[0] if row.image_paths else None for row in rows]
        images = [open_lora_image(path) for path in paths]
        try:
            if use_chat_batch:
                conversations = [
                    lora_messages(row, image) for row, image in zip(rows, images)
                ]
                kwargs = {
                    "add_generation_prompt": True,
                    "tokenize": True,
                    "return_dict": True,
                    "return_tensors": "pt",
                    "padding": True,
                    "truncation": True,
                    "max_length": 1536,
                }
                try:
                    batch = processor.apply_chat_template(
                        conversations, enable_thinking=False, **kwargs
                    )
                except TypeError:
                    batch = processor.apply_chat_template(conversations, **kwargs)
            else:
                prompts = [
                    processor.apply_chat_template(
                        lora_messages(row, path), tokenize=False, add_generation_prompt=True
                    )
                    for row, path in zip(rows, paths)
                ]
                batch = processor(
                    text=prompts,
                    images=images,
                    padding=True,
                    truncation=True,
                    max_length=1536,
                    return_tensors="pt",
                )
        finally:
            for image in images:
                image.close()
        batch = {key: value.to("cuda") for key, value in batch.items()}
        outputs = model(**batch)
        positions = torch.arange(batch["attention_mask"].shape[1], device="cuda")[None, :]
        last = torch.where(batch["attention_mask"].bool(), positions, -1).max(dim=1).values
        logits = outputs.logits[torch.arange(len(rows), device="cuda"), last]
        scores.extend((logits[:, token_one] - logits[:, token_zero]).float().cpu().tolist())
        done = min(start + LORA_BATCH_SIZE, len(frame))
        if done % 400 < len(rows) or done == len(frame):
            print(
                f"lora_scored={done}/{len(frame)} elapsed_min={(time.monotonic()-started)/60:.1f}",
                flush=True,
            )
    return np.asarray(scores, dtype=np.float32)


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks


FUSION = {
    "БАД": {"alpha_text": 0.85, "threshold": 0.24864045896205267},
    "Легковоспламеняющиеся": {"alpha_text": 0.65, "threshold": 0.9591804083988902},
}

LORA_FUSION = {
    "БАД": {
        "weight_base": 0.50,
        "weight_qwen3vl": 0.25,
        "weight_qwen35": 0.25,
        "threshold": 0.27193570137023926,
    },
    "Легковоспламеняющиеся": {
        "weight_base": 0.15,
        "weight_qwen3vl": 0.10,
        "weight_qwen35": 0.75,
        "threshold": 0.953912615776062,
    },
}


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
    if bundle.get("instruction") != INSTRUCTION or bundle.get("embedding_dim") != 2048:
        raise ValueError("classifier and embedding configuration mismatch")
    embedder = load_embedder(MODEL_PATH)
    embeddings = compute_embeddings(embedder, frame)
    if embeddings.shape != (len(frame), 2048) or not np.isfinite(embeddings).all():
        raise ValueError(f"invalid embeddings: {embeddings.shape}")
    text_inputs = [compose_text(name, description) for name, description in zip(
        frame["name"], frame["description"]
    )]
    text_matrix = text_bundle.vectorizer.transform(text_inputs)
    base_scores = np.zeros(len(frame), dtype=np.float32)
    for category, model in bundle["models"].items():
        mask = frame["category"].to_numpy() == category
        if not mask.any():
            print(f"base_category={category!r} rows=0 skipped", flush=True)
            continue
        mm_scores = model.decision_function(embeddings[mask])
        text_scores = text_bundle.category_models[category].decision_function(text_matrix[mask])
        alpha = FUSION[category]["alpha_text"]
        fused = alpha * rank01(text_scores) + (1.0 - alpha) * rank01(mm_scores)
        base_scores[mask] = fused
        print(
            f"base_category={category!r} rows={int(mask.sum())} alpha_text={alpha}", flush=True,
        )

    del embedder, embeddings
    gc.collect()
    torch.cuda.empty_cache()
    qwen3vl_model, qwen3vl_processor = load_lora_model(
        INSTRUCT_MODEL_PATH, QWEN3VL_ADAPTER_PATH, "image_text"
    )
    qwen3vl_scores = compute_lora_scores(qwen3vl_model, qwen3vl_processor, frame)
    del qwen3vl_model, qwen3vl_processor
    gc.collect()
    torch.cuda.empty_cache()
    qwen35_model, qwen35_processor = load_lora_model(
        QWEN35_MODEL_PATH, QWEN35_ADAPTER_PATH, "multimodal"
    )
    qwen35_scores = compute_lora_scores(
        qwen35_model, qwen35_processor, frame, use_chat_batch=True
    )
    predictions = np.zeros(len(frame), dtype=np.int8)
    for category, config in LORA_FUSION.items():
        mask = frame["category"].to_numpy() == category
        if not mask.any():
            print(f"final_category={category!r} rows=0 skipped", flush=True)
            continue
        combined = (
            config["weight_base"] * rank01(base_scores[mask])
            + config["weight_qwen3vl"] * rank01(qwen3vl_scores[mask])
            + config["weight_qwen35"] * rank01(qwen35_scores[mask])
        )
        predictions[mask] = (combined >= config["threshold"]).astype(np.int8)
        print(
            f"final_category={category!r} rows={int(mask.sum())} "
            f"predicted={int(predictions[mask].sum())} "
            f"weights={config['weight_base']}/{config['weight_qwen3vl']}/{config['weight_qwen35']}",
            flush=True,
        )

    # The organizer confirmed that test labels come from the same ambiguous
    # annotation process.  Use leave-one-out-calibrated empirical product-family
    # priors, including majority labels for genuinely conflicting repeats.
    with gzip.open(ANNOTATOR_PRIOR_PATH, "rt", encoding="utf-8") as stream:
        annotator_prior = json.load(stream)
    categories = frame["category"].astype(str).to_numpy()
    for index, (name, text, category) in enumerate(zip(frame["name"], text_inputs, categories)):
        exact_key = f"{category}\t{fingerprint(text)}"
        name_key = f"{category}\t{normalize(name)}"
        if exact_key in annotator_prior["exact"]:
            predictions[index] = annotator_prior["exact"][exact_key]
        elif name_key in annotator_prior["name"]:
            predictions[index] = annotator_prior["name"][name_key]
    frozen_predictions = predictions.copy()
    attach_explanation_adapter(qwen35_model, EXPLANATION_ADAPTER_PATH)
    comments, explanation_statuses = generate_explanations(
        qwen35_model,
        qwen35_processor,
        frame,
        frozen_predictions,
        batch_size=EXPLANATION_BATCH_SIZE,
    )
    if not np.array_equal(predictions, frozen_predictions):
        raise AssertionError("explanation stage changed solution140 verdicts")
    results = []
    for comment, prediction in zip(comments, frozen_predictions, strict=True):
        result = format_result(comment, int(prediction))
        if OUTPUT_RE.fullmatch(result) is None:
            raise ValueError(f"invalid output: {result!r}")
        results.append(result)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": frame["id"], "result": results}).to_csv(output_path, index=False)
    fallback_rows = sum(status != "generated_plain_text" for status in explanation_statuses)
    print(
        f"saved rows={len(results)} fallback_rows={fallback_rows} path={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
