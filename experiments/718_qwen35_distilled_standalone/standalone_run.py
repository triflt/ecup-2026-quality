from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.environ.get(
        "QWEN35_MODEL_PATH",
        os.path.join(
            os.environ.get("SHARED_MODELS_PATH", "/shared_models"),
            "Qwen/Qwen3.5-4B",
        ),
    )
)
ADAPTER_PATH = Path(os.environ.get("QWEN35_LORA_ADAPTER_PATH", ROOT / "adapter_qwen35"))
CONFIG_PATH = Path(os.environ.get("STANDALONE_CONFIG_PATH", ROOT / "standalone_config.json"))
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
BATCH_SIZE = int(os.environ.get("QWEN_LORA_BATCH_SIZE", "8"))
IMAGE_PREPROCESSING = "solution140_first_image_thumbnail_448_lanczos_v1"
OUTPUT_RE = re.compile(r"^<комментарий>(.{50,300})<вердикт>(бан|не бан)$", re.S)
RULES = {
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


def canonical_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def normalize_id(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def compact_text(value: Any, limit: int) -> str:
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    head = int(limit * 0.7)
    return value[:head].rstrip() + " … " + value[-(limit - head) :].lstrip()


def user_text(row: Any) -> str:
    category = str(row.category)
    if category not in RULES:
        raise ValueError(f"unsupported category: {category}")
    return (
        f"Категория: {category}\n"
        f"Название: {compact_text(row.name, 320)}\n"
        f"Описание: {compact_text(row.description, 1800)}\n"
        f"Правило: {RULES[category]}\n"
        "Определи правильность категории. Ответь только одной цифрой: 1 или 0."
    )


def open_image(path: str | None) -> Image.Image:
    image = Image.open(path).convert("RGB") if path else Image.new("RGB", (32, 32), "white")
    image.thumbnail((448, 448), Image.Resampling.LANCZOS)
    return image


def rank_percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("rank input is empty or non-finite")
    return pd.Series(values).rank(method="average", pct=True).to_numpy(np.float32)


def prepare_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"id", "name", "description", "category"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing CSV columns: {sorted(required - set(frame.columns))}")
    if frame["id"].duplicated().any():
        raise ValueError("duplicate input IDs")
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["category"] = frame["category"].astype(str)
    if not set(frame["category"]).issubset(RULES):
        raise ValueError("unsupported input category")
    images_root = path.parent / "images"
    first_images = []
    for value in frame["id"]:
        folder = images_root / normalize_id(value)
        paths = (
            sorted(
                child
                for child in folder.iterdir()
                if child.is_file() and child.suffix.lower() in VALID_SUFFIXES
            )
            if folder.is_dir()
            else []
        )
        first_images.append(str(paths[0]) if paths else None)
    frame["first_image"] = first_images
    return frame


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(config)
    digest = payload.pop("contract_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("standalone runtime config self-hash mismatch")
    expected = {
        "schema_version": "exp718_runtime_config_v1",
        "architecture": "qwen35_only",
        "base_model": "Qwen/Qwen3.5-4B",
        "score": "category_batch_percentile_rank",
        "threshold_rule": "median_of_five_outer_train_thresholds",
        "image_preprocessing": IMAGE_PREPROCESSING,
        "teacher_required_at_inference": False,
    }
    mismatch = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatch or set(config.get("thresholds", {})) != set(RULES):
        raise ValueError(f"standalone runtime config mismatch: {mismatch}")
    for threshold in config["thresholds"].values():
        if not 0.0 < float(threshold) <= 1.0:
            raise ValueError("standalone threshold outside (0, 1]")
    return config


def score(model, processor, frame: pd.DataFrame) -> np.ndarray:
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1 or zero == one:
        raise ValueError("0/1 are not distinct atomic tokens")
    values: list[float] = []
    started = time.monotonic()
    for offset in range(0, len(frame), BATCH_SIZE):
        rows = list(frame.iloc[offset : offset + BATCH_SIZE].itertuples(index=False))
        images = [open_image(row.first_image) for row in rows]
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_text(row)},
                    ],
                }
            ]
            for row, image in zip(rows, images, strict=True)
        ]
        try:
            batch = processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1536,
                enable_thinking=False,
            ).to(model.device)
            with torch.inference_mode():
                output = model(**batch, use_cache=True)
                positions = torch.arange(batch["attention_mask"].shape[1], device=model.device)[None, :]
                last = torch.where(batch["attention_mask"].bool(), positions, -1).max(dim=1).values
                logits = output.logits[torch.arange(len(rows), device=model.device), last]
                values.extend((logits[:, one[0]] - logits[:, zero[0]]).float().cpu().tolist())
        finally:
            for image in images:
                image.close()
        done = offset + len(rows)
        if done % 400 < len(rows) or done == len(frame):
            print(f"scored={done}/{len(frame)} elapsed_min={(time.monotonic()-started)/60:.1f}", flush=True)
    return np.asarray(values, dtype=np.float32)


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
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("standalone runtime requires exactly one visible CUDA GPU")
    config = load_config(CONFIG_PATH)
    frame = prepare_frame(Path(args.input))
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    base = AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH, is_trainable=False)
    model.eval()
    raw_scores = score(model, processor, frame)
    predictions = np.zeros(len(frame), dtype=np.int8)
    categories = frame["category"].astype(str).to_numpy()
    for category, threshold in config["thresholds"].items():
        mask = categories == category
        if mask.any():
            ranks = rank_percentile(raw_scores[mask])
            predictions[mask] = ranks >= float(threshold)
            print(
                f"category={category!r} rows={int(mask.sum())} predicted={int(predictions[mask].sum())} threshold={threshold}",
                flush=True,
            )
    results = []
    for row, prediction in zip(frame.itertuples(index=False), predictions, strict=True):
        verdict = "не бан" if prediction else "бан"
        result = f"<комментарий>{explain(row.category, int(prediction))}<вердикт>{verdict}"
        if OUTPUT_RE.fullmatch(result) is None:
            raise ValueError("invalid output schema")
        results.append(result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": frame["id"], "result": results}).to_csv(output, index=False)
    print(f"saved rows={len(results)} path={output}", flush=True)


if __name__ == "__main__":
    main()
