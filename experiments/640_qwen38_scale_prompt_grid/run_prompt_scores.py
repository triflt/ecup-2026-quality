from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path, *, shard_index: int, num_shards: int, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if int(row["global_index"]) % num_shards != shard_index:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_image(url: str) -> Image.Image:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        image = Image.open(io.BytesIO(response.read())).convert("RGB")
    width, height = image.size
    if width * height > 262144:
        scale = math.sqrt(262144 / (width * height))
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        resized = image.resize((max(28, int(width * scale)), max(28, int(height * scale))), resampling)
        image.close()
        image = resized
    return image


def prompt(row: SimpleNamespace) -> str:
    return (
        f"Категория: {row.category}\n"
        f"Название: {row.name}\n"
        f"Описание: {row.description}\n"
        f"Правило: {RULES[row.category]}\n"
        "Определи правильность категории. Ответь только одной цифрой: 1 или 0."
    )


def messages(row: SimpleNamespace, image: Image.Image) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt(row)},
            ],
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--experiment-id", default="640")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device-map-auto", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    rows = load_rows(args.runtime, shard_index=args.shard_index, num_shards=args.num_shards, limit=args.limit)
    processor = AutoProcessor.from_pretrained(args.model_root, local_files_only=True)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
    }
    if args.device_map_auto:
        model_kwargs["device_map"] = "auto"
        model_kwargs["low_cpu_mem_usage"] = True
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_root,
        **model_kwargs,
    )
    if not args.device_map_auto:
        model = model.to("cuda")
    model = model.eval()
    device_map = getattr(model, "hf_device_map", {})
    if args.device_map_auto and any(str(device) in {"cpu", "disk"} for device in device_map.values()):
        raise RuntimeError(f"model was offloaded outside CUDA: {device_map}")
    input_device = model.device
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1 or zero == one:
        raise ValueError(f"0/1 are not distinct atomic tokens: zero={zero}, one={one}")

    output_path = args.output_dir / "scores.jsonl"
    started = time.monotonic()
    with output_path.open("w", encoding="utf-8") as output:
        for start in range(0, len(rows), args.batch_size):
            chunk = [SimpleNamespace(**row) for row in rows[start : start + args.batch_size]]
            images = [load_image(row.image_url) for row in chunk]
            try:
                conversations = [messages(row, image) for row, image in zip(chunk, images, strict=True)]
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
                ).to(input_device)
                with torch.inference_mode():
                    logits = model(**batch).logits[:, -1, :].float().cpu()
                scores = logits[:, one[0]] - logits[:, zero[0]]
                for row, score in zip(chunk, scores.tolist(), strict=True):
                    output.write(
                        json.dumps(
                            {
                                "global_index": row.global_index,
                                "id": row.id,
                                "fold": row.fold,
                                "category": row.category,
                                "score": float(score),
                                "prediction": int(score >= 0.0),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                output.flush()
            finally:
                for image in images:
                    image.close()
    elapsed = time.monotonic() - started
    report = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "runtime_sha256": sha256_file(args.runtime),
        "output_sha256": sha256_file(output_path),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "rows": len(rows),
        "batch_size": args.batch_size,
        "device_map_auto": args.device_map_auto,
        "cuda_device_count": torch.cuda.device_count(),
        "hf_device_map": {str(key): str(value) for key, value in device_map.items()},
        "elapsed_seconds": elapsed,
        "rows_per_second": len(rows) / elapsed if elapsed else 0.0,
        "thinking": False,
        "threshold": 0.0,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
