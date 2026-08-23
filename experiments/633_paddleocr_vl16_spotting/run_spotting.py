from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
MODEL_REVISION = "c5630abae1d940eafe0697512a0325494b02ab42"
LOC_PATTERN = re.compile(r"([^<\n]+?)((?:<\|LOC_\d+\|>){8})")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_spotting(raw: str, *, width: int, height: int, sequence_confidence: float | None) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for match in LOC_PATTERN.finditer(raw):
        text = match.group(1).strip()
        values = [int(value) for value in re.findall(r"<\|LOC_(\d+)\|>", match.group(2))]
        if not text or len(values) != 8 or any(value < 0 or value > 1000 for value in values):
            continue
        polygon = [
            [round(values[index] * width / 1000), round(values[index + 1] * height / 1000)]
            for index in range(0, 8, 2)
        ]
        polygon = [[min(width, max(0, x)), min(height, max(0, y))] for x, y in polygon]
        detections.append(
            {
                "text": text,
                "polygon": polygon,
                "normalized_polygon": [[values[index], values[index + 1]] for index in range(0, 8, 2)],
                "confidence": sequence_confidence,
                "confidence_type": "sequence_geomean_token_probability",
            }
        )
    return detections


def load_rows(path: Path, *, shard_index: int, num_shards: int, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for global_index, line in enumerate(stream):
            if global_index % num_shards != shard_index:
                continue
            row = json.loads(line)
            if set(row) != {"id", "image_index", "url"}:
                raise ValueError("unexpected manifest row schema")
            row["global_index"] = global_index
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response, path.open("wb") as output:
        output.write(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.model_revision != MODEL_REVISION:
        raise ValueError("model revision differs from the frozen contract")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.image_cache.mkdir(parents=True, exist_ok=True)

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = load_rows(args.manifest, shard_index=args.shard_index, num_shards=args.num_shards, limit=args.limit)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_root,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model_root, local_files_only=True)
    output_path = args.output_dir / "spotting.jsonl"
    started = time.monotonic()
    failures = 0
    detections_total = 0
    parseable_images = 0
    with output_path.open("w", encoding="utf-8") as output:
        for row in rows:
            cache_path = args.image_cache / f"{row['global_index']}.img"
            try:
                download(str(row["url"]), cache_path)
                with Image.open(cache_path) as source:
                    source.load()
                    original = source.convert("RGB")
                width, height = original.size
                if width < 1500 and height < 1500:
                    resampling = getattr(Image, "Resampling", Image).LANCZOS
                    image = original.resize((width * 2, height * 2), resampling)
                else:
                    image = original
                messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Spotting:"}]}]
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    images_kwargs={"size": {"shortest_edge": processor.image_processor.min_pixels, "longest_edge": 2048 * 28 * 28}},
                ).to(model.device)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                prompt_length = inputs["input_ids"].shape[-1]
                tokens = generated.sequences[0, prompt_length:]
                raw = processor.decode(tokens, skip_special_tokens=False)
                selected = []
                for step, scores in enumerate(generated.scores):
                    if step >= len(tokens):
                        break
                    selected.append(float(torch.softmax(scores[0].float(), dim=-1)[tokens[step]].cpu()))
                confidence = math.exp(sum(math.log(max(value, 1e-12)) for value in selected) / len(selected)) if selected else None
                detections = parse_spotting(raw, width=width, height=height, sequence_confidence=confidence)
                detections_total += len(detections)
                parseable_images += bool(detections)
                record = {
                    "id": str(row["id"]),
                    "image_index": int(row["image_index"]),
                    "global_index": int(row["global_index"]),
                    "width": width,
                    "height": height,
                    "raw_generation": raw,
                    "sequence_confidence": confidence,
                    "detections": detections,
                    "error": None,
                }
                original.close()
                if image is not original:
                    image.close()
            except Exception as exc:  # noqa: BLE001 - isolate corrupt or unreadable images per row
                failures += 1
                record = {
                    "id": str(row["id"]),
                    "image_index": int(row["image_index"]),
                    "global_index": int(row["global_index"]),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                cache_path.unlink(missing_ok=True)
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()

    elapsed = time.monotonic() - started
    successful = len(rows) - failures
    report = {
        "schema_version": 1,
        "experiment_id": "633",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "manifest_sha256": sha256_file(args.manifest),
        "output_sha256": sha256_file(output_path),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "requested_images": len(rows),
        "successful_images": successful,
        "failed_images": failures,
        "parseable_images": parseable_images,
        "parseable_fraction_of_successful": parseable_images / successful if successful else 0.0,
        "detections": detections_total,
        "elapsed_seconds": elapsed,
        "images_per_second": len(rows) / elapsed if elapsed else 0.0,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
