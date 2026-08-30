from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import time
from pathlib import Path
from types import ModuleType
from typing import Any

LOC_TOKEN_PATTERN = re.compile(r"<\|LOC_(\d+)\|>")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truncation_reasons(row: dict[str, Any]) -> list[str]:
    if row.get("error") is not None:
        raise ValueError("source OCR contains a processing error")
    raw = str(row["raw_generation"])
    detections = row["detections"]
    if not isinstance(detections, list):
        raise TypeError("source detections must be a list")
    token_count = len(LOC_TOKEN_PATTERN.findall(raw))
    stripped = raw.strip()
    reasons: list[str] = []
    if token_count % 8:
        reasons.append("partial_location_block")
    if not detections and stripped not in {"", "</s>"}:
        reasons.append("noncanonical_unparseable_generation")
    if stripped not in {"", "</s>"} and not stripped.endswith("</s>"):
        reasons.append("generation_reached_limit_without_eos")
    return reasons


def identify_candidates(shard_rows: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for global_index in sorted(shard_rows):
        source = shard_rows[global_index]
        reasons = truncation_reasons(source)
        if reasons:
            candidates.append(
                {
                    "global_index": global_index,
                    "id": str(source["id"]),
                    "image_index": int(source["image_index"]),
                    "reasons": reasons,
                }
            )
    return candidates


def output_is_accepted(raw: str, detections: list[dict[str, Any]]) -> bool:
    stripped = raw.strip()
    return (
        len(LOC_TOKEN_PATTERN.findall(raw)) % 8 == 0
        and (stripped in {"", "</s>"} or stripped.endswith("</s>"))
        and (bool(detections) or stripped in {"", "</s>"})
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair only truncated experiment-633 OCR rows.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--builder-module", type=Path, required=True)
    parser.add_argument("--spotting-module", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--repair-shard-index", type=int, required=True)
    parser.add_argument("--num-repair-shards", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    args = parser.parse_args()

    if not 0 <= args.repair_shard_index < args.num_repair_shards:
        raise ValueError("invalid repair shard index")
    if args.max_new_tokens <= 512:
        raise ValueError("repair generation limit must be greater than the source limit")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.image_cache.mkdir(parents=True, exist_ok=True)

    builder = load_module(args.builder_module, "exp634_builder")
    spotting = load_module(args.spotting_module, "exp633_spotting")
    manifest_sha = sha256_file(args.manifest)
    manifest_rows = builder.load_manifest(
        args.manifest,
        expected_items=builder.EXPECTED_ITEMS,
        expected_images=builder.EXPECTED_IMAGES,
    )
    source_rows, source_provenance = builder.collect_shards(
        args.shard_dir,
        manifest_rows=manifest_rows,
        manifest_sha256=manifest_sha,
    )
    all_candidates = identify_candidates(source_rows)
    selected = [
        candidate
        for repair_order, candidate in enumerate(all_candidates)
        if repair_order % args.num_repair_shards == args.repair_shard_index
    ]
    print(
        "phase=candidate_scan_complete "
        f"all_candidates={len(all_candidates)} selected={len(selected)} "
        f"repair_shard={args.repair_shard_index}/{args.num_repair_shards}",
        flush=True,
    )

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_root,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model_root, local_files_only=True)
    image_size = spotting.image_size_from_model_config(args.model_root)

    output_path = args.output_dir / "spotting.jsonl"
    started = time.monotonic()
    failures = 0
    accepted = 0
    detections_total = 0
    with output_path.open("w", encoding="utf-8") as output:
        for row_index, candidate in enumerate(selected, start=1):
            global_index = int(candidate["global_index"])
            source_manifest = manifest_rows[global_index]
            cache_path = args.image_cache / f"{global_index}.img"
            try:
                spotting.download(str(source_manifest["url"]), cache_path)
                with Image.open(cache_path) as source:
                    source.load()
                    original = source.convert("RGB")
                width, height = original.size
                if width < 1500 and height < 1500:
                    resampling = getattr(Image, "Resampling", Image).LANCZOS
                    image = original.resize((width * 2, height * 2), resampling)
                else:
                    image = original
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "Spotting:"},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    images_kwargs={"size": image_size},
                ).to(model.device)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                prompt_length = inputs["input_ids"].shape[-1]
                tokens = generated.sequences[0, prompt_length:]
                raw = processor.decode(tokens, skip_special_tokens=False)
                selected_probabilities = []
                for step, scores in enumerate(generated.scores):
                    if step >= len(tokens):
                        break
                    probability = torch.softmax(scores[0].float(), dim=-1)[tokens[step]]
                    selected_probabilities.append(float(probability.cpu()))
                confidence = (
                    math.exp(
                        sum(math.log(max(value, 1e-12)) for value in selected_probabilities)
                        / len(selected_probabilities)
                    )
                    if selected_probabilities
                    else None
                )
                detections = spotting.parse_spotting(
                    raw,
                    width=width,
                    height=height,
                    sequence_confidence=confidence,
                )
                detections_total += len(detections)
                accepted += output_is_accepted(raw, detections)
                record = {
                    "id": str(source_manifest["id"]),
                    "image_index": int(source_manifest["image_index"]),
                    "global_index": global_index,
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
            except Exception as exc:  # noqa: BLE001 - preserve row-level failure evidence
                failures += 1
                record = {
                    "id": str(source_manifest["id"]),
                    "image_index": int(source_manifest["image_index"]),
                    "global_index": global_index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                cache_path.unlink(missing_ok=True)
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            if row_index == 1 or row_index % 10 == 0 or row_index == len(selected):
                print(
                    "phase=repair_progress "
                    f"processed={row_index}/{len(selected)} accepted={accepted} failures={failures}",
                    flush=True,
                )

    elapsed = time.monotonic() - started
    report = {
        "schema_version": 1,
        "experiment_id": "655",
        "source_experiment_id": "633",
        "source_manifest_sha256": manifest_sha,
        "source_shards": source_provenance,
        "model_id": spotting.MODEL_ID,
        "model_revision": spotting.MODEL_REVISION,
        "source_max_new_tokens": 512,
        "repair_max_new_tokens": args.max_new_tokens,
        "all_truncation_candidates": len(all_candidates),
        "repair_shard_index": args.repair_shard_index,
        "num_repair_shards": args.num_repair_shards,
        "requested_images": len(selected),
        "successful_images": len(selected) - failures,
        "failed_images": failures,
        "accepted_images": accepted,
        "detections": detections_total,
        "elapsed_seconds": elapsed,
        "output_sha256": sha256_file(output_path),
        "labels_read": 0,
        "folds_read": 0,
        "public_used": False,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failures or accepted != len(selected):
        raise RuntimeError(
            f"repair shard incomplete: requested={len(selected)} failures={failures} accepted={accepted}"
        )


if __name__ == "__main__":
    main()
