from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import time
import unicodedata
from pathlib import Path
from types import ModuleType
from typing import Any

MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
MODEL_REVISION = "c5630abae1d940eafe0697512a0325494b02ab42"
PROMPT = "Spotting:"
SOURCE_MAX_NEW_TOKENS = 512
TILE_MAX_NEW_TOKENS = 512
TARGET_LOC_TOKENS_PER_TILE = 160
MIN_INITIAL_TILES = 2
MAX_INITIAL_TILES = 16
MAX_AXIS_TILES = 8
OVERLAP_FRACTION = 0.10
DEDUPE_IOU_THRESHOLD = 0.75
BOUNDED_SMOKE_ROWS = 8
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


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def truncation_reasons(row: dict[str, Any]) -> list[str]:
    if row.get("error") is not None:
        raise ValueError("source OCR contains a processing error")
    raw = row.get("raw_generation")
    detections = row.get("detections")
    if not isinstance(raw, str):
        raise TypeError("source raw_generation must be a string")
    if not isinstance(detections, list):
        raise TypeError("source detections must be a list")
    stripped = raw.strip()
    reasons: list[str] = []
    if len(LOC_TOKEN_PATTERN.findall(raw)) % 8:
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
        if not reasons:
            continue
        raw = str(source["raw_generation"])
        candidates.append(
            {
                "global_index": global_index,
                "id": str(source["id"]),
                "image_index": int(source["image_index"]),
                "source_loc_tokens": len(LOC_TOKEN_PATTERN.findall(raw)),
                "reasons": reasons,
            }
        )
    return candidates


def _choose_grid(width: int, height: int, *, source_loc_tokens: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if source_loc_tokens < 0:
        raise ValueError("source LOC token count must be nonnegative")
    required = max(
        MIN_INITIAL_TILES,
        math.ceil(source_loc_tokens / TARGET_LOC_TOKENS_PER_TILE),
    )
    required = min(MAX_INITIAL_TILES, required)
    candidates: list[tuple[float, int, float, int, int]] = []
    for rows in range(1, MAX_AXIS_TILES + 1):
        for columns in range(1, MAX_AXIS_TILES + 1):
            cells = rows * columns
            if not required <= cells <= MAX_INITIAL_TILES:
                continue
            tile_aspect = (width / columns) / (height / rows)
            aspect_error = abs(math.log(tile_aspect))
            extra_cells = cells - required
            cost = aspect_error + 0.75 * extra_cells
            candidates.append((cost, extra_cells, aspect_error, rows, columns))
    if not candidates:
        raise ValueError("cannot construct a bounded tile grid")
    _, _, _, rows, columns = min(candidates)
    return rows, columns


def _axis_windows(length: int, parts: int) -> list[tuple[int, int]]:
    if length <= 0 or parts <= 0 or parts > length:
        raise ValueError("invalid axis partition")
    windows: list[tuple[int, int]] = []
    for index in range(parts):
        core_start = (index * length) // parts
        core_end = ((index + 1) * length) // parts
        padding = math.ceil((core_end - core_start) * OVERLAP_FRACTION)
        start = core_start if index == 0 else max(0, core_start - padding)
        end = core_end if index == parts - 1 else min(length, core_end + padding)
        if start >= end:
            raise ValueError("empty tile axis window")
        windows.append((start, end))
    return windows


def plan_tiles(width: int, height: int, *, source_loc_tokens: int) -> list[dict[str, int]]:
    rows, columns = _choose_grid(width, height, source_loc_tokens=source_loc_tokens)
    if columns > width or rows > height:
        raise ValueError("image is too small for the required tile grid")
    x_windows = _axis_windows(width, columns)
    y_windows = _axis_windows(height, rows)
    tiles: list[dict[str, int]] = []
    for row_index, (top, bottom) in enumerate(y_windows):
        for column_index, (left, right) in enumerate(x_windows):
            tiles.append(
                {
                    "tile_index": len(tiles),
                    "row_index": row_index,
                    "column_index": column_index,
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                }
            )
    return tiles


def tile_output_is_accepted(raw: str, detections: list[dict[str, Any]]) -> bool:
    stripped = raw.strip()
    loc_count = len(LOC_TOKEN_PATTERN.findall(raw))
    if not stripped.endswith("</s>") or loc_count % 8:
        return False
    if stripped == "</s>":
        return not detections and loc_count == 0
    return loc_count > 0 and len(detections) == loc_count // 8


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _polygon_bbox(polygon: list[list[int]]) -> tuple[int, int, int, int]:
    if len(polygon) != 4 or any(len(point) != 2 for point in polygon):
        raise ValueError("polygon must contain four [x, y] points")
    xs = [int(point[0]) for point in polygon]
    ys = [int(point[1]) for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _polygon_area(polygon: list[list[int]]) -> float:
    return abs(
        sum(
            polygon[index][0] * polygon[(index + 1) % 4][1]
            - polygon[(index + 1) % 4][0] * polygon[index][1]
            for index in range(4)
        )
    ) / 2.0


def translate_detection(
    detection: dict[str, Any],
    *,
    tile: dict[str, int],
    width: int,
    height: int,
    detection_index: int,
) -> dict[str, Any]:
    text = detection.get("text")
    polygon = detection.get("polygon")
    confidence = detection.get("confidence")
    confidence_type = detection.get("confidence_type")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("tile detection text must be nonempty")
    if not isinstance(polygon, list):
        raise TypeError("tile detection polygon must be a list")
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError("tile detection confidence must be null or finite in [0, 1]")
    if confidence_type != "sequence_geomean_token_probability":
        raise ValueError("unexpected tile detection confidence semantics")
    translated_points: list[list[int]] = []
    for point in polygon:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError("tile detection point must be [x, y]")
        x = min(width, max(0, int(point[0]) + tile["left"]))
        y = min(height, max(0, int(point[1]) + tile["top"]))
        translated_points.append([x, y])
    normalized = [
        [round(point[0] * 1000 / width), round(point[1] * 1000 / height)]
        for point in translated_points
    ]
    canonical_pixel = [
        [round(point[0] * width / 1000), round(point[1] * height / 1000)]
        for point in normalized
    ]
    if len(set(map(tuple, canonical_pixel))) < 3 or _polygon_area(canonical_pixel) <= 0:
        raise ValueError("translated detection is degenerate in original-image coordinates")
    bbox = _polygon_bbox(canonical_pixel)
    tile_width = tile["right"] - tile["left"]
    tile_height = tile["bottom"] - tile["top"]
    edge_margin = min(
        bbox[0] - tile["left"],
        bbox[1] - tile["top"],
        tile["right"] - bbox[2],
        tile["bottom"] - bbox[3],
    ) / max(1, min(tile_width, tile_height))
    return {
        "text": text.strip(),
        "polygon": canonical_pixel,
        "normalized_polygon": normalized,
        "confidence": float(confidence) if confidence is not None else None,
        "confidence_type": confidence_type,
        "_tile_index": tile["tile_index"],
        "_detection_index": detection_index,
        "_edge_margin": edge_margin,
    }


def deduplicate_detections(
    detections: list[dict[str, Any]],
    *,
    iou_threshold: float = DEDUPE_IOU_THRESHOLD,
) -> tuple[list[dict[str, Any]], int]:
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be in (0, 1]")

    def rank(detection: dict[str, Any]) -> tuple[Any, ...]:
        bbox = _polygon_bbox(detection["polygon"])
        return (
            -float(detection["_edge_margin"]),
            int(detection["_tile_index"]),
            int(detection["_detection_index"]),
            bbox,
            _normalize_text(str(detection["text"])),
        )

    kept: list[dict[str, Any]] = []
    duplicate_count = 0
    for candidate in sorted(detections, key=rank):
        candidate_text = _normalize_text(str(candidate["text"]))
        candidate_bbox = _polygon_bbox(candidate["polygon"])
        duplicate = any(
            candidate_text == _normalize_text(str(existing["text"]))
            and _bbox_iou(candidate_bbox, _polygon_bbox(existing["polygon"])) >= iou_threshold
            for existing in kept
        )
        if duplicate:
            duplicate_count += 1
        else:
            kept.append(candidate)

    kept.sort(
        key=lambda detection: (
            min(point[1] for point in detection["polygon"]),
            min(point[0] for point in detection["polygon"]),
            _normalize_text(str(detection["text"])),
            int(detection["_tile_index"]),
            int(detection["_detection_index"]),
        )
    )
    public = [
        {key: value for key, value in detection.items() if not key.startswith("_")}
        for detection in kept
    ]
    return public, duplicate_count


def canonical_raw_generation(detections: list[dict[str, Any]]) -> str:
    if not detections:
        return "</s>"
    chunks: list[str] = []
    for detection in detections:
        values = [value for point in detection["normalized_polygon"] for value in point]
        chunks.append(
            str(detection["text"])
            + "".join(f"<|LOC_{int(value)}|>" for value in values)
        )
    raw = "\n".join(chunks) + "</s>"
    if len(LOC_TOKEN_PATTERN.findall(raw)) != 8 * len(detections):
        raise ValueError("detection text collides with the structured LOC-token grammar")
    return raw


def _sequence_confidence(torch: Any, generated: Any, tokens: Any) -> float | None:
    probabilities: list[float] = []
    for step, scores in enumerate(generated.scores):
        if step >= len(tokens):
            break
        probability = torch.softmax(scores[0].float(), dim=-1)[tokens[step]]
        probabilities.append(float(probability.cpu()))
    if not probabilities:
        return None
    return math.exp(
        sum(math.log(max(value, 1e-12)) for value in probabilities) / len(probabilities)
    )


def _infer_tile(
    *,
    image: Any,
    tile: dict[str, int],
    model: Any,
    processor: Any,
    image_size: dict[str, int],
    torch: Any,
    spotting: ModuleType,
) -> tuple[str, float | None, list[dict[str, Any]]]:
    cropped = image.crop((tile["left"], tile["top"], tile["right"], tile["bottom"]))
    tile_width, tile_height = cropped.size
    inference_image = cropped
    if tile_width < 1500 and tile_height < 1500:
        from PIL import Image

        resampling = getattr(Image, "Resampling", Image).LANCZOS
        inference_image = cropped.resize((tile_width * 2, tile_height * 2), resampling)
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": inference_image},
                    {"type": "text", "text": PROMPT},
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
                max_new_tokens=TILE_MAX_NEW_TOKENS,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        prompt_length = inputs["input_ids"].shape[-1]
        tokens = generated.sequences[0, prompt_length:]
        raw = processor.decode(tokens, skip_special_tokens=False)
        confidence = _sequence_confidence(torch, generated, tokens)
        detections = spotting.parse_spotting(
            raw,
            width=tile_width,
            height=tile_height,
            sequence_confidence=confidence,
        )
        return raw, confidence, detections
    finally:
        cropped.close()
        if inference_image is not cropped:
            inference_image.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair experiment-633 truncated OCR with deterministic overlapping tiles."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--builder-module", type=Path, required=True)
    parser.add_argument("--spotting-module", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--repair-shard-index", type=int, required=True)
    parser.add_argument("--num-repair-shards", type=int, required=True)
    parser.add_argument("--candidate-limit", type=int)
    args = parser.parse_args()

    if args.model_revision != MODEL_REVISION:
        raise ValueError("model revision differs from the frozen contract")
    if not 0 <= args.repair_shard_index < args.num_repair_shards:
        raise ValueError("invalid repair shard index")
    if args.candidate_limit not in {None, BOUNDED_SMOKE_ROWS}:
        raise ValueError("the only allowed bounded limit is the frozen 8-row smoke")
    if args.candidate_limit is not None and (
        args.repair_shard_index != 0 or args.num_repair_shards != 1
    ):
        raise ValueError("bounded smoke must use the global candidate stream")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.image_cache.mkdir(parents=True, exist_ok=True)

    builder = load_module(args.builder_module, "exp634_builder")
    spotting = load_module(args.spotting_module, "exp633_spotting")
    if spotting.MODEL_ID != MODEL_ID or spotting.MODEL_REVISION != MODEL_REVISION:
        raise ValueError("spotting module differs from the frozen model contract")
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
    if args.candidate_limit is not None:
        selected = selected[: args.candidate_limit]
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
    tile_provenance_path = args.output_dir / "tile_provenance.jsonl"
    started = time.monotonic()
    failures = 0
    accepted = 0
    detections_total = 0
    tiles_total = 0
    failed_tiles = 0
    duplicates_removed = 0
    with (
        output_path.open("w", encoding="utf-8") as output,
        tile_provenance_path.open("w", encoding="utf-8") as provenance_output,
    ):
        for row_index, candidate in enumerate(selected, start=1):
            global_index = int(candidate["global_index"])
            source_manifest = manifest_rows[global_index]
            source_row = source_rows[global_index]
            cache_path = args.image_cache / f"{global_index}.img"
            tile_records: list[dict[str, Any]] = []
            try:
                spotting.download(str(source_manifest["url"]), cache_path)
                with Image.open(cache_path) as source:
                    source.load()
                    original = source.convert("RGB")
                width, height = original.size
                tiles = plan_tiles(
                    width,
                    height,
                    source_loc_tokens=int(candidate["source_loc_tokens"]),
                )
                translated: list[dict[str, Any]] = []
                for tile in tiles:
                    tiles_total += 1
                    try:
                        raw, confidence, detections = _infer_tile(
                            image=original,
                            tile=tile,
                            model=model,
                            processor=processor,
                            image_size=image_size,
                            torch=torch,
                            spotting=spotting,
                        )
                    except Exception as exc:
                        failed_tiles += 1
                        tile_records.append(
                            {
                                **tile,
                                "accepted": False,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        raise
                    tile_accepted = tile_output_is_accepted(raw, detections)
                    tile_record = {
                        **tile,
                        "raw_generation": raw,
                        "raw_generation_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                        "sequence_confidence": confidence,
                        "loc_tokens": len(LOC_TOKEN_PATTERN.findall(raw)),
                        "detections": len(detections),
                        "eos": raw.strip().endswith("</s>"),
                        "accepted": tile_accepted,
                    }
                    tile_records.append(tile_record)
                    if not tile_accepted:
                        failed_tiles += 1
                        raise ValueError(
                            f"tile {tile['tile_index']} violates EOS/LOC/parseability gate"
                        )
                    translated.extend(
                        translate_detection(
                            detection,
                            tile=tile,
                            width=width,
                            height=height,
                            detection_index=detection_index,
                        )
                        for detection_index, detection in enumerate(detections)
                    )
                detections, row_duplicates = deduplicate_detections(translated)
                raw_generation = canonical_raw_generation(detections)
                if not tile_output_is_accepted(raw_generation, detections):
                    raise ValueError("merged row violates the strict output grammar")
                duplicates_removed += row_duplicates
                detections_total += len(detections)
                accepted += 1
                record = {
                    "id": str(source_manifest["id"]),
                    "image_index": int(source_manifest["image_index"]),
                    "global_index": global_index,
                    "width": width,
                    "height": height,
                    "raw_generation": raw_generation,
                    "sequence_confidence": None,
                    "detections": detections,
                    "error": None,
                }
                original.close()
            except Exception as exc:  # noqa: BLE001 - preserve fail-closed row evidence
                failures += 1
                record = {
                    "id": str(source_manifest["id"]),
                    "image_index": int(source_manifest["image_index"]),
                    "global_index": global_index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                cache_path.unlink(missing_ok=True)
            provenance_record = {
                "schema_version": 1,
                "experiment_id": "656",
                "id": str(source_manifest["id"]),
                "image_index": int(source_manifest["image_index"]),
                "global_index": global_index,
                "source_row_sha256": sha256_json(source_row),
                "source_candidate_reasons": candidate["reasons"],
                "source_loc_tokens": candidate["source_loc_tokens"],
                "tiles": tile_records,
                "accepted": record.get("error") is None,
            }
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            provenance_output.write(
                json.dumps(provenance_record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            output.flush()
            provenance_output.flush()
            if row_index == 1 or row_index % 10 == 0 or row_index == len(selected):
                print(
                    "phase=tiled_repair_progress "
                    f"processed={row_index}/{len(selected)} accepted={accepted} failures={failures}",
                    flush=True,
                )

    elapsed = time.monotonic() - started
    selected_indices = [int(candidate["global_index"]) for candidate in selected]
    report = {
        "schema_version": 1,
        "experiment_id": "656",
        "source_experiment_id": "633",
        "rejected_parent_experiment_id": "655",
        "source_manifest_sha256": manifest_sha,
        "source_shards": source_provenance,
        "source_builder_sha256": sha256_file(args.builder_module),
        "source_spotting_code_sha256": sha256_file(args.spotting_module),
        "runtime_code_sha256": sha256_file(Path(__file__)),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt": PROMPT,
        "decoding": "greedy",
        "source_max_new_tokens": SOURCE_MAX_NEW_TOKENS,
        "tile_max_new_tokens": TILE_MAX_NEW_TOKENS,
        "target_loc_tokens_per_tile": TARGET_LOC_TOKENS_PER_TILE,
        "overlap_fraction_per_side": OVERLAP_FRACTION,
        "dedupe_iou_threshold": DEDUPE_IOU_THRESHOLD,
        "all_truncation_candidates": len(all_candidates),
        "repair_shard_index": args.repair_shard_index,
        "num_repair_shards": args.num_repair_shards,
        "bounded_candidate_limit": args.candidate_limit,
        "selected_global_indices_sha256": sha256_json(selected_indices),
        "requested_images": len(selected),
        "successful_images": len(selected) - failures,
        "failed_images": failures,
        "accepted_images": accepted,
        "tiles_attempted": tiles_total,
        "failed_tiles": failed_tiles,
        "detections": detections_total,
        "overlap_duplicates_removed": duplicates_removed,
        "elapsed_seconds": elapsed,
        "spotting_sha256": sha256_file(output_path),
        "tile_provenance_sha256": sha256_file(tile_provenance_path),
        "labels_read": 0,
        "categories_read": 0,
        "folds_read": 0,
        "sealed_membership_used": False,
        "public_used": False,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "phase=report_written "
        f"report_sha256={sha256_file(report_path)} accepted={accepted}/{len(selected)}",
        flush=True,
    )
    if failures or accepted != len(selected):
        raise RuntimeError(
            f"tiled repair shard incomplete: requested={len(selected)} "
            f"failures={failures} accepted={accepted}"
        )


if __name__ == "__main__":
    main()
