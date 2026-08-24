from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from types import ModuleType
from typing import Any

EXPERIMENT_ID = "657"
MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
MODEL_REVISION = "c5630abae1d940eafe0697512a0325494b02ab42"
MAX_REFINE_DEPTH = 3
REFINE_OVERLAP_FRACTION = 0.10
BOUNDED_SMOKE_ROWS = 8


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


def split_tile(tile: dict[str, int]) -> list[dict[str, int]]:
    width = tile["right"] - tile["left"]
    height = tile["bottom"] - tile["top"]
    if width < 2 or height < 2:
        raise ValueError("tile is too small to refine")
    if width >= height:
        midpoint = tile["left"] + width // 2
        padding = math.ceil((width / 2) * REFINE_OVERLAP_FRACTION)
        bounds = [
            (tile["left"], tile["top"], min(tile["right"], midpoint + padding), tile["bottom"]),
            (max(tile["left"], midpoint - padding), tile["top"], tile["right"], tile["bottom"]),
        ]
    else:
        midpoint = tile["top"] + height // 2
        padding = math.ceil((height / 2) * REFINE_OVERLAP_FRACTION)
        bounds = [
            (tile["left"], tile["top"], tile["right"], min(tile["bottom"], midpoint + padding)),
            (tile["left"], max(tile["top"], midpoint - padding), tile["right"], tile["bottom"]),
        ]
    children = [
        {"left": left, "top": top, "right": right, "bottom": bottom}
        for left, top, right, bottom in bounds
    ]
    if any(
        child["left"] >= child["right"]
        or child["top"] >= child["bottom"]
        or child == {key: tile[key] for key in ("left", "top", "right", "bottom")}
        for child in children
    ):
        raise ValueError("recursive split did not reduce the tile")
    return children


def process_tree(
    *,
    original: Any,
    initial_tile: dict[str, int],
    initial_path: str,
    width: int,
    height: int,
    model: Any,
    processor: Any,
    image_size: dict[str, int],
    torch: Any,
    spotting: ModuleType,
    tiled: ModuleType,
    tile_index_offset: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int, int]:
    attempts: list[dict[str, Any]] = []
    translated: list[dict[str, Any]] = []
    next_tile_index = tile_index_offset
    refined_attempts = 0
    terminal_failures = 0

    def visit(bounds: dict[str, int], *, path: str, depth: int) -> None:
        nonlocal next_tile_index, refined_attempts, terminal_failures
        tile = {
            "tile_index": next_tile_index,
            "row_index": 0,
            "column_index": 0,
            **bounds,
        }
        next_tile_index += 1
        raw, confidence, detections = tiled._infer_tile(
            image=original,
            tile=tile,
            model=model,
            processor=processor,
            image_size=image_size,
            torch=torch,
            spotting=spotting,
        )
        accepted = tiled.tile_output_is_accepted(raw, detections)
        attempt = {
            "tile_index": tile["tile_index"],
            "tile_path": path,
            "depth": depth,
            "left": tile["left"],
            "top": tile["top"],
            "right": tile["right"],
            "bottom": tile["bottom"],
            "raw_generation": raw,
            "raw_generation_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "sequence_confidence": confidence,
            "loc_tokens": len(tiled.LOC_TOKEN_PATTERN.findall(raw)),
            "detections": len(detections),
            "eos": raw.strip().endswith("</s>"),
            "accepted": accepted,
            "refined": False,
        }
        attempts.append(attempt)
        if accepted:
            translated.extend(
                tiled.translate_detection(
                    detection,
                    tile=tile,
                    width=width,
                    height=height,
                    detection_index=detection_index,
                )
                for detection_index, detection in enumerate(detections)
            )
            return
        if depth >= MAX_REFINE_DEPTH:
            terminal_failures += 1
            return
        attempt["refined"] = True
        refined_attempts += 1
        for child_index, child in enumerate(split_tile(tile)):
            visit(child, path=f"{path}.{child_index}", depth=depth + 1)

    visit(
        {key: int(initial_tile[key]) for key in ("left", "top", "right", "bottom")},
        path=initial_path,
        depth=0,
    )
    return translated, attempts, refined_attempts, terminal_failures, next_tile_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively refine only failed dense OCR tiles under the strict leaf gate."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--builder-module", type=Path, required=True)
    parser.add_argument("--spotting-module", type=Path, required=True)
    parser.add_argument("--tiled-module", type=Path, required=True)
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
    tiled = load_module(args.tiled_module, "exp656_tiled")
    if spotting.MODEL_ID != MODEL_ID or spotting.MODEL_REVISION != MODEL_REVISION:
        raise ValueError("spotting module differs from the frozen model contract")
    if tiled.MODEL_ID != MODEL_ID or tiled.MODEL_REVISION != MODEL_REVISION:
        raise ValueError("tiled module differs from the frozen model contract")
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
    all_candidates = tiled.identify_candidates(source_rows)
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

    model = (
        AutoModelForImageTextToText.from_pretrained(
            args.model_root,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        .to("cuda")
        .eval()
    )
    processor = AutoProcessor.from_pretrained(args.model_root, local_files_only=True)
    image_size = spotting.image_size_from_model_config(args.model_root)

    output_path = args.output_dir / "spotting.jsonl"
    provenance_path = args.output_dir / "tile_provenance.jsonl"
    started = time.monotonic()
    failures = 0
    accepted = 0
    detections_total = 0
    tile_attempts = 0
    refined_attempts = 0
    terminal_failed_tiles = 0
    duplicates_removed = 0
    with (
        output_path.open("w", encoding="utf-8") as output,
        provenance_path.open("w", encoding="utf-8") as provenance_output,
    ):
        for row_index, candidate in enumerate(selected, start=1):
            global_index = int(candidate["global_index"])
            source_manifest = manifest_rows[global_index]
            source_row = source_rows[global_index]
            cache_path = args.image_cache / f"{global_index}.img"
            attempts: list[dict[str, Any]] = []
            original = None
            try:
                spotting.download(str(source_manifest["url"]), cache_path)
                with Image.open(cache_path) as source:
                    source.load()
                    original = source.convert("RGB")
                width, height = original.size
                translated: list[dict[str, Any]] = []
                row_refined = 0
                row_terminal_failures = 0
                row_next_tile_index = 0
                for initial_index, initial_tile in enumerate(
                    tiled.plan_tiles(
                        width,
                        height,
                        source_loc_tokens=int(candidate["source_loc_tokens"]),
                    )
                ):
                    (
                        local,
                        local_attempts,
                        local_refined,
                        local_failures,
                        row_next_tile_index,
                    ) = process_tree(
                        original=original,
                        initial_tile=initial_tile,
                        initial_path=str(initial_index),
                        width=width,
                        height=height,
                        model=model,
                        processor=processor,
                        image_size=image_size,
                        torch=torch,
                        spotting=spotting,
                        tiled=tiled,
                        tile_index_offset=row_next_tile_index,
                    )
                    translated.extend(local)
                    attempts.extend(local_attempts)
                    row_refined += local_refined
                    row_terminal_failures += local_failures
                tile_attempts += len(attempts)
                refined_attempts += row_refined
                terminal_failed_tiles += row_terminal_failures
                if row_terminal_failures:
                    raise ValueError(
                        f"recursive leaf gate failed for {row_terminal_failures} tiles"
                    )
                detections, row_duplicates = tiled.deduplicate_detections(translated)
                raw_generation = tiled.canonical_raw_generation(detections)
                if not tiled.tile_output_is_accepted(raw_generation, detections):
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
            except Exception as exc:  # noqa: BLE001 - preserve fail-closed evidence
                failures += 1
                record = {
                    "id": str(source_manifest["id"]),
                    "image_index": int(source_manifest["image_index"]),
                    "global_index": global_index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                if original is not None:
                    original.close()
                cache_path.unlink(missing_ok=True)
            provenance_record = {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "id": str(source_manifest["id"]),
                "image_index": int(source_manifest["image_index"]),
                "global_index": global_index,
                "source_row_sha256": tiled.sha256_json(source_row),
                "source_candidate_reasons": candidate["reasons"],
                "source_loc_tokens": candidate["source_loc_tokens"],
                "tile_attempts": attempts,
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
                    "phase=recursive_repair_progress "
                    f"processed={row_index}/{len(selected)} accepted={accepted} failures={failures}",
                    flush=True,
                )

    elapsed = time.monotonic() - started
    selected_indices = [int(candidate["global_index"]) for candidate in selected]
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": "633",
        "parent_experiment_id": "656",
        "source_manifest_sha256": manifest_sha,
        "source_shards": source_provenance,
        "source_builder_sha256": sha256_file(args.builder_module),
        "source_spotting_code_sha256": sha256_file(args.spotting_module),
        "parent_tiled_code_sha256": sha256_file(args.tiled_module),
        "runtime_code_sha256": sha256_file(Path(__file__)),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt": tiled.PROMPT,
        "decoding": "greedy",
        "source_max_new_tokens": tiled.SOURCE_MAX_NEW_TOKENS,
        "tile_max_new_tokens": tiled.TILE_MAX_NEW_TOKENS,
        "initial_target_loc_tokens_per_tile": tiled.TARGET_LOC_TOKENS_PER_TILE,
        "initial_overlap_fraction_per_side": tiled.OVERLAP_FRACTION,
        "refine_overlap_fraction_per_side": REFINE_OVERLAP_FRACTION,
        "maximum_refine_depth": MAX_REFINE_DEPTH,
        "dedupe_iou_threshold": tiled.DEDUPE_IOU_THRESHOLD,
        "all_truncation_candidates": len(all_candidates),
        "repair_shard_index": args.repair_shard_index,
        "num_repair_shards": args.num_repair_shards,
        "bounded_candidate_limit": args.candidate_limit,
        "selected_global_indices_sha256": tiled.sha256_json(selected_indices),
        "requested_images": len(selected),
        "successful_images": len(selected) - failures,
        "failed_images": failures,
        "accepted_images": accepted,
        "tile_attempts": tile_attempts,
        "refined_attempts": refined_attempts,
        "terminal_failed_tiles": terminal_failed_tiles,
        "detections": detections_total,
        "overlap_duplicates_removed": duplicates_removed,
        "elapsed_seconds": elapsed,
        "spotting_sha256": sha256_file(output_path),
        "tile_provenance_sha256": sha256_file(provenance_path),
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
            f"recursive repair shard incomplete: requested={len(selected)} "
            f"failures={failures} accepted={accepted}"
        )


if __name__ == "__main__":
    main()
