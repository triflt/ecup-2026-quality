from __future__ import annotations

import importlib.util
import json
from itertools import pairwise
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/656_paddleocr_tiled_repair"


def load(name: str, filename: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, EXP / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPAIR = load("repair_656", "run_tiled_repair.py")
PRESETS = load("presets_656", "build_private_presets.py")


def loc_block(offset: int = 0) -> str:
    return "text" + "".join(f"<|LOC_{offset + value}|>" for value in range(8))


def detection(
    text: str,
    polygon: list[list[int]],
    *,
    tile_index: int,
    detection_index: int = 0,
    edge_margin: float = 0.1,
) -> dict:  # type: ignore[type-arg]
    return {
        "text": text,
        "polygon": polygon,
        "normalized_polygon": polygon,
        "confidence": None,
        "confidence_type": "sequence_geomean_token_probability",
        "_tile_index": tile_index,
        "_detection_index": detection_index,
        "_edge_margin": edge_margin,
    }


def test_tile_gate_requires_eos_complete_blocks_and_full_parse() -> None:
    parsed = [{"text": "text"}]
    assert REPAIR.tile_output_is_accepted(loc_block() + "</s>", parsed) is True
    assert REPAIR.tile_output_is_accepted("</s>", []) is True
    assert REPAIR.tile_output_is_accepted(loc_block(), parsed) is False
    assert REPAIR.tile_output_is_accepted(loc_block()[:-10] + "</s>", parsed) is False
    assert REPAIR.tile_output_is_accepted(loc_block() + "</s>", []) is False
    assert REPAIR.tile_output_is_accepted("unparsed</s>", []) is False


def test_adaptive_plan_is_deterministic_bounded_and_overlapping() -> None:
    sparse = REPAIR.plan_tiles(1200, 800, source_loc_tokens=8)
    dense = REPAIR.plan_tiles(1200, 800, source_loc_tokens=666)
    assert sparse == REPAIR.plan_tiles(1200, 800, source_loc_tokens=8)
    assert 2 <= len(sparse) < len(dense) <= 16
    assert [tile["tile_index"] for tile in dense] == list(range(len(dense)))
    assert all(0 <= tile["left"] < tile["right"] <= 1200 for tile in dense)
    assert all(0 <= tile["top"] < tile["bottom"] <= 800 for tile in dense)
    horizontal_neighbors = [
        (first, second)
        for first, second in pairwise(dense)
        if first["row_index"] == second["row_index"]
    ]
    assert horizontal_neighbors
    assert all(first["right"] > second["left"] for first, second in horizontal_neighbors)


def test_translation_uses_original_image_coordinate_system() -> None:
    tile = {
        "tile_index": 3,
        "row_index": 1,
        "column_index": 1,
        "left": 400,
        "top": 200,
        "right": 800,
        "bottom": 600,
    }
    local = {
        "text": "mark",
        "polygon": [[0, 0], [200, 0], [200, 200], [0, 200]],
        "confidence": 0.875,
        "confidence_type": "sequence_geomean_token_probability",
    }
    translated = REPAIR.translate_detection(
        local,
        tile=tile,
        width=1000,
        height=800,
        detection_index=2,
    )
    assert translated["polygon"] == [[400, 200], [600, 200], [600, 400], [400, 400]]
    assert translated["normalized_polygon"] == [
        [400, 250],
        [600, 250],
        [600, 500],
        [400, 500],
    ]
    assert translated["_tile_index"] == 3
    assert translated["confidence"] == 0.875
    assert translated["confidence_type"] == "sequence_geomean_token_probability"


def test_translation_rejects_invalid_tile_confidence() -> None:
    tile = {
        "tile_index": 0,
        "row_index": 0,
        "column_index": 0,
        "left": 0,
        "top": 0,
        "right": 100,
        "bottom": 100,
    }
    local = {
        "text": "mark",
        "polygon": [[0, 0], [50, 0], [50, 50], [0, 50]],
        "confidence": 1.1,
        "confidence_type": "sequence_geomean_token_probability",
    }
    with pytest.raises(ValueError, match="confidence"):
        REPAIR.translate_detection(
            local,
            tile=tile,
            width=100,
            height=100,
            detection_index=0,
        )


def test_strict_dedupe_is_deterministic_and_keeps_distinct_text_or_geometry() -> None:
    duplicate_edge = detection(
        "same text",
        [[100, 100], [200, 100], [200, 200], [100, 200]],
        tile_index=0,
        edge_margin=0.01,
    )
    duplicate_interior = detection(
        "same   text",
        [[102, 102], [202, 102], [202, 202], [102, 202]],
        tile_index=1,
        edge_margin=0.20,
    )
    different_text = detection(
        "other text",
        [[102, 102], [202, 102], [202, 202], [102, 202]],
        tile_index=2,
    )
    separate = detection(
        "same text",
        [[400, 400], [500, 400], [500, 500], [400, 500]],
        tile_index=3,
    )
    first, removed = REPAIR.deduplicate_detections(
        [duplicate_edge, different_text, separate, duplicate_interior]
    )
    second, removed_again = REPAIR.deduplicate_detections(
        [separate, duplicate_interior, duplicate_edge, different_text]
    )
    assert first == second
    assert removed == removed_again == 1
    assert len(first) == 3
    assert any(item["polygon"][0] == [102, 102] and item["text"] == "same   text" for item in first)
    assert all(not any(key.startswith("_") for key in item) for item in first)


def test_canonical_merge_generation_preserves_eight_loc_tokens_per_detection() -> None:
    detections = [
        {
            "text": "one",
            "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "normalized_polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
        },
        {
            "text": "two",
            "polygon": [[20, 20], [30, 20], [30, 30], [20, 30]],
            "normalized_polygon": [[20, 20], [30, 20], [30, 30], [20, 30]],
        },
    ]
    raw = REPAIR.canonical_raw_generation(detections)
    assert raw.endswith("</s>")
    assert len(REPAIR.LOC_TOKEN_PATTERN.findall(raw)) == 16
    assert REPAIR.tile_output_is_accepted(raw, detections)


def test_candidate_order_and_source_scope_are_deterministic() -> None:
    complete = loc_block() + "</s>"
    rows = {
        9: {
            "id": "nine",
            "image_index": 0,
            "raw_generation": "bad",
            "detections": [],
            "error": None,
        },
        2: {
            "id": "two",
            "image_index": 0,
            "raw_generation": complete,
            "detections": [{"text": "text"}],
            "error": None,
        },
        4: {
            "id": "four",
            "image_index": 0,
            "raw_generation": loc_block(),
            "detections": [{"text": "text"}],
            "error": None,
        },
    }
    candidates = REPAIR.identify_candidates(rows)
    assert [row["global_index"] for row in candidates] == [4, 9]
    assert candidates[0]["source_loc_tokens"] == 8


def test_preset_builder_uses_native_artifacts_and_exact_smoke_bound() -> None:
    base = """job:
  flavor: test-flavor
  region: test-region
  image: test-image
  preemption: forbidden
  input:
    - {type: model_registry, mrid: public/model/revision, dst: /hf_models/}
"""
    artifacts = {shard: f"source-{shard:02d}/output" for shard in range(32)}
    rendered = PRESETS.render(
        base_text=base,
        manifest_artifact="manifest-source/output",
        shard_artifacts=artifacts,
        repair_shard=0,
        num_repair_shards=1,
        smoke=True,
        time_limit="4h0m0s",
    )
    assert rendered.count("type: artifact") == 33
    assert "src: source-00/output" in rendered
    assert "src: source-31/output" in rendered
    assert "--repair-shard-index 0 --num-repair-shards 1 --candidate-limit 8" in rendered
    assert "generate_name: paddleocr" in rendered
    assert "RUNTIME_URL" not in rendered
    assert "https://" not in rendered


def test_source_artifact_manifest_rejects_incomplete_scope(tmp_path: Path) -> None:
    path = tmp_path / "source_artifacts.json"
    path.write_text(
        json.dumps(
            {
                "manifest_artifact": "manifest/output",
                "records": [
                    {"shard": shard, "artifact": f"source-{shard:02d}/output"}
                    for shard in range(31)
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly shards"):
        PRESETS.source_artifacts(path)


def test_source_artifact_manifest_rejects_url(tmp_path: Path) -> None:
    path = tmp_path / "source_artifacts.json"
    path.write_text(
        json.dumps(
            {
                "manifest_artifact": "https://example.invalid/private",
                "records": [
                    {"shard": shard, "artifact": f"source-{shard:02d}/output"}
                    for shard in range(32)
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="manifest artifact"):
        PRESETS.source_artifacts(path)


def test_private_preset_paths_must_live_under_local_mlc(tmp_path: Path) -> None:
    allowed = tmp_path / ".local" / "compute" / "presets"
    assert PRESETS._is_ignored_local_path(allowed) is True
    assert PRESETS._is_ignored_local_path(tmp_path / "tracked") is False
