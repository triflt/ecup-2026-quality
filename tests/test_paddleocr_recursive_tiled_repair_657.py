from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/657_paddleocr_recursive_tiled_repair"


def load(name: str, filename: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, EXP / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPAIR = load("repair_657", "run_recursive_repair.py")
PRESETS = load("presets_657", "build_private_presets.py")


def test_split_tile_bisects_long_axis_with_overlap() -> None:
    wide = {"left": 0, "top": 0, "right": 1000, "bottom": 400}
    wide_children = REPAIR.split_tile(wide)
    assert wide_children == [
        {"left": 0, "top": 0, "right": 550, "bottom": 400},
        {"left": 450, "top": 0, "right": 1000, "bottom": 400},
    ]
    tall = {"left": 0, "top": 0, "right": 400, "bottom": 1000}
    tall_children = REPAIR.split_tile(tall)
    assert tall_children == [
        {"left": 0, "top": 0, "right": 400, "bottom": 550},
        {"left": 0, "top": 450, "right": 400, "bottom": 1000},
    ]


def fake_tiled(*, always_fail: bool = False):  # type: ignore[no-untyped-def]
    def infer(**kwargs):  # type: ignore[no-untyped-def]
        tile = kwargs["tile"]
        width = tile["right"] - tile["left"]
        if always_fail or width > 600:
            return "unfinished", 0.4, []
        return "ok</s>", 0.8, [{"text": "ok"}]

    def translate(detection, *, tile, width, height, detection_index):  # type: ignore[no-untyped-def]
        return {
            "text": detection["text"],
            "tile": tile["tile_index"],
            "width": width,
            "height": height,
            "detection_index": detection_index,
        }

    return SimpleNamespace(
        _infer_tile=infer,
        tile_output_is_accepted=lambda raw, detections: raw.endswith("</s>") and bool(detections),
        translate_detection=translate,
        LOC_TOKEN_PATTERN=SimpleNamespace(findall=lambda raw: []),
    )


def test_process_tree_refines_only_failed_parent_and_keeps_unique_indices() -> None:
    translated, attempts, refined, failures, next_index = REPAIR.process_tree(
        original=object(),
        initial_tile={"left": 0, "top": 0, "right": 1000, "bottom": 400},
        initial_path="3",
        width=1000,
        height=400,
        model=object(),
        processor=object(),
        image_size={},
        torch=object(),
        spotting=SimpleNamespace(),
        tiled=fake_tiled(),
        tile_index_offset=7,
    )
    assert [attempt["tile_path"] for attempt in attempts] == ["3", "3.0", "3.1"]
    assert [attempt["tile_index"] for attempt in attempts] == [7, 8, 9]
    assert attempts[0]["refined"] is True
    assert all(attempt["accepted"] for attempt in attempts[1:])
    assert [row["tile"] for row in translated] == [8, 9]
    assert refined == 1
    assert failures == 0
    assert next_index == 10


def test_process_tree_fails_closed_at_frozen_depth() -> None:
    translated, attempts, refined, failures, _ = REPAIR.process_tree(
        original=object(),
        initial_tile={"left": 0, "top": 0, "right": 1024, "bottom": 1024},
        initial_path="0",
        width=1024,
        height=1024,
        model=object(),
        processor=object(),
        image_size={},
        torch=object(),
        spotting=SimpleNamespace(),
        tiled=fake_tiled(always_fail=True),
    )
    assert translated == []
    assert refined == 7
    assert failures == 8
    assert len(attempts) == 15
    assert max(attempt["depth"] for attempt in attempts) == REPAIR.MAX_REFINE_DEPTH


def test_preset_render_adds_recursive_runner_and_keeps_exact_smoke_scope() -> None:
    parent = PRESETS.parent_builder()
    base = """job:
  flavor: test-flavor
  region: test-region
  image: test-image
  preemption: forbidden
  input:
    - {type: model_registry, mrid: public/model/revision, dst: /hf_models/}
"""
    rendered = PRESETS.render(
        parent=parent,
        base_text=base,
        manifest_artifact="manifest-source/output",
        shard_artifacts={shard: f"source-{shard:02d}/output" for shard in range(32)},
        repair_shard=0,
        num_repair_shards=1,
        smoke=True,
        time_limit="4h0m0s",
    )
    assert "run_recursive_repair.py" in rendered
    assert "--tiled-module /work/code/656/run_tiled_repair.py" in rendered
    assert "--candidate-limit 8" in rendered
    assert "name: ocr_refine_smoke" in rendered
    assert rendered.count("type: artifact") == 33
