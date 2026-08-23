from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/651_qwen_mature_prompt_benchmark"


def test_frozen_models_and_protocol() -> None:
    spec = json.loads((EXPERIMENT / "frozen_spec.json").read_text(encoding="utf-8"))
    assert spec["parent_experiment"] == "640"
    assert spec["runtime_rows"] == 11118
    assert spec["technical_smoke_rows"] == 20
    assert spec["screen_folds"] == [0, 3]
    assert spec["thinking"] is False
    assert spec["threshold"] == 0.0
    assert spec["threshold_tuned"] is False
    assert spec["public_used_for_selection"] is False
    assert spec["sealed_rows_used"] == 0
    models = {item["id"]: item for item in spec["models"]}
    assert models["Qwen/Qwen3.6-27B"]["revision"] == "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    assert models["Qwen/Qwen3.5-122B-A10B"]["revision"] == "dc4d348443bc740c68e2d77492492c11606384d5"
    assert models["Qwen/Qwen3.5-122B-A10B"]["gpus"] == 4


def test_prompt_runner_rejects_cpu_or_disk_offload() -> None:
    source = (
        ROOT / "experiments/640_qwen38_scale_prompt_grid/run_prompt_scores.py"
    ).read_text(encoding="utf-8")
    assert 'model_kwargs["device_map"] = "auto"' in source
    assert '{"cpu", "disk"}' in source
    assert 'parser.add_argument("--experiment-id"' in source
