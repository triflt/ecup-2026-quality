from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/655_paddleocr_truncation_repair"


def load(name: str, filename: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, EXP / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPAIR = load("repair_655", "run_repair.py")
PRESETS = load("presets_655", "build_private_presets.py")


def row(raw: str, detections: list[dict] | None = None) -> dict:  # type: ignore[type-arg]
    return {
        "id": "item",
        "image_index": 0,
        "global_index": 0,
        "raw_generation": raw,
        "detections": detections or [],
        "error": None,
    }


def test_truncation_detector_keeps_complete_or_canonical_empty_output() -> None:
    complete = "text" + "".join(f"<|LOC_{value}|>" for value in range(8)) + "</s>"
    assert REPAIR.truncation_reasons(row(complete, [{"text": "text"}])) == []
    assert REPAIR.truncation_reasons(row("</s>")) == []
    assert REPAIR.output_is_accepted(complete, [{"text": "text"}]) is True
    assert REPAIR.output_is_accepted("</s>", []) is True


def test_truncation_detector_selects_partial_or_missing_eos() -> None:
    partial = "text" + "".join(f"<|LOC_{value}|>" for value in range(7))
    reasons = REPAIR.truncation_reasons(row(partial))
    assert set(reasons) == {
        "partial_location_block",
        "noncanonical_unparseable_generation",
        "generation_reached_limit_without_eos",
    }
    complete_no_eos = "text" + "".join(f"<|LOC_{value}|>" for value in range(8))
    assert REPAIR.truncation_reasons(row(complete_no_eos, [{"text": "text"}])) == [
        "generation_reached_limit_without_eos"
    ]
    assert REPAIR.output_is_accepted(complete_no_eos, [{"text": "text"}]) is False


def test_candidate_order_is_global_and_deterministic() -> None:
    source = {
        5: row("bad"),
        2: row("</s>"),
        3: row("bad"),
    }
    source[5]["global_index"] = 5
    source[3]["global_index"] = 3
    assert [item["global_index"] for item in REPAIR.identify_candidates(source)] == [3, 5]


def test_preset_builder_renders_exact_artifact_grid() -> None:
    base = """job:
  flavor: test-flavor
  region: test-region
  image: test-image
  preemption: forbidden
  input:
    - {type: model_registry, mrid: public/model/revision, dst: /hf_models/}
"""
    jobs = {shard: f"source-{shard:02d}" for shard in range(32)}
    rendered = PRESETS.render(
        base_text=base,
        jobs=jobs,
        repair_shard=0,
        num_repair_shards=16,
        max_new_tokens=1536,
    )
    assert rendered.count("type: artifact") == 32
    assert "source-00/paddleocr_full_s00" in rendered
    assert "source-31/paddleocr_full_s31" in rendered
    assert "--repair-shard-index 0 --num-repair-shards 16" in rendered
    assert "--max-new-tokens 1536" in rendered
    assert "time_limit: 16h0m0s" in rendered
    assert "RUNTIME_URL: ${RUNTIME_URL}" in rendered


def test_source_jobs_rejects_incomplete_scope(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    path.write_text(
        json.dumps(
            {"records": [{"shard": shard, "job_name": f"source-{shard:02d}"} for shard in range(31)]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly shards"):
        PRESETS.source_jobs(path)
