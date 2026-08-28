from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "exp699_remote_compute_grid", ROOT / "remote_compute_ablation_grid.py"
)
assert SPEC and SPEC.loader
grid = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grid
SPEC.loader.exec_module(grid)


def write_spec(path: Path, jobs: list[dict], max_gpus: int = 8) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": grid.SCHEMA,
                "experiment_id": "699",
                "max_gpus": max_gpus,
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )


def job(name: str, gpu: int, fold: int) -> dict:
    return {
        "name": name,
        "gpu": gpu,
        "architecture": "qwen35_4b",
        "source": "v2",
        "mode": "positive_only",
        "cap": 19,
        "fold": fold,
        "run_root": f"/remote_compute/home/runs/exp699/full5/{name}",
        "log_path": f"/remote_compute/home/runs/exp699/full5/logs/{name}.log",
    }


def test_loads_eight_gpu_grid(tmp_path: Path) -> None:
    path = tmp_path / "grid.json"
    write_spec(path, [job(f"exp699-f{fold}", fold, fold) for fold in range(5)])
    jobs = grid.load_spec(path, detected_gpus=8)
    assert [value.fold for value in jobs] == [0, 1, 2, 3, 4]
    assert jobs[4].gpu == 4


@pytest.mark.parametrize("field", ["name", "gpu", "run_root", "log_path"])
def test_rejects_concurrent_collisions(tmp_path: Path, field: str) -> None:
    rows = [job("exp699-a", 0, 0), job("exp699-b", 1, 1)]
    rows[1][field] = rows[0][field]
    path = tmp_path / "grid.json"
    write_spec(path, rows)
    with pytest.raises(ValueError, match="duplicate"):
        grid.load_spec(path, detected_gpus=8)


def test_rejects_more_gpus_than_detected(tmp_path: Path) -> None:
    path = tmp_path / "grid.json"
    write_spec(path, [job("exp699-a", 0, 0)], max_gpus=8)
    with pytest.raises(ValueError, match="only 4 detected"):
        grid.load_spec(path, detected_gpus=4)


def test_command_is_exact_and_secret_free(tmp_path: Path) -> None:
    path = tmp_path / "grid.json"
    write_spec(path, [job("exp699-a", 7, 4)])
    value = grid.command(grid.load_spec(path, detected_gpus=8)[0])
    assert value[value.index("--fold") + 1] == "4"
    assert value[value.index("--cap") + 1] == "19"
    assert value[value.index("--epochs") + 1] == "1"
    assert not any("secret" in item.lower() or "credential" in item.lower() for item in value)


def test_two_epoch_job_is_explicit_and_bounded(tmp_path: Path) -> None:
    row = job("exp699-two-epochs", 5, 3)
    row["epochs"] = 2
    path = tmp_path / "grid.json"
    write_spec(path, [row])
    value = grid.command(grid.load_spec(path, detected_gpus=8)[0])
    assert value[value.index("--epochs") + 1] == "2"
    row["epochs"] = 3
    write_spec(path, [row])
    with pytest.raises(ValueError, match="epochs"):
        grid.load_spec(path, detected_gpus=8)


@pytest.mark.parametrize("repeat", [2, 4])
def test_synthetic_repeat_is_explicit_and_bounded(
    tmp_path: Path, repeat: int
) -> None:
    row = job(f"exp699-repeat{repeat}", 6, 0)
    row["cap"] = 10
    row["synthetic_repeat"] = repeat
    path = tmp_path / "grid.json"
    write_spec(path, [row])
    value = grid.command(grid.load_spec(path, detected_gpus=8)[0])
    assert value[value.index("--synthetic-repeat") + 1] == str(repeat)
    row["synthetic_repeat"] = 3
    write_spec(path, [row])
    with pytest.raises(ValueError, match="synthetic_repeat"):
        grid.load_spec(path, detected_gpus=8)


def test_command_can_bind_immutable_code_snapshot(tmp_path: Path) -> None:
    row = job("exp699-snapshot", 6, 0)
    row["code_root"] = "/remote_compute/home/runs/exp699/code/epochs2-v1"
    path = tmp_path / "grid.json"
    write_spec(path, [row])
    loaded = grid.load_spec(path, detected_gpus=8)[0]
    value = grid.command(loaded)
    assert value[2] == (
        "/remote_compute/home/runs/exp699/code/epochs2-v1/run_remote_compute_append.py"
    )
