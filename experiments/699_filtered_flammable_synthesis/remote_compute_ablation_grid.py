from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "exp699_remote_compute_ablation_grid_v1"
SESSION_RE = re.compile(r"[a-z0-9][a-z0-9-]{2,79}")
RUNS_ROOT = Path("/workspace/runs/exp699")
REPO_ROOT = Path("/workspace/repos/quality")
PYTHON = Path("/workspace/.venv-exp699/bin/python")
DATA_ROOT = Path("/workspace/data/exp699")
IMAGE_CACHE = Path("/workspace/.cache/exp699_images")
VENDOR = Path("/workspace/data/exp699/vendor")
DEFAULT_CODE_ROOT = REPO_ROOT / "experiments/699_filtered_flammable_synthesis"


@dataclass(frozen=True)
class Job:
    name: str
    gpu: int
    architecture: str
    source: str
    mode: str
    cap: int
    epochs: int
    synthetic_repeat: int
    fold: int
    run_root: Path
    log_path: Path
    code_root: Path

    @property
    def contract_path(self) -> Path:
        return self.run_root / "output" / f"fold{self.fold}" / "output_contract.json"


def _path_under(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{label} must be under {root}")
    return resolved


def load_spec(path: Path, *, detected_gpus: int | None = None) -> list[Job]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("grid schema mismatch")
    if payload.get("experiment_id") != "699":
        raise ValueError("grid experiment mismatch")
    max_gpus = payload.get("max_gpus")
    if isinstance(max_gpus, bool) or not isinstance(max_gpus, int) or not 1 <= max_gpus <= 8:
        raise ValueError("max_gpus must be an integer in [1,8]")
    if detected_gpus is not None and max_gpus > detected_gpus:
        raise ValueError(f"grid requests {max_gpus} GPUs but only {detected_gpus} detected")
    rows = payload.get("jobs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("grid jobs must be a non-empty list")
    jobs: list[Job] = []
    for raw in rows:
        if not isinstance(raw, dict):
            raise TypeError("grid job must be an object")
        name = raw.get("name")
        if not isinstance(name, str) or SESSION_RE.fullmatch(name) is None:
            raise ValueError("invalid tmux session name")
        gpu = raw.get("gpu")
        fold = raw.get("fold")
        cap = raw.get("cap")
        epochs = raw.get("epochs", 1)
        synthetic_repeat = raw.get("synthetic_repeat", 1)
        if isinstance(gpu, bool) or not isinstance(gpu, int) or not 0 <= gpu < max_gpus:
            raise ValueError(f"invalid GPU for {name}")
        if isinstance(fold, bool) or not isinstance(fold, int) or fold not in range(5):
            raise ValueError(f"invalid fold for {name}")
        if isinstance(cap, bool) or not isinstance(cap, int) or cap not in {5, 10, 19, 40, 80}:
            raise ValueError(f"invalid cap for {name}")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs not in {1, 2}:
            raise ValueError(f"invalid epochs for {name}")
        if (
            isinstance(synthetic_repeat, bool)
            or not isinstance(synthetic_repeat, int)
            or synthetic_repeat not in {1, 2, 4}
        ):
            raise ValueError(f"invalid synthetic_repeat for {name}")
        architecture = raw.get("architecture")
        source = raw.get("source")
        mode = raw.get("mode")
        if architecture not in {"qwen35_4b", "qwen3vl_2b"}:
            raise ValueError(f"invalid architecture for {name}")
        if source not in {"v1", "v2", "both"} or mode not in {"positive_only", "balanced"}:
            raise ValueError(f"invalid synth selector for {name}")
        run_root = _path_under(Path(raw["run_root"]), RUNS_ROOT, "run_root")
        log_path = _path_under(Path(raw["log_path"]), RUNS_ROOT, "log_path")
        code_root = _path_under(
            Path(raw.get("code_root", str(DEFAULT_CODE_ROOT))),
            Path("/workspace"),
            "code_root",
        )
        jobs.append(
            Job(
                name,
                gpu,
                architecture,
                source,
                mode,
                cap,
                epochs,
                synthetic_repeat,
                fold,
                run_root,
                log_path,
                code_root,
            )
        )
    for label, values in {
        "job name": [job.name for job in jobs],
        "GPU": [job.gpu for job in jobs],
        "run root": [str(job.run_root) for job in jobs],
        "log path": [str(job.log_path) for job in jobs],
    }.items():
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {label} in concurrent grid")
    return jobs


def detected_gpu_count() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    )
    return len([line for line in result.stdout.splitlines() if line.strip()])


def occupied_gpu_indices() -> set[int]:
    gpu_rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    uuid_to_index = {
        uuid.strip(): int(index.strip())
        for index, uuid in (row.split(",", 1) for row in gpu_rows if row.strip())
    }
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    return {
        uuid_to_index[uuid.strip()]
        for uuid, _pid in (row.split(",", 1) for row in processes if row.strip())
        if uuid.strip() in uuid_to_index
    }


def command(job: Job) -> list[str]:
    runner = job.code_root / "run_remote_compute_append.py"
    return [
        str(PYTHON),
        "-u",
        str(runner),
        "--architecture",
        job.architecture,
        "--source",
        job.source,
        "--mode",
        job.mode,
        "--cap",
        str(job.cap),
        "--epochs",
        str(job.epochs),
        "--synthetic-repeat",
        str(job.synthetic_repeat),
        "--fold",
        str(job.fold),
        "--data-root",
        str(DATA_ROOT),
        "--run-root",
        str(job.run_root),
        "--image-cache",
        str(IMAGE_CACHE),
        "--vendor",
        str(VENDOR),
    ]


def session_exists(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name], capture_output=True, check=False
    ).returncode == 0


def launch(jobs: list[Job], *, live: bool) -> list[dict[str, Any]]:
    occupied = occupied_gpu_indices()
    planned: list[dict[str, Any]] = []
    for job in jobs:
        runner = job.code_root / "run_remote_compute_append.py"
        if not runner.is_file() or runner.is_symlink():
            raise FileNotFoundError(f"regular immutable runner missing for {job.name}")
        if session_exists(job.name):
            raise FileExistsError(f"tmux session already exists: {job.name}")
        if job.run_root.exists() or job.log_path.exists():
            raise FileExistsError(f"refusing overwrite for {job.name}")
        if job.gpu in occupied:
            raise RuntimeError(f"GPU {job.gpu} is already occupied")
        argv = command(job)
        planned.append(
            {
                "name": job.name,
                "gpu": job.gpu,
                "fold": job.fold,
                "run_root": str(job.run_root),
                "log_path": str(job.log_path),
                "code_root": str(job.code_root),
                "argv": argv,
            }
        )
    if not live:
        return planned
    for job, item in zip(jobs, planned):
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        shell_command = (
            f"CUDA_VISIBLE_DEVICES={job.gpu} "
            + shlex.join(item["argv"])
            + " > "
            + shlex.quote(str(job.log_path))
            + " 2>&1"
        )
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", job.name, shell_command], check=True
        )
    return planned


def _latest_training_event(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    latest = None
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.startswith("{") or '"phase": "train"' not in line:
                continue
            try:
                latest = json.loads(line)
            except json.JSONDecodeError:
                continue
    return latest


def status(jobs: list[Job]) -> list[dict[str, Any]]:
    result = []
    for job in jobs:
        contract = None
        if job.contract_path.is_file():
            contract = json.loads(job.contract_path.read_text(encoding="utf-8"))
        result.append(
            {
                "name": job.name,
                "gpu": job.gpu,
                "fold": job.fold,
                "state": (
                    "TERMINAL_ARTIFACT"
                    if contract is not None
                    else "RUNNING"
                    if session_exists(job.name)
                    else "NOT_STARTED_OR_FAILED"
                ),
                "latest_training_event": _latest_training_event(job.log_path),
                "contract_decision": None if contract is None else contract.get("decision"),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "launch", "status"))
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    jobs = load_spec(args.spec, detected_gpus=detected_gpu_count())
    if args.action == "plan":
        payload = launch(jobs, live=False)
    elif args.action == "launch":
        payload = launch(jobs, live=True)
    else:
        payload = status(jobs)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
