from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    title: str
    status: str
    entrypoint: Path
    result_file: Path
    submission_source: Path | None
    arguments: tuple[str, ...]
    data_version: str
    evaluation_version: str


def repository_root(config_path: Path) -> Path:
    for parent in [config_path.parent, *config_path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("repository root with pyproject.toml was not found")


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    root = repository_root(config_path)
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    experiment = raw["experiment"]
    execution = raw.get("execution", {})
    data = raw["data"]
    validation = raw["validation"]
    submission = experiment.get("submission_source")
    return ExperimentConfig(
        experiment_id=str(experiment["id"]),
        title=str(experiment["title"]),
        status=str(experiment["status"]),
        entrypoint=root / experiment["entrypoint"],
        result_file=root / experiment["result_file"],
        submission_source=(root / submission) if submission else None,
        arguments=tuple(str(value) for value in execution.get("arguments", [])),
        data_version=str(data["version"]),
        evaluation_version=str(validation["primary_version"]),
    )


def run_entrypoint(config_path: str | Path, extra_arguments: Sequence[str]) -> int:
    config = load_config(config_path)
    if not config.entrypoint.exists():
        raise FileNotFoundError(config.entrypoint)
    portable = argparse.ArgumentParser(add_help=False)
    portable.add_argument("--data")
    portable.add_argument("--images")
    portable.add_argument("--output-dir")
    portable.add_argument("--model-root")
    portable.add_argument("--set", action="append", default=[])
    known, native_arguments = portable.parse_known_args(list(extra_arguments))
    command = [sys.executable, "-u", str(config.entrypoint), *config.arguments, *native_arguments]
    environment = os.environ.copy()
    root = repository_root(Path(config_path).resolve())
    source = str(root / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    portable_environment = {
        "ECUP_DATA": known.data,
        "ECUP_IMAGES": known.images,
        "ECUP_OUTPUT_DIR": known.output_dir,
        "ECUP_MODEL_ROOT": known.model_root,
    }
    for key, value in portable_environment.items():
        if value:
            environment[key] = str(Path(value).expanduser().resolve())
    for assignment in known.set:
        if "=" not in assignment:
            raise ValueError(f"--set expects KEY=VALUE, got {assignment!r}")
        key, value = assignment.split("=", 1)
        if not key or not key.replace("_", "").isalnum() or not key[0].isalpha():
            raise ValueError(f"invalid environment key: {key!r}")
        environment[key] = value
    return subprocess.run(command, cwd=root, env=environment, check=False).returncode
