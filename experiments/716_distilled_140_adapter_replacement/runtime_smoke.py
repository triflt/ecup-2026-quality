from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

EXPECTED_DATA_SHA256 = "4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510"
OUTPUT_RE = re.compile(r"^<комментарий>(.{50,300})<вердикт>(бан|не бан)$", re.S)
REQUIRED_PACKAGE_MEMBERS = {
    "run.py",
    "metadata.json",
    "adapter_qwen3vl/adapter_config.json",
    "adapter_qwen3vl/adapter_model.safetensors",
    "adapter_qwen35/adapter_config.json",
    "adapter_qwen35/adapter_model.safetensors",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_zip(path: Path) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("submission ZIP integrity failure")
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("submission ZIP contains duplicate paths")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                raise ValueError("submission ZIP contains an unsafe path")
        if not REQUIRED_PACKAGE_MEMBERS.issubset(names):
            missing = sorted(REQUIRED_PACKAGE_MEMBERS - set(names))
            raise ValueError(f"submission ZIP lacks required runtime members: {missing}")
        return infos


def output_contract(output: pd.DataFrame, expected_ids: set[str]) -> tuple[bool, bool]:
    if list(output.columns) != ["id", "result"]:
        return False, False
    actual_ids = output["id"].astype(str)
    ids_valid = (
        len(output) == len(expected_ids)
        and actual_ids.nunique() == len(expected_ids)
        and set(actual_ids) == expected_ids
    )
    schema_valid = output["result"].astype(str).map(
        lambda value: OUTPUT_RE.fullmatch(value) is not None
    ).all()
    return bool(ids_valid), bool(schema_valid)


def gpu_snapshot(device: int) -> tuple[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
            f"--id={device}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    memory, utilization = result.stdout.strip().split(",")
    return int(memory.strip()), int(utilization.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--package-report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--qwen-embed-model", type=Path, required=True)
    parser.add_argument("--qwen-instruct-model", type=Path, required=True)
    parser.add_argument("--qwen35-model", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 1:
        raise ValueError("smoke row count must be positive")
    if not args.python.is_file():
        raise FileNotFoundError(f"runtime Python is missing: {args.python}")
    if not args.images.is_dir():
        raise FileNotFoundError(f"canonical image directory is missing: {args.images}")
    if args.work_dir.exists() and any(args.work_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty smoke work directory")
    if args.report.exists():
        raise FileExistsError("refusing to overwrite smoke report")
    if sha256(args.data) != EXPECTED_DATA_SHA256:
        raise ValueError("canonical data checksum mismatch")
    package = json.loads(args.package_report.read_text(encoding="utf-8"))
    if package.get("schema_version") != "exp716_submission_package_v1":
        raise ValueError("package report schema mismatch")
    if package.get("output_sha256") != sha256(args.submission):
        raise ValueError("package report/submission checksum mismatch")
    if package.get("teacher_in_submission") or not package.get("under_5_gib"):
        raise ValueError("package violates deployment constraints")
    for path in (
        args.qwen_embed_model,
        args.qwen_instruct_model,
        args.qwen35_model,
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"mounted model is missing: {path}")

    frame = pd.read_csv(args.data, dtype={"id": str})
    if args.rows > len(frame):
        raise ValueError("smoke row count exceeds canonical data")
    selected = frame.sample(n=args.rows, random_state=20260828).copy()
    args.work_dir.mkdir(parents=True)
    package_dir = args.work_dir / "package"
    package_dir.mkdir()
    safe_zip(args.submission)
    with zipfile.ZipFile(args.submission) as archive:
        archive.extractall(package_dir)
    input_dir = args.work_dir / "input"
    input_dir.mkdir()
    input_path = input_dir / "data.csv"
    selected.to_csv(input_path, index=False)
    (input_dir / "images").symlink_to(args.images.resolve(), target_is_directory=True)
    output_path = args.work_dir / "predictions.csv"
    stdout_path = args.work_dir / "stdout.log"
    stderr_path = args.work_dir / "stderr.log"
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.device),
            "QWEN_EMBED_MODEL_PATH": str(args.qwen_embed_model.resolve()),
            "QWEN_INSTRUCT_MODEL_PATH": str(args.qwen_instruct_model.resolve()),
            "QWEN35_MODEL_PATH": str(args.qwen35_model.resolve()),
            "QWEN_LORA_BATCH_SIZE": "8",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    command = [
        str(args.python),
        "-u",
        str(package_dir / "run.py"),
        "-i",
        str(input_path),
        "-o",
        str(output_path),
    ]
    started = time.monotonic()
    peak_memory_mib = 0
    max_utilization = 0
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=args.work_dir,
            env=environment,
            stdout=stdout,
            stderr=stderr,
        )
        while process.poll() is None:
            try:
                memory, utilization = gpu_snapshot(args.device)
                peak_memory_mib = max(peak_memory_mib, memory)
                max_utilization = max(max_utilization, utilization)
            except Exception:
                pass
            time.sleep(1)
        return_code = process.returncode
    elapsed = time.monotonic() - started
    result = {
        "schema_version": "exp716_runtime_smoke_v1",
        "experiment_id": "716",
        "rows": args.rows,
        "submission_sha256": sha256(args.submission),
        "package_report_sha256": sha256(args.package_report),
        "return_code": return_code,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_mib": peak_memory_mib,
        "max_gpu_utilization_percent": max_utilization,
        "cuda_visible_devices": str(args.device),
        "model_mounts": {
            "Qwen/Qwen3-VL-Embedding-2B": str(args.qwen_embed_model),
            "Qwen/Qwen3-VL-2B-Instruct": str(args.qwen_instruct_model),
            "Qwen/Qwen3.5-4B": str(args.qwen35_model),
        },
        "output_schema_valid": False,
        "unique_ids_valid": False,
        "decision": "RUNTIME_SMOKE_FAILED",
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    if return_code == 0 and output_path.is_file():
        output = pd.read_csv(output_path, dtype={"id": str})
        expected_ids = set(selected["id"].astype(str))
        ids_valid, schema_valid = output_contract(output, expected_ids)
        result["unique_ids_valid"] = ids_valid
        result["output_schema_valid"] = schema_valid
        result["output_sha256"] = sha256(output_path)
        if args.rows >= 600:
            result["projected_public_minutes_1600"] = elapsed / args.rows * 1600 / 60
            result["projected_private_minutes_3800"] = elapsed / args.rows * 3800 / 60
        if result["unique_ids_valid"] and result["output_schema_valid"]:
            result["decision"] = "RUNTIME_SMOKE_PASS"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["decision"] != "RUNTIME_SMOKE_PASS":
        raise SystemExit(return_code or 1)


if __name__ == "__main__":
    main()
