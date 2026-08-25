from __future__ import annotations

import argparse
import json
import re
from pathlib import PurePosixPath, Path


EXPERIMENT_DIR = "experiments/685_qwen35_4b_structured_rank_distillation"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
STAGE_FOLDS = {"outer0": (0,), "screen": (0, 3), "full": (0, 1, 2, 3, 4)}


def scalar(text: str, key: str, indent: int = 2) -> str:
    match = re.search(
        rf"^{' ' * indent}{re.escape(key)}:\s*(.+?)\s*$", text, re.MULTILINE
    )
    if match is None:
        raise ValueError(f"missing base-preset key: {key}")
    return match.group(1)


def safe_s3_path(value: str, allowed_prefix: str) -> str:
    path = PurePosixPath(value)
    if (
        not value.startswith(allowed_prefix.rstrip("/") + "/")
        or not path.is_absolute()
        or ".." in path.parts
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"unsafe or out-of-scope S3 path: {value}")
    return value.rstrip("/")


def input_spec(
    *,
    bucket: str,
    src: str,
    dst: str,
    file: str | None = None,
    cluster_cache: bool = False,
) -> list[str]:
    lines = [
        "    - type: s3msk",
        f"      src: {json.dumps(src)}",
        f"      dst: {json.dumps(dst)}",
        f"      bucket: {json.dumps(bucket)}",
    ]
    if file:
        if PurePosixPath(file).name != file or file in {".", ".."}:
            raise ValueError("S3 file must be one safe basename")
        lines.insert(2, f"      file: {json.dumps(file)}")
    if cluster_cache:
        lines.extend(
            [
                "      cache:",
                "        enable: true",
                "        location: cluster",
            ]
        )
    return lines


def output_spec(*, bucket: str, dst: str) -> list[str]:
    return [
        "    - type: s3msk",
        "      src: /work/output",
        f"      dst: {json.dumps(dst)}",
        f"      bucket: {json.dumps(bucket)}",
        "      upload_policies:",
        "        - when: on_job_status=succeeded",
    ]


def parse_fold_sources(values: list[str], folds: tuple[int, ...], label: str) -> dict[int, str]:
    output: dict[int, str] = {}
    for value in values:
        fold_text, separator, src = value.partition("=")
        if not separator or not fold_text.isdigit():
            raise ValueError(f"{label} source must use FOLD=/absolute/s3/prefix")
        fold = int(fold_text)
        if fold in output:
            raise ValueError(f"duplicate {label} fold")
        output[fold] = src
    if set(output) != set(folds):
        raise ValueError(f"{label} sources do not exactly cover stage folds")
    return output


def base_values(args: argparse.Namespace) -> dict[str, str]:
    base = args.base_preset.read_text(encoding="utf-8")
    return {
        "time_limit": args.time_limit or scalar(base, "time_limit"),
        "flavor": args.flavor or scalar(base, "flavor"),
        "region": scalar(base, "region"),
        "image": scalar(base, "image"),
        "preemption": scalar(base, "preemption"),
        "work_dir": scalar(base, "work_dir"),
        "tokenizers": scalar(base, "TOKENIZERS_PARALLELISM", indent=4),
        "allocator": scalar(base, "PYTORCH_ALLOC_CONF", indent=4),
    }


def header(args: argparse.Namespace, *, name: str, command: str) -> list[str]:
    values = base_values(args)
    return [
        "job:",
        f"  generate_name: {name}",
        f"  time_limit: {values['time_limit']}",
        f"  flavor: {values['flavor']}",
        f"  region: {values['region']}",
        f"  image: {values['image']}",
        f"  preemption: {values['preemption']}",
        f"  work_dir: {values['work_dir']}",
        "  env:",
        f"    TOKENIZERS_PARALLELISM: {values['tokenizers']}",
        f"    PYTORCH_ALLOC_CONF: {values['allocator']}",
        "  entrypoint: /bin/bash",
        "  args:",
        "    - -lc",
        "    - >-",
        f"      {command}",
    ]


def build_prepare(args: argparse.Namespace) -> str:
    source = safe_s3_path(args.source_src, args.allowed_prefix)
    teacher_runtime = safe_s3_path(args.teacher_runtime_src, args.allowed_prefix)
    teacher_artifact = safe_s3_path(args.teacher_artifact_src, args.allowed_prefix)
    teacher_aggregate = safe_s3_path(args.teacher_aggregate_src, args.allowed_prefix)
    command = (
        "mkdir -p /work/output/runtime && "
        f"PYTHONPATH=/work/code/{EXPERIMENT_DIR} python3 -u "
        f"/work/code/{EXPERIMENT_DIR}/build_pair_runtime.py "
        f"--source-runtime /work/source --teacher-runtime /work/teacher_runtime "
        f"--teacher-artifact /work/teacher_artifact/{args.teacher_artifact_file} "
        f"--teacher-aggregate /work/teacher_aggregate/{args.teacher_aggregate_file} "
        f"--output /work/output/runtime --fold {args.fold} && "
        f"PYTHONPATH=/work/code/{EXPERIMENT_DIR} python3 -u "
        f"/work/code/{EXPERIMENT_DIR}/verify_pair_runtime.py "
        "--runtime /work/output/runtime --output /work/output/r0_acceptance.json"
    )
    lines = header(args, name=f"kd-r0-f{args.fold}", command=command)
    lines.append("  input:")
    for spec in (
        input_spec(bucket=args.bucket, src=args.code_src, dst="/work/code", cluster_cache=True),
        input_spec(bucket=args.bucket, src=source, dst="/work/source"),
        input_spec(
            bucket=args.bucket,
            src=teacher_runtime,
            dst="/work/teacher_runtime",
        ),
        input_spec(
            bucket=args.bucket,
            src=teacher_artifact,
            file=args.teacher_artifact_file,
            dst=f"/work/teacher_artifact/{args.teacher_artifact_file}",
        ),
        input_spec(
            bucket=args.bucket,
            src=teacher_aggregate,
            file=args.teacher_aggregate_file,
            dst=f"/work/teacher_aggregate/{args.teacher_aggregate_file}",
        ),
    ):
        lines.extend(spec)
    lines.append("  output:")
    lines.extend(output_spec(bucket=args.bucket, dst=args.output_dst))
    return "\n".join(lines) + "\n"


def build_train(args: argparse.Namespace) -> str:
    smoke = " --technical-smoke" if args.technical_smoke else ""
    command = (
        "mkdir -p /work/vendor /work/output && "
        "python3 -m zipfile -e "
        "/work/code/research/peft-vendor-extracted/peft-0.20.0.zip /work/vendor && "
        f"PYTHONPATH=/work/code/{EXPERIMENT_DIR}:/work/code/experiments/645_qwen_scale_2x3_gate:/work/vendor "
        f"python3 -u /work/code/{EXPERIMENT_DIR}/train_pair_fold.py "
        f"--fold {args.fold} --runtime-dir /work/source --pair-runtime /work/pair "
        f"--images /work/images --model-root /hf_models --model-revision {MODEL_REVISION} "
        f"--vendor /work/vendor --output-dir /work/output --runtime-backend legacy_eager "
        f"--micro-batch-size-override 2 --mode {args.mode}{smoke} && "
        f"PYTHONPATH=/work/code/{EXPERIMENT_DIR}:/work/code/experiments/645_qwen_scale_2x3_gate:/work/vendor "
        f"python3 -u /work/code/{EXPERIMENT_DIR}/verify_training_artifact.py "
        f"--output-dir /work/output --fold {args.fold} --mode {args.mode} "
        f"--source-runtime /work/source --pair-runtime /work/pair{smoke} "
        "--output /work/output/acceptance.json"
    )
    lines = header(args, name=f"kd-{args.mode[:4]}-f{args.fold}", command=command)
    lines.append("  input:")
    for spec in (
        input_spec(bucket=args.bucket, src=args.code_src, dst="/work/code", cluster_cache=True),
        input_spec(bucket=args.bucket, src=args.source_src, dst="/work/source"),
        input_spec(bucket=args.bucket, src=args.pair_src, dst="/work/pair"),
        input_spec(
            bucket=args.bucket,
            src=args.images_src,
            dst="/work/images",
            cluster_cache=True,
        ),
    ):
        lines.extend(spec)
    model_input = args.model_input_line_file.read_text(encoding="utf-8").strip()
    if "type: model_registry" not in model_input or "/hf_models/" not in model_input:
        raise ValueError("model registry input line is invalid")
    lines.append(f"    - {model_input}")
    lines.append("  output:")
    lines.extend(output_spec(bucket=args.bucket, dst=args.output_dst))
    return "\n".join(lines) + "\n"


def build_eval(args: argparse.Namespace) -> str:
    folds = STAGE_FOLDS[args.eval_stage]
    candidates = parse_fold_sources(args.candidate_src, folds, "candidate")
    controls = parse_fold_sources(args.control_src, folds, "control")
    candidate_args = " ".join(
        f"--candidate-score /work/candidate/fold{fold}/predictions.jsonl "
        f"--candidate-acceptance /work/candidate/fold{fold}/acceptance.json"
        for fold in folds
    )
    control_args = " ".join(
        f"--control-score /work/control/fold{fold}/predictions.jsonl "
        f"--control-acceptance /work/control/fold{fold}/acceptance.json"
        for fold in folds
    )
    command = (
        "mkdir -p /work/output && "
        f"PYTHONPATH=/work/code/{EXPERIMENT_DIR} python3 -u "
        f"/work/code/{EXPERIMENT_DIR}/evaluate.py --stage {args.eval_stage} "
        f"--bundle /work/replay/{args.bundle_file} "
        f"--replay-contract /work/replay/{args.replay_contract_file} "
        f"--registry /work/replay/{args.registry_file} "
        f"{candidate_args} {control_args} --output /work/output/evaluation.json"
    )
    lines = header(args, name=f"kd-eval-{args.eval_stage}", command=command)
    lines.append("  input:")
    lines.extend(
        input_spec(bucket=args.bucket, src=args.code_src, dst="/work/code", cluster_cache=True)
    )
    lines.extend(input_spec(bucket=args.bucket, src=args.replay_src, dst="/work/replay"))
    for fold in folds:
        lines.extend(
            input_spec(
                bucket=args.bucket,
                src=candidates[fold],
                dst=f"/work/candidate/fold{fold}",
            )
        )
        lines.extend(
            input_spec(
                bucket=args.bucket,
                src=controls[fold],
                dst=f"/work/control/fold{fold}",
            )
        )
    lines.append("  output:")
    lines.extend(output_spec(bucket=args.bucket, dst=args.output_dst))
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--stage", choices=("prepare", "train", "eval"), required=True)
    result.add_argument("--base-preset", type=Path, required=True)
    result.add_argument("--bucket", required=True)
    result.add_argument("--allowed-prefix", required=True)
    result.add_argument("--code-src", required=True)
    result.add_argument("--output-dst", required=True)
    result.add_argument("--time-limit")
    result.add_argument("--flavor")
    result.add_argument("--fold", type=int, choices=range(5))
    result.add_argument("--source-src")
    result.add_argument("--teacher-runtime-src")
    result.add_argument("--teacher-artifact-src")
    result.add_argument("--teacher-artifact-file")
    result.add_argument("--teacher-aggregate-src")
    result.add_argument("--teacher-aggregate-file")
    result.add_argument("--pair-src")
    result.add_argument("--images-src")
    result.add_argument("--mode", choices=("paired_hard_control", "rank_candidate"))
    result.add_argument("--technical-smoke", action="store_true")
    result.add_argument("--model-input-line-file", type=Path)
    result.add_argument("--eval-stage", choices=tuple(STAGE_FOLDS))
    result.add_argument("--replay-src")
    result.add_argument("--bundle-file")
    result.add_argument("--replay-contract-file")
    result.add_argument("--registry-file")
    result.add_argument("--candidate-src", action="append", default=[])
    result.add_argument("--control-src", action="append", default=[])
    result.add_argument("--output", type=Path, required=True)
    return result


def require(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = [name for name in names if getattr(args, name) in (None, [], "")]
    if missing:
        raise ValueError(f"missing stage arguments: {missing}")


if __name__ == "__main__":
    args = parser().parse_args()
    args.code_src = safe_s3_path(args.code_src, args.allowed_prefix)
    args.output_dst = safe_s3_path(args.output_dst, args.allowed_prefix)
    if args.stage == "prepare":
        require(
            args,
            (
                "fold",
                "source_src",
                "teacher_runtime_src",
                "teacher_artifact_src",
                "teacher_artifact_file",
                "teacher_aggregate_src",
                "teacher_aggregate_file",
            ),
        )
        payload = build_prepare(args)
    elif args.stage == "train":
        require(
            args,
            (
                "fold",
                "source_src",
                "pair_src",
                "images_src",
                "mode",
                "model_input_line_file",
            ),
        )
        for name in ("source_src", "pair_src", "images_src"):
            setattr(args, name, safe_s3_path(getattr(args, name), args.allowed_prefix))
        payload = build_train(args)
    else:
        require(
            args,
            (
                "eval_stage",
                "replay_src",
                "bundle_file",
                "replay_contract_file",
                "registry_file",
                "candidate_src",
                "control_src",
            ),
        )
        args.replay_src = safe_s3_path(args.replay_src, args.allowed_prefix)
        args.candidate_src = [
            f"{value.partition('=')[0]}={safe_s3_path(value.partition('=')[2], args.allowed_prefix)}"
            for value in args.candidate_src
        ]
        args.control_src = [
            f"{value.partition('=')[0]}={safe_s3_path(value.partition('=')[2], args.allowed_prefix)}"
            for value in args.control_src
        ]
        payload = build_eval(args)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite preset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
