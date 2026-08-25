from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path, PurePosixPath

EXPERIMENT_DIR = "experiments/685_qwen35_4b_structured_rank_distillation"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
STAGE_FOLDS = {"outer0": (0,), "screen": (0, 3), "full": (0, 1, 2, 3, 4)}
TEACHER_AGGREGATE = (
    "experiments/662_qwen36_27b_outer_train_scoring/results/full_target_set_acceptance.json"
)


def scalar(text: str, key: str, indent: int = 2) -> str:
    match = re.search(rf"^{' ' * indent}{re.escape(key)}:\s*(.+?)\s*$", text, re.MULTILINE)
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


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe relative path: {value}")
    return path.as_posix()


def sha256_value(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("SHA-256 must be 64 lowercase hexadecimal characters")
    return value


def s3_auth(args: argparse.Namespace) -> tuple[str | None, str | None, str | None]:
    auth_role = getattr(args, "vault_auth_role", None)
    access_ref = getattr(args, "s3_access_key_vault_ref", None)
    secret_ref = getattr(args, "s3_secret_key_vault_ref", None)
    configured = [value is not None for value in (auth_role, access_ref, secret_ref)]
    if any(configured) and not all(configured):
        raise ValueError("vault auth role and both S3 Vault references must be configured together")
    if not any(configured):
        return None, None, None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", auth_role):
        raise ValueError("unsafe Vault auth role")
    vault_pattern = re.compile(r"vault:[A-Za-z0-9_.-]+/data/[A-Za-z0-9_./-]+#[A-Za-z0-9_.-]+")
    for reference in (access_ref, secret_ref):
        if not vault_pattern.fullmatch(reference):
            raise ValueError("unsafe Vault secret reference")
        secret_path = reference.removeprefix("vault:").partition("#")[0]
        if ".." in PurePosixPath(secret_path).parts:
            raise ValueError("unsafe Vault secret path")
    return auth_role, access_ref, secret_ref


def input_spec(
    *,
    bucket: str,
    src: str,
    dst: str,
    file: str | None = None,
    cluster_cache: bool = False,
    access_key_ref: str | None = None,
    secret_key_ref: str | None = None,
) -> list[str]:
    lines = [
        "    - type: s3msk",
        f"      src: {json.dumps(src)}",
        f"      dst: {json.dumps(dst)}",
        f"      bucket: {json.dumps(bucket)}",
    ]
    if (access_key_ref is None) != (secret_key_ref is None):
        raise ValueError("both S3 Vault references are required together")
    if access_key_ref is not None:
        lines.extend(
            [
                f"      access_key: {json.dumps(access_key_ref)}",
                f"      secret_key: {json.dumps(secret_key_ref)}",
            ]
        )
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


def output_spec(
    *,
    bucket: str,
    dst: str,
    access_key_ref: str | None = None,
    secret_key_ref: str | None = None,
) -> list[str]:
    if (access_key_ref is None) != (secret_key_ref is None):
        raise ValueError("both S3 Vault references are required together")
    lines = [
        "    - type: s3msk",
        "      src: /work/output",
        f"      dst: {json.dumps(dst)}",
        f"      bucket: {json.dumps(bucket)}",
    ]
    if access_key_ref is not None:
        lines.extend(
            [
                f"      access_key: {json.dumps(access_key_ref)}",
                f"      secret_key: {json.dumps(secret_key_ref)}",
            ]
        )
    lines.extend(
        [
            "      upload_policies:",
            "        - when: on_job_status=succeeded",
        ]
    )
    return lines


def artifact_input_spec(*, src: str, dst: str) -> list[str]:
    if not re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", src):
        raise ValueError("artifact source must be exact JOB/OUTPUT")
    return [
        "    - type: artifact",
        f"      src: {json.dumps(src)}",
        f"      dst: {json.dumps(dst)}",
    ]


def extract_segment(archive: str, destination: str, expected_sha256: str) -> str:
    expected_sha256 = sha256_value(expected_sha256)
    python = (
        "import pathlib,sys,tarfile;"
        "archive=pathlib.Path(sys.argv[1]);destination=pathlib.Path(sys.argv[2]);"
        "handle=tarfile.open(archive);members=handle.getmembers();"
        "bad=[m.name for m in members if pathlib.PurePosixPath(m.name).is_absolute() "
        "or '..' in pathlib.PurePosixPath(m.name).parts or not (m.isfile() or m.isdir())];"
        "bad and (_ for _ in ()).throw(ValueError('unsafe tar members'));"
        "handle.extractall(destination,members=members)"
    )
    return (
        f"test \"$(sha256sum {shlex.quote(archive)} | cut -d' ' -f1)\" = "
        f'"{expected_sha256}" && mkdir -p {shlex.quote(destination)} && '
        f"python3 -c {shlex.quote(python)} {shlex.quote(archive)} "
        f"{shlex.quote(destination)}"
    )


def code_bootstrap(args: argparse.Namespace) -> str:
    archive = f"/work/input/code/{args.code_bundle_file}"
    return extract_segment(archive, "/work/code", args.code_bundle_sha256)


def code_input(args: argparse.Namespace) -> list[str]:
    _, access_ref, secret_ref = s3_auth(args)
    return input_spec(
        bucket=args.bucket,
        src=args.code_bundle_src,
        file=args.code_bundle_file,
        dst=f"/work/input/code/{args.code_bundle_file}",
        cluster_cache=True,
        access_key_ref=access_ref,
        secret_key_ref=secret_ref,
    )


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
    auth_role, _, _ = s3_auth(args)
    lines = [
        "job:",
        f"  generate_name: {name}",
        f"  time_limit: {values['time_limit']}",
        f"  flavor: {values['flavor']}",
        f"  region: {values['region']}",
        f"  image: {values['image']}",
        f"  preemption: {values['preemption']}",
        f"  work_dir: {values['work_dir']}",
    ]
    if auth_role is not None:
        lines.extend(
            [
                "  vault:",
                f"    auth_role: {json.dumps(auth_role)}",
            ]
        )
    lines.extend(
        [
            "  env:",
            f"    TOKENIZERS_PARALLELISM: {values['tokenizers']}",
            f"    PYTORCH_ALLOC_CONF: {values['allocator']}",
            "  entrypoint: /bin/bash",
            "  args:",
            "    - -lc",
            "    - >-",
            f"      {command}",
        ]
    )
    return lines


def build_bridge(args: argparse.Namespace) -> str:
    _, access_ref, secret_ref = s3_auth(args)
    folds = (0, 3)
    artifacts = parse_fold_sources(args.artifact_src, folds, "legacy artifact")
    score_sha = parse_fold_sources(args.expected_score_sha, folds, "score SHA")
    archive_sha = parse_fold_sources(args.expected_archive_sha, folds, "archive SHA")
    for digest in [*score_sha.values(), *archive_sha.values()]:
        sha256_value(digest)
    fold_args = " ".join(
        f"--fold-input {fold}=/work/legacy/fold{fold} "
        f"--expected-archive-sha {fold}={archive_sha[fold]} "
        f"--expected-score-sha {fold}={score_sha[fold]}"
        for fold in folds
    )
    command = (
        f"{code_bootstrap(args)} && mkdir -p /work/output_parent && "
        f"PYTHONPATH=/work/code/{EXPERIMENT_DIR} python3 -u "
        f"/work/code/{EXPERIMENT_DIR}/bridge_legacy_teacher.py "
        f"{fold_args} --output /work/output"
    )
    lines = header(args, name="kd-bridge-662", command=command)
    lines.append("  input:")
    lines.extend(code_input(args))
    for fold in folds:
        lines.extend(artifact_input_spec(src=artifacts[fold], dst=f"/work/legacy/fold{fold}"))
    lines.append("  output:")
    lines.extend(
        output_spec(
            bucket=args.bucket,
            dst=args.output_dst,
            access_key_ref=access_ref,
            secret_key_ref=secret_ref,
        )
    )
    return "\n".join(lines) + "\n"


def build_prepare(args: argparse.Namespace) -> str:
    _, access_ref, secret_ref = s3_auth(args)
    source_runtime = f"/work/source/{safe_relative(args.source_runtime_rel)}"
    teacher_runtime = f"/work/teacher/{safe_relative(args.teacher_runtime_rel)}"
    teacher_artifact = f"/work/teacher_scores/{safe_relative(args.teacher_artifact_rel)}"
    source_archive = f"/work/input/source/{args.source_bundle_file}"
    teacher_archive = f"/work/input/teacher/{args.teacher_bundle_file}"
    command = (
        f"{code_bootstrap(args)} && "
        f"{extract_segment(source_archive, '/work/source', args.source_bundle_sha256)} && "
        f"{extract_segment(teacher_archive, '/work/teacher', args.teacher_bundle_sha256)} && "
        "mkdir -p /work/output/runtime && "
        f"PYTHONPATH=/work/code/{EXPERIMENT_DIR} python3 -u "
        f"/work/code/{EXPERIMENT_DIR}/build_pair_runtime.py "
        f"--source-runtime {shlex.quote(source_runtime)} "
        f"--teacher-runtime {shlex.quote(teacher_runtime)} "
        f"--teacher-artifact {shlex.quote(teacher_artifact)} "
        f"--teacher-aggregate /work/code/{TEACHER_AGGREGATE} "
        f"--output /work/output/runtime --fold {args.fold} && "
        f"PYTHONPATH=/work/code/{EXPERIMENT_DIR} python3 -u "
        f"/work/code/{EXPERIMENT_DIR}/verify_pair_runtime.py "
        "--runtime /work/output/runtime --output /work/output/r0_acceptance.json && "
        f"cp -a {shlex.quote(source_runtime)} /work/output/source_runtime"
    )
    lines = header(args, name=f"kd-r0-f{args.fold}", command=command)
    lines.append("  input:")
    for spec in (
        code_input(args),
        input_spec(
            bucket=args.bucket,
            src=args.source_bundle_src,
            file=args.source_bundle_file,
            dst=source_archive,
            cluster_cache=True,
            access_key_ref=access_ref,
            secret_key_ref=secret_ref,
        ),
        input_spec(
            bucket=args.bucket,
            src=args.teacher_bundle_src,
            file=args.teacher_bundle_file,
            dst=teacher_archive,
            cluster_cache=True,
            access_key_ref=access_ref,
            secret_key_ref=secret_ref,
        ),
        input_spec(
            bucket=args.bucket,
            src=args.teacher_scores_src,
            dst="/work/teacher_scores",
            access_key_ref=access_ref,
            secret_key_ref=secret_ref,
        ),
    ):
        lines.extend(spec)
    lines.append("  output:")
    lines.extend(
        output_spec(
            bucket=args.bucket,
            dst=args.output_dst,
            access_key_ref=access_ref,
            secret_key_ref=secret_ref,
        )
    )
    return "\n".join(lines) + "\n"


def build_train(args: argparse.Namespace) -> str:
    _, access_ref, secret_ref = s3_auth(args)
    smoke = " --technical-smoke" if args.technical_smoke else ""
    command = (
        f"{code_bootstrap(args)} && "
        "mkdir -p /work/vendor /work/images /work/output && "
        "python3 -m zipfile -e "
        "/work/code/research/peft-vendor-extracted/peft-0.20.0.zip /work/vendor && "
        f"PYTHONPATH=/work/code/{EXPERIMENT_DIR}:/work/code/experiments/645_qwen_scale_2x3_gate:/work/vendor "
        f"python3 -u /work/code/{EXPERIMENT_DIR}/train_pair_fold.py "
        f"--fold {args.fold} --runtime-dir /work/pair/source_runtime "
        "--pair-runtime /work/pair/runtime "
        f"--images /work/images --model-root /hf_models --model-revision {MODEL_REVISION} "
        f"--vendor /work/vendor --output-dir /work/output --runtime-backend legacy_eager "
        f"--micro-batch-size-override 2 --mode {args.mode}{smoke} && "
        f"PYTHONPATH=/work/code/{EXPERIMENT_DIR}:/work/code/experiments/645_qwen_scale_2x3_gate:/work/vendor "
        f"python3 -u /work/code/{EXPERIMENT_DIR}/verify_training_artifact.py "
        f"--output-dir /work/output --fold {args.fold} --mode {args.mode} "
        "--source-runtime /work/pair/source_runtime "
        f"--pair-runtime /work/pair/runtime{smoke} "
        "--output /work/output/acceptance.json"
    )
    lines = header(args, name=f"kd-{args.mode[:4]}-f{args.fold}", command=command)
    lines.append("  input:")
    for spec in (
        code_input(args),
        input_spec(
            bucket=args.bucket,
            src=args.pair_src,
            dst="/work/pair",
            access_key_ref=access_ref,
            secret_key_ref=secret_ref,
        ),
    ):
        lines.extend(spec)
    model_input = args.model_input_line_file.read_text(encoding="utf-8").strip()
    if "type: model_registry" not in model_input or "/hf_models/" not in model_input:
        raise ValueError("model registry input line is invalid")
    lines.append(f"    - {model_input}")
    lines.append("  output:")
    lines.extend(
        output_spec(
            bucket=args.bucket,
            dst=args.output_dst,
            access_key_ref=access_ref,
            secret_key_ref=secret_ref,
        )
    )
    return "\n".join(lines) + "\n"


def build_eval(args: argparse.Namespace) -> str:
    _, access_ref, secret_ref = s3_auth(args)
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
        f"{code_bootstrap(args)} && mkdir -p /work/output && "
        f"PYTHONPATH=/work/code/{EXPERIMENT_DIR} python3 -u "
        f"/work/code/{EXPERIMENT_DIR}/evaluate.py --stage {args.eval_stage} "
        f"--bundle /work/replay/{args.bundle_file} "
        f"--replay-contract /work/replay/{args.replay_contract_file} "
        f"--registry /work/replay/{args.registry_file} "
        f"{candidate_args} {control_args} --output /work/output/evaluation.json"
    )
    lines = header(args, name=f"kd-eval-{args.eval_stage}", command=command)
    lines.append("  input:")
    lines.extend(code_input(args))
    lines.extend(
        input_spec(
            bucket=args.bucket,
            src=args.replay_src,
            dst="/work/replay",
            access_key_ref=access_ref,
            secret_key_ref=secret_ref,
        )
    )
    for fold in folds:
        lines.extend(
            input_spec(
                bucket=args.bucket,
                src=candidates[fold],
                dst=f"/work/candidate/fold{fold}",
                access_key_ref=access_ref,
                secret_key_ref=secret_ref,
            )
        )
        lines.extend(
            input_spec(
                bucket=args.bucket,
                src=controls[fold],
                dst=f"/work/control/fold{fold}",
                access_key_ref=access_ref,
                secret_key_ref=secret_ref,
            )
        )
    lines.append("  output:")
    lines.extend(
        output_spec(
            bucket=args.bucket,
            dst=args.output_dst,
            access_key_ref=access_ref,
            secret_key_ref=secret_ref,
        )
    )
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--stage", choices=("bridge", "prepare", "train", "eval"), required=True)
    result.add_argument("--base-preset", type=Path, required=True)
    result.add_argument("--bucket", required=True)
    result.add_argument("--vault-auth-role")
    result.add_argument("--s3-access-key-vault-ref")
    result.add_argument("--s3-secret-key-vault-ref")
    result.add_argument("--allowed-prefix", required=True)
    result.add_argument("--code-bundle-src", required=True)
    result.add_argument("--code-bundle-file", required=True)
    result.add_argument("--code-bundle-sha256", required=True)
    result.add_argument("--output-dst", required=True)
    result.add_argument("--time-limit")
    result.add_argument("--flavor")
    result.add_argument("--fold", type=int, choices=range(5))
    result.add_argument("--artifact-src", action="append", default=[])
    result.add_argument("--expected-score-sha", action="append", default=[])
    result.add_argument("--expected-archive-sha", action="append", default=[])
    result.add_argument("--source-bundle-src")
    result.add_argument("--source-bundle-file")
    result.add_argument("--source-bundle-sha256")
    result.add_argument("--source-runtime-rel")
    result.add_argument("--teacher-bundle-src")
    result.add_argument("--teacher-bundle-file")
    result.add_argument("--teacher-bundle-sha256")
    result.add_argument("--teacher-runtime-rel")
    result.add_argument("--teacher-scores-src")
    result.add_argument("--teacher-artifact-rel")
    result.add_argument("--pair-src")
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
    args.code_bundle_src = safe_s3_path(args.code_bundle_src, args.allowed_prefix)
    sha256_value(args.code_bundle_sha256)
    args.output_dst = safe_s3_path(args.output_dst, args.allowed_prefix)
    if args.stage == "bridge":
        require(args, ("artifact_src", "expected_score_sha", "expected_archive_sha"))
        payload = build_bridge(args)
    elif args.stage == "prepare":
        require(
            args,
            (
                "fold",
                "source_bundle_src",
                "source_bundle_file",
                "source_bundle_sha256",
                "source_runtime_rel",
                "teacher_bundle_src",
                "teacher_bundle_file",
                "teacher_bundle_sha256",
                "teacher_runtime_rel",
                "teacher_scores_src",
                "teacher_artifact_rel",
            ),
        )
        for name in ("source_bundle_src", "teacher_bundle_src", "teacher_scores_src"):
            setattr(args, name, safe_s3_path(getattr(args, name), args.allowed_prefix))
        payload = build_prepare(args)
    elif args.stage == "train":
        require(
            args,
            (
                "fold",
                "pair_src",
                "mode",
                "model_input_line_file",
            ),
        )
        args.pair_src = safe_s3_path(args.pair_src, args.allowed_prefix)
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
