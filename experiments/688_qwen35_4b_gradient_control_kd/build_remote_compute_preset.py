from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "686_qwen35_4b_additive_rank_kd"
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_gradient_control import (
    CONTROL_MODE,
    MODEL_REVISION,
    load_terminal_probe_selection,
    sha256_file,
)

EXP688_DIR = "experiments/688_qwen35_4b_gradient_control_kd"
EXP687_DIR = "experiments/687_qwen35_4b_gradient_conflict_probe"
EXP686_DIR = "experiments/686_qwen35_4b_additive_rank_kd"
SHARED_DIR = "experiments/645_qwen_scale_2x3_gate"
MODEL_INPUT_LINE_SHA256 = (
    "30e01413e4346a5081e9104a48214c7330fb290d147bcc78b806721c6e81ba9a"
)
SCOPE = "paired_technical_smoke"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _scalar(text: str, key: str, indent: int = 2) -> str:
    match = re.search(
        rf"^{' ' * indent}{re.escape(key)}:\s*(.+?)\s*$", text, re.MULTILINE
    )
    if match is None:
        raise ValueError(f"missing base-preset key: {key}")
    return match.group(1)


def _lower_hex(value: str, length: int) -> str:
    if not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError(f"expected {length}-character lowercase hexadecimal value")
    return value


def _safe_basename(value: str) -> str:
    if PurePosixPath(value).name != value or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", value
    ):
        raise ValueError("bundle filename must be one shell-safe basename")
    return value


def _safe_s3(value: str, prefix: str) -> str:
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or not value.startswith(prefix.rstrip("/") + "/")
        or ".." in path.parts
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"S3 path escapes frozen scope: {value}")
    return value.rstrip("/")


def _load_self_hashed(path: Path, hash_field: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required acceptance is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    body = dict(value)
    digest = body.pop(hash_field, None)
    if digest != canonical_sha256(body):
        raise ValueError(f"acceptance self-hash mismatch: {path.name}")
    return value


def _validate_code_acceptance(
    path: Path,
    *,
    experiment_id: str,
    decision: str,
    revision: str,
    bundle_sha256: str,
    scope: str | None = None,
) -> dict[str, Any]:
    value = _load_self_hashed(path, "acceptance_sha256")
    expected = {
        "experiment_id": experiment_id,
        "git_revision": revision,
        "bundle_sha256": bundle_sha256,
        "decision": decision,
    }
    if scope is not None:
        expected["scope"] = scope
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError(f"experiment-{experiment_id} code acceptance mismatch")
    return value


def _read_s3_credentials(path: Path) -> tuple[str, str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("S3 environment file must be a regular ignored file")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if not separator:
            raise ValueError("invalid S3 environment-file line")
        values[key.strip()] = value.strip().strip("'\"")
    access = [
        value
        for key, value in values.items()
        if key.endswith("_ACCESS_KEY") and not key.endswith("_SECRET_ACCESS_KEY")
    ]
    secret = [value for key, value in values.items() if key.endswith("_SECRET_ACCESS_KEY")]
    if len(access) != 1 or len(secret) != 1 or not access[0] or not secret[0]:
        raise ValueError("S3 environment file must contain exactly one access/secret pair")
    return access[0], secret[0]


def _require_sensitive_output_ignored(repo: Path, path: Path) -> None:
    resolved_repo = repo.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_repo):
        return
    relative = resolved_path.relative_to(resolved_repo)
    result = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", "--quiet", "--", str(relative)],
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("raw and secret-override presets must be git-ignored")


def _extract(archive: str, destination: str, expected_sha256: str) -> str:
    python = (
        "import pathlib,sys,tarfile;"
        "a=pathlib.Path(sys.argv[1]);d=pathlib.Path(sys.argv[2]);"
        "h=tarfile.open(a);m=h.getmembers();"
        "p=[pathlib.PurePosixPath(x.name) for x in m];"
        "n=[x.as_posix().removeprefix('./') for x in p];"
        "b=[x.name for x,y,z in zip(m,p,n) if y.is_absolute() or '..' in y.parts "
        "or any(q == '__MACOSX' or q == '.DS_Store' or q.startswith('._') for q in y.parts) "
        "or not z or not (x.isfile() or x.isdir())];"
        "b and (_ for _ in ()).throw(ValueError('unsafe tar members'));"
        "len(n)!=len(set(n)) and (_ for _ in ()).throw(ValueError('duplicate tar members'));"
        "h.extractall(d,members=m)"
    )
    return (
        f'test "$(sha256sum {shlex.quote(archive)} | cut -d\' \' -f1)" = '
        f'"{expected_sha256}" && mkdir -p {shlex.quote(destination)} && '
        f"python3 -c {shlex.quote(python)} {shlex.quote(archive)} "
        f"{shlex.quote(destination)}"
    )


def _input_spec(
    *,
    name: str,
    bucket: str,
    src: str,
    dst: str,
    credentials: tuple[str, str] | None,
    file: str | None = None,
    cache: bool = False,
) -> list[str]:
    lines = [
        "    - type: s3msk",
        f"      name: {json.dumps(name)}",
        f"      src: {json.dumps(src)}",
    ]
    if file is not None:
        lines.append(f"      file: {json.dumps(file)}")
    lines.extend(
        [
            f"      dst: {json.dumps(dst)}",
            f"      bucket: {json.dumps(bucket)}",
        ]
    )
    if credentials is not None:
        lines.extend(
            [
                f"      access_key: {json.dumps(credentials[0])}",
                f"      secret_key: {json.dumps(credentials[1])}",
            ]
        )
    if cache:
        lines.extend(["      cache:", "        enable: true", "        location: cluster"])
    return lines


def _output_spec(
    *, name: str, bucket: str, dst: str, credentials: tuple[str, str] | None
) -> list[str]:
    lines = [
        "    - type: s3msk",
        f"      name: {json.dumps(name)}",
        "      src: /work/output",
        f"      dst: {json.dumps(dst)}",
        f"      bucket: {json.dumps(bucket)}",
    ]
    if credentials is not None:
        lines.extend(
            [
                f"      access_key: {json.dumps(credentials[0])}",
                f"      secret_key: {json.dumps(credentials[1])}",
            ]
        )
    lines.extend(["      upload_policies:", "        - when: on_job_status=succeeded"])
    return lines


def _header(base: str, command: str) -> list[str]:
    values = {
        "time_limit": _scalar(base, "time_limit"),
        "flavor": _scalar(base, "flavor"),
        "region": _scalar(base, "region"),
        "image": _scalar(base, "image"),
        "preemption": _scalar(base, "preemption"),
        "work_dir": _scalar(base, "work_dir"),
        "tokenizers": _scalar(base, "TOKENIZERS_PARALLELISM", indent=4),
        "allocator": _scalar(base, "PYTORCH_ALLOC_CONF", indent=4),
    }
    if (
        values["flavor"] != "h100-1x"
        or values["region"] != "ix-m5-sm11"
        or values["time_limit"] != "3h"
    ):
        raise ValueError("base preset is not the frozen one-H100 exp687 recipe")
    return [
        "job:",
        "  generate_name: kd-gctrl-paired-smoke-f3",
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


def _train_command(args: argparse.Namespace, mode: str, output_dir: str) -> str:
    pythonpath = (
        f"/work/code/{EXP688_DIR}:/work/code/{EXP687_DIR}:"
        f"/work/code/{EXP686_DIR}:/work/code/{SHARED_DIR}:/work/vendor"
    )
    common = (
        "--fold 3 --runtime-dir /work/pair_clean/source_runtime "
        "--pair-runtime /work/pair_clean/runtime "
        "--transport-acceptance /work/pair_clean/transport_acceptance.json "
        "--parent-code-acceptance /work/parent_code_acceptance.json "
        "--probe-code-acceptance /work/probe_code_acceptance.json "
        "--probe-report /work/probe_artifact/gradient_conflict_report.json "
        "--probe-acceptance /work/probe_artifact/acceptance.json "
        f"--code-bundle /work/input/exp688_code/{args.exp688_bundle_file} "
        f"--expected-code-bundle-sha256 {args.exp688_bundle_sha256} "
        f"--code-revision {args.exp688_code_revision} "
        "--vendor-acceptance /work/input/vendor/acceptance.json "
        "--vendor-archive /work/input/vendor/peft-0.20.0.zip "
        "--images /work/images --model-root /hf_models "
        f"--model-revision {MODEL_REVISION} --vendor /work/vendor "
        f"--output-dir {output_dir} --runtime-backend legacy_eager "
        f"--micro-batch-size-override 2 --technical-smoke --mode {mode}"
    )
    return (
        f"PYTHONPATH={pythonpath} python3 -u /work/code/{EXP688_DIR}/"
        f"train_gradient_control.py {common} && "
        f"PYTHONPATH={pythonpath} python3 -u /work/code/{EXP688_DIR}/"
        f"verify_training_artifact.py {common} --output {output_dir}/acceptance.json"
    )


def _command(args: argparse.Namespace, selected_mode: str) -> str:
    parent_archive = f"/work/input/parent_code/{args.parent_bundle_file}"
    probe_archive = f"/work/input/probe_code/{args.probe_bundle_file}"
    exp688_archive = f"/work/input/exp688_code/{args.exp688_bundle_file}"
    pythonpath = (
        f"/work/code/{EXP688_DIR}:/work/code/{EXP687_DIR}:"
        f"/work/code/{EXP686_DIR}:/work/code/{SHARED_DIR}:/work/vendor"
    )
    segments = [
        _extract(parent_archive, "/work/code", args.parent_bundle_sha256),
        (
            f"PYTHONPATH=/work/code/{EXP686_DIR} python3 -u "
            f"/work/code/{EXP686_DIR}/verify_code_bundle.py --root /work/code "
            f"--expected-revision {args.parent_code_revision} --expected-scope training "
            f"--archive {parent_archive} --expected-bundle-sha256 "
            f"{args.parent_bundle_sha256} --output /work/parent_code_acceptance.json"
        ),
        _extract(probe_archive, "/work/probe_overlay", args.probe_bundle_sha256),
        (
            f"python3 -u /work/probe_overlay/{EXP687_DIR}/verify_probe_code_bundle.py "
            f"--root /work/probe_overlay --archive {probe_archive} "
            f"--expected-revision {args.probe_code_revision} "
            f"--expected-bundle-sha256 {args.probe_bundle_sha256} "
            "--output /work/probe_code_acceptance.json"
        ),
        f"test ! -e /work/code/{EXP687_DIR}",
        f"cp -R /work/probe_overlay/{EXP687_DIR} /work/code/experiments/",
        _extract(exp688_archive, "/work/exp688_overlay", args.exp688_bundle_sha256),
        "mkdir -p /work/output",
        (
            f"PYTHONPATH=/work/exp688_overlay/{EXP688_DIR} python3 -u "
            f"/work/exp688_overlay/{EXP688_DIR}/verify_code_bundle.py "
            f"--root /work/exp688_overlay --archive {exp688_archive} "
            f"--expected-revision {args.exp688_code_revision} "
            f"--expected-bundle-sha256 {args.exp688_bundle_sha256} "
            "--output /work/output/code_acceptance.json"
        ),
        f"test ! -e /work/code/{EXP688_DIR}",
        f"cp -R /work/exp688_overlay/{EXP688_DIR} /work/code/experiments/",
        (
            f'test "$(sha256sum /work/probe_artifact/gradient_conflict_report.json '
            f"| cut -d' ' -f1)\" = \"{args.probe_report_file_sha256}\""
        ),
        (
            f'test "$(sha256sum /work/probe_artifact/acceptance.json '
            f"| cut -d' ' -f1)\" = \"{args.probe_acceptance_file_sha256}\""
        ),
        (
            f'test "$(sha256sum /work/input/vendor/peft-0.20.0.zip | cut -d\' \' -f1)" '
            f'= "{args.vendor_sha256}"'
        ),
        "mkdir -p /work/vendor /work/images",
        "python3 -m zipfile -e /work/input/vendor/peft-0.20.0.zip /work/vendor",
        (
            f"PYTHONPATH=/work/code/{EXP686_DIR} python3 -u "
            f"/work/code/{EXP686_DIR}/stage_training_input.py --source /work/pair_raw "
            f"--output /work/pair_clean --fold 3 --expected-pair-acceptance-sha256 "
            f"{args.expected_pair_acceptance_sha256} "
            f"--expected-pair-runtime-contract-sha256 "
            f"{args.expected_pair_runtime_contract_sha256} "
            f"--expected-r0-code-acceptance-sha256 "
            f"{args.expected_r0_code_acceptance_sha256}"
        ),
        _train_command(args, CONTROL_MODE, "/work/output/control"),
        _train_command(args, selected_mode, "/work/output/candidate"),
        (
            f"PYTHONPATH={pythonpath} python3 -u /work/code/{EXP688_DIR}/"
            "verify_paired_smoke.py --root /work/output "
            "--probe-report /work/probe_artifact/gradient_conflict_report.json "
            "--probe-acceptance /work/probe_artifact/acceptance.json "
            "--code-acceptance /work/output/code_acceptance.json "
            "--output /work/output/paired_acceptance.json"
        ),
    ]
    return " && ".join(segments)


def _preset(
    args: argparse.Namespace,
    command: str,
    credentials: tuple[str, str] | None,
) -> str:
    lines = _header(args.base_preset.read_text(encoding="utf-8"), command)
    lines.append("  input:")
    specs = (
        ("parent_code", args.parent_bundle_src, f"/work/input/parent_code/{args.parent_bundle_file}", args.parent_bundle_file, True),
        ("probe_code", args.probe_bundle_src, f"/work/input/probe_code/{args.probe_bundle_file}", args.probe_bundle_file, True),
        ("exp688_code", args.exp688_bundle_src, f"/work/input/exp688_code/{args.exp688_bundle_file}", args.exp688_bundle_file, True),
        ("pair", args.pair_src, "/work/pair_raw", None, False),
        ("vendor", args.vendor_src, "/work/input/vendor", None, True),
        ("probe_artifact", args.probe_artifact_src, "/work/probe_artifact", None, False),
    )
    for name, src, dst, file, cache in specs:
        lines.extend(
            _input_spec(
                name=f"exp688_{name}",
                bucket=args.bucket,
                src=src,
                dst=dst,
                file=file,
                cache=cache,
                credentials=credentials,
            )
        )
    model_line = args.model_input_line_file.read_text(encoding="utf-8").strip()
    if (
        hashlib.sha256((model_line + "\n").encode()).hexdigest()
        != MODEL_INPUT_LINE_SHA256
        or not model_line.startswith("{")
        or "type: model_registry" not in model_line
        or "/hf_models/" not in model_line
    ):
        raise ValueError("model registry input differs from frozen exp686")
    lines.append(f"    - {model_line}")
    lines.append("  output:")
    lines.extend(
        _output_spec(
            name="exp688_output",
            bucket=args.bucket,
            dst=args.output_dst,
            credentials=credentials,
        )
    )
    return "\n".join(lines) + "\n"


def _write_new(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_paths = (args.raw_output, args.clean_output, args.overrides_output)
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("raw, clean and override output paths must be distinct")
    if any(path.exists() for path in output_paths):
        raise FileExistsError("refusing to overwrite exp688 preset material")
    for value in (
        args.parent_bundle_sha256,
        args.probe_bundle_sha256,
        args.exp688_bundle_sha256,
        args.expected_pair_acceptance_sha256,
        args.expected_pair_runtime_contract_sha256,
        args.expected_r0_code_acceptance_sha256,
        args.vendor_sha256,
        args.probe_report_file_sha256,
        args.probe_acceptance_file_sha256,
    ):
        _lower_hex(value, 64)
    for value in (
        args.parent_code_revision,
        args.probe_code_revision,
        args.exp688_code_revision,
    ):
        _lower_hex(value, 40)
    for value in (
        args.parent_bundle_file,
        args.probe_bundle_file,
        args.exp688_bundle_file,
    ):
        _safe_basename(value)
    args.parent_bundle_src = _safe_s3(
        args.parent_bundle_src, "/d.strizhakov/ecup/experiments/686"
    )
    args.probe_bundle_src = _safe_s3(
        args.probe_bundle_src, "/d.strizhakov/ecup/experiments/687"
    )
    args.exp688_bundle_src = _safe_s3(
        args.exp688_bundle_src, "/d.strizhakov/ecup/experiments/688/code"
    )
    args.pair_src = _safe_s3(args.pair_src, "/d.strizhakov/ecup/experiments/686")
    args.vendor_src = _safe_s3(args.vendor_src, "/d.strizhakov/ecup")
    args.probe_artifact_src = _safe_s3(
        args.probe_artifact_src, "/d.strizhakov/ecup/experiments/687/probe/fold3"
    )
    args.output_dst = _safe_s3(
        args.output_dst, "/d.strizhakov/ecup/experiments/688/technical_smoke/fold3"
    )
    parent_code_acceptance = _validate_code_acceptance(
        args.parent_code_acceptance,
        experiment_id="686",
        decision="ACCEPT_CODE_BUNDLE",
        revision=args.parent_code_revision,
        bundle_sha256=args.parent_bundle_sha256,
        scope="training",
    )
    probe_code_acceptance = _validate_code_acceptance(
        args.probe_code_acceptance,
        experiment_id="687",
        decision="ACCEPT_PROBE_CODE_BUNDLE",
        revision=args.probe_code_revision,
        bundle_sha256=args.probe_bundle_sha256,
    )
    _validate_code_acceptance(
        args.exp688_code_acceptance,
        experiment_id="688",
        decision="ACCEPT_EXP688_CODE_BUNDLE",
        revision=args.exp688_code_revision,
        bundle_sha256=args.exp688_bundle_sha256,
        scope=SCOPE,
    )
    if sha256_file(args.probe_report) != args.probe_report_file_sha256:
        raise ValueError("terminal exp687 report file SHA mismatch")
    if sha256_file(args.probe_acceptance) != args.probe_acceptance_file_sha256:
        raise ValueError("terminal exp687 acceptance file SHA mismatch")
    selection = load_terminal_probe_selection(args.probe_report, args.probe_acceptance)
    frozen_input_bindings = {
        "pair_runtime_contract_sha256": args.expected_pair_runtime_contract_sha256,
        "pair_runtime_acceptance_sha256": args.expected_pair_acceptance_sha256,
        "parent_code_bundle_sha256": args.parent_bundle_sha256,
        "parent_code_revision": args.parent_code_revision,
        "parent_code_acceptance_sha256": parent_code_acceptance[
            "acceptance_sha256"
        ],
        "probe_code_bundle_sha256": args.probe_bundle_sha256,
        "probe_code_revision": args.probe_code_revision,
        "probe_code_acceptance_sha256": probe_code_acceptance[
            "acceptance_sha256"
        ],
        "vendor_zip_sha256": args.vendor_sha256,
    }
    mismatch = {
        field: {"selector": selection.get(field), "preset": value}
        for field, value in frozen_input_bindings.items()
        if selection.get(field) != value
    }
    if mismatch:
        raise ValueError(f"terminal selector/preset input mismatch: {mismatch}")
    selected_mode = selection["selected_candidate_mode"]
    _require_sensitive_output_ignored(args.repo, args.s3_env_file)
    credentials = _read_s3_credentials(args.s3_env_file)
    _require_sensitive_output_ignored(args.repo, args.raw_output)
    _require_sensitive_output_ignored(args.repo, args.overrides_output)
    command = _command(args, selected_mode)
    raw_payload = _preset(args, command, credentials)
    clean_payload = _preset(args, command, None)
    overrides = {
        "job": {
            "input": [
                {
                    "name": f"exp688_{name}",
                    "access_key": credentials[0],
                    "secret_key": credentials[1],
                }
                for name in (
                    "parent_code",
                    "probe_code",
                    "exp688_code",
                    "pair",
                    "vendor",
                    "probe_artifact",
                )
            ],
            "output": [
                {
                    "name": "exp688_output",
                    "access_key": credentials[0],
                    "secret_key": credentials[1],
                }
            ],
        }
    }
    _write_new(args.raw_output, raw_payload)
    _write_new(args.clean_output, clean_payload)
    _write_new(args.overrides_output, json.dumps(overrides, indent=2) + "\n")
    return {
        "candidate_mode": selected_mode,
        "fold": 3,
        "flavor": "h100-1x",
        "optimizer_steps_per_arm": 1,
        "sequential_arms": [CONTROL_MODE, selected_mode],
        "raw_preset_sha256": hashlib.sha256(raw_payload.encode()).hexdigest(),
        "clean_preset_sha256": hashlib.sha256(clean_payload.encode()).hexdigest(),
        "overrides_sha256": hashlib.sha256(
            (json.dumps(overrides, indent=2) + "\n").encode()
        ).hexdigest(),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--repo", type=Path, required=True)
    result.add_argument("--base-preset", type=Path, required=True)
    result.add_argument("--bucket", required=True)
    result.add_argument("--s3-env-file", type=Path, required=True)
    for prefix in ("parent", "probe", "exp688"):
        result.add_argument(f"--{prefix}-bundle-src", required=True)
        result.add_argument(f"--{prefix}-bundle-file", required=True)
        result.add_argument(f"--{prefix}-bundle-sha256", required=True)
        result.add_argument(f"--{prefix}-code-revision", required=True)
        result.add_argument(f"--{prefix}-code-acceptance", type=Path, required=True)
    result.add_argument("--pair-src", required=True)
    result.add_argument("--expected-pair-acceptance-sha256", required=True)
    result.add_argument("--expected-pair-runtime-contract-sha256", required=True)
    result.add_argument("--expected-r0-code-acceptance-sha256", required=True)
    result.add_argument("--vendor-src", required=True)
    result.add_argument("--vendor-sha256", required=True)
    result.add_argument("--probe-artifact-src", required=True)
    result.add_argument("--probe-report", type=Path, required=True)
    result.add_argument("--probe-acceptance", type=Path, required=True)
    result.add_argument("--probe-report-file-sha256", required=True)
    result.add_argument("--probe-acceptance-file-sha256", required=True)
    result.add_argument("--model-input-line-file", type=Path, required=True)
    result.add_argument("--output-dst", required=True)
    result.add_argument("--raw-output", type=Path, required=True)
    result.add_argument("--clean-output", type=Path, required=True)
    result.add_argument("--overrides-output", type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(build(parser().parse_args()), indent=2, sort_keys=True))
