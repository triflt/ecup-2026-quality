from __future__ import annotations

import argparse
import copy
import os
import re
import shlex
from pathlib import Path

import yaml

PARENT_DIR = "experiments/686_qwen35_4b_additive_rank_kd"
PROBE_DIR = "experiments/687_qwen35_4b_gradient_conflict_probe"
SHARED_DIR = "experiments/645_qwen_scale_2x3_gate"
EXPECTED_PARENT_BUNDLE_SHA256 = (
    "ba5527b65548fd68fafa6eaf8d6fedfc906b6469d32c2e08fdc56ebccf9fd999"
)
EXPECTED_PARENT_REVISION = "9899e2039d0063a8d503eade07be12cf7f1db729"
EXPECTED_PAIR_ACCEPTANCE = (
    "c51eb01f115c6cb06d0d5bf04866c6ea39fd940e72a3471db9e4f3b34015c684"
)


def _safe_extract_command(archive: str, destination: str, required_prefix: str) -> str:
    return (
        "python3 -c 'import pathlib,sys,tarfile;"
        "archive=pathlib.Path(sys.argv[1]);destination=pathlib.Path(sys.argv[2]);"
        "handle=tarfile.open(archive);members=handle.getmembers();"
        "paths=[pathlib.PurePosixPath(m.name) for m in members];"
        "names=[p.as_posix().removeprefix(\"./\") for p in paths];"
        "bad=[m.name for m,p,n in zip(members,paths,names) "
        "if p.is_absolute() or \"..\" in p.parts or any(x == \"__MACOSX\" "
        "or x == \".DS_Store\" or x.startswith(\"._\") for x in p.parts) "
        "or not n or (n != \"EXP687_CODE_BUNDLE_MANIFEST.json\" "
        "and n not in {\"experiments\",sys.argv[3].rstrip(\"/\")} "
        "and not n.startswith(sys.argv[3].rstrip(\"/\")+\"/\")) "
        "or not (m.isfile() or m.isdir())];"
        "bad and (_ for _ in ()).throw(ValueError(\"unsafe tar members\"));"
        "len(names)!=len(set(names)) and (_ for _ in ()).throw(ValueError(\"duplicate tar members\"));"
        "handle.extractall(destination,members=members)' "
        f"{shlex.quote(archive)} {shlex.quote(destination)} {shlex.quote(required_prefix)}"
    )


def build(args: argparse.Namespace) -> dict:
    generated_paths = [args.output]
    if args.clean_output is not None:
        generated_paths.extend([args.clean_output, args.overrides_output])
    if any(path.exists() for path in generated_paths):
        raise FileExistsError("refusing to overwrite probe preset")
    if (args.clean_output is None) != (args.overrides_output is None):
        raise ValueError("clean and overrides outputs must be provided together")
    if (
        Path(args.probe_bundle_file).name != args.probe_bundle_file
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.probe_bundle_file)
        or not re.fullmatch(r"[0-9a-f]{64}", args.probe_bundle_sha256)
        or not re.fullmatch(r"[0-9a-f]{40}", args.probe_code_revision)
    ):
        raise ValueError("probe bundle identity is not shell-safe and immutable")
    expected_code_prefix = "/project/experiments/687/code/"
    expected_output_prefix = "/project/experiments/687/probe/fold3/"
    if (
        not args.probe_bundle_src.startswith(expected_code_prefix)
        or not args.output_dst.startswith(expected_output_prefix)
        or ".." in Path(args.probe_bundle_src).parts
        or ".." in Path(args.output_dst).parts
        or any(character.isspace() for character in args.probe_bundle_src + args.output_dst)
    ):
        raise ValueError("probe S3 paths escape the frozen experiment scope")
    payload = yaml.safe_load(args.base_preset.read_text(encoding="utf-8"))
    job = payload.get("job")
    if not isinstance(job, dict):
        raise TypeError("base preset has no job block")
    if job.get("flavor") != "h100-1x" or job.get("region") != "ix-m5-sm11":
        raise ValueError("base preset is not the proven one-H100 sm11 recipe")
    if job.get("time_limit") != "3h":
        raise ValueError("base time limit differs from frozen parent")
    command = job.get("args", [None, None])[1]
    if not isinstance(command, str):
        raise TypeError("base command is missing")
    required = (
        EXPECTED_PARENT_BUNDLE_SHA256,
        EXPECTED_PARENT_REVISION,
        EXPECTED_PAIR_ACCEPTANCE,
        "--fold 3",
        "--mode rank_candidate",
        f"/{PARENT_DIR}/stage_training_input.py",
        f"/{PARENT_DIR}/train_pair_fold.py",
    )
    if any(value not in command for value in required):
        raise ValueError("base preset differs from accepted experiment-686 candidate")
    if "--technical-smoke" in command or "promotion" in command.lower():
        raise ValueError("probe must not inherit smoke-only or promotion inputs")

    train_marker = f"python3 -u /work/code/{PARENT_DIR}/train_pair_fold.py"
    train_at = command.index(train_marker)
    tail_at = command.rfind("&& PYTHONPATH=", 0, train_at)
    if tail_at < 0:
        raise ValueError("cannot isolate parent training tail")
    prefix = command[:tail_at].rstrip()
    probe_archive = f"/work/input/probe_code/{args.probe_bundle_file}"
    pythonpath = f"/work/code/{PROBE_DIR}:/work/code/{PARENT_DIR}:/work/code/{SHARED_DIR}:/work/vendor"
    probe_command = " ".join(
        [
            prefix,
            "&&",
            f'test "$(sha256sum {probe_archive} | cut -d\' \' -f1)" = "{args.probe_bundle_sha256}"',
            "&&",
            "rm -rf /work/probe_overlay",
            "&&",
            "mkdir -p /work/probe_overlay /work/probe_code_acceptance",
            "&&",
            _safe_extract_command(probe_archive, "/work/probe_overlay", PROBE_DIR),
            "&&",
            f"python3 -u /work/probe_overlay/{PROBE_DIR}/verify_probe_code_bundle.py",
            "--root /work/probe_overlay",
            f"--archive {probe_archive}",
            f"--expected-revision {args.probe_code_revision}",
            f"--expected-bundle-sha256 {args.probe_bundle_sha256}",
            "--output /work/probe_code_acceptance/acceptance.json",
            "&&",
            f"test ! -e /work/code/{PROBE_DIR}",
            "&&",
            f"cp -R /work/probe_overlay/{PROBE_DIR} /work/code/experiments/",
            "&&",
            f"python3 -u /work/code/{PROBE_DIR}/sanitize_probe_source.py",
            "--source-runtime /work/pair_clean/source_runtime",
            "--pair-runtime /work/pair_clean/runtime",
            "--transport-acceptance /work/pair_clean/transport_acceptance.json",
            "--output /work/pair_clean/sanitizer_acceptance.json",
            "&&",
            f"PYTHONPATH={pythonpath}",
            f"python3 -u /work/code/{PROBE_DIR}/probe_gradient_conflict.py",
            "--fold 3",
            "--runtime-dir /work/pair_clean/source_runtime",
            "--pair-runtime /work/pair_clean/runtime",
            "--transport-acceptance /work/pair_clean/transport_acceptance.json",
            "--sanitizer-acceptance /work/pair_clean/sanitizer_acceptance.json",
            "--parent-code-acceptance /work/code_acceptance.json",
            f"--probe-code-bundle {probe_archive}",
            f"--expected-probe-code-sha256 {args.probe_bundle_sha256}",
            f"--probe-code-revision {args.probe_code_revision}",
            "--probe-code-acceptance /work/probe_code_acceptance/acceptance.json",
            "--vendor-acceptance /work/input/vendor/acceptance.json",
            "--vendor-archive /work/input/vendor/peft-0.20.0.zip",
            "--images /work/images",
            "--model-root /hf_models",
            "--model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "--vendor /work/vendor",
            "--output-dir /work/output",
            "--runtime-backend legacy_eager",
            "--micro-batch-size-override 2",
            "&&",
            f"PYTHONPATH={pythonpath}",
            f"python3 -u /work/code/{PROBE_DIR}/verify_probe_artifact.py",
            "--report /work/output/gradient_conflict_report.json",
            "--output /work/output/acceptance.json",
        ]
    )

    job["generate_name"] = "kd-gradprobe-f3"
    job["region"] = args.region
    job["args"] = ["-lc", probe_command]
    inputs = job.get("input")
    outputs = job.get("output")
    if not isinstance(inputs, list) or len(inputs) != 4:
        raise ValueError("base input contract differs from parent")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError("base output contract differs from parent")
    s3_template = next(
        (spec for spec in inputs if spec.get("type") == "s3msk"), None
    )
    if not isinstance(s3_template, dict):
        raise TypeError("base preset has no S3 input template")
    probe_input = {
        "type": "s3msk",
        "src": args.probe_bundle_src,
        "file": args.probe_bundle_file,
        "dst": probe_archive,
        "bucket": s3_template["bucket"],
        "cache": {"enable": True, "location": "cluster"},
    }
    for field in ("access_key", "secret_key"):
        if field not in s3_template:
            raise ValueError("base raw preset is missing ignored credentials")
        probe_input[field] = s3_template[field]
    inputs.insert(1, probe_input)
    output_spec = copy.deepcopy(outputs[0])
    output_spec["dst"] = args.output_dst
    job["output"] = [output_spec]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    os.chmod(args.output, 0o600)
    if args.clean_output is not None:
        clean = copy.deepcopy(payload)
        overrides_job: dict[str, list[dict[str, str]]] = {}
        credential_pairs = 0
        for section in ("input", "output"):
            specs = clean["job"].get(section)
            if not isinstance(specs, list) or not specs:
                raise ValueError(f"clean preset lacks {section} specs")
            override_specs: list[dict[str, str]] = []
            for index, spec in enumerate(specs):
                if "name" in spec:
                    raise ValueError(f"unexpected pre-existing {section} name")
                name = f"exp687_{section}_{index}"
                spec["name"] = name
                access = spec.pop("access_key", None)
                secret = spec.pop("secret_key", None)
                if (access is None) != (secret is None):
                    raise ValueError("S3 credential pair is incomplete")
                if access is not None:
                    override_specs.append(
                        {"name": name, "access_key": access, "secret_key": secret}
                    )
                    credential_pairs += 1
            overrides_job[section] = override_specs
        if credential_pairs == 0:
            raise ValueError("raw preset contains no ignored credential inputs")
        args.clean_output.parent.mkdir(parents=True, exist_ok=True)
        args.clean_output.write_text(
            yaml.safe_dump(clean, sort_keys=False), encoding="utf-8"
        )
        args.overrides_output.write_text(
            yaml.safe_dump({"job": overrides_job}, sort_keys=False),
            encoding="utf-8",
        )
        os.chmod(args.clean_output, 0o600)
        os.chmod(args.overrides_output, 0o600)
    return {
        "region": args.region,
        "flavor": job["flavor"],
        "time_limit": job["time_limit"],
        "inputs": len(job["input"]),
        "outputs": len(job["output"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-preset", type=Path, required=True)
    parser.add_argument("--probe-bundle-src", required=True)
    parser.add_argument("--probe-bundle-file", required=True)
    parser.add_argument("--probe-bundle-sha256", required=True)
    parser.add_argument("--probe-code-revision", required=True)
    parser.add_argument("--output-dst", required=True)
    parser.add_argument("--region", choices=("ix-m5-sm11", "ix-m5-sm12"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean-output", type=Path)
    parser.add_argument("--overrides-output", type=Path)
    values = build(parser.parse_args())
    print(yaml.safe_dump(values, sort_keys=True).strip())
