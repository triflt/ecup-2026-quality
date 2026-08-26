"""Build a credential-free remote compute CPU preset for exp689 source PREPARE."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path, PurePosixPath

EXPERIMENT = "experiments/689_qwen35_4b_grounded_transaction_graph_kd"
ALLOWED_REGIONS = {"ix-m5-sm11", "ix-m5-sm12"}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def safe_key(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or len(path.parts) < 3
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"unsafe S3 key: {value}")
    return value


def safe_extract(archive: str, destination: str, expected_sha256: str) -> str:
    program = (
        "import pathlib,sys,tarfile;"
        "a=pathlib.Path(sys.argv[1]);d=pathlib.Path(sys.argv[2]);"
        "t=tarfile.open(a);m=t.getmembers();p=[pathlib.PurePosixPath(x.name) for x in m];"
        "n=[x.as_posix().removeprefix('./') for x in p];"
        "bad=[x.name for x,y,z in zip(m,p,n) if y.is_absolute() or '..' in y.parts "
        "or any(q=='__MACOSX' or q=='.DS_Store' or q.startswith('._') for q in y.parts) "
        "or not z or not (x.isfile() or x.isdir())];"
        "bad and (_ for _ in ()).throw(ValueError('unsafe tar members'));"
        "len(n)!=len(set(n)) and (_ for _ in ()).throw(ValueError('duplicate tar members'));"
        "d.mkdir(parents=True,exist_ok=False);t.extractall(d,members=m)"
    )
    return (
        f'test "$(sha256sum {shlex.quote(archive)} | cut -d\' \' -f1)" = '
        f'"{expected_sha256}" && python3 -c {shlex.quote(program)} '
        f"{shlex.quote(archive)} {shlex.quote(destination)}"
    )


def input_lines(name: str, bucket: str, key: str, dst: str, filename: str) -> list[str]:
    return [
        "    - type: s3msk",
        f"      name: {json.dumps(name)}",
        f"      src: {json.dumps(key)}",
        f"      file: {json.dumps(filename)}",
        f"      dst: {json.dumps(dst)}",
        f"      bucket: {json.dumps(bucket)}",
    ]


def build(args: argparse.Namespace) -> str:
    if args.region not in ALLOWED_REGIONS:
        raise ValueError("source PREPARE region is not approved")
    if not HEX40.fullmatch(args.revision):
        raise ValueError("revision must be exact lowercase Git SHA")
    frozen_hashes = (
        args.bundle_sha256,
        args.manifest_sha256,
        args.source_f03_sha256,
        args.source_f124_sha256,
        args.exclusion_670_sha256,
        args.exclusion_672_sha256,
    )
    if any(not HEX64.fullmatch(value) for value in frozen_hashes):
        raise ValueError("all input hashes must be exact lowercase SHA-256")
    keys = tuple(
        safe_key(value)
        for value in (
            args.bundle_key,
            args.manifest_key,
            args.source_f03_key,
            args.source_f124_key,
            args.exclusion_670_key,
            args.exclusion_672_key,
            args.output_prefix,
        )
    )
    (
        bundle_key,
        manifest_key,
        source_f03_key,
        source_f124_key,
        exclusion_670_key,
        exclusion_672_key,
        output_prefix,
    ) = keys
    bundle_path = "/work/input/code/source_prepare_bundle.tar.gz"
    source_f03_path = "/work/input/source_f03/source_f03.tar.gz"
    source_f124_path = "/work/input/source_f124/source_f124.tar.gz"
    segments = [
        safe_extract(bundle_path, "/work/code", args.bundle_sha256),
        (
            f'test "$(sha256sum /work/input/manifest/bundle_manifest.json | cut -d\' \' -f1)" '
            f'= "{args.manifest_sha256}"'
        ),
        safe_extract(source_f03_path, "/work/source_f03", args.source_f03_sha256),
        safe_extract(source_f124_path, "/work/source_f124", args.source_f124_sha256),
        (
            f"PYTHONPATH=/work/code/{EXPERIMENT} python3 -c "
            + shlex.quote(
                "from pathlib import Path;import verify_source_prepare as v;"
                f"v._validate_bundle(Path('/work/code'),Path('/work/input/manifest/bundle_manifest.json'),"
                f"'{args.manifest_sha256}','{args.revision}')"
            )
        ),
        (
            f"PYTHONPATH=/work/code/{EXPERIMENT} python3 -u /work/code/{EXPERIMENT}/"
            "prepare_source_universe.py "
            "--runtime-dir /work/source_f03/experiments/641_qwen35_4b_class_only_lora/.local/runtime/fold0 "
            "--runtime-dir /work/source_f124/runtime/fold1 "
            "--runtime-dir /work/source_f124/runtime/fold2 "
            "--runtime-dir /work/source_f03/experiments/641_qwen35_4b_class_only_lora/.local/runtime/fold3 "
            "--runtime-dir /work/source_f124/runtime/fold4 "
            "--exclusion-670 /work/input/exclusion_670/exp670.csv "
            "--exclusion-672 /work/input/exclusion_672/exp672.json "
            f"--exclusion-670-sha256 {args.exclusion_670_sha256} "
            f"--exclusion-672-sha256 {args.exclusion_672_sha256} "
            f"--builder-revision {args.revision} --output-dir /work/output/prepared"
        ),
    ]
    command = " && ".join(segments)
    lines = [
        "job:",
        "  generate_name: exp689-source-prepare",
        "  time_limit: 2h",
        "  flavor: 8cpu-128ram",
        f"  region: {args.region}",
        "  image: odsai/ecup26-quality-baseline:1.0",
        "  preemption: false",
        "  work_dir: /work",
        "  entrypoint: /bin/bash",
        "  args:",
        "    - -lc",
        "    - >-",
        f"      {command}",
        "  inputs:",
    ]
    specs = (
        ("code", bundle_key, "/work/input/code", "source_prepare_bundle.tar.gz"),
        ("manifest", manifest_key, "/work/input/manifest", "bundle_manifest.json"),
        ("source_f03", source_f03_key, "/work/input/source_f03", "source_f03.tar.gz"),
        ("source_f124", source_f124_key, "/work/input/source_f124", "source_f124.tar.gz"),
        ("exclusion_670", exclusion_670_key, "/work/input/exclusion_670", "exp670.csv"),
        ("exclusion_672", exclusion_672_key, "/work/input/exclusion_672", "exp672.json"),
    )
    for name, key, dst, filename in specs:
        lines.extend(input_lines(name, args.bucket, key, dst, filename))
    lines.extend(
        [
            "  outputs:",
            "    - type: s3msk",
            "      name: source_prepare",
            "      src: /work/output",
            f"      dst: {json.dumps(output_prefix)}",
            f"      bucket: {json.dumps(args.bucket)}",
            "      upload_policies:",
            "        - when: on_job_status=succeeded",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--bundle-key", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--manifest-key", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-f03-key", required=True)
    parser.add_argument("--source-f03-sha256", required=True)
    parser.add_argument("--source-f124-key", required=True)
    parser.add_argument("--source-f124-sha256", required=True)
    parser.add_argument("--exclusion-670-key", required=True)
    parser.add_argument("--exclusion-670-sha256", required=True)
    parser.add_argument("--exclusion-672-key", required=True)
    parser.add_argument("--exclusion-672-sha256", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite clean preset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(args), encoding="utf-8")


if __name__ == "__main__":
    main()
