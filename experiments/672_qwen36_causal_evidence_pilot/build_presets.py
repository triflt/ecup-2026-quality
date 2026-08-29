#!/usr/bin/env python3
"""Derive two private one-GPU presets from already proven model presets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "frozen_spec.json").read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def derive(
    *, template: Path, bundle: Path, context_dir: Path, output: Path, alias: str
) -> dict[str, Any]:
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    if set(payload) != {"job"} or not isinstance(payload["job"], dict):
        raise ValueError("template is not a single remote compute job preset")
    job = payload["job"]
    inputs = job.get("input")
    if not isinstance(inputs, list) or len(inputs) < 2 or inputs[1].get("type") != "model_registry":
        raise ValueError("template does not contain the proven model-registry input")
    expected = SPEC["candidate_models"][alias]
    if expected["revision"] not in str(inputs[1].get("mrid", "")):
        raise ValueError("template model revision differs from frozen candidate")
    job["generate_name"] = f"exp672-{alias}-"
    job["time_limit"] = "45m"
    job["flavor"] = "h100-1x"
    job["work_dir"] = "/work"
    job["entrypoint"] = "bash"
    job.setdefault("env", {}).pop("RUNTIME_URL", None)
    resolved_bundle = bundle.resolve()
    resolved_context = context_dir.resolve()
    if not resolved_bundle.is_relative_to(resolved_context):
        raise ValueError("bundle must be inside the explicit remote compute custom context")
    job["input"] = [
        {
            "type": "files",
            "src": os.path.relpath(resolved_bundle, resolved_context),
            "dst": "/work/input",
        },
        {**inputs[1], "dst": "/work/model"},
    ]
    job["output"] = [
        {
            "type": "files",
            "name": "e672_q35" if alias == "qwen35_4b" else "e672_q36",
            "src": "/work/output",
            "mask": "**/*",
        }
    ]
    command = (
        "python -m pip install -q --break-system-packages --no-cache-dir --upgrade "
        "'transformers>=5,<6' 'accelerate>=1.12,<2' pillow && "
        "python /work/input/run_explanations.py "
        "--runtime /work/input/pilot_runtime.jsonl "
        "--images-dir /work/input/images "
        "--model-root /work/model "
        f"--model-id {expected['model_id']} "
        f"--model-revision {expected['revision']} "
        f"--candidate-alias {alias} "
        "--output-dir /work/output"
    )
    job["args"] = ["-lc", command]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return {
        "alias": alias,
        "template_sha256": sha256_file(template),
        "preset_sha256": sha256_file(output),
        "flavor": job["flavor"],
        "time_limit": job["time_limit"],
        "model_revision": expected["revision"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen35-template", type=Path, required=True)
    parser.add_argument("--qwen36-template", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        derive(
            template=args.qwen35_template,
            bundle=args.bundle,
            context_dir=args.context_dir,
            output=args.output_dir / "qwen35_4b.yml",
            alias="qwen35_4b",
        ),
        derive(
            template=args.qwen36_template,
            bundle=args.bundle,
            context_dir=args.context_dir,
            output=args.output_dir / "qwen36_27b.yml",
            alias="qwen36_27b",
        ),
    ]
    report = {"schema_version": "exp672_preset_build_v1", "presets": results}
    report_path = args.output_dir / "build_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
