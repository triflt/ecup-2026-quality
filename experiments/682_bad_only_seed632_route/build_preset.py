"""Build an ignored, environment-explicit remote compute preset for the one allowed refit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contract import load_json, load_spec, require_sha, verify_self_hash


def _q(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def render_preset(args: argparse.Namespace) -> str:
    spec = load_spec(args.spec)
    gate = load_json(args.launch_gate)
    verify_self_hash(gate, "gate_sha256")
    if spec["full_refit"]["enabled"] is not True:
        raise PermissionError("frozen spec blocks preset generation")
    if gate.get("decision") != "GO_GPU" or gate.get("preset_generation_allowed") is not True:
        raise PermissionError("launch gate blocks preset generation")
    if ".local" not in args.output.parts:
        raise ValueError("preset output must remain under an ignored .local directory")
    require_sha(
        args.parent,
        "c30e690ad260af72fcc625c8d3e6d9ab9c5a096d8443d6d9f5f7adbcaa52123c",
        name="parent runner",
    )
    require_sha(
        args.experiment_dir / "train_full.py",
        spec["full_refit"]["full_train_runner_sha256"],
        name="full train runner",
    )
    runtime = load_json(args.runtime_dir / "runtime_audit.json")
    verify_self_hash(runtime, "runtime_sha256")
    if runtime.get("decision") != "GO" or runtime.get("sealed_rows_written") != 0:
        raise PermissionError("full runtime audit is not accepted")
    lines = [
        "job:",
        "  generate_name: ecup-exp682-bad-seed632-full",
        f"  time_limit: {_q(args.time_limit)}",
        f"  flavor: {_q(args.flavor)}",
        f"  region: {_q(args.region)}",
        f"  image: {_q(args.image)}",
        "  preemption: forbidden",
        "  work_dir: /work/repo",
        "  env:",
        "    PYTORCH_ALLOC_CONF: expandable_segments:True",
        '    TOKENIZERS_PARALLELISM: "false"',
        "  entrypoint: python",
        "  args:",
        "    - -u",
        "    - /work/repo/experiments/682_bad_only_seed632_route/train_full.py",
        "    - --runtime-dir",
        "    - /work/runtime",
        "    - --parent",
        "    - /work/repo/research/qwen3vl_lora_holdout.py",
        "    - --upstream-dir",
        "    - /work/repo/experiments/623_semantic_v3_multitask_span_head",
        "    - --images",
        "    - /work/images",
        "    - --model-root",
        "    - /hf_models",
        "    - --model-revision",
        f"    - {spec['frozen_inputs']['qwen35_model_revision']}",
        "    - --vendor",
        "    - /work/vendor",
        "    - --output-dir",
        "    - /work/output",
        "  input:",
        f"    - {{type: files, src: {_q(args.experiment_dir)}, dst: /work/repo/experiments/682_bad_only_seed632_route, mask: \"*.py\"}}",
        f"    - {{type: files, src: {_q(args.upstream_dir)}, dst: /work/repo/experiments/623_semantic_v3_multitask_span_head, mask: \"*.py\"}}",
        f"    - {{type: files, src: {_q(args.parent)}, dst: /work/repo/research/qwen3vl_lora_holdout.py}}",
        f"    - {{type: files, src: {_q(args.runtime_dir)}, dst: /work/runtime, mask: \"**/*\"}}",
        f"    - {{type: files, src: {_q(args.vendor_dir)}, dst: /work/vendor, mask: \"**/*\"}}",
        f"    - {{type: model_registry, mrid: {_q(args.model_mrid)}, dst: /hf_models/}}",
        "  output:",
        "    - {type: files, name: exp682_bad_seed632_full, src: /work/output/, mask: \"**/*\"}",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=here / "frozen_spec.json")
    parser.add_argument("--launch-gate", type=Path, default=here / "results/launch_gate.json")
    parser.add_argument("--experiment-dir", type=Path, default=here)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument(
        "--model-mrid",
        default="huggingface-proxy/Qwen/Qwen3.5-4B/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    )
    parser.add_argument("--flavor", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--time-limit", default="3h")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rendered = render_preset(args)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite preset")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
