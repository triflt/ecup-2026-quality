from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

VARIANTS = (
    ("v1p40", "v1", "positive_only", 40),
    ("v2p19", "v2", "positive_only", 19),
    ("bp20", "both", "positive_only", 20),
    ("bb80", "both", "balanced", 80),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_input(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = [item for item in payload["job"]["input"] if item.get("type") == "model_registry"]
    if len(values) != 1:
        raise ValueError("parent preset must contain exactly one model-registry input")
    item = dict(values[0])
    item["dst"] = "/hf_models/"
    return item


def build_presets(
    *,
    repo: Path,
    bundle: Path,
    bundle_manifest: Path,
    qwen35_parent: Path,
    qwen3vl_parent: Path,
    output_dir: Path,
) -> dict:
    manifest = json.loads(bundle_manifest.read_text(encoding="utf-8"))
    payload = dict(manifest)
    declared_self = payload.pop("self_sha256", None)
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if declared_self != hashlib.sha256(canonical).hexdigest():
        raise ValueError("bundle manifest self-hash mismatch")
    if manifest["bundle_sha256"] != sha256(bundle):
        raise ValueError("bundle SHA mismatch")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite preset directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    parents = {
        "qwen35_4b": yaml.safe_load(qwen35_parent.read_text(encoding="utf-8"))["job"],
        "qwen3vl_2b": yaml.safe_load(qwen3vl_parent.read_text(encoding="utf-8"))["job"],
    }
    model_inputs = {
        "qwen35_4b": model_input(qwen35_parent),
        "qwen3vl_2b": model_input(qwen3vl_parent),
    }
    runner_source = "experiments/699_filtered_flammable_synthesis/run_screen_job.py"
    result = {}
    for architecture, short in (("qwen35_4b", "35"), ("qwen3vl_2b", "2")):
        parent = parents[architecture]
        for variant, source, mode, cap in VARIANTS:
            alias = f"s699_{short}_{variant}"
            preset = {
                "job": {
                    "generate_name": "synth-screen",
                    "time_limit": "6h",
                    "flavor": "h100-1x",
                    "region": parent["region"],
                    "image": parent["image"],
                    "preemption": "forbidden",
                    "work_dir": "/work",
                    "env": {
                        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                        "TOKENIZERS_PARALLELISM": "false",
                        "FILTER_FOLD0_URL": "override_on_submit",
                        "FILTER_FOLD3_URL": "override_on_submit",
                        "BUNDLE_URL": "override_on_submit",
                        "FILTER_FOLD0_SHA256": "e9403ce50513c28f4321eac8c7eaae6ad56973c93d843414449fcc5f941c3e2e",
                        "FILTER_FOLD3_SHA256": "46a6119d2043f639d08a96d2e293d019050d84b27c0dd13757e2641bdb364441",
                    },
                    "entrypoint": "python",
                    "args": [
                        "-u",
                        "/work/code/run_screen_job.py",
                        "--architecture",
                        architecture,
                        "--source",
                        source,
                        "--mode",
                        mode,
                        "--cap",
                        str(cap),
                        "--bundle",
                        "/work/input/screen_bundle.tar.gz",
                        "--bundle-sha256",
                        manifest["bundle_sha256"],
                        "--model-root",
                        "/hf_models",
                        "--output-root",
                        "/work/output",
                    ],
                    "input": [
                        {
                            "type": "files",
                            "src": runner_source,
                            "dst": "/work/code/run_screen_job.py",
                        },
                        model_inputs[architecture],
                    ],
                    "output": [
                        {
                            "type": "files",
                            "name": alias,
                            "src": "/work/output/",
                            "mask": "**/*",
                        }
                    ],
                }
            }
            destination = output_dir / f"{alias}.yml"
            destination.write_text(
                yaml.safe_dump(preset, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            result[alias] = {"path": str(destination), "sha256": sha256(destination)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--qwen35-parent", type=Path, required=True)
    parser.add_argument("--qwen3vl-parent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_presets(
                repo=args.repo,
                bundle=args.bundle,
                bundle_manifest=args.bundle_manifest,
                qwen35_parent=args.qwen35_parent,
                qwen3vl_parent=args.qwen3vl_parent,
                output_dir=args.output_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
