"""Package the BAD-only adapter into the frozen experiment-140 archive.

This builder is inert until a separate post-training package gate proves the
full adapter and a direct runtime smoke.  The original Qwen3.5 adapter remains
inside the archive and scores every row first; only BAD positions are then
overwritten by the new adapter.  No auxiliary span/concept head is loaded or
packaged for the verdict.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from contract import (
    add_self_hash,
    load_json,
    load_spec,
    require_sha,
    sha256_file,
    verify_self_hash,
)

BAD_ADAPTER_DIR = "adapter_qwen35_bad_seed632"
ALLOWED_ADAPTER_FILES = {"adapter_config.json", "adapter_model.safetensors", "README.md"}

CONSTANT_ANCHOR = """QWEN35_ADAPTER_PATH = Path(
    os.environ.get(\"QWEN35_LORA_ADAPTER_PATH\", ROOT / \"adapter_qwen35\")
)
"""
CONSTANT_REPLACEMENT = CONSTANT_ANCHOR + """QWEN35_BAD_ADAPTER_PATH = Path(
    os.environ.get(\"QWEN35_BAD_LORA_ADAPTER_PATH\", ROOT / \"adapter_qwen35_bad_seed632\")
)
"""
SCORING_ANCHOR = """    qwen35_scores = compute_lora_scores(
        qwen35_model, qwen35_processor, frame, use_chat_batch=True
    )
    predictions = np.zeros(len(frame), dtype=np.int8)
"""
SCORING_REPLACEMENT = """    # Preserve the champion's original logits for every row first.  A second
    # adapter pass overwrites only BAD positions; flammable positions never move.
    qwen35_scores = compute_lora_scores(
        qwen35_model, qwen35_processor, frame, use_chat_batch=True
    )
    bad_mask = frame[\"category\"].to_numpy() == \"БАД\"
    if bad_mask.any():
        if not QWEN35_BAD_ADAPTER_PATH.is_dir():
            raise FileNotFoundError(f\"BAD LoRA adapter not found: {QWEN35_BAD_ADAPTER_PATH}\")
        qwen35_model.load_adapter(
            str(QWEN35_BAD_ADAPTER_PATH), adapter_name=\"bad_seed632\", is_trainable=False
        )
        qwen35_model.set_adapter(\"bad_seed632\")
        bad_scores = compute_lora_scores(
            qwen35_model,
            qwen35_processor,
            frame.loc[bad_mask].reset_index(drop=True),
            use_chat_batch=True,
        )
        qwen35_scores[bad_mask] = bad_scores
    predictions = np.zeros(len(frame), dtype=np.int8)
"""


def patch_champion_run(source: str) -> str:
    if source.count(CONSTANT_ANCHOR) != 1 or source.count(SCORING_ANCHOR) != 1:
        raise ValueError("champion run.py anchors changed")
    patched = source.replace(CONSTANT_ANCHOR, CONSTANT_REPLACEMENT).replace(
        SCORING_ANCHOR, SCORING_REPLACEMENT
    )
    if "auxiliary_head" in patched:
        raise ValueError("auxiliary head is forbidden in production verdict path")
    return patched


def _adapter_payload(path: Path) -> tuple[dict[str, bytes], dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"adapter ZIP CRC failure: {bad}")
        files: dict[str, bytes] = {}
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = PurePosixPath(info.filename).name
            if name not in ALLOWED_ADAPTER_FILES or len(PurePosixPath(info.filename).parts) != 1:
                raise ValueError(f"unexpected adapter member: {info.filename}")
            if "auxiliary" in name.lower() or "head" in name.lower():
                raise ValueError("auxiliary head is forbidden in submission adapter")
            files[name] = archive.read(info)
    required = {"adapter_config.json", "adapter_model.safetensors"}
    if not required.issubset(files):
        raise ValueError("adapter ZIP is incomplete")
    import hashlib

    hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in files.items()}
    return files, hashes


def _verify_package_inputs(args: argparse.Namespace, spec: dict[str, Any]) -> tuple[dict, dict]:
    package_gate = load_json(args.package_gate)
    verify_self_hash(package_gate, "gate_sha256")
    if package_gate.get("schema_version") != "exp682_package_gate_v1":
        raise ValueError("package gate schema mismatch")
    if package_gate.get("decision") != "GO_PACKAGE" or package_gate.get("package_allowed") is not True:
        raise PermissionError("package gate is closed")
    adapter_contract = load_json(args.adapter_contract)
    verify_self_hash(adapter_contract, "contract_sha256")
    expected_adapter = {
        "schema_version": "exp682_full_adapter_contract_v1",
        "experiment_id": "682",
        "seed": 31415,
        "full_train": True,
        "sealed_training_rows": 0,
        "auxiliary_head_packaged": False,
        "verdict_inference": "token-1 minus token-0 logit",
        "decision": "GO",
    }
    mismatch = {
        key: {"expected": value, "actual": adapter_contract.get(key)}
        for key, value in expected_adapter.items()
        if adapter_contract.get(key) != value
    }
    if mismatch:
        raise ValueError(f"full adapter contract mismatch: {mismatch}")
    if adapter_contract.get("adapter_zip_sha256") != sha256_file(args.bad_adapter):
        raise ValueError("full adapter ZIP checksum mismatch")
    runtime = load_json(args.runtime_report)
    verify_self_hash(runtime, "report_sha256")
    expected_runtime = {
        "schema_version": "exp682_runtime_smoke_v1",
        "experiment_id": "682",
        "rows": 600,
        "input_schema_valid": True,
        "output_schema_valid": True,
        "flammable_original_scores_byte_identical": True,
        "optimized_predictions_identical": True,
        "public_feedback_used": False,
        "sealed_rows_used": 0,
    }
    mismatch = {
        key: {"expected": value, "actual": runtime.get(key)}
        for key, value in expected_runtime.items()
        if runtime.get(key) != value
    }
    if mismatch:
        raise ValueError(f"runtime report mismatch: {mismatch}")
    production = spec["production_design"]
    if runtime.get("projected_public_minutes", float("inf")) > production[
        "maximum_public_runtime_minutes"
    ]:
        raise ValueError("projected Public runtime exceeds frozen limit")
    if runtime.get("projected_private_minutes", float("inf")) > production[
        "maximum_private_runtime_minutes"
    ]:
        raise ValueError("projected Private runtime exceeds frozen limit")
    return adapter_contract, runtime


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("refusing to overwrite submission archive")
    spec = load_spec(args.spec)
    require_sha(
        args.champion,
        spec["frozen_inputs"]["champion_archive_sha256"],
        name="champion archive",
    )
    adapter_contract, runtime = _verify_package_inputs(args, spec)
    adapter_files, adapter_hashes = _adapter_payload(args.bad_adapter)

    with zipfile.ZipFile(args.champion) as source:
        bad = source.testzip()
        if bad is not None:
            raise ValueError(f"champion ZIP CRC failure: {bad}")
        run_bytes = source.read("run.py")
        import hashlib

        if hashlib.sha256(run_bytes).hexdigest() != spec["frozen_inputs"]["champion_run_sha256"]:
            raise ValueError("champion run.py checksum mismatch")
        patched_run = patch_champion_run(run_bytes.decode("utf-8")).encode("utf-8")
        original_adapter = source.read("adapter_qwen35/adapter_model.safetensors")
        original_adapter_sha = hashlib.sha256(original_adapter).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        manifest = add_self_hash(
            {
                "schema_version": "exp682_production_manifest_v1",
                "experiment_id": "682",
                "parent_archive_sha256": sha256_file(args.champion),
                "original_flammable_adapter_model_sha256": original_adapter_sha,
                "bad_adapter_zip_sha256": sha256_file(args.bad_adapter),
                "bad_adapter_member_sha256": adapter_hashes,
                "adapter_contract_sha256": sha256_file(args.adapter_contract),
                "runtime_report_sha256": sha256_file(args.runtime_report),
                "route": spec["candidate_route"],
                "auxiliary_head_packaged": False,
            },
            "manifest_sha256",
        )
        with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as target:
            for info in source.infolist():
                if info.is_dir():
                    target.writestr(info, b"")
                    continue
                payload = patched_run if info.filename == "run.py" else source.read(info)
                target.writestr(info, payload)
            for name, payload in sorted(adapter_files.items()):
                target.writestr(f"{BAD_ADAPTER_DIR}/{name}", payload)
            target.writestr(
                "exp682_production_manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
                + b"\n",
            )

    with zipfile.ZipFile(args.output) as check:
        if check.testzip() is not None:
            raise ValueError("built submission failed CRC test")
        if hashlib.sha256(check.read("adapter_qwen35/adapter_model.safetensors")).hexdigest() != original_adapter_sha:
            raise AssertionError("original flammable adapter changed during packaging")
        forbidden = [name for name in check.namelist() if "auxiliary_head" in name.lower()]
        if forbidden:
            raise ValueError(f"auxiliary head entered submission: {forbidden}")
    return {
        "submission_sha256": sha256_file(args.output),
        "parent_sha256": sha256_file(args.champion),
        "bad_adapter_sha256": sha256_file(args.bad_adapter),
        "original_flammable_adapter_model_sha256": original_adapter_sha,
        "projected_public_minutes": runtime["projected_public_minutes"],
        "projected_private_minutes": runtime["projected_private_minutes"],
        "adapter_contract": adapter_contract["contract_sha256"],
    }


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=here / "frozen_spec.json")
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--bad-adapter", type=Path, required=True)
    parser.add_argument("--adapter-contract", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--package-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
