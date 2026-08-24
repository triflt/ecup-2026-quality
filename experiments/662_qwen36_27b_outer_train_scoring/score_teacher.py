from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "654_qwen36_27b_class_only_lora"
PREFLIGHT = ROOT / "653_qwen36_27b_lora_runtime_preflight"
for path in (SOURCE, PREFLIGHT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from technical_smoke import first_parameter_device, validate_device_map
from train_fold import MODEL_ID, MODEL_REVISION, one_score, open_image

EXPERIMENT_ID = "662"
REQUIRED_INPUT_FIELDS = {
    "global_index", "id", "category", "fold", "occurrence_index",
    "name", "description", "image_url",
}
OUTPUT_FIELDS = {"global_index", "id", "category", "fold", "occurrence_index", "score"}
FORBIDDEN_FIELDS = {
    "label", "target", "gold", "answer", "evidence_target", "sealed",
    "public", "is_banned", "y_true",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def adapter_manifest(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): sha256_file(item)
        for item in sorted(path.rglob("*")) if item.is_file()
    }


def read_runtime(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_path = path / "score_input.jsonl"
    audit_path = path / "runtime_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    payload = dict(audit)
    digest = payload.pop("contract_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("runtime self-hash mismatch")
    if not (
        audit.get("experiment_id") == EXPERIMENT_ID
        and audit.get("source_experiment_id") == "654"
        and audit.get("source_validation_rows_bundled") == 0
        and audit.get("source_validation_labels_bundled") == 0
        and audit.get("labels_bundled") == 0
        and audit.get("sealed_rows") == 0
        and audit.get("public_used") is False
        and audit.get("decision") == "GO_TEACHER_TRAIN_TARGET_SCORING"
        and audit.get("model_revision") == MODEL_REVISION
        and audit.get("score_input_sha256") == sha256_file(input_path)
    ):
        raise ValueError("runtime contract mismatch")
    with input_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != int(audit["rows"]):
        raise ValueError("runtime row count mismatch")
    if any(set(row) != REQUIRED_INPUT_FIELDS for row in rows):
        raise ValueError("runtime input schema mismatch")
    if any(FORBIDDEN_FIELDS.intersection(row) for row in rows):
        raise ValueError("runtime contains forbidden supervision")
    if any(int(row["fold"]) == int(audit["outer_fold"]) for row in rows):
        raise ValueError("outer-validation row reached teacher scoring")
    keys = [
        [row[key] for key in ("global_index", "id", "category", "fold", "occurrence_index")]
        for row in rows
    ]
    if canonical_sha256(keys) != audit["ordered_occurrence_key_sha256"]:
        raise ValueError("ordered occurrence key SHA mismatch")
    return rows, audit


def package(output: Path, report: Path, scores: Path) -> Path:
    archive = output / "teacher_outer_train_scores.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(report, "report.json")
        bundle.write(scores, "teacher_scores.jsonl")
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt ZIP member: {bad}")
    return archive


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, runtime = read_runtime(args.runtime_dir)
    manifest = adapter_manifest(args.adapter_dir)
    if manifest != runtime.get("teacher_adapter_manifest"):
        raise ValueError("adapter manifest differs from frozen runtime")

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if torch.cuda.device_count() < args.expected_cuda_devices:
        raise RuntimeError("requested CUDA devices are unavailable")
    if not args.vendor.is_dir():
        raise FileNotFoundError("vendored PEFT directory is missing")
    sys.path.insert(0, str(args.vendor.resolve()))
    import peft
    from peft import PeftModel

    if peft.__version__ != "0.20.0":
        raise RuntimeError("exact vendored PEFT 0.20.0 is required")
    processor = AutoProcessor.from_pretrained(args.model_root, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1 or zero == one:
        raise RuntimeError("0 and 1 are not distinct atomic tokens")
    started = time.monotonic()
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_root,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map="balanced",
        attn_implementation="eager",
    )
    device_summary = validate_device_map(
        getattr(model, "hf_device_map", {}), minimum_cuda_devices=args.expected_cuda_devices
    )
    model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False)
    model.eval()
    input_device = first_parameter_device(model)
    scores_path = args.output_dir / "teacher_scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as output:
        for row in rows:
            image = open_image(args.images, row)
            try:
                with torch.inference_mode():
                    score = float(
                        one_score(model, processor, row, image, zero[0], one[0], input_device)
                        .float().cpu()[0]
                    )
            finally:
                image.close()
            if not math.isfinite(score):
                raise RuntimeError("non-finite teacher score")
            payload = {
                "global_index": int(row["global_index"]),
                "id": str(row["id"]),
                "category": str(row["category"]),
                "fold": int(row["fold"]),
                "occurrence_index": int(row["occurrence_index"]),
                "score": score,
            }
            if set(payload) != OUTPUT_FIELDS:
                raise RuntimeError("output schema drift")
            output.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": "654",
        "outer_fold": int(runtime["outer_fold"]),
        "mode": runtime["mode"],
        "rows": len(rows),
        "score_semantics": "raw_last_token_logit_1_minus_logit_0",
        "source_train_sha256": runtime["source_train_sha256"],
        "runtime_contract_sha256": runtime["contract_sha256"],
        "teacher_adapter_manifest_sha256": runtime["teacher_adapter_manifest_sha256"],
        "ordered_occurrence_key_sha256": runtime["ordered_occurrence_key_sha256"],
        "teacher_scores_sha256": sha256_file(scores_path),
        "validation_rows_read": 0,
        "validation_labels_read": 0,
        "labels_read": 0,
        "sealed_rows": 0,
        "public_used": False,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "expected_cuda_devices": args.expected_cuda_devices,
        "device_map_summary": device_summary,
        "cpu_or_disk_offload": False,
        "packages": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "peft": peft.__version__,
        },
        "elapsed_seconds": time.monotonic() - started,
        "decision": "READY_FOR_FAIL_CLOSED_ACCEPTANCE",
    }
    report["report_contract_sha256"] = canonical_sha256(report)
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    archive = package(args.output_dir, report_path, scores_path)
    (args.output_dir / "delivery.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "archive": archive.name,
                "archive_sha256": sha256_file(archive),
                "report_sha256": sha256_file(report_path),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--runtime-dir", type=Path, required=True)
    result.add_argument("--adapter-dir", type=Path, required=True)
    result.add_argument("--images", type=Path, required=True)
    result.add_argument("--model-root", type=Path, required=True)
    result.add_argument("--vendor", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--expected-cuda-devices", type=int, default=4)
    return result


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
