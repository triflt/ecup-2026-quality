from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


DISTILL_PREFIX = "adapter_qwen35_distill/"
PARENT_PREFIX = "adapter_qwen35/"
MAX_BYTES = 5 * 1024**3

PATH_BLOCK = '''QWEN35_ADAPTER_PATH = Path(
    os.environ.get("QWEN35_LORA_ADAPTER_PATH", ROOT / "adapter_qwen35")
)
'''
PATH_REPLACEMENT = PATH_BLOCK + '''QWEN35_DISTILL_ADAPTER_PATH = Path(
    os.environ.get("QWEN35_DISTILL_LORA_ADAPTER_PATH", ROOT / "adapter_qwen35_distill")
)
QWEN35_DISTILL_ALPHA = float(os.environ.get("QWEN35_DISTILL_ALPHA", "{alpha}"))
'''

INFERENCE_BLOCK = '''    qwen35_model, qwen35_processor = load_lora_model(
        QWEN35_MODEL_PATH, QWEN35_ADAPTER_PATH, "multimodal"
    )
    qwen35_scores = compute_lora_scores(
        qwen35_model, qwen35_processor, frame, use_chat_batch=True
    )
'''
INFERENCE_REPLACEMENT = '''    qwen35_model, qwen35_processor = load_lora_model(
        QWEN35_MODEL_PATH, QWEN35_ADAPTER_PATH, "multimodal"
    )
    original_adapter = qwen35_model.active_adapter
    qwen35_model.load_adapter(
        QWEN35_DISTILL_ADAPTER_PATH, adapter_name="distill", is_trainable=False
    )
    qwen35_model.set_adapter(original_adapter)
    qwen35_scores = compute_lora_scores(
        qwen35_model, qwen35_processor, frame, use_chat_batch=True
    )
    flammable_mask = frame["category"].to_numpy() == "Легковоспламеняющиеся"
    if flammable_mask.any() and QWEN35_DISTILL_ALPHA > 0.0:
        qwen35_model.set_adapter("distill")
        distill_scores = compute_lora_scores(
            qwen35_model,
            qwen35_processor,
            frame.loc[flammable_mask],
            use_chat_batch=True,
        )
        qwen35_scores[flammable_mask] = (
            (1.0 - QWEN35_DISTILL_ALPHA) * qwen35_scores[flammable_mask]
            + QWEN35_DISTILL_ALPHA * distill_scores
        )
        print(
            f"qwen35_route original=BAD raw_logit_blend=flammable "
            f"distill_alpha={QWEN35_DISTILL_ALPHA}",
            flush=True,
        )
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--distill-adapter", type=Path, required=True)
    parser.add_argument("--full-refit-contract", type=Path, required=True)
    parser.add_argument("--allow-continuation", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite final package")

    contract = json.loads(args.full_refit_contract.read_text(encoding="utf-8"))
    contract_payload = dict(contract)
    contract_digest = contract_payload.pop("contract_sha256", None)
    if contract_digest != canonical_sha256(contract_payload):
        raise ValueError("full-refit contract self-hash mismatch")
    is_continuation = contract.get("schema_version") == "exp706_parent_anchored_continuation_v1"
    if is_continuation and not args.allow_continuation:
        raise ValueError("speculative continuation package requires explicit authorization")
    expected = {
        "schema_version": (
            "exp706_parent_anchored_continuation_v1"
            if is_continuation
            else "exp706_parent_anchored_full_v1"
        ),
        "experiment_id": "706",
        "selected_from_strict_holdout_update": 40,
        "full_oof": True,
        "strict_oof": True,
        "teacher_folds": [0, 1, 2, 3, 4],
        "bad_route": "immutable_original_exp140_adapter",
        "teacher_at_inference": False,
        "submission_base_model": "Qwen/Qwen3.5-4B",
        "technical_smoke": False,
    }
    mismatch = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise ValueError(f"full-refit contract mismatch: {mismatch}")
    if is_continuation:
        if contract.get("selection_status") != "SPECULATIVE_NO_NEW_HOLDOUT":
            raise ValueError("continuation selection status is not explicit")
        if int(contract.get("additional_updates", 0)) < 1:
            raise ValueError("continuation has no additional optimizer updates")
    if int(contract.get("unique_flammable_rows", 0)) < 5000:
        raise ValueError("full-refit does not contain the complete flammable set")
    if contract.get("unique_flammable_rows_covered") != contract.get("unique_flammable_rows"):
        raise ValueError("full-refit coverage is incomplete")

    adapter_files = {
        path.name: path.read_bytes()
        for path in sorted(args.distill_adapter.iterdir())
        if path.is_file()
    }
    required_adapter = {"adapter_config.json", "adapter_model.safetensors"}
    if not required_adapter.issubset(adapter_files):
        raise ValueError("distill adapter is incomplete")
    if bytes_sha256(adapter_files["adapter_model.safetensors"]) != contract["adapter_model_sha256"]:
        raise ValueError("distill adapter checksum mismatch")

    with zipfile.ZipFile(args.base_submission) as source:
        if source.testzip() is not None:
            raise ValueError("base ZIP integrity failure")
        names = source.namelist()
        if len(names) != len(set(names)) or any(not safe_name(name) for name in names):
            raise ValueError("base ZIP has duplicate or unsafe members")
        if "src/ocr_stage.py" in names:
            raise ValueError("base ZIP is not the no-OCR exp140 runtime")
        parent_member = f"{PARENT_PREFIX}adapter_model.safetensors"
        if parent_member not in names:
            raise ValueError("base ZIP lacks the exp140 Qwen3.5 adapter")
        if bytes_sha256(source.read(parent_member)) != contract["parent_adapter_sha256"]:
            raise ValueError("base ZIP parent adapter is not byte-exact training parent")
        run = source.read("run.py").decode("utf-8")
        if run.count(PATH_BLOCK) != 1 or run.count(INFERENCE_BLOCK) != 1:
            raise ValueError("base run.py does not match frozen exp140 insertion points")
        routed = run.replace(
            PATH_BLOCK, PATH_REPLACEMENT.format(alpha=format(args.alpha, ".8g"))
        ).replace(INFERENCE_BLOCK, INFERENCE_REPLACEMENT)
        if "ocr_stage" in routed.lower() or "paddleocr" in routed.lower():
            raise ValueError("final runtime unexpectedly references OCR")
        metadata = json.loads(source.read("metadata.json"))
        continuation = (
            f" plus {contract['additional_updates']} conservative continuation updates"
            if is_continuation
            else ""
        )
        metadata["description"] = (
            "exp140 byte-exact for BAD; strict five-fold OOF parent-anchored "
            f"raw-logit Qwen3.5 distillation for flammable alpha={args.alpha}{continuation}"
        )
        metadata["routed_distillation"] = {
            "bad_adapter": "original_exp140_byte_exact",
            "flammable_distill_mode": "strict_oof_parent_anchored_raw_logit",
            "flammable_distill_alpha": args.alpha,
            "teacher_at_inference": False,
            "full_refit_contract_sha256": sha256(args.full_refit_contract),
            "selection_status": contract.get("selection_status", "STRICT_HOLDOUT_SELECTED"),
        }
        metadata_bytes = (
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()

        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with zipfile.ZipFile(temporary, "w", allowZip64=True) as destination:
                for info in source.infolist():
                    if info.filename == "run.py":
                        destination.writestr(info, routed.encode("utf-8"))
                    elif info.filename == "metadata.json":
                        destination.writestr(info, metadata_bytes)
                    else:
                        destination.writestr(info, source.read(info))
                for name, payload in sorted(adapter_files.items()):
                    info = zipfile.ZipInfo(f"{DISTILL_PREFIX}{name}")
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o644 << 16
                    destination.writestr(info, payload)
            temporary.replace(args.output)
        finally:
            temporary.unlink(missing_ok=True)

    if args.output.stat().st_size >= MAX_BYTES:
        raise ValueError("submission exceeds 5 GiB")
    with zipfile.ZipFile(args.output) as result:
        if result.testzip() is not None:
            raise ValueError("final ZIP integrity failure")
        result_names = result.namelist()
        if "src/ocr_stage.py" in result_names:
            raise ValueError("OCR member survived final packaging")
        if not {
            f"{DISTILL_PREFIX}adapter_config.json",
            f"{DISTILL_PREFIX}adapter_model.safetensors",
        }.issubset(result_names):
            raise ValueError("final ZIP lacks distill adapter")
    report = {
        "schema_version": "exp706_final_submission_v1",
        "base_submission_sha256": sha256(args.base_submission),
        "full_refit_contract_sha256": sha256(args.full_refit_contract),
        "bad_original_adapter_byte_exact": True,
        "flammable_only_distill_route": True,
        "blend_space": "raw_logit",
        "distill_alpha": args.alpha,
        "selection_status": contract.get("selection_status", "STRICT_HOLDOUT_SELECTED"),
        "teacher_at_inference": False,
        "ocr_runtime_absent": True,
        "output_sha256": sha256(args.output),
        "output_bytes": args.output.stat().st_size,
        "under_5_gib": True,
        "zip_integrity": "PASS",
        "runtime_smoke": "PENDING",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
