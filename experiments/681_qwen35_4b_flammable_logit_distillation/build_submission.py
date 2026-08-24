from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "681"
BASE_ARCHIVE_SHA256 = "6cc2fda9d17d959880050889d964c3b971b92adbfa606505df7f04b46a819cd3"
BASE_RUN_SHA256 = "9318db4cc0f26eabaa73ae9856dbf9daffd06787a162e6f5db71f249f6db7146"
FLAMMABLE_ADAPTER_PREFIX = "adapter_qwen35_flammable/"
FIXED_ZIP_TIME = (2026, 8, 24, 0, 0, 0)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def verify_full_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(contract)
    digest = payload.pop("contract_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("full-refit output contract self-hash mismatch")
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "control_experiment_id": "641",
        "full_refit": True,
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "objective": "class_only",
        "changed_factor": "hard_bce_to_fixed_hard_plus_teacher_soft_bce",
        "category_scope": "Легковоспламеняющиеся",
        "temperature": 2.0,
        "soft_loss_weight": 0.5,
        "teacher_target_scope": (
            "development_rowwise_oof_plus_label_free_sealed_teacher0"
        ),
        "submission_base_model": "Qwen/Qwen3.5-4B",
        "uses_27b_at_training": False,
        "uses_27b_at_training_targets": True,
        "uses_27b_at_inference": False,
        "train_occurrences": 2590,
        "validation_rows": 0,
        "optimizer_steps_executed": 162,
        "technical_smoke": False,
        "sealed_rows_used": 0,
        "public_used": False,
        "decision": "GO_PACKAGE_AFTER_FULL_CV",
    }
    mismatch = {
        key: {"expected": expected_value, "actual": contract.get(key)}
        for key, expected_value in expected.items()
        if contract.get(key) != expected_value
    }
    if mismatch:
        raise ValueError(f"full-refit output contract mismatch: {mismatch}")
    return contract


def verify_full_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not (
        report.get("experiment_id") == EXPERIMENT_ID
        and report.get("stage") == "full"
        and report.get("decision") == "ACCEPT_FOR_REFIT"
        and (report.get("passed") is True or report.get("ablation_submission_eligible") is True)
        and report.get("public_used") is False
        and report.get("sealed_rows") == 0
        and report.get("uses_27b_at_inference") is False
        and report.get("ordinary_oof_merge_used") is False
        and report.get("gates", {}).get("bad_route_byte_identical") is True
    ):
        raise ValueError("full CV report does not authorize packaging")
    return report


def patched_run(payload: bytes) -> bytes:
    if sha256_bytes(payload) != BASE_RUN_SHA256:
        raise ValueError("base run.py checksum mismatch")
    text = payload.decode()
    path_anchor = '''QWEN35_ADAPTER_PATH = Path(
    os.environ.get("QWEN35_LORA_ADAPTER_PATH", ROOT / "adapter_qwen35")
)
'''
    path_replacement = path_anchor + '''QWEN35_FLAMMABLE_ADAPTER_PATH = Path(
    os.environ.get(
        "QWEN35_FLAMMABLE_LORA_ADAPTER_PATH", ROOT / "adapter_qwen35_flammable"
    )
)
'''
    loader_anchor = '''def open_lora_image(path: str | None) -> Image.Image:
    image = Image.open(path).convert("RGB") if path else Image.new("RGB", (32, 32), "white")
    # Training normalized every first image to this bound before both Qwen
    # processors. It also keeps Qwen3.5 visual tokens inside the 1,536-token
    # context instead of truncating multimodal special tokens.
    image.thumbnail((448, 448), Image.Resampling.LANCZOS)
    return image


'''
    loader_replacement = loader_anchor + '''def open_qwen35_distillation_image(path: str | None) -> Image.Image:
    image = Image.open(path).convert("RGB") if path else Image.new("RGB", (32, 32), "white")
    width, height = image.size
    if width * height > 262144:
        scale = (262144 / (width * height)) ** 0.5
        resized = image.resize(
            (max(28, int(width * scale)), max(28, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
        image.close()
        image = resized
    return image


'''
    score_signature_anchor = '''def compute_lora_scores(
    model, processor, frame: pd.DataFrame, *, use_chat_batch: bool = False
) -> np.ndarray:
'''
    score_signature_replacement = '''def compute_lora_scores(
    model,
    processor,
    frame: pd.DataFrame,
    *,
    use_chat_batch: bool = False,
    image_loader=open_lora_image,
) -> np.ndarray:
'''
    image_open_anchor = '''        images = [open_lora_image(path) for path in paths]
'''
    image_open_replacement = '''        images = [image_loader(path) for path in paths]
'''
    route_anchor = '''    qwen35_model, qwen35_processor = load_lora_model(
        QWEN35_MODEL_PATH, QWEN35_ADAPTER_PATH, "multimodal"
    )
    qwen35_scores = compute_lora_scores(
        qwen35_model, qwen35_processor, frame, use_chat_batch=True
    )
'''
    route_replacement = '''    qwen35_model, qwen35_processor = load_lora_model(
        QWEN35_MODEL_PATH, QWEN35_ADAPTER_PATH, "multimodal"
    )
    qwen35_scores = compute_lora_scores(
        qwen35_model, qwen35_processor, frame, use_chat_batch=True
    )
    qwen35_model.load_adapter(
        QWEN35_FLAMMABLE_ADAPTER_PATH,
        adapter_name="flammable_distillation",
        is_trainable=False,
    )
    qwen35_model.set_adapter("flammable_distillation")
    flammable_mask = frame["category"].to_numpy() == "Легковоспламеняющиеся"
    if flammable_mask.any():
        qwen35_scores[flammable_mask] = compute_lora_scores(
            qwen35_model,
            qwen35_processor,
            frame.loc[flammable_mask].reset_index(drop=True),
            use_chat_batch=True,
            image_loader=open_qwen35_distillation_image,
        )
'''
    anchors = (path_anchor, loader_anchor, score_signature_anchor, image_open_anchor, route_anchor)
    if any(text.count(anchor) != 1 for anchor in anchors):
        raise ValueError("base run.py patch anchor mismatch")
    text = (
        text.replace(path_anchor, path_replacement)
        .replace(loader_anchor, loader_replacement)
        .replace(score_signature_anchor, score_signature_replacement)
        .replace(image_open_anchor, image_open_replacement)
        .replace(route_anchor, route_replacement)
    )
    if "Qwen3.6-27B" in text or "27b" in text.lower():
        raise ValueError("submission run.py unexpectedly references 27B")
    return text.encode()


def zip_info(name: str, *, mode: int = 0o644) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = mode << 16
    return info


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("refusing to overwrite submission archive")
    if sha256_file(args.base_archive) != BASE_ARCHIVE_SHA256:
        raise ValueError("base submission archive checksum mismatch")
    full_contract = verify_full_contract(args.full_artifact / "output_contract.json")
    full_report = verify_full_report(args.full_report)
    adapter_dir = args.full_artifact / "adapter"
    adapter_files = {
        path.name: path
        for path in adapter_dir.iterdir()
        if path.is_file() and not path.name.startswith("._")
    }
    required_adapter = {"adapter_config.json", "adapter_model.safetensors"}
    if not required_adapter <= set(adapter_files):
        raise ValueError("full-refit adapter files are incomplete")
    adapter_config = json.loads(adapter_files["adapter_config.json"].read_text(encoding="utf-8"))
    if adapter_config.get("base_model_name_or_path") not in {
        "Qwen/Qwen3.5-4B",
        "/hf_models",
    }:
        raise ValueError("full-refit adapter is not bound to deployable Qwen3.5-4B")
    adapter_manifest = {
        name: sha256_file(path) for name, path in sorted(adapter_files.items())
    }

    with zipfile.ZipFile(args.base_archive) as source:
        if source.testzip() is not None:
            raise ValueError("corrupt base submission archive")
        source_names = source.namelist()
        if any(name.startswith("/") or ".." in Path(name).parts for name in source_names):
            raise ValueError("unsafe base archive member")
        if any(name.startswith(FLAMMABLE_ADAPTER_PREFIX) for name in source_names):
            raise ValueError("base archive already contains the candidate adapter path")
        run_payload = patched_run(source.read("run.py"))
        manifest = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "architecture": "category_routed_qwen35_4b_adapter",
            "base_archive_sha256": BASE_ARCHIVE_SHA256,
            "base_run_sha256": BASE_RUN_SHA256,
            "patched_run_sha256": sha256_bytes(run_payload),
            "full_cv_report_sha256": sha256_file(args.full_report),
            "full_cv_acceptance_tier": full_report.get("acceptance_tier"),
            "full_refit_contract_sha256": full_contract["contract_sha256"],
            "flammable_adapter_manifest": adapter_manifest,
            "bad_adapter_changed": False,
            "qwen3vl_adapter_changed": False,
            "embedder_and_classifiers_changed": False,
            "uses_27b_at_inference": False,
            "public_used_for_tuning": False,
        }
        manifest["contract_sha256"] = canonical_sha256(manifest)
        manifest_payload = (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, "w") as target:
            for name in source_names:
                if name == "run.py":
                    target.writestr(zip_info(name), run_payload)
                elif not name.endswith("/"):
                    target.writestr(zip_info(name), source.read(name))
            for name, path in sorted(adapter_files.items()):
                target.writestr(zip_info(FLAMMABLE_ADAPTER_PREFIX + name), path.read_bytes())
            target.writestr(zip_info("submission_manifest.json"), manifest_payload)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise ValueError("built submission archive is corrupt")
        names = archive.namelist()
        if any("27b" in name.lower() for name in names):
            raise ValueError("built submission archive contains a 27B-named member")
        if not {
            "run.py",
            "metadata.json",
            "adapter_qwen35/adapter_model.safetensors",
            FLAMMABLE_ADAPTER_PREFIX + "adapter_model.safetensors",
            "submission_manifest.json",
        } <= set(names):
            raise ValueError("built submission archive is incomplete")
    return {
        **manifest,
        "archive": str(args.output),
        "archive_size": args.output.stat().st_size,
        "archive_sha256": sha256_file(args.output),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--base-archive", type=Path, required=True)
    result.add_argument("--full-artifact", type=Path, required=True)
    result.add_argument("--full-report", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    print(json.dumps(build(parser().parse_args()), ensure_ascii=False, indent=2, sort_keys=True))
