from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

BASELINE_ARCHIVE_SHA256 = "6cc2fda9d17d959880050889d964c3b971b92adbfa606505df7f04b46a819cd3"
BASELINE_RUN_SHA256 = "9318db4cc0f26eabaa73ae9856dbf9daffd06787a162e6f5db71f249f6db7146"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_once(text: str, before: str, after: str) -> str:
    if text.count(before) != 1:
        raise ValueError("baseline run.py patch anchor is not unique")
    return text.replace(before, after)


def patch_run(payload: bytes) -> bytes:
    if sha256_bytes(payload) != BASELINE_RUN_SHA256:
        raise ValueError("baseline run.py checksum mismatch")
    text = payload.decode("utf-8")
    text = replace_once(
        text,
        '''QWEN35_ADAPTER_PATH = Path(
    os.environ.get("QWEN35_LORA_ADAPTER_PATH", ROOT / "adapter_qwen35")
)
''',
        '''QWEN35_ADAPTER_PATH = Path(
    os.environ.get("QWEN35_LORA_ADAPTER_PATH", ROOT / "adapter_qwen35")
)
QWEN35_FLAMMABLE_ADAPTER_PATH = Path(
    os.environ.get(
        "QWEN35_FLAMMABLE_LORA_ADAPTER_PATH", ROOT / "adapter_qwen35_flammable"
    )
)
''',
    )
    text = replace_once(
        text,
        '''def open_lora_image(path: str | None) -> Image.Image:
    image = Image.open(path).convert("RGB") if path else Image.new("RGB", (32, 32), "white")
    # Training normalized every first image to this bound before both Qwen
    # processors. It also keeps Qwen3.5 visual tokens inside the 1,536-token
    # context instead of truncating multimodal special tokens.
    image.thumbnail((448, 448), Image.Resampling.LANCZOS)
    return image
''',
        '''def open_lora_image(
    path: str | None,
    *,
    preprocessing: str = "thumbnail_448",
) -> Image.Image:
    image = Image.open(path).convert("RGB") if path else Image.new("RGB", (32, 32), "white")
    if preprocessing == "thumbnail_448":
        # Preserve the frozen BAD route byte-for-byte.
        image.thumbnail((448, 448), Image.Resampling.LANCZOS)
    elif preprocessing == "area_262144":
        # The flammable adapter is evaluated with its train-identical image view.
        width, height = image.size
        if width * height > 262144:
            scale = (262144 / (width * height)) ** 0.5
            resized = image.resize(
                (max(28, int(width * scale)), max(28, int(height * scale))),
                Image.Resampling.LANCZOS,
            )
            image.close()
            image = resized
    else:
        raise ValueError(f"unknown Qwen3.5 preprocessing route: {preprocessing}")
    return image
''',
    )
    text = replace_once(
        text,
        '''def compute_lora_scores(
    model, processor, frame: pd.DataFrame, *, use_chat_batch: bool = False
) -> np.ndarray:
''',
        '''def compute_lora_scores(
    model,
    processor,
    frame: pd.DataFrame,
    *,
    use_chat_batch: bool = False,
    image_preprocessing: str = "thumbnail_448",
) -> np.ndarray:
''',
    )
    text = replace_once(
        text,
        '''        images = [open_lora_image(path) for path in paths]
''',
        '''        images = [
            open_lora_image(path, preprocessing=image_preprocessing) for path in paths
        ]
''',
    )
    text = replace_once(
        text,
        '''    qwen35_model, qwen35_processor = load_lora_model(
        QWEN35_MODEL_PATH, QWEN35_ADAPTER_PATH, "multimodal"
    )
    qwen35_scores = compute_lora_scores(
        qwen35_model, qwen35_processor, frame, use_chat_batch=True
    )
''',
        '''    qwen35_model, qwen35_processor = load_lora_model(
        QWEN35_MODEL_PATH, QWEN35_ADAPTER_PATH, "multimodal"
    )
    if not QWEN35_FLAMMABLE_ADAPTER_PATH.is_dir():
        raise FileNotFoundError(
            f"Flammable Qwen3.5 adapter not found: {QWEN35_FLAMMABLE_ADAPTER_PATH}"
        )
    qwen35_model.load_adapter(
        QWEN35_FLAMMABLE_ADAPTER_PATH,
        adapter_name="flammable",
        is_trainable=False,
    )
    qwen35_scores = np.zeros(len(frame), dtype=np.float32)
    category_values = frame["category"].to_numpy()
    adapter_routes = {
        "БАД": ("default", "thumbnail_448"),
        "Легковоспламеняющиеся": ("flammable", "area_262144"),
    }
    for category, (adapter_name, image_preprocessing) in adapter_routes.items():
        mask = category_values == category
        if not mask.any():
            print(f"qwen35_category={category!r} rows=0 skipped", flush=True)
            continue
        qwen35_model.set_adapter(adapter_name, inference_mode=True)
        qwen35_scores[mask] = compute_lora_scores(
            qwen35_model,
            qwen35_processor,
            frame.loc[mask].reset_index(drop=True),
            use_chat_batch=True,
            image_preprocessing=image_preprocessing,
        )
        print(
            f"qwen35_category={category!r} rows={int(mask.sum())} "
            f"adapter={adapter_name} preprocessing={image_preprocessing}",
            flush=True,
        )
''',
    )
    compile(text, "run.py", "exec")
    return text.encode("utf-8")


def validate_full_report(path: Path) -> str:
    payload = path.read_bytes()
    report = json.loads(payload)
    if not (
        report.get("experiment_id") == "680"
        and report.get("stage") == "full"
        and (
            report.get("passed") is True
            or report.get("ablation_submission_eligible") is True
        )
        and report.get("decision") == "ACCEPT_FOR_REFIT"
        and report.get("public_used") is False
        and report.get("sealed_rows") == 0
        and (
            all(report.get("gates", {}).values())
            or all(report.get("ablation_gates", {}).values())
        )
    ):
        raise ValueError("full report does not authorize submission packaging")
    return sha256_bytes(payload)


def load_full_adapter(path: Path) -> tuple[dict[str, bytes], dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("full-refit artifact is corrupt")
        contract = json.loads(archive.read("output_contract.json"))
        required = {
            "adapter/README.md",
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
        }
        if not required <= set(archive.namelist()):
            raise ValueError("full-refit adapter members are missing")
        members = {name: archive.read(name) for name in sorted(required)}
    expected = {
        "experiment_id": "680",
        "control_experiment_id": "641",
        "full_refit": True,
        "changed_factor": "remove_bad_training_occurrences",
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": MODEL_REVISION,
        "train_occurrences": 2590,
        "validation_rows": 0,
        "optimizer_steps_executed": 162,
        "uses_27b_at_training": False,
        "uses_27b_at_inference": False,
        "decision": "GO_PACKAGE_AFTER_FULL_CV",
    }
    mismatch = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise ValueError(f"full-refit output contract mismatch: {mismatch}")
    config = json.loads(members["adapter/adapter_config.json"])
    if config.get("base_model_name_or_path") not in {"Qwen/Qwen3.5-4B", "/hf_models"}:
        raise ValueError("full-refit adapter is not compatible with the deployable 4B base")
    return members, contract


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("refusing to overwrite submission archive")
    if sha256_file(args.baseline_archive) != BASELINE_ARCHIVE_SHA256:
        raise ValueError("baseline submission archive checksum mismatch")
    full_report_sha256 = validate_full_report(args.full_report)
    adapter_members, contract = load_full_adapter(args.full_adapter_artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.baseline_archive) as source, zipfile.ZipFile(
        args.output,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as target:
        if source.testzip() is not None:
            raise ValueError("baseline submission archive is corrupt")
        names = source.namelist()
        if any(name.startswith("adapter_qwen35_flammable/") for name in names):
            raise ValueError("baseline unexpectedly already contains a flammable adapter")
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "run.py":
                payload = patch_run(payload)
            target.writestr(info, payload)
        for source_name, payload in adapter_members.items():
            suffix = source_name.removeprefix("adapter/")
            info = zipfile.ZipInfo(f"adapter_qwen35_flammable/{suffix}")
            info.date_time = (2026, 8, 24, 0, 0, 0)
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(info, payload, compresslevel=6)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise ValueError("built submission archive is corrupt")
        run_sha256 = sha256_bytes(archive.read("run.py"))
        required = {
            "adapter_qwen35/adapter_model.safetensors",
            "adapter_qwen35_flammable/adapter_model.safetensors",
            "metadata.json",
            "run.py",
        }
        if not required <= set(archive.namelist()):
            raise ValueError("built submission archive is incomplete")
    return {
        "schema_version": 1,
        "experiment_id": "680",
        "baseline_archive_sha256": BASELINE_ARCHIVE_SHA256,
        "full_report_sha256": full_report_sha256,
        "full_adapter_artifact_sha256": sha256_file(args.full_adapter_artifact),
        "full_adapter_contract_sha256": contract["contract_sha256"],
        "run_sha256": run_sha256,
        "submission_archive_sha256": sha256_file(args.output),
        "uses_27b": False,
        "category_routed_single_forward": True,
        "bad_preprocessing_unchanged": True,
        "flammable_preprocessing": "area_262144",
        "decision": "GO_RUNTIME_SMOKE",
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--baseline-archive", type=Path, required=True)
    result.add_argument("--full-adapter-artifact", type=Path, required=True)
    result.add_argument("--full-report", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2, sort_keys=True))
