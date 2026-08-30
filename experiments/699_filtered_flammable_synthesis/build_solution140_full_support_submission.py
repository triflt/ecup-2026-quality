from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Any

EXCLUDED_PARTS = {"__pycache__", ".git", ".pytest_cache", ".DS_Store"}
ADAPTER_MEMBERS = {"README.md", "adapter_config.json", "adapter_model.safetensors"}
EXPECTED_REAL_SELECTION = {
    "selected_real_ordered_indices_sha256": (
        "2006a56d7499cf754fe9db34781117350a738fae513eec334eaa6864ad1f309d"
    ),
    "selected_real_multiset_sha256": (
        "0a31883f4e0b68a5fd9f6f803f660d02d67300cde30dd4f60564253e90a4af63"
    ),
    "selected_real_unique_ids_sha256": (
        "3e9e0bbdad4d2f832521375ab1a305a42de0e8b31e12bae29bf869b87df47ba1"
    ),
}


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"submission member is symlinked: {relative}")
        if path.is_file():
            result[relative.as_posix()] = path
    return result


def load_self_hashed(path: Path, field: str) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing regular contract: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(value)
    declared = payload.pop(field, None)
    if declared != canonical_sha256(payload):
        raise ValueError(f"contract self-hash mismatch: {path}")
    return value, str(declared)


def verify_refit(
    *,
    runtime_dir: Path,
    refit_output: Path,
    expected_runtime_contract: str,
    expected_synth_ids_sha256: str,
    expected_synth_payload_sha256: str,
    synth_cap: int,
    preprocessing: str,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    runtime, runtime_contract = load_self_hashed(
        runtime_dir / "runtime_audit.json", "contract_sha256"
    )
    expected_runtime = {
        "schema": "exp699_solution140_full_support_refit_runtime_v1",
        "decision": "GO_FULL_REFIT_SOLUTION140_FULL_SUPPORT",
        "outer_fold": -1,
        "source": "v2",
        "mode": "positive_only_append",
        "cap": synth_cap,
        "preprocessing_key": preprocessing,
        "original_real_occurrences": 6790,
        "original_real_unique": 5998,
        "train_occurrences": 6790 + synth_cap,
        "synthetic_occurrences": synth_cap,
        "removed_real_occurrences": 0,
        "expected_optimizer_steps": 425,
        "validation_labels_written": 0,
        "sealed_rows_written": 0,
        "public_rows_used": 0,
        "selected_synth_ids_sha256": expected_synth_ids_sha256,
        "selected_synth_payload_sha256": expected_synth_payload_sha256,
        **EXPECTED_REAL_SELECTION,
    }
    mismatch = {
        key: {"expected": value, "actual": runtime.get(key)}
        for key, value in expected_runtime.items()
        if runtime.get(key) != value
    }
    if mismatch or runtime_contract != expected_runtime_contract:
        raise ValueError(f"full-support runtime gate mismatch: {mismatch}")

    contract, contract_self = load_self_hashed(
        refit_output / "output_contract.json", "contract_sha256"
    )
    expected_output = {
        "experiment_id": "699",
        "architecture": "qwen35_4b",
        "fold": -1,
        "runtime_contract_sha256": expected_runtime_contract,
        "source": "v2",
        "mode": "positive_only_append",
        "cap": synth_cap,
        "preprocessing_key": preprocessing,
        "epochs": 1,
        "train_occurrences": 6790 + synth_cap,
        "training_occurrences_seen": 6790 + synth_cap,
        "synthetic_occurrences": synth_cap,
        "optimizer_steps": 425,
        "validation_rows": 0,
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_rows_used": 0,
        "decision": "GO_PACKAGE",
    }
    output_mismatch = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected_output.items()
        if contract.get(key) != value
    }
    smoke = contract.get("technical_smoke", {})
    if (
        output_mismatch
        or smoke.get("finite") is not True
        or not math.isfinite(float(smoke.get("loss", math.nan)))
        or not math.isfinite(float(smoke.get("grad_norm", math.nan)))
        or not math.isfinite(float(contract.get("last_grad_norm", math.nan)))
        or not math.isfinite(float(contract.get("training_runtime_minutes", math.nan)))
    ):
        raise ValueError(f"full-support output gate mismatch: {output_mismatch}")

    predictions = refit_output / "predictions.jsonl"
    adapter_zip = refit_output / "adapter.zip"
    actual_artifacts = {
        "predictions.jsonl": sha256_file(predictions),
        "adapter.zip": sha256_file(adapter_zip),
    }
    if contract.get("artifacts") != actual_artifacts or predictions.stat().st_size != 0:
        raise ValueError("refit artifact binding mismatch")
    adapter = refit_output / "adapter"
    members = regular_files(adapter)
    if set(members) != ADAPTER_MEMBERS:
        raise ValueError(f"unexpected refit adapter inventory: {sorted(members)}")
    adapter_hashes = {name: sha256_file(path) for name, path in members.items()}
    with zipfile.ZipFile(adapter_zip) as archive:
        zip_members = {
            info.filename: hashlib.sha256(archive.read(info)).hexdigest()
            for info in archive.infolist()
            if not info.is_dir()
        }
    if zip_members != adapter_hashes:
        raise ValueError("adapter ZIP differs from adapter directory")
    return contract, adapter_hashes, {
        "runtime_contract_sha256": runtime_contract,
        "runtime_audit_file_sha256": sha256_file(runtime_dir / "runtime_audit.json"),
        "output_contract_sha256": contract_self,
        "output_contract_file_sha256": sha256_file(
            refit_output / "output_contract.json"
        ),
        "adapter_zip_sha256": actual_artifacts["adapter.zip"],
        "selected_synth_ids_sha256": expected_synth_ids_sha256,
        "selected_synth_payload_sha256": expected_synth_payload_sha256,
    }


def patch_area_cap(source: str) -> str:
    replacements = {
        "import json\nimport os\n": "import json\nimport math\nimport os\n",
        '''def open_lora_image(path: str | None) -> Image.Image:
    image = Image.open(path).convert("RGB") if path else Image.new("RGB", (32, 32), "white")
    # Training normalized every first image to this bound before both Qwen
    # processors. It also keeps Qwen3.5 visual tokens inside the 1,536-token
    # context instead of truncating multimodal special tokens.
    image.thumbnail((448, 448), Image.Resampling.LANCZOS)
    return image
''': '''def open_lora_image(
    path: str | None, *, max_pixels: int | None = None
) -> Image.Image:
    image = Image.open(path).convert("RGB") if path else Image.new("RGB", (32, 32), "white")
    if max_pixels is None:
        image.thumbnail((448, 448), Image.Resampling.LANCZOS)
        return image
    width, height = image.size
    if width * height > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
        resized = image.resize(
            (max(28, int(width * scale)), max(28, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
        image.close()
        image = resized
    return image
''',
        '''def compute_lora_scores(
    model, processor, frame: pd.DataFrame, *, use_chat_batch: bool = False
) -> np.ndarray:
''': '''def compute_lora_scores(
    model,
    processor,
    frame: pd.DataFrame,
    *,
    use_chat_batch: bool = False,
    max_pixels: int | None = None,
) -> np.ndarray:
''',
        '''        images = [open_lora_image(path) for path in paths]
''': '''        images = [open_lora_image(path, max_pixels=max_pixels) for path in paths]
''',
        '''    qwen35_scores = compute_lora_scores(
        qwen35_model, qwen35_processor, frame, use_chat_batch=True
    )
''': '''    qwen35_scores = compute_lora_scores(
        qwen35_model,
        qwen35_processor,
        frame,
        use_chat_batch=True,
        max_pixels=262144,
    )
''',
    }
    output = source
    for old, new in replacements.items():
        if output.count(old) != 1:
            raise ValueError("solution140 area-cap patch anchor mismatch")
        output = output.replace(old, new)
    return output


def patch_flammable_route(source: str) -> str:
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
    scoring_anchor = '''    qwen35_model, qwen35_processor = load_lora_model(
        QWEN35_MODEL_PATH, QWEN35_ADAPTER_PATH, "multimodal"
    )
    qwen35_scores = compute_lora_scores(
        qwen35_model, qwen35_processor, frame, use_chat_batch=True
    )
'''
    scoring_replacement = '''    qwen35_model, qwen35_processor = load_lora_model(
        QWEN35_MODEL_PATH, QWEN35_ADAPTER_PATH, "multimodal"
    )
    # Preserve the original solution140 Qwen3.5 pass byte-for-byte, including
    # row order and batch composition. Only flammable scores are overwritten.
    qwen35_scores = compute_lora_scores(
        qwen35_model, qwen35_processor, frame, use_chat_batch=True
    )
    del qwen35_model, qwen35_processor
    gc.collect()
    torch.cuda.empty_cache()
    qwen35_model, qwen35_processor = load_lora_model(
        QWEN35_MODEL_PATH, QWEN35_FLAMMABLE_ADAPTER_PATH, "multimodal"
    )
    qwen35_flammable_scores = compute_lora_scores(
        qwen35_model, qwen35_processor, frame, use_chat_batch=True
    )
    qwen35_categories = frame["category"].astype(str).to_numpy()
    qwen35_flammable_mask = qwen35_categories == "Легковоспламеняющиеся"
    qwen35_scores[qwen35_flammable_mask] = qwen35_flammable_scores[
        qwen35_flammable_mask
    ]
    print(
        f"qwen35_route='БАД' rows={int((~qwen35_flammable_mask).sum())} "
        f"adapter={QWEN35_ADAPTER_PATH.name}",
        flush=True,
    )
    print(
        f"qwen35_route='Легковоспламеняющиеся' "
        f"rows={int(qwen35_flammable_mask.sum())} "
        f"adapter={QWEN35_FLAMMABLE_ADAPTER_PATH.name}",
        flush=True,
    )
    del qwen35_flammable_scores
'''
    if source.count(path_anchor) != 1 or source.count(scoring_anchor) != 1:
        raise ValueError("solution140 category-route patch anchor mismatch")
    return source.replace(path_anchor, path_replacement).replace(
        scoring_anchor, scoring_replacement
    )


def build(
    *,
    source: Path,
    runtime_dir: Path,
    refit_output: Path,
    destination: Path,
    archive: Path,
    route: str,
    preprocessing: str,
    synth_cap: int,
    expected_source_run_sha256: str,
    expected_runtime_contract: str,
    expected_synth_ids_sha256: str,
    expected_synth_payload_sha256: str,
) -> dict[str, Any]:
    if destination.exists() or archive.exists():
        raise FileExistsError("refusing to overwrite submission candidate")
    if route == "flammable_only" and (synth_cap, preprocessing) != (10, "thumbnail448"):
        raise ValueError("flammable-only route is preregistered only for A")
    source_files = regular_files(source)
    required = {"run.py", "metadata.json", "adapter_qwen35/adapter_model.safetensors"}
    if not required.issubset(source_files):
        raise ValueError("solution140 source inventory is incomplete")
    if sha256_file(source_files["run.py"]) != expected_source_run_sha256:
        raise ValueError("solution140 runtime SHA mismatch")
    contract, adapter_hashes, refit_binding = verify_refit(
        runtime_dir=runtime_dir,
        refit_output=refit_output,
        expected_runtime_contract=expected_runtime_contract,
        expected_synth_ids_sha256=expected_synth_ids_sha256,
        expected_synth_payload_sha256=expected_synth_payload_sha256,
        synth_cap=synth_cap,
        preprocessing=preprocessing,
    )
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", ".DS_Store", ".pytest_cache"),
    )
    original_qwen35_hashes = {
        relative.removeprefix("adapter_qwen35/"): sha256_file(path)
        for relative, path in source_files.items()
        if relative.startswith("adapter_qwen35/")
    }
    if set(original_qwen35_hashes) != ADAPTER_MEMBERS:
        raise ValueError("solution140 original Qwen3.5 adapter inventory mismatch")
    source_run = source_files["run.py"].read_text(encoding="utf-8")
    if route == "all":
        shutil.rmtree(destination / "adapter_qwen35")
        shutil.copytree(refit_output / "adapter", destination / "adapter_qwen35")
        candidate_run = (
            source_run if preprocessing == "thumbnail448" else patch_area_cap(source_run)
        )
        expected_changed_prefixes = {"adapter_qwen35/"}
        route_contract = {
            "БАД": "refit_adapter",
            "Легковоспламеняющиеся": "refit_adapter",
        }
    else:
        shutil.copytree(
            refit_output / "adapter", destination / "adapter_qwen35_flammable"
        )
        candidate_run = patch_flammable_route(source_run)
        expected_changed_prefixes = {"adapter_qwen35_flammable/"}
        route_contract = {
            "БАД": "original_solution140_qwen35_adapter",
            "Легковоспламеняющиеся": "refit_adapter",
        }
    (destination / "run.py").write_text(candidate_run, encoding="utf-8")

    unchanged: dict[str, str] = {}
    for relative, source_path in source_files.items():
        if relative == "run.py" or any(
            relative.startswith(prefix) for prefix in expected_changed_prefixes
        ):
            continue
        candidate_path = destination / relative
        source_hash = sha256_file(source_path)
        if sha256_file(candidate_path) != source_hash:
            raise RuntimeError(f"unrelated solution140 member drifted: {relative}")
        unchanged[relative] = source_hash
    if route == "flammable_only":
        routed_original = {
            relative.removeprefix("adapter_qwen35/"): sha256_file(path)
            for relative, path in regular_files(destination).items()
            if relative.startswith("adapter_qwen35/")
        }
        if routed_original != original_qwen35_hashes:
            raise RuntimeError("BAD-route original Qwen3.5 adapter drifted")

    manifest: dict[str, Any] = {
        "schema": "exp699_solution140_full_support_candidate_v1",
        "experiment_id": "699",
        "parent_submission": "140",
        "route": route,
        "category_route": route_contract,
        "preprocessing_key": preprocessing,
        "synth_cap": synth_cap,
        "source_run_sha256": expected_source_run_sha256,
        "candidate_run_sha256": sha256_file(destination / "run.py"),
        "refit_output_contract_sha256": contract["contract_sha256"],
        "refit_binding": refit_binding,
        "refit_adapter_members_sha256": adapter_hashes,
        "original_qwen35_adapter_members_sha256": original_qwen35_hashes,
        "unchanged_files_sha256": unchanged,
        "unrelated_solution140_members_changed": 0,
        "qwen3vl_adapter_unchanged": True,
        "base_models_unchanged": True,
        "fusion_weights_unchanged": True,
        "thresholds_unchanged": True,
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "decision": "GO_PACKAGE_SMOKE",
    }
    manifest["self_sha256"] = canonical_sha256(manifest)
    (destination / "exp699_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as output:
        for relative, path in regular_files(destination).items():
            output.write(path, relative)
    manifest["archive_sha256"] = sha256_file(archive)
    manifest["archive_size"] = archive.stat().st_size
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--refit-output", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--route", choices=("all", "flammable_only"), required=True)
    parser.add_argument(
        "--preprocessing", choices=("thumbnail448", "area_cap262144"), required=True
    )
    parser.add_argument("--synth-cap", type=int, choices=(5, 10), required=True)
    parser.add_argument("--expected-source-run-sha256", required=True)
    parser.add_argument("--expected-runtime-contract", required=True)
    parser.add_argument("--expected-synth-ids-sha256", required=True)
    parser.add_argument("--expected-synth-payload-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                source=args.source,
                runtime_dir=args.runtime_dir,
                refit_output=args.refit_output,
                destination=args.destination,
                archive=args.archive,
                route=args.route,
                preprocessing=args.preprocessing,
                synth_cap=args.synth_cap,
                expected_source_run_sha256=args.expected_source_run_sha256,
                expected_runtime_contract=args.expected_runtime_contract,
                expected_synth_ids_sha256=args.expected_synth_ids_sha256,
                expected_synth_payload_sha256=args.expected_synth_payload_sha256,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
