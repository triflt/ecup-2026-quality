from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

EXCLUDED_PARTS = {"__pycache__", ".git", ".pytest_cache", ".DS_Store"}
ADAPTER_MEMBERS = {"README.md", "adapter_config.json", "adapter_model.safetensors"}


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
        if any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts):
            continue
        if path.is_symlink():
            raise ValueError(f"submission member is symlinked: {path.relative_to(root)}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path
    return result


def patch_submission_run(source: str) -> str:
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
        # Qwen3-VL keeps the frozen solution-140 448px preprocessing.
        image.thumbnail((448, 448), Image.Resampling.LANCZOS)
        return image
    # The accepted Qwen3.5 candidate reuses the exact experiment-641
    # preprocessing: preserve aspect ratio and cap the first image by area.
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
            raise ValueError("solution-140 runtime patch anchor mismatch")
        output = output.replace(old, new)
    return output


def verify_refit(
    refit_output: Path, expected_runtime_contract: str
) -> tuple[dict[str, Any], dict[str, str]]:
    contract_path = refit_output / "output_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    payload = dict(contract)
    declared = payload.pop("contract_sha256", None)
    if declared != canonical_sha256(payload):
        raise ValueError("refit output contract self-hash mismatch")
    expected = {
        "experiment_id": "699",
        "architecture": "qwen35_4b",
        "fold": -1,
        "source": "v2",
        "mode": "positive_only_append",
        "cap": 10,
        "epochs": 1,
        "train_occurrences": 5460,
        "synthetic_occurrences": 10,
        "optimizer_steps": 342,
        "validation_rows": 0,
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_rows_used": 0,
        "decision": "GO_PACKAGE",
    }
    mismatch = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise ValueError(f"refit output gate mismatch: {mismatch}")
    if contract.get("runtime_contract_sha256") != expected_runtime_contract:
        raise ValueError("refit runtime contract binding mismatch")
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
    return contract, {name: sha256_file(path) for name, path in members.items()}


def build(
    *,
    source: Path,
    refit_output: Path,
    destination: Path,
    archive: Path,
    expected_source_run_sha256: str,
    expected_runtime_contract: str,
) -> dict[str, Any]:
    if destination.exists() or archive.exists():
        raise FileExistsError("refusing to overwrite submission candidate")
    source_files = regular_files(source)
    if not {"run.py", "metadata.json", "adapter_qwen35/adapter_model.safetensors"}.issubset(
        source_files
    ):
        raise ValueError("solution-140 source inventory is incomplete")
    if sha256_file(source_files["run.py"]) != expected_source_run_sha256:
        raise ValueError("solution-140 runtime SHA mismatch")
    contract, adapter_hashes = verify_refit(refit_output, expected_runtime_contract)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", ".DS_Store", ".pytest_cache"),
    )
    shutil.rmtree(destination / "adapter_qwen35")
    shutil.copytree(refit_output / "adapter", destination / "adapter_qwen35")
    patched = patch_submission_run(source_files["run.py"].read_text(encoding="utf-8"))
    (destination / "run.py").write_text(patched, encoding="utf-8")

    unchanged = {}
    for relative, source_path in source_files.items():
        if relative == "run.py" or relative.startswith("adapter_qwen35/"):
            continue
        candidate_path = destination / relative
        source_hash = sha256_file(source_path)
        if sha256_file(candidate_path) != source_hash:
            raise RuntimeError(f"unchanged solution-140 file drifted: {relative}")
        unchanged[relative] = source_hash
    manifest: dict[str, Any] = {
        "schema": "exp699_solution140_q35_refit_candidate_v1",
        "experiment_id": "699",
        "parent_submission": "140",
        "changed_factor": "qwen35_adapter_v2_positive_append_cap10_full_refit",
        "source_run_sha256": expected_source_run_sha256,
        "candidate_run_sha256": sha256_file(destination / "run.py"),
        "refit_output_contract_sha256": contract["contract_sha256"],
        "refit_runtime_contract_sha256": expected_runtime_contract,
        "adapter_members_sha256": adapter_hashes,
        "unchanged_files_sha256": unchanged,
        "qwen3vl_adapter_unchanged": True,
        "base_models_unchanged": True,
        "fusion_weights_unchanged": True,
        "thresholds_unchanged": True,
        "public_used": False,
        "sealed_rows_used": 0,
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
    parser.add_argument("--refit-output", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-source-run-sha256", required=True)
    parser.add_argument("--expected-runtime-contract", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                source=args.source,
                refit_output=args.refit_output,
                destination=args.destination,
                archive=args.archive,
                expected_source_run_sha256=args.expected_source_run_sha256,
                expected_runtime_contract=args.expected_runtime_contract,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
