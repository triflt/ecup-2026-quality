"""Build the immutable exact-base Qwen3.6-27B model-tree contract for exp689."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from build_target_audit import (
    ContractError,
    canonical_json_bytes,
    require_remote_path,
    sha256_bytes,
    sha256_file,
    with_self_hash,
    write_json,
)

MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
FORBIDDEN_FILE_NAMES = {
    "adapter_config.json",
    "adapter_model.bin",
    "adapter_model.safetensors",
}
FORBIDDEN_PATH_PARTS = {"adapter", "adapters", "lora", "peft"}
PROCESSOR_MARKERS = (
    "processor",
    "tokenizer",
    "preprocessor",
    "chat_template",
    "vocab",
    "merges",
    "special_tokens",
)


def is_processor_file(path: Path) -> bool:
    name = path.name.lower()
    return any(marker in name for marker in PROCESSOR_MARKERS)


def scan_tree(
    root: Path, *, predicate: Callable[[Path], bool] | None = None
) -> tuple[str, int]:
    if not root.is_dir():
        raise ContractError("teacher model root is not a directory")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ContractError(f"teacher model tree contains a forbidden symlink: {relative}")
        lowered_parts = {part.lower() for part in relative.parts}
        if path.is_file() and (
            path.name.lower() in FORBIDDEN_FILE_NAMES
            or lowered_parts & FORBIDDEN_PATH_PARTS
            or path.name.lower().startswith("adapter_model.")
        ):
            raise ContractError(f"teacher model tree contains an adapter/class-LoRA: {relative}")
        if not path.is_file() or (predicate is not None and not predicate(path)):
            continue
        entries.append(
            {
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise ContractError("teacher model tree selection is empty")
    return sha256_bytes(canonical_json_bytes(entries)), len(entries)


def build(*, remote_root: Path, model_root: Path, output_path: Path) -> dict[str, Any]:
    remote_root = remote_root.resolve(strict=True)
    model_root = require_remote_path(
        remote_root, model_root, context="teacher model root", must_exist=True
    )
    output_path = require_remote_path(
        remote_root, output_path, context="teacher model contract output", must_exist=False
    )
    if output_path.exists():
        raise FileExistsError("refusing to overwrite immutable teacher model contract")
    if output_path.is_relative_to(model_root):
        raise ContractError("teacher model contract output must remain outside the model tree")
    model_sha, model_files = scan_tree(model_root)
    processor_sha, processor_files = scan_tree(model_root, predicate=is_processor_file)
    contract = with_self_hash(
        {
            "schema_version": "exp689_teacher_model_contract_v1",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_tree_sha256": model_sha,
            "model_tree_files": model_files,
            "processor_sha256": processor_sha,
            "processor_files": processor_files,
            "base_only": True,
            "class_lora_present": False,
            "self_sha256": None,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, contract)
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = build(
        remote_root=args.remote_root,
        model_root=args.model_root,
        output_path=args.output,
    )
    print(contract["self_sha256"])


if __name__ == "__main__":
    main()
