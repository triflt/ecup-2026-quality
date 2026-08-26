"""Run the frozen base Qwen3.6-27B exp689 extraction-only teacher."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import struct
import time
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from build_target_audit import (
    ContractError,
    canonical_json_bytes,
    expect_exact_keys,
    load_json,
    load_spec,
    read_jsonl,
    require_hex64,
    require_remote_path,
    sha256_bytes,
    sha256_file,
    validate_selection_manifest,
    validate_self_hash,
    validate_teacher_request,
    validate_teacher_selection,
    with_self_hash,
    write_json,
    write_jsonl,
)

MODEL_ID = "Qwen/Qwen3.6-27B"
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
MAX_NEW_TOKENS = 160
PROMPT_VERSION = "exp689_extraction_only_image_grounding_v1"
SYSTEM_PROMPT = """You are an extraction-only annotation engine.
Use ONLY the supplied listing text/OCR and the visible first image. Do not use prior knowledge.
UNKNOWN is preferred whenever the supplied evidence is insufficient or ambiguous.
Return exactly one JSON object and nothing else: no markdown, free text, reasoning, chain-of-thought, verdict, confidence, score, probability, rank, label, or family ID.
Use only the closed enum values and candidate IDs printed in the request. Bind separate nonempty evidence ID lists for sold_object, substance, and relation when support_status is supported. For unsupported or ambiguous, set supervise=false and all three evidence lists empty.
Image regions are full, q00=top-left, q01=top-right, q10=bottom-left, q11=bottom-right.
"""
PROMPT_SHA256 = sha256_bytes(
    canonical_json_bytes(
        {
            "prompt_version": PROMPT_VERSION,
            "system_prompt": SYSTEM_PROMPT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "enable_thinking": False,
        }
    )
)
IMAGE_MANIFEST_FIELDS = {
    "schema_version",
    "audit_id",
    "reference",
    "relative_path",
    "content_sha256",
    "decoded_rgb_sha256",
    "pixel_sha256",
    "media_type",
    "width",
    "height",
}


class RGBImage:
    """Minimal RGB image used only when Pillow is absent for synthetic PNG tests."""

    def __init__(self, width: int, height: int, pixels: bytes) -> None:
        self.size = (width, height)
        self._pixels = pixels

    def tobytes(self) -> bytes:
        return self._pixels

    def crop(self, box: tuple[int, int, int, int]) -> RGBImage:
        left, top, right, bottom = box
        width, _ = self.size
        pieces = []
        for row in range(top, bottom):
            start = (row * width + left) * 3
            pieces.append(self._pixels[start : start + (right - left) * 3])
        return RGBImage(right - left, bottom - top, b"".join(pieces))

    def close(self) -> None:
        return None


def _decode_smoke_png(path: Path) -> RGBImage:
    payload = path.read_bytes()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ContractError("Pillow is required for non-PNG teacher images")
    offset = 8
    width = height = 0
    compressed = bytearray()
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise ContractError("synthetic PNG is truncated")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, depth, color, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", data
            )
            if (depth, color, compression, filtering, interlace) != (8, 2, 0, 0, 0):
                raise ContractError("synthetic PNG format is not exact RGB8")
        elif kind == b"IDAT":
            compressed.extend(data)
        elif kind == b"IEND":
            break
    raw = zlib.decompress(bytes(compressed))
    stride = width * 3
    rows = []
    for offset in range(0, len(raw), stride + 1):
        if raw[offset] != 0:
            raise ContractError("synthetic PNG uses an unsupported row filter")
        rows.append(raw[offset + 1 : offset + stride + 1])
    pixels = b"".join(rows)
    if len(rows) != height or len(pixels) != width * height * 3:
        raise ContractError("synthetic PNG decoded size mismatch")
    return RGBImage(width, height, pixels)


def decode_rgb_image(path: Path) -> Any:
    try:
        from PIL import Image
    except ModuleNotFoundError:
        return _decode_smoke_png(path)
    with Image.open(path) as opened:
        opened.load()
        return opened.convert("RGB")


def tree_sha256(root: Path, *, predicate: Callable[[Path], bool] | None = None) -> tuple[str, int]:
    if not root.is_dir():
        raise ContractError("model root is not a directory")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError("model tree may not contain symlinks")
        if not path.is_file() or (predicate is not None and not predicate(path)):
            continue
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise ContractError("model tree selection is empty")
    return sha256_bytes(canonical_json_bytes(entries)), len(entries)


def is_processor_file(path: Path) -> bool:
    name = path.name.lower()
    return any(
        marker in name
        for marker in (
            "processor",
            "tokenizer",
            "preprocessor",
            "chat_template",
            "vocab",
            "merges",
            "special_tokens",
        )
    )


def validate_base_model_tree(model_root: Path, model_contract: dict[str, Any]) -> dict[str, Any]:
    expect_exact_keys(
        model_contract,
        {
            "schema_version",
            "model_id",
            "model_revision",
            "model_tree_sha256",
            "model_tree_files",
            "processor_sha256",
            "processor_files",
            "base_only",
            "class_lora_present",
            "self_sha256",
        },
        "teacher model contract",
    )
    validate_self_hash(model_contract, "teacher model contract")
    if model_contract["schema_version"] != "exp689_teacher_model_contract_v1":
        raise ContractError("teacher model contract: schema version mismatch")
    if model_contract["model_id"] != MODEL_ID or model_contract["model_revision"] != MODEL_REVISION:
        raise ContractError("teacher model contract: frozen model or revision mismatch")
    if model_contract["base_only"] is not True or model_contract["class_lora_present"] is not False:
        raise ContractError("teacher model contract: base-only model is required")
    forbidden_names = {
        "adapter_config.json",
        "adapter_model.bin",
        "adapter_model.safetensors",
    }
    if any(path.is_file() and path.name in forbidden_names for path in model_root.rglob("*")):
        raise ContractError("teacher model tree: adapter/class-LoRA artifact is forbidden")
    model_sha, model_files = tree_sha256(model_root)
    processor_sha, processor_files = tree_sha256(model_root, predicate=is_processor_file)
    expected = {
        "model_tree_sha256": model_sha,
        "model_tree_files": model_files,
        "processor_sha256": processor_sha,
        "processor_files": processor_files,
    }
    for field, value in expected.items():
        if model_contract[field] != value:
            raise ContractError(f"teacher model contract: {field} mismatch")
    return expected


def validate_smoke_input_contract(
    contract: dict[str, Any], request_path: Path, image_manifest_path: Path
) -> None:
    expect_exact_keys(
        contract,
        {
            "schema_version",
            "experiment_id",
            "scope",
            "teacher_request_sha256",
            "teacher_request_rows",
            "image_manifest_sha256",
            "pixel_set_sha256",
            "runner_code_sha256",
            "prompt_sha256",
            "model_id",
            "model_revision",
            "structural_enum_coverage",
            "evidence_kind_coverage",
            "image_region_coverage",
            "source_kind_coverage",
            "labels_present",
            "sealed_rows",
            "public_used",
            "quality_evaluated",
            "full_teacher_authorized",
            "student_gpu_authorized",
            "jobs_launched",
            "uploads",
            "presets_built",
            "bundles_built",
            "self_sha256",
        },
        "teacher smoke input contract",
    )
    validate_self_hash(contract, "teacher smoke input contract")
    spec = load_spec()
    exact = {
        "schema_version": "exp689_teacher_smoke_input_contract_v1",
        "experiment_id": "689",
        "scope": "synthetic_technical_smoke",
        "teacher_request_sha256": sha256_file(request_path),
        "teacher_request_rows": 12,
        "image_manifest_sha256": sha256_file(image_manifest_path),
        "runner_code_sha256": sha256_file(Path(__file__)),
        "prompt_sha256": PROMPT_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "structural_enum_coverage": {
            key: spec["enums"][key] for key in ("sold_object", "substance", "relation")
        },
        "evidence_kind_coverage": ["image_region", "text_span"],
        "image_region_coverage": ["full", "q00", "q01", "q10", "q11"],
        "source_kind_coverage": ["name", "description", "ocr"],
        "labels_present": 0,
        "sealed_rows": 0,
        "public_used": False,
        "quality_evaluated": False,
        "full_teacher_authorized": False,
        "student_gpu_authorized": False,
        "jobs_launched": 0,
        "uploads": 0,
        "presets_built": 0,
        "bundles_built": 0,
    }
    for field, expected in exact.items():
        if contract[field] != expected:
            raise ContractError(f"teacher smoke input contract: {field} mismatch")


def validate_full_input_contract(contract: dict[str, Any], request_path: Path) -> None:
    spec = load_spec()
    validate_selection_manifest(contract, spec)
    if contract["teacher_request_sha256"] != sha256_file(request_path):
        raise ContractError("full teacher request SHA differs from selection manifest")
    if contract["teacher_request_rows"] != 300:
        raise ContractError("full teacher run requires exactly 300 requests")


def validate_smoke_acceptance(
    acceptance_path: Path,
    expected_sha256: str,
    *,
    model_binding: dict[str, Any],
) -> dict[str, Any]:
    require_hex64(expected_sha256, "accepted smoke expected SHA")
    if sha256_file(acceptance_path) != expected_sha256:
        raise ContractError("accepted smoke file SHA mismatch")
    acceptance = load_json(acceptance_path, "accepted teacher smoke")
    validate_self_hash(acceptance, "accepted teacher smoke")
    if acceptance.get("schema_version") != "exp689_teacher_run_acceptance_v1":
        raise ContractError("accepted teacher smoke: schema mismatch")
    required = {
        "experiment_id": "689",
        "scope": "technical_smoke",
        "status": "accepted",
        "decision": "TECHNICAL_SMOKE_PASS_NOT_QUALITY",
        "rows": 12,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_tree_sha256": model_binding["model_tree_sha256"],
        "processor_sha256": model_binding["processor_sha256"],
        "code_sha256": sha256_file(Path(__file__)),
        "prompt_sha256": PROMPT_SHA256,
        "quality_evaluated": False,
        "full_teacher_authorized": False,
        "student_gpu_authorized": False,
    }
    for field, expected in required.items():
        if acceptance.get(field) != expected:
            raise ContractError(f"accepted teacher smoke: {field} mismatch")
    checks = acceptance.get("technical_checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        raise ContractError("accepted teacher smoke: technical checks are incomplete")
    return acceptance


def load_requests(path: Path, *, scope: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path, "teacher request")
    expected_rows = 12 if scope == "technical_smoke" else 300
    if len(rows) != expected_rows:
        raise ContractError(f"{scope}: expected exactly {expected_rows} request rows")
    spec = load_spec()
    for index, row in enumerate(rows, 1):
        validate_teacher_request(row, index, spec)
    return rows


def _quadrant_rgb(image: Any, region: str) -> bytes:
    if region == "full":
        return image.tobytes()
    width, height = image.size
    col = 0 if region[2] == "0" else width // 2
    row = 0 if region[1] == "0" else height // 2
    return image.crop((col, row, col + width // 2, row + height // 2)).tobytes()


def load_images(
    image_root: Path,
    image_manifest_path: Path,
    requests: list[dict[str, Any]],
) -> tuple[list[Any], str]:
    manifest = read_jsonl(image_manifest_path, "teacher image manifest")
    if len(manifest) != len(requests):
        raise ContractError("image manifest row count differs from teacher request")
    images: list[Any] = []
    pixel_bindings: list[str] = []
    seen_paths: set[Path] = set()
    for index, (row, request) in enumerate(zip(manifest, requests, strict=True), 1):
        expect_exact_keys(row, IMAGE_MANIFEST_FIELDS, f"image manifest row {index}")
        if row["schema_version"] != "exp689_teacher_image_manifest_row_v1":
            raise ContractError(f"image manifest row {index}: schema mismatch")
        if row["audit_id"] != request["audit_id"]:
            raise ContractError(f"image manifest row {index}: audit binding mismatch")
        first_image = request["first_image"]
        for field in (
            "reference",
            "content_sha256",
            "decoded_rgb_sha256",
            "media_type",
            "width",
            "height",
        ):
            if row[field] != first_image[field]:
                raise ContractError(f"image manifest row {index}: {field} binding mismatch")
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError(f"image manifest row {index}: relative path escapes image root")
        path = (image_root / relative).resolve(strict=True)
        if not path.is_relative_to(image_root.resolve(strict=True)):
            raise ContractError(f"image manifest row {index}: image path escapes image root")
        if path in seen_paths:
            raise ContractError(f"image manifest row {index}: duplicate image path")
        seen_paths.add(path)
        if sha256_file(path) != row["content_sha256"]:
            raise ContractError(f"image manifest row {index}: encoded image SHA mismatch")
        try:
            image = decode_rgb_image(path)
        except Exception as error:
            raise ContractError(f"image manifest row {index}: image decode failed") from error
        pixel_sha = sha256_bytes(image.tobytes())
        if pixel_sha != row["decoded_rgb_sha256"] or pixel_sha != row["pixel_sha256"]:
            image.close()
            raise ContractError(f"image manifest row {index}: decoded pixel SHA mismatch")
        if image.size != (row["width"], row["height"]):
            image.close()
            raise ContractError(f"image manifest row {index}: decoded dimensions mismatch")
        for candidate in request["evidence_candidates"]:
            if candidate["evidence_kind"] == "image_region":
                actual = sha256_bytes(_quadrant_rgb(image, candidate["image_region"]))
                if actual != candidate["evidence_sha256"]:
                    image.close()
                    raise ContractError(
                        f"image manifest row {index}: image-region evidence SHA mismatch"
                    )
        images.append(image)
        pixel_bindings.append(pixel_sha)
    return images, sha256_bytes(canonical_json_bytes(pixel_bindings))


def row_prompt(request: dict[str, Any]) -> str:
    allowed = {
        "schema_version": "exp689_teacher_selection_row_v2",
        "audit_id": request["audit_id"],
        "record_id": request["record_id"],
        "request_row_sha256": request["request_row_sha256"],
        "sold_object": "<closed enum>",
        "substance": "<closed enum>",
        "relation": "<closed enum>",
        "object_evidence_candidate_ids": ["<candidate id>"],
        "substance_evidence_candidate_ids": ["<candidate id>"],
        "relation_evidence_candidate_ids": ["<candidate id>"],
        "support_status": "<supported|unsupported|ambiguous>",
        "supervise": True,
    }
    payload = {
        "binding": {
            "audit_id": request["audit_id"],
            "record_id": request["record_id"],
            "request_row_sha256": request["request_row_sha256"],
        },
        "closed_enums": {
            key: load_spec()["enums"][key]
            for key in ("sold_object", "substance", "relation", "support_status")
        },
        "sources": request["sources"],
        "evidence_candidates": request["evidence_candidates"],
        "exact_output_shape": allowed,
    }
    return SYSTEM_PROMPT + "\nREQUEST=" + canonical_json_bytes(payload).decode("utf-8")


def parse_generated_target(
    raw: str, request: dict[str, Any], spec: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        raise ContractError("teacher output is empty")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ContractError("teacher output is not exactly one JSON object; no repair allowed") from error
    if not isinstance(value, dict):
        raise ContractError("teacher output must be exactly one JSON object")
    validate_teacher_selection(value, request, spec or load_spec(), 1)
    return value


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("torch", "transformers", "Pillow", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _cuda_backend(
    args: argparse.Namespace,
    requests: list[dict[str, Any]],
    images: list[Any],
) -> tuple[list[str], dict[str, Any]]:
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("teacher requires exactly one visible CUDA device")
    device_name = torch.cuda.get_device_name(0)
    if "H100" not in device_name.upper():
        raise RuntimeError("teacher requires exactly one H100 GPU")
    torch.cuda.reset_peak_memory_stats(0)
    processor = AutoProcessor.from_pretrained(args.model_root, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_root,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        attn_implementation="eager",
    )
    if hasattr(model, "peft_config"):
        raise RuntimeError("PEFT/class-LoRA model is forbidden for the exp689 teacher")
    device_map = getattr(model, "hf_device_map", {"": "cuda:0"})

    def normalize_device(value: Any) -> str:
        if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
            return f"cuda:{value}"
        return "cuda:0" if str(value).lower() == "cuda" else str(value).lower()

    normalized = {normalize_device(value) for value in device_map.values()}
    if normalized != {"cuda:0"}:
        raise RuntimeError("teacher model must reside entirely on cuda:0 without CPU/disk offload")
    model.eval()
    raw_outputs: list[str] = []
    pixel_rows = 0
    started = time.monotonic()
    for request, image in zip(requests, images, strict=True):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": row_prompt(request)},
                ],
            }
        ]
        batch = processor.apply_chat_template(
            [messages],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
            enable_thinking=False,
        )
        pixel_tensors = [
            value
            for key, value in batch.items()
            if "pixel" in key and hasattr(value, "numel") and value.numel() > 0
        ]
        if not pixel_tensors:
            raise RuntimeError("processor did not produce image pixel tensors")
        pixel_rows += 1
        device_batch = {
            key: value.to("cuda:0") if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        input_tokens = int(device_batch["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = model.generate(
                **device_batch,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
            )
        new_tokens = generated[:, input_tokens:]
        raw_outputs.append(processor.batch_decode(new_tokens, skip_special_tokens=True)[0])
    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated(0))
    total = int(torch.cuda.get_device_properties(0).total_memory)
    return raw_outputs, {
        "cuda_device_count": 1,
        "cuda_device_name": device_name,
        "cuda_forward_rows": len(raw_outputs),
        "pixel_tensor_rows": pixel_rows,
        "cpu_offload": False,
        "disk_offload": False,
        "base_only": True,
        "class_lora_present": False,
        "model_class": type(model).__name__,
        "runtime_seconds": elapsed,
        "peak_cuda_bytes": peak,
        "total_cuda_bytes": total,
        "packages": _package_versions(),
    }


def _validate_backend_report(report: dict[str, Any], row_count: int) -> None:
    expected_fields = {
        "cuda_device_count",
        "cuda_device_name",
        "cuda_forward_rows",
        "pixel_tensor_rows",
        "cpu_offload",
        "disk_offload",
        "base_only",
        "class_lora_present",
        "model_class",
        "runtime_seconds",
        "peak_cuda_bytes",
        "total_cuda_bytes",
        "packages",
    }
    expect_exact_keys(report, expected_fields, "teacher backend report")
    if report["cuda_device_count"] != 1 or "H100" not in report["cuda_device_name"].upper():
        raise ContractError("teacher backend did not use exactly one H100")
    if report["cuda_forward_rows"] != row_count or report["pixel_tensor_rows"] != row_count:
        raise ContractError("teacher backend did not consume every row and image")
    if report["cpu_offload"] is not False or report["disk_offload"] is not False:
        raise ContractError("teacher backend used forbidden offload")
    if report["base_only"] is not True or report["class_lora_present"] is not False:
        raise ContractError("teacher backend was not exact base-only inference")
    if not isinstance(report["runtime_seconds"], (int, float)) or report["runtime_seconds"] <= 0:
        raise ContractError("teacher backend runtime is invalid")
    if not isinstance(report["peak_cuda_bytes"], int) or report["peak_cuda_bytes"] <= 0:
        raise ContractError("teacher backend peak CUDA memory is invalid")
    if not isinstance(report["total_cuda_bytes"], int) or report["total_cuda_bytes"] <= 0:
        raise ContractError("teacher backend CUDA capacity is invalid")
    if not isinstance(report["packages"], dict) or not report["packages"]:
        raise ContractError("teacher backend package inventory is missing")


def run(
    args: argparse.Namespace,
    *,
    backend: Callable[
        [argparse.Namespace, list[dict[str, Any]], list[Any]],
        tuple[list[str], dict[str, Any]],
    ] = _cuda_backend,
) -> dict[str, Any]:
    if args.scope not in {"technical_smoke", "full"}:
        raise ContractError("teacher scope must be technical_smoke or full")
    if args.scope == "full" and (
        args.accepted_smoke is None or args.accepted_smoke_sha256 is None
    ):
        raise ContractError("full teacher run requires exact accepted smoke path and SHA")
    if args.scope == "technical_smoke" and (
        args.accepted_smoke is not None or args.accepted_smoke_sha256 is not None
    ):
        raise ContractError("technical smoke must not consume a prior smoke gate")
    remote_root = args.remote_root.resolve(strict=True)
    request_path = require_remote_path(
        remote_root, args.teacher_request, context="teacher request", must_exist=True
    )
    input_contract_path = require_remote_path(
        remote_root, args.input_contract, context="teacher input contract", must_exist=True
    )
    image_root = require_remote_path(
        remote_root, args.image_root, context="teacher image root", must_exist=True
    )
    image_manifest_path = require_remote_path(
        remote_root, args.image_manifest, context="teacher image manifest", must_exist=True
    )
    model_root = require_remote_path(
        remote_root, args.model_root, context="teacher model root", must_exist=True
    )
    model_contract_path = require_remote_path(
        remote_root, args.model_contract, context="teacher model contract", must_exist=True
    )
    output_dir = require_remote_path(
        remote_root, args.output_dir, context="teacher output", must_exist=False
    )
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite immutable teacher output")

    input_contract = load_json(input_contract_path, "teacher input contract")
    if args.scope == "technical_smoke":
        validate_smoke_input_contract(input_contract, request_path, image_manifest_path)
    else:
        validate_full_input_contract(input_contract, request_path)
    requests = load_requests(request_path, scope=args.scope)
    if args.scope == "full":
        for request, binding in zip(
            requests, input_contract["selected_bindings"], strict=True
        ):
            for field in ("audit_id", "record_id", "source_card_sha256", "request_row_sha256"):
                if request[field] != binding[field]:
                    raise ContractError(f"full teacher request: {field} binding mismatch")
    model_contract = load_json(model_contract_path, "teacher model contract")
    model_binding = validate_base_model_tree(model_root, model_contract)
    smoke_gate: dict[str, Any] | None = None
    if args.scope == "full":
        smoke_path = require_remote_path(
            remote_root, args.accepted_smoke, context="accepted teacher smoke", must_exist=True
        )
        smoke_gate = validate_smoke_acceptance(
            smoke_path, args.accepted_smoke_sha256, model_binding=model_binding
        )

    images, pixel_set_sha = load_images(image_root, image_manifest_path, requests)
    if args.scope == "technical_smoke" and pixel_set_sha != input_contract["pixel_set_sha256"]:
        raise ContractError("teacher smoke: pixel-set SHA mismatch")
    try:
        raw_outputs, backend_report = backend(args, requests, images)
    finally:
        for image in images:
            image.close()
    if len(raw_outputs) != len(requests):
        raise ContractError("teacher backend output row count mismatch")
    _validate_backend_report(backend_report, len(requests))
    spec = load_spec()
    targets = [
        parse_generated_target(raw, request, spec)
        for raw, request in zip(raw_outputs, requests, strict=True)
    ]

    output_dir.mkdir(parents=True)
    targets_path = output_dir / "targets.jsonl"
    write_jsonl(targets_path, targets)
    selection_payload_sha = sha256_bytes(canonical_json_bytes(targets))
    package_sha = sha256_bytes(canonical_json_bytes(backend_report["packages"]))
    technical_checks = {
        "exact_one_h100": True,
        "base_model_only": True,
        "no_cpu_or_disk_offload": True,
        "cuda_forward_all_rows": True,
        "pixel_tensor_all_rows": True,
        "closed_schema_all_rows": True,
        "request_binding_all_rows": True,
        "evidence_binding_all_rows": True,
        "artifact_write_complete": True,
    }
    report = with_self_hash(
        {
            "schema_version": "exp689_teacher_run_report_v1",
            "experiment_id": "689",
            "scope": args.scope,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_tree_sha256": model_binding["model_tree_sha256"],
            "processor_sha256": model_binding["processor_sha256"],
            "code_sha256": sha256_file(Path(__file__)),
            "prompt_sha256": PROMPT_SHA256,
            "source_sha256": sha256_file(request_path),
            "image_manifest_sha256": sha256_file(image_manifest_path),
            "pixel_set_sha256": pixel_set_sha,
            "input_contract_self_sha256": input_contract["self_sha256"],
            "accepted_smoke_self_sha256": smoke_gate["self_sha256"] if smoke_gate else None,
            "selection_payload_sha256": selection_payload_sha,
            "targets_sha256": sha256_file(targets_path),
            "rows": len(targets),
            "decoding": {
                "do_sample": False,
                "enable_thinking": False,
                "max_new_tokens": MAX_NEW_TOKENS,
                "repair_attempts": 0,
            },
            "runtime_seconds": backend_report["runtime_seconds"],
            "peak_cuda_bytes": backend_report["peak_cuda_bytes"],
            "total_cuda_bytes": backend_report["total_cuda_bytes"],
            "cuda_device_name": backend_report["cuda_device_name"],
            "model_class": backend_report["model_class"],
            "packages": backend_report["packages"],
            "packages_sha256": package_sha,
            "technical_checks": technical_checks,
            "labels_read": 0,
            "sealed_rows": 0,
            "public_used": False,
            "quality_evaluated": False,
            "student_gpu_authorized": False,
            "self_sha256": None,
        }
    )
    report_path = output_dir / "report.json"
    write_json(report_path, report)
    acceptance = with_self_hash(
        {
            "schema_version": "exp689_teacher_run_acceptance_v1",
            "experiment_id": "689",
            "scope": args.scope,
            "status": "accepted",
            "decision": (
                "TECHNICAL_SMOKE_PASS_NOT_QUALITY"
                if args.scope == "technical_smoke"
                else "TARGETS_READY_FOR_DUAL_BLIND_REVIEW"
            ),
            "rows": len(targets),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_tree_sha256": model_binding["model_tree_sha256"],
            "processor_sha256": model_binding["processor_sha256"],
            "code_sha256": sha256_file(Path(__file__)),
            "prompt_sha256": PROMPT_SHA256,
            "source_sha256": sha256_file(request_path),
            "pixel_set_sha256": pixel_set_sha,
            "selection_payload_sha256": selection_payload_sha,
            "targets_sha256": sha256_file(targets_path),
            "report_sha256": sha256_file(report_path),
            "runtime_seconds": backend_report["runtime_seconds"],
            "peak_cuda_bytes": backend_report["peak_cuda_bytes"],
            "technical_checks": technical_checks,
            "labels_read": 0,
            "sealed_rows": 0,
            "public_used": False,
            "quality_evaluated": False,
            "full_teacher_authorized": False,
            "student_gpu_authorized": False,
            "jobs_launched": 0,
            "uploads": 0,
            "presets_built": 0,
            "bundles_built": 0,
            "self_sha256": None,
        }
    )
    write_json(output_dir / "acceptance.json", acceptance)
    return acceptance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("technical_smoke", "full"), required=True)
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--teacher-request", type=Path, required=True)
    parser.add_argument("--input-contract", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--accepted-smoke", type=Path)
    parser.add_argument("--accepted-smoke-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    acceptance = run(parse_args())
    print(json.dumps(acceptance, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
