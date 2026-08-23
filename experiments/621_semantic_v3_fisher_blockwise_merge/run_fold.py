"""One-GPU semantic-v3 Fisher collection, merge, and label-free scoring.

The command consumes only the runtime directory produced by
``build_fold_runtime.py``.  It never receives the source competition CSV or a
validation label file.  Transformers, PEFT, and Pillow are imported lazily so
the public preflight and contract tests do not require a model installation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from build_fold_runtime import DEVELOPMENT_FOLDS, canonical_sha256, sha256_file
from fold_worker import (
    ADAPTER_CONFIG,
    ADAPTER_WEIGHTS,
    estimate_fisher_from_gradients,
    merge_fold_adapters,
)

FISHER_MAX_ROWS = 512
FISHER_BATCH_SIZE = 4
DESCRIPTION_LIMIT = 1800
NAME_LIMIT = 320
MAX_LENGTH = 1536
FIRST_IMAGE_MAX_EDGE = 448
IMAGE_DOWNLOAD_WORKERS = 32
PROMPT_VERSION = "qwen35_parent_multimodal_binary_v1"
ORIGINAL_PARENT_SHA256 = "c30e690ad260af72fcc625c8d3e6d9ab9c5a096d8443d6d9f5f7adbcaa52123c"
SPECIALIST_PARENT_SHA256 = "404f6d07965f551dfe7c7ee0120ce0a16132e1103f1db7d13b3622993f4fd6c0"
LABEL_FREE_PREDICTION_COLUMNS = ("id", "category", "fold", "lora_score")
SPECIALIST_CATEGORY = "Легковоспламеняющиеся"
RULES = {
    "БАД": (
        "Метка 1 только если в описании или на упаковке есть прямое указание БАД "
        "или dietary supplement. Спортивное питание без такой маркировки, явное "
        "отрицание или отсутствие маркировки — метка 0."
    ),
    SPECIALIST_CATEGORY: (
        "Метка 1 для самостоятельного источника огня, горючего вещества или газа, "
        "либо если такой товар входит в комплект. Пустое оборудование, встроенный "
        "источник, горючий материал только как компонент или предмет не в комплекте — 0."
    ),
}


def _load_rows(path: Path, *, require_label: bool) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        required = {
            "id",
            "name",
            "description",
            "category",
            "semantic_component",
            "development_fold",
        }
        if require_label:
            required.add("label")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
        if not require_label and "label" in reader.fieldnames:
            raise ValueError("label column is forbidden in label-free runtime input")
        rows = [{str(key): str(value) for key, value in raw.items()} for raw in reader]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("runtime rows must have non-empty unique IDs")
    return rows


def verify_parent_source(path: Path, expected_sha256: str, *, component: str) -> str:
    """Fail closed unless the exact frozen parent runner is mounted."""

    if not path.is_file():
        raise FileNotFoundError(f"{component} parent runner is missing")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"exact {component} parent checksum mismatch")
    return actual


def verify_runtime_audit(runtime_dir: Path, audit: Mapping[str, Any]) -> None:
    """Bind every scoped runtime input to the builder's signed-by-hash audit."""

    claimed_audit_sha256 = audit.get("audit_sha256")
    audit_payload = dict(audit)
    audit_payload.pop("audit_sha256", None)
    if claimed_audit_sha256 != canonical_sha256(audit_payload):
        raise ValueError("runtime audit self-checksum mismatch")
    files = audit.get("files_sha256")
    if not isinstance(files, dict) or not files:
        raise ValueError("runtime audit has no file checksums")
    expected_names = {
        "train_data.csv",
        "validation_data.csv",
        "development_data_label_free.csv",
        "development_membership.csv",
        "development_image_manifest.tsv",
    }
    if set(files) != expected_names:
        raise ValueError("runtime audit file set mismatch")
    for name in sorted(expected_names):
        path = runtime_dir / name
        if not path.is_file() or sha256_file(path) != files[name]:
            raise ValueError(f"runtime input checksum mismatch: {name}")


def stable_row_key(row: Mapping[str, str], *, seed: int = 42) -> str:
    payload = f"{seed}|{row['id']}|{row['semantic_component']}".encode()
    return hashlib.sha256(payload).hexdigest()


def select_fisher_rows(
    rows: Iterable[Mapping[str, str]],
    *,
    task: str,
    max_rows: int = FISHER_MAX_ROWS,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Select one stable row per component and balance fixed label strata."""

    if task not in {"original", "specialist"}:
        raise ValueError("task must be original or specialist")
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    candidates = [dict(row) for row in rows]
    if task == "specialist":
        candidates = [row for row in candidates if row["category"] == SPECIALIST_CATEGORY]
    if not candidates:
        raise ValueError(f"no rows available for Fisher task {task}")
    candidates.sort(key=lambda row: stable_row_key(row, seed=seed))
    # A semantic component is the leakage unit.  Pick its deterministic
    # representative first, then stratify, so a component can never contribute
    # two near-duplicate rows merely because its labels disagree.
    by_component: dict[str, dict[str, str]] = {}
    for row in candidates:
        if row.get("label") not in {"0", "1"}:
            raise ValueError("Fisher rows must carry binary train labels")
        by_component.setdefault(row["semantic_component"], row)
    strata: dict[str, list[dict[str, str]]] = {"0": [], "1": []}
    for component in sorted(by_component):
        row = by_component[component]
        strata[row["label"]].append(row)
    if not strata["0"] or not strata["1"]:
        raise ValueError(f"Fisher task {task} lacks both label strata")
    per_class = min(len(strata["0"]), len(strata["1"]), max_rows // 2)
    if per_class <= 0:
        raise ValueError("Fisher cap leaves no balanced samples")
    selected = strata["0"][:per_class] + strata["1"][:per_class]
    selected.sort(key=lambda row: stable_row_key(row, seed=seed))
    return selected


def _safe_extract_zip(archive: Path, destination: Path) -> Path:
    """Extract one adapter archive while rejecting traversal and symlinks."""

    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for info in source.infolist():
            name = info.filename
            target = (destination / name).resolve()
            if not str(target).startswith(str(root) + "/") and target != root:
                raise ValueError("adapter archive contains a path traversal")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ValueError("adapter archive symlinks are forbidden")
            if info.file_size > 2 * 1024 * 1024 * 1024:
                raise ValueError("adapter archive member is too large")
            source.extract(info, destination)
    configs = list(destination.rglob(ADAPTER_CONFIG))
    weights = list(destination.rglob(ADAPTER_WEIGHTS))
    if len(configs) != 1 or len(weights) != 1:
        raise ValueError("adapter archive must contain exactly one config and one weights file")
    if configs[0].parent != weights[0].parent:
        raise ValueError("adapter config and weights must share a directory")
    return configs[0].parent


def unpack_adapter(path: Path, temporary_root: Path) -> Path:
    if path.is_dir():
        if not (path / ADAPTER_CONFIG).is_file() or not (path / ADAPTER_WEIGHTS).is_file():
            raise ValueError("adapter directory lacks config or weights")
        return path
    if path.suffix.lower() != ".zip":
        raise ValueError("adapter input must be a directory or ZIP archive")
    return _safe_extract_zip(path, temporary_root / path.stem)


def _load_image_manifest(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None or not {"id", "image_url"}.issubset(reader.fieldnames):
            raise ValueError("runtime image manifest lacks id or image_url")
        image_field = "image_url"
        result: dict[str, str] = {}
        for raw in reader:
            item_id = str(raw["id"])
            if item_id in result:
                raise ValueError("runtime image manifest contains duplicate IDs")
            result[item_id] = str(raw[image_field])
    return result


def _resolve_image(image_ref: str, item_id: str, image_root: Path):
    from PIL import Image

    candidates = []
    parsed = urlparse(image_ref)
    basename = Path(parsed.path).name
    if basename:
        candidates.append(image_root / item_id / basename)
    candidates.extend(
        image_root / item_id / name for name in ("0.jpg", "0.jpeg", "0.png", "image.jpg")
    )
    candidates.append(image_root / f"{item_id}.jpg")
    candidates.append(image_root / basename if basename else image_root / item_id)
    for path in candidates:
        if path.is_file():
            with Image.open(path) as image:
                return image.convert("RGB")
    raise FileNotFoundError(f"no local first image for development item {item_id}")


def _download_image(item_id: str, url: str, image_root: Path) -> tuple[str, str | None]:
    """Download and decode one first image without a synthetic fallback."""

    from PIL import Image

    destination = image_root / f"{item_id}.jpg"
    last_error: Exception | None = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            with Image.open(io.BytesIO(payload)) as source:
                image = source.convert("RGB")
                image.thumbnail(
                    (FIRST_IMAGE_MAX_EDGE, FIRST_IMAGE_MAX_EDGE), Image.Resampling.LANCZOS
                )
                image.save(destination, format="JPEG", quality=92)
            return item_id, None
        except (OSError, ValueError, urllib.error.URLError) as error:  # pragma: no cover
            last_error = error
    return item_id, f"{type(last_error).__name__}: {last_error}"


def predownload_images(
    item_ids: Iterable[str], image_manifest: Mapping[str, str], image_root: Path
) -> dict[str, Any]:
    """Strictly materialize every image needed by Fisher or validation scoring."""

    ordered = sorted({str(item_id) for item_id in item_ids})
    missing_refs = [item_id for item_id in ordered if not image_manifest.get(item_id)]
    if missing_refs:
        raise ValueError(f"image manifest misses {len(missing_refs)} required IDs")
    if image_root.exists():
        if not image_root.is_dir():
            raise NotADirectoryError(image_root)
        if any(image_root.iterdir()):
            raise FileExistsError("refusing to reuse a nonempty image directory")
    else:
        image_root.mkdir(parents=True, exist_ok=False)
    failures: list[tuple[str, str]] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=IMAGE_DOWNLOAD_WORKERS) as pool:
        futures = {
            pool.submit(_download_image, item_id, image_manifest[item_id], image_root): item_id
            for item_id in ordered
        }
        for index, future in enumerate(as_completed(futures), 1):
            item_id, error = future.result()
            if error is not None:
                failures.append((item_id, error))
            if index % 500 == 0 or index == len(futures):
                print(
                    f"images={index}/{len(futures)} failures={len(failures)} "
                    f"elapsed_min={(time.monotonic() - started) / 60:.1f}",
                    flush=True,
                )
    if failures:
        raise RuntimeError(f"strict image download failed for {len(failures)} items")
    return {
        "required": len(ordered),
        "downloaded": len(ordered),
        "failures": 0,
        "max_edge": FIRST_IMAGE_MAX_EDGE,
    }


def compact_text(value: Any, limit: int = DESCRIPTION_LIMIT) -> str:
    """Exact text normalization used by both frozen experiment-600 parents."""

    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    return text[:head].rstrip() + " … " + text[-(limit - head) :].lstrip()


def parent_user_text(row: Mapping[str, str]) -> str:
    category = row["category"]
    if category not in RULES:
        raise ValueError(f"unsupported parent category: {category}")
    return (
        f"Категория: {category}\n"
        f"Название: {compact_text(row['name'], NAME_LIMIT)}\n"
        f"Описание: {compact_text(row['description'], DESCRIPTION_LIMIT)}\n"
        f"Правило: {RULES[category]}\n"
        "Определи правильность категории. Ответь только одной цифрой: 1 или 0."
    )


def build_messages(row: Mapping[str, str], image: Any) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "image", "image": image},
        {"type": "text", "text": parent_user_text(row)},
    ]
    return [{"role": "user", "content": content}]


def _import_model_stack():
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Qwen/PEFT runtime dependencies are not installed") from exc
    return torch, PeftModel, AutoModelForMultimodalLM, AutoProcessor


def _load_model(base_model_id: str, base_model_revision: str, adapter_dir: Path, device: str):
    torch, PeftModel, AutoModelForMultimodalLM, AutoProcessor = _import_model_stack()

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    base = AutoModelForMultimodalLM.from_pretrained(
        base_model_id,
        revision=base_model_revision,
        dtype=dtype,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)
    model.to(device)
    processor = AutoProcessor.from_pretrained(
        base_model_id, revision=base_model_revision, trust_remote_code=True
    )
    return torch, model, processor


def _target_ids(processor: Any) -> tuple[int, int]:
    tokenizer = processor.tokenizer
    zero = tokenizer("0", add_special_tokens=False).input_ids
    one = tokenizer("1", add_special_tokens=False).input_ids
    if len(zero) != 1 or len(one) != 1:
        raise ValueError("binary answer tokens must each be one tokenizer token")
    return int(zero[0]), int(one[0])


def _encode(
    processor: Any, messages: list[dict[str, Any]], torch: Any, device: str
) -> dict[str, Any]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
        "padding": True,
        "truncation": True,
        "max_length": MAX_LENGTH,
    }
    try:
        encoded = processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        # Older compatible transformer builds do not expose this keyword; this
        # is the exact fallback in both parent runners.
        encoded = processor.apply_chat_template(messages, **kwargs)
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()
    }


def _binary_loss(
    model: Any, encoded: dict[str, Any], target: int, zero_id: int, one_id: int, torch: Any
):
    output = model(**encoded, use_cache=False)
    logits = output.logits[:, -1, :]
    selected = logits[:, [zero_id, one_id]]
    labels = torch.tensor([target], dtype=torch.long, device=selected.device)
    return torch.nn.functional.cross_entropy(selected, labels), float(
        (logits[0, one_id] - logits[0, zero_id]).detach().cpu()
    )


def _normalise_gradient_name(name: str) -> str | None:
    for suffix in (
        ".lora_A.default.weight",
        ".lora_B.default.weight",
        ".lora_A.weight",
        ".lora_B.weight",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def collect_fisher(
    *,
    model: Any,
    processor: Any,
    rows: list[dict[str, str]],
    image_manifest: Mapping[str, str],
    image_root: Path,
    device: str,
    task: str,
    outer_fold: int,
    batch_size: int,
    torch: Any,
) -> dict[str, Any]:
    if task not in {"original", "specialist"}:
        raise ValueError("task must be original or specialist")
    if outer_fold not in DEVELOPMENT_FOLDS:
        raise ValueError("invalid outer fold")
    zero_id, one_id = _target_ids(processor)
    gradients: list[dict[str, np.ndarray]] = []
    model.train()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        model.zero_grad(set_to_none=True)
        loss_total = None
        for row in batch:
            image = _resolve_image(image_manifest[row["id"]], row["id"], image_root)
            target = int(row["label"])
            encoded = _encode(processor, build_messages(row, image), torch, device)
            loss, _ = _binary_loss(model, encoded, target, zero_id, one_id, torch)
            loss_total = loss if loss_total is None else loss_total + loss
        (loss_total / len(batch)).backward()
        gradient_parts: dict[str, list[np.ndarray]] = {}
        for name, parameter in model.named_parameters():
            module = _normalise_gradient_name(name)
            if module is None or parameter.grad is None:
                continue
            values = parameter.grad.detach().float().cpu().numpy()
            gradient_parts.setdefault(module, []).append(values.reshape(-1))
        # LoRA A and B generally have incompatible matrix shapes.  Concatenate
        # their flattened gradients; adding them can broadcast incorrectly or
        # fail outright and does not estimate a module-level squared gradient.
        batch_gradients = {
            module: np.concatenate(parts) for module, parts in sorted(gradient_parts.items())
        }
        if not batch_gradients:
            raise RuntimeError("no trainable LoRA gradients were observed")
        gradients.append(batch_gradients)
    return estimate_fisher_from_gradients(
        gradients,
        train_rows=len(rows),
        outer_fold=outer_fold,
        train_ids_sha256=canonical_sha256([row["id"] for row in rows]),
    )


def score_validation(
    *,
    model: Any,
    processor: Any,
    rows: list[dict[str, str]],
    image_manifest: Mapping[str, str],
    image_root: Path,
    device: str,
    torch: Any,
) -> list[dict[str, str]]:
    zero_id, one_id = _target_ids(processor)
    model.eval()
    predictions: list[dict[str, str]] = []
    with torch.no_grad():
        for row in rows:
            image = _resolve_image(image_manifest[row["id"]], row["id"], image_root)
            encoded = _encode(processor, build_messages(row, image), torch, device)
            output = model(**encoded, use_cache=False)
            logits = output.logits[:, -1, :]
            score = float((logits[0, one_id] - logits[0, zero_id]).detach().cpu())
            predictions.append(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "fold": row["development_fold"],
                    "lora_score": repr(score),
                }
            )
    return predictions


def _write_predictions(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(LABEL_FREE_PREDICTION_COLUMNS), extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(rows)


def run_fold(
    *,
    runtime_dir: Path,
    original_adapter: Path,
    specialist_adapter: Path,
    original_parent: Path,
    specialist_parent: Path,
    base_model_id: str,
    base_model_revision: str,
    image_root: Path,
    output_dir: Path,
    outer_fold: int,
    device: str = "cuda",
    max_fisher_rows: int = FISHER_MAX_ROWS,
    batch_size: int = FISHER_BATCH_SIZE,
) -> dict[str, Any]:
    if outer_fold not in DEVELOPMENT_FOLDS:
        raise ValueError(f"outer_fold must be one of {DEVELOPMENT_FOLDS}")
    original_parent_sha256 = verify_parent_source(
        original_parent, ORIGINAL_PARENT_SHA256, component="original"
    )
    specialist_parent_sha256 = verify_parent_source(
        specialist_parent, SPECIALIST_PARENT_SHA256, component="specialist"
    )
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(output_dir)
        if any(output_dir.iterdir()):
            raise FileExistsError("refusing to overwrite fold output")
    audit_path = runtime_dir / "runtime_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    verify_runtime_audit(runtime_dir, audit)
    if audit.get("outer_fold") != outer_fold or audit.get("decision") != "GO":
        raise ValueError("runtime audit does not match requested fold")
    if audit.get("validation_labels_written") is not False or audit.get("sealed_rows_written") != 0:
        raise ValueError("runtime audit is not label-isolated")
    train_rows = _load_rows(runtime_dir / "train_data.csv", require_label=True)
    validation_rows = _load_rows(runtime_dir / "validation_data.csv", require_label=False)
    if any(int(row["development_fold"]) == outer_fold for row in train_rows):
        raise ValueError("outer validation row entered Fisher training data")
    if any("label" in row for row in validation_rows):
        raise ValueError("validation row carries a label")
    if any(int(row["development_fold"]) != outer_fold for row in validation_rows):
        raise ValueError("validation runtime contains another fold")
    if max_fisher_rows <= 0 or batch_size <= 0:
        raise ValueError("Fisher cap and batch size must be positive")
    image_manifest = _load_image_manifest(runtime_dir / "development_image_manifest.tsv")
    original_rows = select_fisher_rows(train_rows, task="original", max_rows=max_fisher_rows)
    specialist_rows = select_fisher_rows(train_rows, task="specialist", max_rows=max_fisher_rows)
    image_audit = predownload_images(
        [
            *(row["id"] for row in original_rows),
            *(row["id"] for row in specialist_rows),
            *(row["id"] for row in validation_rows),
        ],
        image_manifest,
        image_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        temp_root = Path(temporary)
        original_dir = unpack_adapter(original_adapter, temp_root / "original_unpack")
        specialist_dir = unpack_adapter(specialist_adapter, temp_root / "specialist_unpack")
        torch, original_model, original_processor = _load_model(
            base_model_id, base_model_revision, original_dir, device
        )
        original_fisher = collect_fisher(
            model=original_model,
            processor=original_processor,
            rows=original_rows,
            image_manifest=image_manifest,
            image_root=image_root,
            device=device,
            task="original",
            outer_fold=outer_fold,
            batch_size=batch_size,
            torch=torch,
        )
        del original_model, original_processor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        torch, specialist_model, specialist_processor = _load_model(
            base_model_id, base_model_revision, specialist_dir, device
        )
        specialist_fisher = collect_fisher(
            model=specialist_model,
            processor=specialist_processor,
            rows=specialist_rows,
            image_manifest=image_manifest,
            image_root=image_root,
            device=device,
            task="specialist",
            outer_fold=outer_fold,
            batch_size=batch_size,
            torch=torch,
        )
        fisher_report = dict(original_fisher)
        fisher_report["original_fisher_sample_rows"] = original_fisher["train_rows"]
        fisher_report["original_fisher_sample_ids_sha256"] = original_fisher["train_ids_sha256"]
        fisher_report["specialist_fisher_sample_rows"] = specialist_fisher["train_rows"]
        fisher_report["specialist_fisher_sample_ids_sha256"] = specialist_fisher["train_ids_sha256"]
        fisher_report["specialist_gradient_batches"] = specialist_fisher["gradient_batches"]
        fisher_report["outer_fold"] = outer_fold
        fisher_report["train_rows"] = len(train_rows)
        fisher_report["train_ids_sha256"] = canonical_sha256([row["id"] for row in train_rows])
        fisher_report["fisher_specialist"] = specialist_fisher["fisher"]
        fisher_path = output_dir / "train_only_fisher.json"
        fisher_path.write_text(
            json.dumps(fisher_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        merged_dir = output_dir / "merged_adapter"
        merge_fold_adapters(
            original_dir=original_dir,
            specialist_dir=specialist_dir,
            data_path=runtime_dir / "development_data_label_free.csv",
            folds_path=runtime_dir / "development_membership.csv",
            fisher_report_path=fisher_path,
            output_dir=merged_dir,
            outer_fold=outer_fold,
            base_model_id=base_model_id,
            base_model_revision=base_model_revision,
        )
        del specialist_model, specialist_processor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        torch, merged_model, merged_processor = _load_model(
            base_model_id, base_model_revision, merged_dir, device
        )
        predictions = score_validation(
            model=merged_model,
            processor=merged_processor,
            rows=validation_rows,
            image_manifest=image_manifest,
            image_root=image_root,
            device=device,
            torch=torch,
        )
    prediction_path = output_dir / "validation_predictions.csv"
    _write_predictions(prediction_path, predictions)
    if "label" in prediction_path.read_text(encoding="utf-8").splitlines()[0].lower():
        raise AssertionError("label leaked into prediction header")
    contract = {
        "protocol": "621_semantic_v3_fold_runtime_v1",
        "outer_fold": outer_fold,
        "prompt_version": PROMPT_VERSION,
        "original_parent_sha256": original_parent_sha256,
        "specialist_parent_sha256": specialist_parent_sha256,
        "fisher_max_rows": max_fisher_rows,
        "fisher_batch_size": batch_size,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "original_fisher_rows": len(original_rows),
        "specialist_fisher_rows": len(specialist_rows),
        "image_audit": image_audit,
        "runtime_audit_sha256": sha256_file(audit_path),
        "fisher_report_sha256": sha256_file(fisher_path),
        "merge_manifest_sha256": sha256_file(output_dir / "merged_adapter" / "merge_manifest.json"),
        "predictions_sha256": sha256_file(prediction_path),
        "validation_labels_loaded": False,
        "sealed_labels_loaded": False,
        "sealed_rows_loaded": 0,
        "decision": "READY_FOR_FROZEN_EXTERNAL_EVALUATION",
    }
    (output_dir / "fold_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one label-isolated 621 fold.")
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--original-adapter", required=True, type=Path)
    parser.add_argument("--specialist-adapter", required=True, type=Path)
    parser.add_argument("--original-parent", required=True, type=Path)
    parser.add_argument("--specialist-parent", required=True, type=Path)
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument("--base-model-revision", required=True)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--outer-fold", required=True, type=int, choices=DEVELOPMENT_FOLDS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-fisher-rows", type=int, default=FISHER_MAX_ROWS)
    parser.add_argument("--batch-size", type=int, default=FISHER_BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            run_fold(
                runtime_dir=args.runtime_dir,
                original_adapter=args.original_adapter,
                specialist_adapter=args.specialist_adapter,
                original_parent=args.original_parent,
                specialist_parent=args.specialist_parent,
                base_model_id=args.base_model_id,
                base_model_revision=args.base_model_revision,
                image_root=args.image_root,
                output_dir=args.output_dir,
                outer_fold=args.outer_fold,
                device=args.device,
                max_fisher_rows=args.max_fisher_rows,
                batch_size=args.batch_size,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
