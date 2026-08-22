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
import json
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
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
PROMPT_VERSION = "qwen35_hard_first_image_binary_v1"
LABEL_FREE_PREDICTION_COLUMNS = ("id", "category", "fold", "lora_score")
SPECIALIST_CATEGORY = "Легковоспламеняющиеся"


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
    by_component_stratum: dict[tuple[str, str], dict[str, str]] = {}
    for row in candidates:
        if row.get("label") not in {"0", "1"}:
            raise ValueError("Fisher rows must carry binary train labels")
        stratum = (row["semantic_component"], row["label"])
        by_component_stratum.setdefault(stratum, row)
    strata: dict[str, list[dict[str, str]]] = {"0": [], "1": []}
    for (_, label), row in sorted(by_component_stratum.items()):
        strata[label].append(row)
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
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise ValueError("runtime image manifest lacks id")
        image_field = "image_url" if "image_url" in reader.fieldnames else reader.fieldnames[-1]
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
    candidates.append(image_root / basename if basename else image_root / item_id)
    for path in candidates:
        if path.is_file():
            with Image.open(path) as image:
                return image.convert("RGB")
    raise FileNotFoundError(f"no local first image for development item {item_id}")


def build_messages(row: Mapping[str, str], image: Any) -> list[dict[str, Any]]:
    text = (
        "Определи, относится ли товар к целевой категории. Ответь одним числом: 1 или 0.\n"
        f"Название: {row['name']}\n"
        f"Описание: {row['description'][:DESCRIPTION_LIMIT]}"
    )
    content: list[dict[str, Any]] = [
        {"type": "image", "image": image},
        {"type": "text", "text": text},
    ]
    return [{"role": "user", "content": content}]


def _import_model_stack():
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Qwen/PEFT runtime dependencies are not installed") from exc
    return torch, PeftModel, AutoProcessor


def _load_model(base_model_id: str, base_model_revision: str, adapter_dir: Path, device: str):
    torch, PeftModel, AutoProcessor = _import_model_stack()
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            revision=base_model_revision,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    except (ImportError, ValueError):
        from transformers import AutoModelForMultimodalLM

        base = AutoModelForMultimodalLM.from_pretrained(
            base_model_id,
            revision=base_model_revision,
            torch_dtype=dtype,
            trust_remote_code=True,
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
    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
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
    batch_size: int,
    torch: Any,
) -> dict[str, Any]:
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
        batch_gradients: dict[str, np.ndarray] = {}
        for name, parameter in model.named_parameters():
            module = _normalise_gradient_name(name)
            if module is None or parameter.grad is None:
                continue
            values = parameter.grad.detach().float().cpu().numpy()
            batch_gradients[module] = (
                values if module not in batch_gradients else batch_gradients[module] + values
            )
        if not batch_gradients:
            raise RuntimeError("no trainable LoRA gradients were observed")
        gradients.append(batch_gradients)
    return estimate_fisher_from_gradients(
        gradients,
        train_rows=len(rows),
        outer_fold=int(rows[0]["development_fold"]),
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
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite fold output")
    audit_path = runtime_dir / "runtime_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
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
    output_dir.mkdir(parents=True, exist_ok=False)
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
            batch_size=batch_size,
            torch=torch,
        )
        fisher_report = dict(original_fisher)
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
        "fisher_max_rows": max_fisher_rows,
        "fisher_batch_size": batch_size,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "original_fisher_rows": len(original_rows),
        "specialist_fisher_rows": len(specialist_rows),
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
