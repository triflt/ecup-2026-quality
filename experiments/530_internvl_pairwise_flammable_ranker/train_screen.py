from __future__ import annotations

"""Locked native InternVL pairwise training for folds 0 and 3 only."""

import argparse
import hashlib
import importlib.util
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    # The pinned dependency bootstrap uses runpy and therefore does not add
    # this script directory to sys.path as `python file.py` would.
    sys.path.insert(0, str(EXPERIMENT_DIR))

from contract import (
    DESCRIPTION_LIMIT,
    GRADIENT_ACCUMULATION,
    IMAGE_SIZE,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    NAME_LIMIT,
    OPTIMIZER_UPDATES,
    PAIR_BATCH_SIZE,
    PINNED_SELECTOR_MANIFEST_SHA256,
    SCREEN_FOLDS,
    SEED,
    WEIGHT_DECAY,
)
from image_source import ImagePreparationError, ManifestImageStore, prepare_manifest_images
from pairwise_objective import cyclic_index_batches, pairwise_logistic_loss
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = EXPERIMENT_DIR / "analysis"
PAIR_AUDIT = ANALYSIS / "pair_manifest_audit.json"
MEMBERSHIP = ANALYSIS / "selector_membership.csv"
VERIFIED_NATIVE_LOADER = ROOT / "research/internvl_native_backward_smoke.py"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_native_module():
    spec = importlib.util.spec_from_file_location(
        "_verified_internvl_native_loader", VERIFIED_NATIVE_LOADER
    )
    if spec is None or spec.loader is None:
        raise ImportError("verified native InternVL loader is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def compact(value: object, limit: int) -> str:
    raw = "" if pd.isna(value) else str(value)
    normalized = re.sub(r"\s+", " ", raw).strip()
    if len(normalized) <= limit:
        return normalized
    head = int(limit * 0.7)
    return normalized[:head].rstrip() + " … " + normalized[-(limit - head) :].lstrip()


def row_field(row: object, name: str) -> object:
    if isinstance(row, pd.Series):
        return row[name]
    return getattr(row, name)


def strict_boolean(values: pd.Series, *, name: str) -> pd.Series:
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized) - {"true", "false"})
    if unexpected:
        raise ValueError(f"invalid boolean values in {name}: {unexpected}")
    return normalized == "true"


def question(row: object) -> str:
    return (
        "<image>\n"
        "Товар проверяется для категории «Легковоспламеняющиеся».\n"
        f"Название: {compact(row_field(row, 'name'), NAME_LIMIT)}\n"
        f"Описание: {compact(row_field(row, 'description'), DESCRIPTION_LIMIT)}\n"
        "Определи правильность категории по фактическому товару и его комплекту. "
        "Ответь только одной цифрой: 1 или 0."
    )


def image_tensors(images: ManifestImageStore, ids: list[str]) -> torch.Tensor:
    values = [TRANSFORM(images.rgb(item_id)) for item_id in ids]
    return torch.stack(values).to(device="cuda", dtype=torch.bfloat16)


def validate_inputs(
    *, fold: int, selector_manifest: Path, pair_manifest: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if fold not in SCREEN_FOLDS:
        raise ValueError(f"fold must be one of {SCREEN_FOLDS}")
    if sha256(selector_manifest) != PINNED_SELECTOR_MANIFEST_SHA256:
        raise ValueError("frozen selector manifest checksum mismatch")
    expected_pair_path = ANALYSIS / f"pair_manifest_fold_{fold}.csv"
    if pair_manifest.resolve() != expected_pair_path.resolve():
        raise ValueError("pair manifest must be the checked-in fold manifest")
    audit = json.loads(PAIR_AUDIT.read_text(encoding="utf-8"))
    expected_pair_sha = audit["output_sha256"][f"pair_manifest_fold_{fold}"]
    if sha256(pair_manifest) != expected_pair_sha:
        raise ValueError("pair manifest checksum mismatch")
    if sha256(MEMBERSHIP) != audit["output_sha256"]["selector_membership"]:
        raise ValueError("selector membership checksum mismatch")
    pairs = pd.read_csv(pair_manifest, dtype={"positive_id": str, "negative_id": str})
    membership = pd.read_csv(MEMBERSHIP, dtype={"id": str})
    selector = pd.read_csv(selector_manifest, compression="gzip", dtype={"id": str})
    if selector.id.duplicated().any() or membership.id.duplicated().any():
        raise ValueError("selector identities must be unique")
    if set(selector.id) != set(membership.id):
        raise ValueError("selector source/membership id mismatch")
    if not (set(pairs.positive_id) | set(pairs.negative_id)) <= set(selector.id):
        raise ValueError("pair manifest contains an unknown selector id")
    if len(pairs) % PAIR_BATCH_SIZE:
        raise ValueError("pair manifest must form complete fixed-size batches")
    return pairs, membership, selector, audit


def atomic_digit_tokens(tokenizer) -> tuple[int, int]:
    zero = tokenizer.encode("0", add_special_tokens=False)
    one = tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1 or zero[0] == one[0]:
        raise ValueError(f"digit tokens are not distinct atomic tokens: zero={zero} one={one}")
    return int(zero[0]), int(one[0])


def model_scores(
    *,
    model,
    tokenizer,
    native,
    rows: list[object],
    images: ManifestImageStore,
    token_zero: int,
    token_one: int,
) -> torch.Tensor:
    underlying = model.base_model.model
    prompts = [native.query(underlying, tokenizer, question(row), None) for row in rows]
    batch = tokenizer(prompts, padding=True, return_tensors="pt")
    batch = {key: value.to("cuda") for key, value in batch.items()}
    output = model(
        pixel_values=image_tensors(images, [str(row_field(row, "id")) for row in rows]),
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        image_flags=torch.ones((len(rows), 1), dtype=torch.long, device="cuda"),
        return_dict=True,
    )
    positions = torch.arange(batch["attention_mask"].shape[1], device="cuda")[None, :]
    last = torch.where(batch["attention_mask"].bool(), positions, -1).max(dim=1).values
    logits = output.logits[torch.arange(len(rows), device="cuda"), last]
    return logits[:, token_one].float() - logits[:, token_zero].float()


@torch.inference_mode()
def score_frame(
    *,
    model,
    tokenizer,
    native,
    frame: pd.DataFrame,
    images: ManifestImageStore,
    token_zero: int,
    token_one: int,
    batch_size: int = 8,
) -> np.ndarray:
    model.eval()
    scores: list[float] = []
    for start in range(0, len(frame), batch_size):
        rows = list(frame.iloc[start : start + batch_size].itertuples(index=False))
        values = model_scores(
            model=model,
            tokenizer=tokenizer,
            native=native,
            rows=rows,
            images=images,
            token_zero=token_zero,
            token_one=token_one,
        )
        scores.extend(values.cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=SCREEN_FOLDS)
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--image-source", choices=("manifest",), default="manifest")
    parser.add_argument("--image-cache", type=Path)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pair_manifest = ANALYSIS / f"pair_manifest_fold_{args.fold}.csv"
    pairs, membership, selector, pair_audit = validate_inputs(
        fold=args.fold,
        selector_manifest=args.selector_manifest,
        pair_manifest=pair_manifest,
    )
    targets = (
        args.output_dir / "adapter",
        args.output_dir / "adapter.zip",
        args.output_dir / "internvl_scores.csv",
        args.output_dir / "training_report.json",
    )
    image_report_path = args.output_dir / "image_download_report.json"
    existing = [path for path in (*targets, image_report_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite training outputs: " + ", ".join(map(str, existing))
        )
    pair_ids = sorted(set(pairs.positive_id) | set(pairs.negative_id))
    donor_ids = sorted(
        membership.loc[
            (membership.fold.astype(int) != args.fold)
            & strict_boolean(membership.safe_for_selection, name="safe_for_selection"),
            "id",
        ].astype(str)
    )
    outer_ids = sorted(membership.loc[membership.fold.astype(int) == args.fold, "id"].astype(str))
    score_ids = donor_ids + outer_ids
    expected_image_ids = set(pair_ids) | set(score_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_cache = args.image_cache or (args.output_dir / "image_cache")
    try:
        images, image_report = prepare_manifest_images(
            manifest=selector,
            expected_ids=expected_image_ids,
            cache_dir=image_cache,
        )
    except ImagePreparationError as error:
        image_report_path.write_text(
            json.dumps(error.report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    image_report_path.write_text(
        json.dumps(image_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Keep the identity as both index and column: pair lookup uses the index,
    # while image loading reads the explicit field from each selected row.
    selector = selector.set_index("id", drop=False)
    native = load_verified_native_module()
    native.install_peft()
    from peft import LoraConfig, get_peft_model

    tokenizer = native.AutoTokenizer.from_pretrained(
        args.model_root,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    token_zero, token_one = atomic_digit_tokens(tokenizer)
    model = native.AutoModel.from_pretrained(
        args.model_root,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
        use_flash_attn=False,
    ).to("cuda")
    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj")
    target_modules = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.startswith("language_model.")
        and name.endswith(suffixes)
    ]
    if not target_modules:
        raise ValueError("verified native model exposes no language attention targets")
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=target_modules,
            bias="none",
            use_rslora=True,
        ),
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.base_model.model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable or any(
        "language_model." not in name or "lora_" not in name for name, _ in trainable
    ):
        raise ValueError("trainable parameters are not restricted to language LoRA")
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    warmup = max(1, int(OPTIMIZER_UPDATES * 0.05))

    def schedule(update: int) -> float:
        if update < warmup:
            return (update + 1) / warmup
        progress = (update - warmup) / max(1, OPTIMIZER_UPDATES - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    row_by_id = {item_id: selector.loc[item_id] for item_id in selector.index}
    microbatches = OPTIMIZER_UPDATES * GRADIENT_ACCUMULATION
    batches = cyclic_index_batches(
        length=len(pairs),
        batch_size=PAIR_BATCH_SIZE,
        batches=microbatches,
        seed=SEED + args.fold,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    running_loss = 0.0
    for microbatch_index, positions in enumerate(batches, 1):
        local = pairs.iloc[positions]
        positive_rows = [row_by_id[item_id] for item_id in local.positive_id]
        negative_rows = [row_by_id[item_id] for item_id in local.negative_id]
        rows = [*positive_rows, *negative_rows]
        values = model_scores(
            model=model,
            tokenizer=tokenizer,
            native=native,
            rows=rows,
            images=images,
            token_zero=token_zero,
            token_one=token_one,
        )
        loss = pairwise_logistic_loss(values[:PAIR_BATCH_SIZE], values[PAIR_BATCH_SIZE:])
        (loss / GRADIENT_ACCUMULATION).backward()
        running_loss += float(loss.detach().cpu())
        if microbatch_index % GRADIENT_ACCUMULATION == 0:
            torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update = microbatch_index // GRADIENT_ACCUMULATION
            if update % 24 == 0 or update == OPTIMIZER_UPDATES:
                print(
                    json.dumps(
                        {
                            "fold": args.fold,
                            "update": update,
                            "updates": OPTIMIZER_UPDATES,
                            "loss": running_loss / microbatch_index,
                            "elapsed_minutes": (time.monotonic() - started) / 60,
                        }
                    ),
                    flush=True,
                )
    if microbatch_index // GRADIENT_ACCUMULATION != OPTIMIZER_UPDATES:
        raise RuntimeError("locked optimizer update count was not reached")

    scoring = selector.loc[score_ids].reset_index()
    raw_scores = score_frame(
        model=model,
        tokenizer=tokenizer,
        native=native,
        frame=scoring,
        images=images,
        token_zero=token_zero,
        token_one=token_one,
    )
    roles = np.where(scoring.id.isin(outer_ids), "outer", "donor")
    prediction_frame = pd.DataFrame(
        {
            "id": scoring.id.astype(str),
            "fold": scoring.fold.astype(int),
            "row_role": roles,
            "internvl_score": raw_scores,
        }
    )
    model.save_pretrained(targets[0])
    shutil.make_archive(str(targets[0]), "zip", targets[0])
    prediction_frame.to_csv(targets[2], index=False)
    report = {
        "experiment_id": "530",
        "status": "screen_fold_complete",
        "holdout_fold": args.fold,
        "selector_version": pair_audit["selector_version"],
        "selector_manifest_sha256": sha256(args.selector_manifest),
        "selector_membership_sha256": sha256(MEMBERSHIP),
        "pair_manifest_sha256": sha256(pair_manifest),
        "pair_manifest_ordered_sha256": pair_audit["folds"][str(args.fold)]["ordered_pair_sha256"],
        "pairs": len(pairs),
        "optimizer_updates": OPTIMIZER_UPDATES,
        "pair_batch_size": PAIR_BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "image_tiles": 1,
        "image_source": args.image_source,
        "image_download_report": image_report,
        "image_download_report_sha256": sha256(image_report_path),
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "trainable_parameter_count": int(sum(parameter.numel() for _, parameter in trainable)),
        "language_attention_only": True,
        "vision_frozen": True,
        "atomic_digit_tokens": True,
        "token_zero": token_zero,
        "token_one": token_one,
        "loss": "softplus(-(score_positive-score_negative))",
        "donor_score_rows": len(donor_ids),
        "outer_score_rows": len(outer_ids),
        "runtime_minutes": (time.monotonic() - started) / 60,
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "predictions_sha256": sha256(targets[2]),
        "adapter_zip_sha256": sha256(targets[1]),
    }
    targets[3].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    images.close()
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
