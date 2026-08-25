from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
SHARED = Path(__file__).resolve().parents[1] / "645_qwen_scale_2x3_gate"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import train_lora as control
from build_pair_runtime import canonical_sha256, occurrence_key, read_jsonl
from verify_pair_runtime import verify as verify_pair_runtime

EXPERIMENT_ID = "685"
SOURCE_EXPERIMENT_ID = "641"
FLAMMABLE = "Легковоспламеняющиеся"
SEED = 42
EPOCHS = 1
LEARNING_RATE = 2e-4
MICRO_BATCH_PAIRS = 1
MICRO_BATCH_ROWS = 2
GRADIENT_ACCUMULATION_PAIRS = 8
EFFECTIVE_BATCH_ROWS = 16
RANK_LOSS_WEIGHT = 0.5
MODES = ("paired_hard_control", "rank_candidate")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_tree_sha256(root: Path) -> tuple[str, int]:
    resolved_root = root.resolve()
    files: list[tuple[str, Path]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if any(
            part == "__MACOSX" or part == ".DS_Store" or part.startswith("._")
            for part in Path(relative).parts
        ):
            raise ValueError("model registry input contains transport metadata")
        resolved = path.resolve()
        if not resolved.is_relative_to(resolved_root):
            raise ValueError("model registry input escapes its delivery root")
        if path.is_file():
            files.append((relative, path))
        elif not path.is_dir():
            raise ValueError("model registry input contains a special filesystem entry")
    if not files:
        raise ValueError("model registry input is empty")
    digest = hashlib.sha256()
    for relative, path in files:
        payload_sha = sha256_file(path)
        digest.update(
            json.dumps(
                [relative, path.stat().st_size, payload_sha],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest(), len(files)


def trainable_state_sha256(model: Any) -> str:
    import torch

    digest = hashlib.sha256()
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named:
        raise ValueError("model has no trainable parameters")
    for name, parameter in named:
        tensor = parameter.detach().contiguous().view(torch.uint8).cpu()
        digest.update(
            json.dumps(
                [name, list(parameter.shape), str(parameter.dtype)],
                separators=(",", ":"),
            ).encode()
        )
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def load_code_acceptance(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    body = dict(value)
    digest = body.pop("acceptance_sha256", None)
    if digest != canonical_sha256(body):
        raise ValueError("code-bundle acceptance self-hash mismatch")
    if (
        value.get("experiment_id") != EXPERIMENT_ID
        or value.get("decision") != "ACCEPT_CODE_BUNDLE"
        or not isinstance(value.get("files"), int)
        or int(value["files"]) <= 0
        or not isinstance(value.get("directories"), int)
        or int(value["directories"]) <= 0
    ):
        raise ValueError("code-bundle acceptance is invalid")
    for field in ("bundle_sha256", "manifest_sha256"):
        candidate = value.get(field)
        if not isinstance(candidate, str) or len(candidate) != 64 or any(
            character not in "0123456789abcdef" for character in candidate
        ):
            raise ValueError(f"code-bundle {field} is invalid")
    revision = value.get("git_revision")
    if not isinstance(revision, str) or len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("code-bundle revision is invalid")
    return value


def load_vendor_acceptance(path: Path, archive: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    body = dict(value)
    digest = body.pop("bridge_sha256", None)
    if digest != canonical_sha256(body):
        raise ValueError("PEFT vendor acceptance self-hash mismatch")
    if (
        value.get("experiment_id") != EXPERIMENT_ID
        or value.get("stage") != "PEFT_VENDOR_REMOTE_BRIDGE"
        or value.get("metadata_version") != "0.20.0"
        or value.get("decision") != "ACCEPT_PEFT_VENDOR_BRIDGE"
        or int(value.get("labels_read", -1)) != 0
        or int(value.get("sealed_rows_used", -1)) != 0
        or value.get("public_used") is not False
        or value.get("vendor_zip_sha256") != sha256_file(archive)
        or int(value.get("vendor_zip_size", -1)) != archive.stat().st_size
    ):
        raise ValueError("PEFT vendor acceptance or archive binding is invalid")
    return value


def load_transport_acceptance(
    path: Path,
    *,
    fold: int,
    pair_acceptance: dict[str, Any],
    source_audit: dict[str, Any],
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    body = dict(value)
    digest = body.pop("transport_acceptance_sha256", None)
    if digest != canonical_sha256(body):
        raise ValueError("training-input transport self-hash mismatch")
    expected = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "TRAINING_INPUT_EXACT_WHITELIST",
        "outer_fold": fold,
        "pair_runtime_contract_sha256": pair_acceptance["runtime_contract_sha256"],
        "pair_runtime_acceptance_sha256": pair_acceptance["acceptance_sha256"],
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "decision": "ACCEPT_TRAINING_INPUT",
    }
    mismatch = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatch:
        raise ValueError(f"training-input transport mismatch: {mismatch}")
    source_files = value.get("accepted_files", {})
    expected_source = {
        "source_runtime/runtime_audit.json": sha256_file(path.parent / "source_runtime/runtime_audit.json"),
        "source_runtime/train.jsonl": sha256_file(path.parent / "source_runtime/train.jsonl"),
        "source_runtime/validation.jsonl": sha256_file(path.parent / "source_runtime/validation.jsonl"),
    }
    if any(
        source_files.get(relative, {}).get("sha256") != payload_sha
        for relative, payload_sha in expected_source.items()
    ):
        raise ValueError("staged source-runtime bytes differ from transport acceptance")
    return value


def pair_loss(
    model: Any,
    processor: Any,
    positive: SimpleNamespace,
    negative: SimpleNamespace,
    images: list[Any],
    zero_token: int,
    one_token: int,
    pair_target: float,
    mode: str,
):
    import torch
    from torch.nn import functional

    if mode not in MODES:
        raise ValueError("unknown pair-training mode")
    rows = [positive, negative]
    conversations = [
        control.messages(row, image, prompt_text=control.base_prompt(row))
        for row, image in zip(rows, images, strict=True)
    ]
    batch = control._processor_batch(
        processor, conversations, add_generation_prompt=True
    ).to(model.device)
    scores = control._last_logits(model, batch, zero_token, one_token).float()
    hard_targets = torch.tensor([1.0, 0.0], dtype=torch.float32, device=model.device)
    hard = functional.binary_cross_entropy_with_logits(scores, hard_targets)
    target = torch.tensor(float(pair_target), dtype=torch.float32, device=model.device)
    rank = functional.binary_cross_entropy_with_logits(scores[0] - scores[1], target)
    if mode == "paired_hard_control":
        total = hard
    else:
        total = (1.0 - RANK_LOSS_WEIGHT) * hard + RANK_LOSS_WEIGHT * rank
    return total, hard.detach(), rank.detach()


def load_inputs(
    source_runtime: Path, pair_runtime: Path, fold: int
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    pair_acceptance = verify_pair_runtime(pair_runtime)
    if (
        int(pair_acceptance["outer_fold"]) != fold
        or pair_acceptance["decision"] != "ACCEPT_R0_OPEN_TECHNICAL_SMOKE"
    ):
        raise ValueError("pair runtime does not open this fold")
    source_train, source_validation, source_audit = control.load_runtime(
        source_runtime, SOURCE_EXPERIMENT_ID, fold
    )
    train = [row for row in source_train if row["category"] == FLAMMABLE]
    validation = [
        row for row in source_validation if row["category"] == FLAMMABLE
    ]
    targets = read_jsonl(pair_runtime / "train_targets.jsonl")
    pairs = read_jsonl(pair_runtime / "pairs.jsonl")
    pair_audit = json.loads((pair_runtime / "runtime_audit.json").read_text())
    if (
        pair_audit["source_641_runtime_contract_sha256"]
        != source_audit["contract_sha256"]
    ):
        raise ValueError("source runtime contract differs from R0 binding")
    filtered_sha256 = {
        "train.jsonl": hashlib.sha256(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in train
            ).encode()
        ).hexdigest(),
        "validation.jsonl": hashlib.sha256(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in validation
            ).encode()
        ).hexdigest(),
    }
    if pair_audit["derived_680_output_sha256"] != filtered_sha256:
        raise ValueError("derived flammable selector differs from R0 binding")
    if [occurrence_key(row) for row in train] != [occurrence_key(row) for row in targets]:
        raise ValueError("sanitized targets do not bind to model-input rows")
    for row, target in zip(train, targets, strict=True):
        if int(row["label"]) != int(target["label"]):
            raise ValueError("hard-label binding mismatch")
        if str(row["semantic_component"]) != str(target["semantic_component"]):
            raise ValueError("semantic-component binding mismatch")
    return train, validation, pairs, pair_acceptance, source_audit


def run(args: Any) -> dict[str, Any]:
    import numpy as np
    import torch

    if args.mode not in MODES:
        raise ValueError("mode must be a frozen paired control or rank candidate")
    if args.runtime_backend != "legacy_eager" or args.micro_batch_size_override != 2:
        raise ValueError("experiment 685 requires the proven 641 legacy-eager micro2 path")
    spec = control.CELL_SPECS[SOURCE_EXPERIMENT_ID]
    if args.model_revision != spec.model_revision:
        raise ValueError("model revision differs from the frozen 641 contract")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, pairs, pair_acceptance, source_audit = load_inputs(
        args.runtime_dir, args.pair_runtime, args.fold
    )
    transport_acceptance = load_transport_acceptance(
        args.transport_acceptance,
        fold=args.fold,
        pair_acceptance=pair_acceptance,
        source_audit=source_audit,
    )
    code_acceptance = load_code_acceptance(args.code_acceptance)
    vendor_acceptance = load_vendor_acceptance(
        args.vendor_acceptance, args.vendor_archive
    )
    pair_lookup = {occurrence_key(row): row for row in train}
    if len(pair_lookup) != len(train):
        raise ValueError("duplicate occurrence key in source train runtime")
    if args.technical_smoke:
        pairs = pairs[:8]
        validation = validation[:2]
    if len(pairs) % GRADIENT_ACCUMULATION_PAIRS != 0:
        raise ValueError("pair count must divide the frozen accumulation exactly")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    if not args.vendor.is_dir():
        raise FileNotFoundError("vendored PEFT 0.20.0 directory is missing")
    sys.path.insert(0, str(args.vendor.resolve()))
    import peft

    if peft.__version__ != "0.20.0":
        raise ValueError("exact vendored PEFT 0.20.0 is required")
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    model_tree_digest, model_tree_files = model_tree_sha256(args.model_root)
    print(
        json.dumps(
            {
                "phase": "model_load_start",
                "mode": args.mode,
                "model_tree_sha256": model_tree_digest,
                "model_tree_files": model_tree_files,
            }
        ),
        flush=True,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_root.resolve(), local_files_only=True, trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1 or zero == one:
        raise ValueError("0/1 must be distinct atomic tokens")
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_root.resolve(),
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_rslora=True,
        ),
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    initial_trainable_state_sha256 = trainable_state_sha256(model)
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)
    updates = len(pairs) // GRADIENT_ACCUMULATION_PAIRS
    warmup = max(1, int(updates * 0.05))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, updates - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    optimizer.zero_grad(set_to_none=True)
    pair_indices = list(range(len(pairs)))
    random.Random(SEED).shuffle(pair_indices)
    ordered_pair_index_sha256 = hashlib.sha256(
        json.dumps(pair_indices, separators=(",", ":")).encode()
    ).hexdigest()
    started = time.monotonic()
    optimizer_steps = 0
    last_reported = started
    print(
        json.dumps(
            {
                "phase": "training_start",
                "mode": args.mode,
                "pairs": len(pairs),
                "optimizer_steps": updates,
            }
        ),
        flush=True,
    )
    for position, pair_index in enumerate(pair_indices, start=1):
        pair = pairs[pair_index]
        positive = pair_lookup[tuple(pair["positive_key"])]
        negative = pair_lookup[tuple(pair["negative_key"])]
        rows = [SimpleNamespace(**positive), SimpleNamespace(**negative)]
        images = [
            control.open_image(args.images, positive),
            control.open_image(args.images, negative),
        ]
        try:
            loss, hard, rank = pair_loss(
                model,
                processor,
                rows[0],
                rows[1],
                images,
                zero[0],
                one[0],
                float(pair["pair_target"]),
                args.mode,
            )
            if (position <= 8 or position % 100 == 0) and not all(
                torch.isfinite(value).item() for value in (loss, hard, rank)
            ):
                raise FloatingPointError("non-finite pair loss")
            (loss / GRADIENT_ACCUMULATION_PAIRS).backward()
        finally:
            for image in images:
                image.close()
        if position % GRADIENT_ACCUMULATION_PAIRS == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        now = time.monotonic()
        if position == len(pairs) or position % 100 == 0 or now - last_reported >= 300:
            elapsed = now - started
            rate = position / max(elapsed, 1e-9)
            remaining = (len(pairs) - position) / max(rate, 1e-9)
            print(
                json.dumps(
                    {
                        "phase": "training_progress",
                        "mode": args.mode,
                        "pairs_done": position,
                        "pairs_total": len(pairs),
                        "optimizer_steps_done": optimizer_steps,
                        "pairs_per_second": rate,
                        "eta_seconds_p50": remaining,
                        "eta_seconds_p90": remaining * 1.5,
                    }
                ),
                flush=True,
            )
            last_reported = now
    if optimizer_steps != updates:
        raise RuntimeError("optimizer update count drifted")

    model.eval()
    model.config.use_cache = True
    print(json.dumps({"phase": "validation_start", "rows": len(validation)}), flush=True)
    predictions = control.predict_class_only(
        model, processor, validation, args.images, spec, zero[0], one[0]
    )
    predictions_path = args.output_dir / "predictions.jsonl"
    control.write_jsonl(predictions_path, predictions)
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(adapter_dir)

    reload_max_abs_score_delta: float | None = None
    reload_prediction_mismatches: int | None = None
    if args.technical_smoke:
        base_model = model.unload()
        reloaded = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
        reloaded.eval()
        reloaded.config.use_cache = True
        reloaded_predictions = control.predict_class_only(
            reloaded, processor, validation, args.images, spec, zero[0], one[0]
        )
        reload_max_abs_score_delta = max(
            abs(float(before["score"]) - float(after["score"]))
            for before, after in zip(predictions, reloaded_predictions, strict=True)
        )
        reload_prediction_mismatches = sum(
            int(before["prediction"]) != int(after["prediction"])
            for before, after in zip(predictions, reloaded_predictions, strict=True)
        )
        if reload_max_abs_score_delta > 1e-5 or reload_prediction_mismatches:
            raise RuntimeError("adapter save/reload prediction parity failed")

    artifacts = {
        "predictions.jsonl": control.sha256_file(predictions_path),
        "adapter/README.md": control.sha256_file(adapter_dir / "README.md"),
        "adapter/adapter_config.json": control.sha256_file(
            adapter_dir / "adapter_config.json"
        ),
        "adapter/adapter_model.safetensors": control.sha256_file(
            adapter_dir / "adapter_model.safetensors"
        ),
    }
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "outer_fold": args.fold,
        "mode": args.mode,
        "changed_factor": (
            "add_fixed_rank_loss_weight_0.5"
            if args.mode == "rank_candidate"
            else "paired_hard_bce_shadow_control"
        ),
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "objective": "class_only_pairwise_rank_distillation",
        "pair_runtime_contract_sha256": pair_acceptance[
            "runtime_contract_sha256"
        ],
        "pair_runtime_acceptance_sha256": pair_acceptance["acceptance_sha256"],
        "transport_acceptance_sha256": transport_acceptance[
            "transport_acceptance_sha256"
        ],
        "source_641_runtime_contract_sha256": source_audit["contract_sha256"],
        "derived_680_runtime_contract_sha256": json.loads(
            (args.pair_runtime / "runtime_audit.json").read_text()
        )["derived_680_runtime_contract_sha256"],
        "code_bundle_sha256": code_acceptance["bundle_sha256"],
        "code_revision": code_acceptance["git_revision"],
        "code_manifest_sha256": code_acceptance["manifest_sha256"],
        "code_acceptance_sha256": code_acceptance["acceptance_sha256"],
        "vendor_zip_sha256": vendor_acceptance["vendor_zip_sha256"],
        "vendor_bridge_sha256": vendor_acceptance["bridge_sha256"],
        "vendor_source_bundle_sha256": vendor_acceptance[
            "source_bundle_sha256"
        ],
        "ordered_pair_index_sha256": ordered_pair_index_sha256,
        "initial_trainable_state_sha256": initial_trainable_state_sha256,
        "model_tree_sha256": model_tree_digest,
        "model_tree_files": model_tree_files,
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "micro_batch_pairs": MICRO_BATCH_PAIRS,
        "micro_batch_rows": MICRO_BATCH_ROWS,
        "gradient_accumulation_pairs": GRADIENT_ACCUMULATION_PAIRS,
        "effective_batch_rows": EFFECTIVE_BATCH_ROWS,
        "rank_loss_weight": (
            RANK_LOSS_WEIGHT if args.mode == "rank_candidate" else 0.0
        ),
        "pairs": len(pairs),
        "optimizer_steps_executed": optimizer_steps,
        "validation_rows": len(validation),
        "technical_smoke": bool(args.technical_smoke),
        "runtime_backend": "legacy_eager",
        "runtime_packages": {
            "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch"),
            "peft": peft.__version__,
        },
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "runtime_minutes": (time.monotonic() - started) / 60,
        "reload_max_abs_score_delta": reload_max_abs_score_delta,
        "reload_prediction_mismatches": reload_prediction_mismatches,
        "artifacts": artifacts,
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "threshold": 0.0,
        "threshold_tuned": False,
        "decision": "TECHNICAL_SMOKE_ONLY" if args.technical_smoke else "GO_EVALUATE",
    }
    report["contract_sha256"] = canonical_sha256(report)
    (args.output_dir / "output_contract.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"phase": "complete", "contract": report}), flush=True)
    return report


def parser():
    result = control.parser_for(SOURCE_EXPERIMENT_ID)
    result.add_argument("--pair-runtime", type=Path, required=True)
    result.add_argument("--transport-acceptance", type=Path, required=True)
    result.add_argument("--code-acceptance", type=Path, required=True)
    result.add_argument("--vendor-acceptance", type=Path, required=True)
    result.add_argument("--vendor-archive", type=Path, required=True)
    result.add_argument("--mode", choices=MODES, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
