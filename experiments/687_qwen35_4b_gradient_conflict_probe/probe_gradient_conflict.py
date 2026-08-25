from __future__ import annotations

import argparse
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
PARENT = HERE.parent / "686_qwen35_4b_additive_rank_kd"
SHARED = HERE.parent / "645_qwen_scale_2x3_gate"
for path in (HERE, PARENT, SHARED):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_lora as control
import train_pair_fold as parent
from build_pair_runtime import canonical_sha256, occurrence_key, read_jsonl
from gradient_metrics import (
    CHECKPOINT_STEPS,
    DIAGNOSTIC_EFFECTIVE_BATCHES,
    gradient_metrics,
    pcgrad_gate,
)

EXPERIMENT_ID = "687"
PARENT_EXPERIMENT_ID = "686"
SOURCE_EXPERIMENT_ID = "641"
FLAMMABLE = "Легковоспламеняющиеся"
SEED = 42
LEARNING_RATE = 2e-4
GRADIENT_ACCUMULATION_PAIRS = 8
EXPECTED_PAIRS = 5440
EXPECTED_UPDATES = 680
EXPECTED_TRAIN_ROWS = 2280


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_probe_inputs(
    source_runtime: Path,
    pair_runtime: Path,
    transport_path: Path,
    *,
    fold: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if fold != 3:
        raise ValueError("experiment 687 is a blind fold3 train-only probe")
    validation_path = source_runtime / "validation.jsonl"
    if validation_path.exists():
        raise ValueError("outer-validation rows must be physically absent from probe input")
    train_path = source_runtime / "train.jsonl"
    audit_path = source_runtime / "runtime_audit.json"
    for path in (train_path, audit_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"probe source file missing or empty: {path.name}")

    source_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    body = dict(source_audit)
    digest = body.pop("contract_sha256", None)
    if digest != canonical_sha256(body):
        raise ValueError("source runtime self-hash mismatch")
    spec = control.CELL_SPECS[SOURCE_EXPERIMENT_ID]
    expected_source = {
        "experiment_id": SOURCE_EXPERIMENT_ID,
        "objective": spec.objective,
        "outer_fold": fold,
        "grid_contract_sha256": control.GRID_CONTRACT_SHA256,
        "sealed_rows_written": 0,
        "validation_labels_written": 0,
        "outer_validation_occurrences": 0,
        "decision": "GO",
    }
    if any(source_audit.get(key) != value for key, value in expected_source.items()):
        raise ValueError("source runtime scope mismatch")
    if source_audit.get("output_sha256", {}).get("train.jsonl") != sha256_file(train_path):
        raise ValueError("source train checksum mismatch")

    pair_acceptance = parent.verify_pair_runtime(pair_runtime)
    if (
        int(pair_acceptance.get("outer_fold", -1)) != fold
        or pair_acceptance.get("decision") != "ACCEPT_R0_OPEN_TECHNICAL_SMOKE"
    ):
        raise ValueError("pair runtime does not open blind fold3")
    pair_audit = json.loads((pair_runtime / "runtime_audit.json").read_text(encoding="utf-8"))
    if pair_audit.get("source_641_runtime_contract_sha256") != source_audit["contract_sha256"]:
        raise ValueError("pair/source runtime contract mismatch")

    transport = json.loads(transport_path.read_text(encoding="utf-8"))
    transport_body = dict(transport)
    transport_digest = transport_body.pop("transport_acceptance_sha256", None)
    if transport_digest != canonical_sha256(transport_body):
        raise ValueError("transport acceptance self-hash mismatch")
    expected_transport = {
        "experiment_id": "685",
        "stage": "TRAINING_INPUT_EXACT_WHITELIST",
        "outer_fold": fold,
        "pair_runtime_contract_sha256": pair_acceptance["runtime_contract_sha256"],
        "pair_runtime_acceptance_sha256": pair_acceptance["acceptance_sha256"],
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "decision": "ACCEPT_TRAINING_INPUT",
    }
    if any(transport.get(key) != value for key, value in expected_transport.items()):
        raise ValueError("transport scope mismatch")
    accepted_train = transport.get("accepted_files", {}).get(
        "source_runtime/train.jsonl", {}
    )
    if accepted_train.get("sha256") != sha256_file(train_path):
        raise ValueError("probe train bytes differ from accepted transport")

    source_train = control.read_jsonl(train_path)
    if any(int(row["fold"]) == fold for row in source_train):
        raise ValueError("outer fold entered probe training rows")
    train = [row for row in source_train if row["category"] == FLAMMABLE]
    targets = read_jsonl(pair_runtime / "train_targets.jsonl")
    pairs = read_jsonl(pair_runtime / "pairs.jsonl")
    if len(train) != EXPECTED_TRAIN_ROWS or len(pairs) != EXPECTED_PAIRS:
        raise ValueError("fold3 row/pair cardinality differs from frozen contract")
    if [occurrence_key(row) for row in train] != [occurrence_key(row) for row in targets]:
        raise ValueError("sanitized targets do not bind to probe rows")
    for row, target in zip(train, targets, strict=True):
        if int(row["label"]) != int(target["label"]):
            raise ValueError("hard-label binding mismatch")
        if str(row["semantic_component"]) != str(target["semantic_component"]):
            raise ValueError("semantic-component binding mismatch")
    return train, pairs, pair_acceptance, source_audit


def separate_pair_losses(
    model: Any,
    processor: Any,
    positive: SimpleNamespace,
    negative: SimpleNamespace,
    images: list[Any],
    zero_token: int,
    one_token: int,
    pair_target: float,
):
    import torch
    from torch.nn import functional

    rows = [positive, negative]
    conversations = [
        control.messages(row, image, prompt_text=control.base_prompt(row))
        for row, image in zip(rows, images, strict=True)
    ]
    batch = control._processor_batch(
        processor, conversations, add_generation_prompt=True
    ).to(model.device)
    scores = control._last_logits(model, batch, zero_token, one_token).float()
    hard = functional.binary_cross_entropy_with_logits(
        scores,
        torch.tensor([1.0, 0.0], dtype=torch.float32, device=model.device),
    )
    rank = functional.binary_cross_entropy_with_logits(
        scores[0] - scores[1],
        torch.tensor(float(pair_target), dtype=torch.float32, device=model.device),
    )
    return hard, rank


def parameter_group(name: str) -> str:
    groups = [group for group in ("q_proj", "k_proj", "v_proj", "o_proj") if group in name]
    if len(groups) != 1:
        raise ValueError(f"unexpected trainable parameter outside q/k/v/o: {name}")
    return groups[0]


def capture_rng() -> dict[str, Any]:
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state: dict[str, Any]) -> None:
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def rng_state_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(repr(state["python"]).encode())
    numpy_state = state["numpy"]
    digest.update(str(numpy_state[0]).encode())
    digest.update(numpy_state[1].tobytes())
    digest.update(repr(tuple(numpy_state[2:])).encode())
    digest.update(state["torch"].cpu().numpy().tobytes())
    for cuda_state in state["cuda"]:
        digest.update(cuda_state.cpu().numpy().tobytes())
    return digest.hexdigest()


def measure_effective_batch(
    *,
    model: Any,
    processor: Any,
    named_trainable: list[tuple[str, Any]],
    pair_indices: list[int],
    pairs: list[dict[str, Any]],
    pair_lookup: dict[tuple[Any, ...], dict[str, Any]],
    images_dir: Path,
    zero_token: int,
    one_token: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    import torch

    if len(pair_indices) != GRADIENT_ACCUMULATION_PAIRS:
        raise ValueError("diagnostic effective batch must contain eight pairs")
    rng = capture_rng()
    expected_rng_sha256 = rng_state_sha256(rng)
    was_training = model.training
    model.eval()
    hard_grads: dict[str, Any] = {}
    loss_means: dict[str, float] = {}
    try:
        for loss_name in ("hard", "rank"):
            model.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), dtype=torch.float32, device=model.device)
            for pair_index in pair_indices:
                pair = pairs[pair_index]
                positive = pair_lookup[tuple(pair["positive_key"])]
                negative = pair_lookup[tuple(pair["negative_key"])]
                rows = [SimpleNamespace(**positive), SimpleNamespace(**negative)]
                images = [
                    control.open_image(images_dir, positive),
                    control.open_image(images_dir, negative),
                ]
                try:
                    hard, rank = separate_pair_losses(
                        model,
                        processor,
                        rows[0],
                        rows[1],
                        images,
                        zero_token,
                        one_token,
                        float(pair["pair_target"]),
                    )
                    loss = hard if loss_name == "hard" else rank
                    if not torch.isfinite(loss).item():
                        raise FloatingPointError("non-finite diagnostic loss")
                    loss_sum.add_(loss.detach())
                    (loss / len(pair_indices)).backward()
                finally:
                    for image in images:
                        image.close()
            loss_means[loss_name] = float(loss_sum.item() / len(pair_indices))
            if loss_name == "hard":
                hard_grads = {
                    name: parameter.grad.detach().clone()
                    for name, parameter in named_trainable
                    if parameter.grad is not None
                }
                if len(hard_grads) != len(named_trainable):
                    raise ValueError("hard loss did not reach every LoRA parameter")

        totals = {
            "all": [torch.zeros((), device=model.device) for _ in range(3)],
            **{
                group: [torch.zeros((), device=model.device) for _ in range(3)]
                for group in ("q_proj", "k_proj", "v_proj", "o_proj")
            },
        }
        for name, parameter in named_trainable:
            if parameter.grad is None or name not in hard_grads:
                raise ValueError("rank/hard gradient parameter coverage mismatch")
            hard = hard_grads[name].float()
            rank = parameter.grad.detach().float()
            group = parameter_group(name)
            values = (
                torch.sum(hard * hard),
                torch.sum(rank * rank),
                torch.sum(hard * rank),
            )
            for bucket in ("all", group):
                for index, value in enumerate(values):
                    totals[bucket][index].add_(value)
        metrics = {
            bucket: gradient_metrics(*(float(value.item()) for value in values))
            for bucket, values in totals.items()
        }
        metrics["all"]["hard_loss"] = loss_means["hard"]
        metrics["all"]["rank_loss"] = loss_means["rank"]
        return metrics["all"], {key: value for key, value in metrics.items() if key != "all"}
    finally:
        model.zero_grad(set_to_none=True)
        restore_rng(rng)
        model.train(was_training)
        if rng_state_sha256(capture_rng()) != expected_rng_sha256:
            raise RuntimeError("diagnostic pass changed the training RNG state")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    if args.fold != 3 or args.runtime_backend != "legacy_eager":
        raise ValueError("experiment 687 requires blind fold3 legacy-eager runtime")
    if args.model_revision != control.CELL_SPECS[SOURCE_EXPERIMENT_ID].model_revision:
        raise ValueError("model revision differs from frozen 641/686 contract")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if sha256_file(args.probe_code_bundle) != args.expected_probe_code_sha256:
        raise ValueError("probe overlay bundle SHA mismatch")
    if len(args.probe_code_revision) != 40:
        raise ValueError("probe code revision must be a full git SHA")
    probe_code_acceptance = json.loads(
        args.probe_code_acceptance.read_text(encoding="utf-8")
    )
    acceptance_body = dict(probe_code_acceptance)
    acceptance_digest = acceptance_body.pop("acceptance_sha256", None)
    if acceptance_digest != canonical_sha256(acceptance_body):
        raise ValueError("probe-code acceptance self-hash mismatch")
    expected_probe_code = {
        "experiment_id": EXPERIMENT_ID,
        "git_revision": args.probe_code_revision,
        "bundle_sha256": args.expected_probe_code_sha256,
        "decision": "ACCEPT_PROBE_CODE_BUNDLE",
    }
    if any(
        probe_code_acceptance.get(key) != value
        for key, value in expected_probe_code.items()
    ):
        raise ValueError("probe-code acceptance provenance mismatch")

    train, pairs, pair_acceptance, source_audit = load_probe_inputs(
        args.runtime_dir, args.pair_runtime, args.transport_acceptance, fold=args.fold
    )
    parent_code = parent.load_code_acceptance(args.parent_code_acceptance)
    vendor_acceptance = parent.load_vendor_acceptance(
        args.vendor_acceptance, args.vendor_archive
    )
    pair_lookup = {occurrence_key(row): row for row in train}
    if len(pair_lookup) != len(train):
        raise ValueError("duplicate occurrence key in train-only probe")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    sys.path.insert(0, str(args.vendor.resolve()))
    import peft

    if peft.__version__ != "0.20.0":
        raise ValueError("exact vendored PEFT 0.20.0 is required")
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    model_tree_digest, model_tree_files = parent.model_tree_sha256(args.model_root)
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
    named_trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_trainable:
        raise ValueError("model has no trainable LoRA parameters")
    initial_state_sha256 = parent.trainable_state_sha256(model)
    trainable = [parameter for _, parameter in named_trainable]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=0.01)

    pair_indices = list(range(len(pairs)))
    random.Random(SEED).shuffle(pair_indices)
    ordered_pair_index_sha256 = hashlib.sha256(
        json.dumps(pair_indices, separators=(",", ":")).encode()
    ).hexdigest()
    diagnostic_batches = [
        pair_indices[index * GRADIENT_ACCUMULATION_PAIRS : (index + 1) * GRADIENT_ACCUMULATION_PAIRS]
        for index in range(DIAGNOSTIC_EFFECTIVE_BATCHES)
    ]
    diagnostic_index_sha256 = hashlib.sha256(
        json.dumps(diagnostic_batches, separators=(",", ":")).encode()
    ).hexdigest()
    updates = len(pairs) // GRADIENT_ACCUMULATION_PAIRS
    if updates != EXPECTED_UPDATES:
        raise ValueError("optimizer steps differ from frozen trajectory")
    warmup = max(1, int(updates * 0.05))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, updates - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    optimizer.zero_grad(set_to_none=True)
    measurements: list[dict[str, Any]] = []
    group_measurements: list[dict[str, Any]] = []
    started = time.monotonic()

    def measure_checkpoint(step: int) -> None:
        for batch_index, indices in enumerate(diagnostic_batches):
            overall, grouped = measure_effective_batch(
                model=model,
                processor=processor,
                named_trainable=named_trainable,
                pair_indices=indices,
                pairs=pairs,
                pair_lookup=pair_lookup,
                images_dir=args.images,
                zero_token=zero[0],
                one_token=one[0],
            )
            row = {"checkpoint_step": step, "batch_index": batch_index, **overall}
            if not all(
                math.isfinite(float(value))
                for key, value in row.items()
                if key not in {"checkpoint_step", "batch_index", "conflict"}
            ):
                raise FloatingPointError("non-finite gradient measurement")
            measurements.append(row)
            for group, values in grouped.items():
                group_measurements.append(
                    {
                        "checkpoint_step": step,
                        "batch_index": batch_index,
                        "group": group,
                        **values,
                    }
                )
            if step == 0 and batch_index == 1:
                print(json.dumps({"phase": "inline_technical_preflight_pass"}), flush=True)
        print(
            json.dumps(
                {
                    "phase": "gradient_checkpoint_complete",
                    "optimizer_step": step,
                    "measurements": DIAGNOSTIC_EFFECTIVE_BATCHES,
                }
            ),
            flush=True,
        )

    measure_checkpoint(0)
    optimizer_steps = 0
    loss_total = torch.zeros((), dtype=torch.float32, device=model.device)
    model.train()
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
            loss, _, _ = parent.pair_loss(
                model,
                processor,
                rows[0],
                rows[1],
                images,
                zero[0],
                one[0],
                float(pair["pair_target"]),
                "rank_candidate",
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError("non-finite trajectory loss")
            loss_total.add_(loss.detach())
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
            if optimizer_steps in CHECKPOINT_STEPS[1:]:
                measure_checkpoint(optimizer_steps)
        if position % 400 == 0 or position == len(pair_indices):
            elapsed = time.monotonic() - started
            rate = position / max(elapsed, 1e-9)
            print(
                json.dumps(
                    {
                        "phase": "trajectory_progress",
                        "pairs_done": position,
                        "pairs_total": len(pair_indices),
                        "optimizer_steps_done": optimizer_steps,
                        "pairs_per_second_including_probe": rate,
                    }
                ),
                flush=True,
            )
    if optimizer_steps != EXPECTED_UPDATES:
        raise RuntimeError("optimizer trajectory did not reach frozen final step")

    by_checkpoint = {
        step: [row for row in measurements if int(row["checkpoint_step"]) == step]
        for step in CHECKPOINT_STEPS
    }
    gate = pcgrad_gate(by_checkpoint)
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "parent_experiment_id": PARENT_EXPERIMENT_ID,
        "outer_fold": 3,
        "objective": "outer_validation_label_free_gradient_conflict_probe",
        "changed_factor": "measure_hard_vs_rank_gradient_geometry",
        "decision": gate["decision"],
        "pcgrad_gate": gate,
        "measurements": measurements,
        "group_measurements": group_measurements,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "diagnostic_effective_batches": DIAGNOSTIC_EFFECTIVE_BATCHES,
        "diagnostic_pair_indices_sha256": diagnostic_index_sha256,
        "ordered_pair_index_sha256": ordered_pair_index_sha256,
        "pair_runtime_contract_sha256": pair_acceptance["runtime_contract_sha256"],
        "pair_runtime_acceptance_sha256": pair_acceptance["acceptance_sha256"],
        "source_runtime_contract_sha256": source_audit["contract_sha256"],
        "parent_code_bundle_sha256": parent_code["bundle_sha256"],
        "parent_code_revision": parent_code["git_revision"],
        "parent_code_acceptance_sha256": parent_code["acceptance_sha256"],
        "probe_code_bundle_sha256": args.expected_probe_code_sha256,
        "probe_code_revision": args.probe_code_revision,
        "probe_code_acceptance_sha256": acceptance_digest,
        "vendor_zip_sha256": vendor_acceptance["vendor_zip_sha256"],
        "vendor_bridge_sha256": vendor_acceptance["bridge_sha256"],
        "model_id": control.CELL_SPECS[SOURCE_EXPERIMENT_ID].model_id,
        "model_revision": args.model_revision,
        "model_tree_sha256": model_tree_digest,
        "model_tree_files": model_tree_files,
        "initial_trainable_state_sha256": initial_state_sha256,
        "final_trainable_state_sha256": parent.trainable_state_sha256(model),
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "rank_loss_weight": 0.5,
        "hard_loss_weight": 1.0,
        "pairs": len(pairs),
        "train_rows": len(train),
        "optimizer_steps_executed": optimizer_steps,
        "training_loss_mean": float(loss_total.item() / len(pairs)),
        "runtime_backend": "legacy_eager",
        "runtime_packages": {
            "transformers": importlib.metadata.version("transformers"),
            "torch": importlib.metadata.version("torch"),
            "peft": peft.__version__,
        },
        "runtime_minutes": (time.monotonic() - started) / 60,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "inline_technical_preflight": True,
        "diagnostic_rng_noninterference": True,
        "outer_validation_transport_checksum_verified": True,
        "outer_validation_rows_consumed_by_probe": 0,
        "outer_validation_labels_read": 0,
        "outer_quality_metrics_computed": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "submission_artifact": False,
    }
    report["report_sha256"] = canonical_sha256(report)
    (args.output_dir / "gradient_conflict_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "phase": "complete",
                "decision": report["decision"],
                "report_sha256": report["report_sha256"],
            }
        ),
        flush=True,
    )
    return report


def parser() -> argparse.ArgumentParser:
    result = control.parser_for(SOURCE_EXPERIMENT_ID)
    result.add_argument("--pair-runtime", type=Path, required=True)
    result.add_argument("--transport-acceptance", type=Path, required=True)
    result.add_argument("--parent-code-acceptance", type=Path, required=True)
    result.add_argument("--probe-code-bundle", type=Path, required=True)
    result.add_argument("--probe-code-acceptance", type=Path, required=True)
    result.add_argument("--expected-probe-code-sha256", required=True)
    result.add_argument("--probe-code-revision", required=True)
    result.add_argument("--vendor-acceptance", type=Path, required=True)
    result.add_argument("--vendor-archive", type=Path, required=True)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
