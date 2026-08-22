from __future__ import annotations

import json
import os
import random
import types
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pair_selector import build_pair_plan, canonical_sha256
from parent_recipe import load_parent_module
from parent_selector import select_parent_training_records

PAIRWISE_WEIGHT = 0.10
PAIRWISE_MARGIN = 1.0
MIN_CONSTRAINTS = 64
STRICT_AUDIT_KEYS = (
    "holdout_fold",
    "training_records",
    "ordered_training_records",
    "parent_record_multiset_sha256",
    "candidate_record_multiset_sha256",
    "eligible_parent_intersection_pairs",
    "pair_batches",
    "realized_pairs",
    "same_cue_mask_pairs",
    "global_fallback_pairs",
    "max_pairs_per_positive",
    "max_negative_reuse",
    "outer_validation_pairs",
    "unsafe_pairs",
    "multiplicity_unchanged",
    "manifest_sha256",
    "decision",
)


class PairOrderedRecords(list[int]):
    """Marker list whose frozen pair-aware order must survive parent shuffle."""


@dataclass
class PairState:
    expected_batches: dict[int, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    current_local_pairs: list[tuple[int, int]] = field(default_factory=list)
    batch_calls: int = 0
    token_zero: int | None = None
    token_one: int | None = None
    forward_calls: int = 0
    constraints: int = 0
    auxiliary_loss_sum: float = 0.0
    actual_order: list[int] = field(default_factory=list)


def _load_expected_audit() -> dict[str, Any]:
    raw = os.environ.get("ECUP_PAIR_AUDIT_JSON")
    if not raw:
        raise ValueError("ECUP_PAIR_AUDIT_JSON is required; run the audit first")
    audit = json.loads(Path(raw).resolve().read_text(encoding="utf-8"))
    claimed = audit.pop("audit_sha256", None)
    if claimed != canonical_sha256(audit):
        raise ValueError("pair audit checksum mismatch")
    audit["audit_sha256"] = claimed
    if audit.get("decision") != "GO":
        raise ValueError("pair audit did not authorize launch")
    return audit


def main() -> int:
    expected = _load_expected_audit()
    parent = load_parent_module()
    if expected.get("holdout_fold") != parent.HOLDOUT_FOLD:
        raise ValueError("pair audit fold differs from training fold")
    state = PairState()
    original_select = parent.select_training
    original_batch = parent.training_batch
    original_random_class = random.Random
    original_loader = parent.AutoModelForMultimodalLM

    class PairAwareRandom(original_random_class):
        def shuffle(self, sequence) -> None:
            if isinstance(sequence, PairOrderedRecords):
                return
            super().shuffle(sequence)

    def select_with_frozen_pairs(frame, oof):
        true_parent_records, parent_audit = original_select(frame, oof)
        light_records, _ = select_parent_training_records(
            frame, oof, seed=parent.SEED, holdout_fold=parent.HOLDOUT_FOLD
        )
        if true_parent_records != light_records:
            raise ValueError("exact runtime parent selector differs from lightweight selector")
        ordered, manifest, actual = build_pair_plan(
            frame,
            oof["fold_ids"].astype("int8"),
            true_parent_records,
            holdout_fold=parent.HOLDOUT_FOLD,
        )
        mismatch = {
            key: {"expected": expected.get(key), "actual": actual.get(key)}
            for key in STRICT_AUDIT_KEYS
            if expected.get(key) != actual.get(key)
        }
        if mismatch:
            raise ValueError(f"frozen pair contract mismatch: {mismatch}")
        # Ordered hashes intentionally do not gate membership across runtimes.
        # Exact parent-vs-light order equality above remains mandatory within
        # this runtime, while multiset and canonical pair topology are strict.
        for row in manifest:
            state.expected_batches[int(row["batch_index"])].append(row)
        runtime_audit = dict(actual)
        runtime_audit["frozen_ordered_records_sha256_diagnostic"] = expected.get(
            "ordered_records_sha256_diagnostic_only"
        )
        runtime_audit["ordered_diagnostic_matches_frozen"] = expected.get(
            "ordered_records_sha256_diagnostic_only"
        ) == actual.get("ordered_records_sha256_diagnostic_only")
        parent.OUTPUT.mkdir(parents=True, exist_ok=True)
        (parent.OUTPUT / "pair_manifest.runtime.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest),
            encoding="utf-8",
        )
        (parent.OUTPUT / "pair_audit.runtime.json").write_text(
            json.dumps(runtime_audit, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        parent.random.Random = PairAwareRandom
        parent_audit = dict(parent_audit)
        parent_audit["flammable_hard_pairwise"] = {
            "weight": PAIRWISE_WEIGHT,
            "margin": PAIRWISE_MARGIN,
            "realized_pairs": actual["realized_pairs"],
            "manifest_sha256": actual["manifest_sha256"],
        }
        return PairOrderedRecords(ordered), parent_audit

    def training_batch_with_pairs(processor, rows):
        batch = original_batch(processor, rows)
        batch_index = state.batch_calls
        state.batch_calls += 1
        row_indices = [int(row.name) for row in rows]
        state.actual_order.extend(row_indices)
        expected_pairs = state.expected_batches.get(batch_index, [])
        local_pairs = []
        zero = processor.tokenizer.encode("0", add_special_tokens=False)
        one = processor.tokenizer.encode("1", add_special_tokens=False)
        if len(zero) != 1 or len(one) != 1:
            raise ValueError("verdict tokens are not atomic")
        state.token_zero, state.token_one = zero[0], one[0]
        labels = batch["labels"]
        for item in expected_pairs:
            positive_slot = int(item["positive_slot"])
            negative_slot = int(item["negative_slot"])
            if row_indices[positive_slot] != int(item["positive_index"]):
                raise ValueError("runtime positive row differs from frozen pair manifest")
            if row_indices[negative_slot] != int(item["negative_index"]):
                raise ValueError("runtime negative row differs from frozen pair manifest")
            for slot, target in (
                (positive_slot, state.token_one),
                (negative_slot, state.token_zero),
            ):
                supervised = labels[slot].ne(-100).nonzero(as_tuple=False).flatten()
                if len(supervised) == 0 or int(labels[slot, supervised[0]]) != target:
                    raise ValueError("first supervised target differs from donor label")
            local_pairs.append((positive_slot, negative_slot))
        state.current_local_pairs = local_pairs
        return batch

    class PairwiseLoader:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            model = original_loader.from_pretrained(*args, **kwargs)
            original_forward = model.forward

            def forward_with_pairwise(self, *forward_args, **forward_kwargs):
                outputs = original_forward(*forward_args, **forward_kwargs)
                labels = forward_kwargs.get("labels")
                if self.training and labels is not None and state.current_local_pairs:
                    if state.token_zero is None or state.token_one is None:
                        raise RuntimeError("pairwise token ids were not initialized")
                    first_positions = []
                    for row_labels in labels:
                        supervised = row_labels.ne(-100).nonzero(as_tuple=False).flatten()
                        if len(supervised) == 0 or int(supervised[0]) == 0:
                            raise ValueError("cannot locate first supervised verdict position")
                        first_positions.append(int(supervised[0]) - 1)
                    scores = [
                        outputs.logits[row_index, position, state.token_one]
                        - outputs.logits[row_index, position, state.token_zero]
                        for row_index, position in enumerate(first_positions)
                    ]
                    losses = [
                        (PAIRWISE_MARGIN - scores[positive] + scores[negative]).clamp_min(0)
                        for positive, negative in state.current_local_pairs
                    ]
                    auxiliary = sum(losses) / len(losses)
                    outputs.loss = outputs.loss + PAIRWISE_WEIGHT * auxiliary
                    state.forward_calls += 1
                    state.constraints += len(losses)
                    state.auxiliary_loss_sum += float(auxiliary.detach().cpu())
                return outputs

            model.forward = types.MethodType(forward_with_pairwise, model)
            return model

    parent.select_training = select_with_frozen_pairs
    parent.training_batch = training_batch_with_pairs
    parent.AutoModelForMultimodalLM = PairwiseLoader
    parent.main()
    runtime_multiset = canonical_sha256(sorted(Counter(state.actual_order).items()))
    if runtime_multiset != expected["candidate_record_multiset_sha256"]:
        raise RuntimeError("runtime training multiset differs from frozen parent multiset")
    if state.constraints != expected["realized_pairs"]:
        raise RuntimeError("runtime pairwise constraint count differs from frozen manifest")
    if state.forward_calls != expected["pair_batches"]:
        raise RuntimeError("runtime pairwise batch count differs from frozen manifest")
    if state.constraints < MIN_CONSTRAINTS:
        raise RuntimeError("runtime executed too few pairwise constraints")
    ordered_runtime_sha = canonical_sha256(state.actual_order)
    runtime = {
        "experiment_id": "560",
        "holdout_fold": parent.HOLDOUT_FOLD,
        "pairwise_weight": PAIRWISE_WEIGHT,
        "pairwise_margin": PAIRWISE_MARGIN,
        "pairwise_forward_calls": state.forward_calls,
        "pairwise_constraints": state.constraints,
        "mean_unweighted_pairwise_loss": state.auxiliary_loss_sum / state.forward_calls,
        "manifest_sha256": expected["manifest_sha256"],
        "runtime_ordered_records_sha256_diagnostic_only": ordered_runtime_sha,
        "ordered_hash_is_membership_gate": False,
    }
    (parent.OUTPUT / "pairwise_runtime.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    report_path = parent.OUTPUT / "lora_holdout_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["experiment_id"] = "560"
    report["pairwise"] = runtime
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
