from __future__ import annotations

import hashlib
import json
import os
import random
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from pair_selector import build_pair_plan
from parent_recipe import load_parent_module
from parent_selector import select_parent_training_records

PAIRWISE_WEIGHT = 0.10
PAIRWISE_MARGIN = 1.0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PairOrderedRecords(list[int]):
    pass


@dataclass
class PairState:
    pairs_by_indices: set[tuple[int, int]] = field(default_factory=set)
    current_local_pairs: list[tuple[int, int]] = field(default_factory=list)
    token_zero: int | None = None
    token_one: int | None = None
    forward_calls: int = 0
    constraints: int = 0
    auxiliary_loss_sum: float = 0.0


def _load_expected_audit() -> dict[str, Any]:
    raw = os.environ.get("ECUP_PAIR_AUDIT_JSON")
    if not raw:
        raise ValueError("ECUP_PAIR_AUDIT_JSON is required; run build_manifest.py first")
    audit = json.loads(Path(raw).resolve().read_text(encoding="utf-8"))
    if audit.get("decision") != "GO":
        raise ValueError("pair audit did not authorize launch")
    return audit


def main() -> int:
    expected_audit = _load_expected_audit()
    parent = load_parent_module()
    if expected_audit.get("holdout_fold") != parent.HOLDOUT_FOLD:
        raise ValueError("pair audit fold differs from training fold")
    guard_path = Path(os.environ["ECUP_CONNECTED_GUARD"]).resolve()
    if _file_sha256(guard_path) != expected_audit.get("input_sha256", {}).get("connected_guard"):
        raise ValueError("runtime connected guard differs from frozen pair audit")
    guard = pd.read_csv(guard_path, dtype={"id": str, "connected_component": str})
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
        records, parent_audit = original_select(frame, oof)
        light_records, _ = select_parent_training_records(
            frame, oof, seed=parent.SEED, holdout_fold=parent.HOLDOUT_FOLD
        )
        if records != light_records:
            raise ValueError("dependency-light selector differs from frozen parent order")
        ordered, manifest, audit = build_pair_plan(
            frame,
            guard,
            oof["fold_ids"].astype("int8"),
            records,
            holdout_fold=parent.HOLDOUT_FOLD,
        )
        frozen_keys = (
            "holdout_fold",
            "training_records",
            "ordered_training_records",
            "parent_record_multiset_sha256",
            "candidate_record_multiset_sha256",
            "pair_batches",
            "realized_pairs",
            "same_family_pairs",
            "nearest_family_pairs",
            "manifest_sha256",
            "decision",
        )
        # The parent's NumPy sampler preserves membership but may order an
        # otherwise identical sampled set differently across NumPy versions.
        # Pair topology and manifests are canonicalized independently; inside
        # this runtime the true parent and light selector still agree exactly.
        mismatch = {
            key: {"expected": expected_audit.get(key), "actual": audit.get(key)}
            for key in frozen_keys
            if expected_audit.get(key) != audit.get(key)
        }
        if mismatch:
            raise ValueError(f"frozen pair manifest mismatch: {mismatch}")
        state.pairs_by_indices = {
            (int(row["positive_index"]), int(row["negative_index"])) for row in manifest
        }
        parent.OUTPUT.mkdir(parents=True, exist_ok=True)
        (parent.OUTPUT / "pair_manifest.runtime.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest),
            encoding="utf-8",
        )
        (parent.OUTPUT / "pair_audit.runtime.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        parent.random.Random = PairAwareRandom
        parent_audit = dict(parent_audit)
        parent_audit["mixed_family_bad_pairwise"] = {
            "weight": PAIRWISE_WEIGHT,
            "margin": PAIRWISE_MARGIN,
            "realized_pairs": audit["realized_pairs"],
            "manifest_sha256": audit["manifest_sha256"],
        }
        return PairOrderedRecords(ordered), parent_audit

    def training_batch_with_pairs(processor, rows):
        batch = original_batch(processor, rows)
        indices = [int(row.name) for row in rows]
        positions = {index: position for position, index in enumerate(indices)}
        state.current_local_pairs = [
            (positions[positive], positions[negative])
            for positive, negative in state.pairs_by_indices
            if positive in positions and negative in positions
        ]
        zero = processor.tokenizer.encode("0", add_special_tokens=False)
        one = processor.tokenizer.encode("1", add_special_tokens=False)
        if len(zero) != 1 or len(one) != 1:
            raise ValueError("verdict tokens are not atomic")
        state.token_zero, state.token_one = zero[0], one[0]
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
    runtime = {
        "experiment_id": "540",
        "holdout_fold": parent.HOLDOUT_FOLD,
        "pairwise_weight": PAIRWISE_WEIGHT,
        "pairwise_margin": PAIRWISE_MARGIN,
        "pairwise_forward_calls": state.forward_calls,
        "pairwise_constraints": state.constraints,
        "mean_unweighted_pairwise_loss": (
            state.auxiliary_loss_sum / state.forward_calls if state.forward_calls else None
        ),
        "manifest_sha256": expected_audit["manifest_sha256"],
    }
    (parent.OUTPUT / "pairwise_runtime.json").write_text(
        json.dumps(runtime, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    report_path = parent.OUTPUT / "lora_holdout_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["experiment_id"] = "540"
    report["pairwise"] = runtime
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
