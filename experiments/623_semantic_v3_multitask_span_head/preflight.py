from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from multitask_head import LossWeights, SpanConceptHead, attach_span_targets, multitask_loss
from protocol import CONCEPTS, FORBIDDEN_LABEL_COLUMNS, read_jsonl
from renderer import render_explanation


def run_preflight(runtime_dir: Path | None = None) -> dict[str, object]:
    torch.manual_seed(623)
    head = SpanConceptHead(hidden_size=12, concept_count=len(CONCEPTS))
    hidden = torch.randn(3, 9, 12, requires_grad=True)
    attention = torch.ones(3, 9, dtype=torch.bool)
    text_mask = torch.zeros(3, 9, dtype=torch.bool)
    text_mask[:, 2:8] = True
    auxiliary = head(hidden, attention, text_mask)
    loss, parts = multitask_loss(
        verdict_logits=torch.tensor([0.2, -0.4, 0.7], requires_grad=True),
        auxiliary=auxiliary,
        verdict_targets=torch.tensor([1, 0, 1]),
        start_targets=torch.tensor([3, 9, 4]),
        end_targets=torch.tensor([4, 9, 6]),
        concept_targets=torch.tensor([0, -1, 3]),
        quality_weights=torch.tensor([1.0, 0.0, 1.0]),
        weights=LossWeights(),
    )
    loss.backward()
    start, end, mask = attach_span_targets(
        full_input_ids=[91, 10, 11, 12, 13, 92],
        canonical_input_ids=[10, 11, 12, 13],
        canonical_offsets=[(0, 2), (2, 4), (4, 7), (7, 9)],
        rationale={"has_evidence": True, "char_start": 2, "char_end": 7},
    )
    rendered = render_explanation(
        name="Горелка", description="без топлива", char_start=21, char_end=32, concept="NEGATION"
    )
    checks = {
        "loss_finite": bool(torch.isfinite(loss)),
        "backward_reaches_hidden_states": hidden.grad is not None,
        "span_target_alignment": (start, end) == (2, 3),
        "canonical_text_mask": sum(mask) == 4,
        "closed_renderer": rendered["concept"] in CONCEPTS,
        "masked_rationale_rows": int(parts["quality_rows"]) == 2,
    }
    if runtime_dir is not None:
        train = read_jsonl(runtime_dir / "train.jsonl")
        validation = read_jsonl(runtime_dir / "validation.jsonl")
        validation_keys = set().union(*(row.keys() for row in validation))
        runtime_checks = {
            "runtime_audit_go": json.loads(
                (runtime_dir / "runtime_audit.json").read_text(encoding="utf-8")
            ).get("decision")
            == "GO",
            "train_nonempty": bool(train),
            "validation_nonempty": bool(validation),
            "validation_has_no_labels": not bool(validation_keys & FORBIDDEN_LABEL_COLUMNS),
            "validation_has_no_rationales": "rationale" not in validation_keys,
            "train_validation_disjoint": not (
                {row["id"] for row in train} & {row["id"] for row in validation}
            ),
        }
        checks.update(runtime_checks)
    decision = "GO" if all(checks.values()) else "NO_GO"
    return {"experiment_id": "623", "checks": checks, "decision": decision}


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU preflight for experiment 623.")
    parser.add_argument("--runtime-dir", type=Path)
    args = parser.parse_args()
    report = run_preflight(args.runtime_dir.resolve() if args.runtime_dir else None)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
