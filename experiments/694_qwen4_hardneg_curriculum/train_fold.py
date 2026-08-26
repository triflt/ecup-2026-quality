from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

SHARED = Path(__file__).resolve().parents[1] / "645_qwen_scale_2x3_gate"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import train_lora as control

EXPERIMENT_ID = "694"
SOURCE_EXPERIMENT_ID = "641"
FLAMMABLE = "Легковоспламеняющиеся"
MODES = ("hard_bce_control", "hardneg_candidate")


def validate_teacher_contract(audit: dict[str, Any], fold: int) -> str:
    teacher = audit.get("teacher_contract", {})
    expected = {
        "schema": "qwen27_all_outer_safe_v1",
        "teacher_recipe": "qwen27-all",
        "outer_fold": fold,
        "train_scope": "outer_train_only",
        "outer_validation_labels_read": 0,
        "target_category": FLAMMABLE,
    }
    if any(teacher.get(key) != value for key, value in expected.items()):
        raise ValueError("teacher artifact contract is missing or not outer-safe")
    digest = teacher.get("artifact_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("teacher artifact SHA-256 is invalid")
    return digest


def validate_teacher_rows(
    train: list[dict[str, Any]], validation: list[dict[str, Any]], fold: int
) -> None:
    for row in validation:
        if "teacher_score" in row:
            raise ValueError("outer-validation row contains a forbidden teacher score")
    for row in train:
        if row["category"] == FLAMMABLE:
            score = float(row.get("teacher_score", float("nan")))
            if not math.isfinite(score):
                raise ValueError("flammable train row lacks a finite teacher score")
            if int(row.get("teacher_outer_fold", -1)) != fold:
                raise ValueError("teacher score is not bound to this outer fold")
        elif "teacher_score" in row:
            raise ValueError("KD signal is present outside flammable")


def curriculum_key(row: dict[str, Any]) -> tuple[int, float, int]:
    if row["category"] != FLAMMABLE:
        stage, hardness = 3, 0.0
    else:
        score = float(row["teacher_score"])
        label = int(row["label"])
        if label == 0:
            stage, hardness = 0, -score  # strongest teacher false-positive risk first
        elif label == 1:
            stage, hardness = 1, score  # strongest missed-positive risk first
        else:
            raise ValueError("hard label must be binary")
    return stage, hardness, int(row["global_index"])


def arrange_for_frozen_shuffle(
    rows: list[dict[str, Any]], *, seed: int = 42
) -> list[dict[str, Any]]:
    """Invert the parent's fixed shuffle so its observed sequence is the curriculum."""
    desired = sorted(rows, key=curriculum_key)
    permutation = list(range(len(rows)))
    random.Random(seed).shuffle(permutation)
    arranged: list[dict[str, Any] | None] = [None] * len(rows)
    for desired_index, source_index in enumerate(permutation):
        arranged[source_index] = desired[desired_index]
    return [row for row in arranged if row is not None]


def run(args: Any) -> dict[str, Any]:
    if args.mode not in MODES:
        raise ValueError("unknown mode")
    original_load = control.load_runtime

    teacher_binding: dict[str, str] = {}

    def load_runtime(runtime_dir: Path, spec_id: str, fold: int):
        train, validation, audit = original_load(runtime_dir, spec_id, fold)
        teacher_binding["artifact_sha256"] = validate_teacher_contract(audit, fold)
        validate_teacher_rows(train, validation, fold)
        if args.mode == "hardneg_candidate":
            train = arrange_for_frozen_shuffle(train)
        return train, validation, audit

    control.load_runtime = load_runtime
    try:
        report = control.run(SOURCE_EXPERIMENT_ID, args)
    finally:
        control.load_runtime = original_load
    report.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "source_experiment_id": SOURCE_EXPERIMENT_ID,
            "mode": args.mode,
            "hard_bce_scope": "all_categories",
            "hard_bce_coefficient": 1.0,
            "teacher_in_loss": False,
            "kd_category": FLAMMABLE,
            "changed_factor": "training_order" if args.mode == "hardneg_candidate" else "none",
            "teacher_outer_safe_required": True,
            "teacher_artifact_sha256": teacher_binding["artifact_sha256"],
        }
    )
    report.pop("contract_sha256", None)
    report["contract_sha256"] = control.canonical_sha256(report)
    (args.output_dir / "output_contract.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = control.parser_for(SOURCE_EXPERIMENT_ID)
    parser.add_argument("--mode", choices=MODES, required=True)
    parsed = parser.parse_args()
    print(json.dumps(run(parsed), ensure_ascii=False, indent=2, sort_keys=True))
