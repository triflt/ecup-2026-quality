from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/590_qwen3vl_cross_listing_recombination"
sys.path.insert(0, str(EXPERIMENT))
SPEC = importlib.util.spec_from_file_location(
    "recombination_plan_590", EXPERIMENT / "recombination_plan.py"
)
assert SPEC is not None and SPEC.loader is not None
PLAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLAN
SPEC.loader.exec_module(PLAN)


def _pair(left: str, right: str, component: str = "c") -> object:
    return PLAN.StrongPair(
        left_id=left,
        right_id=right,
        semantic_component=component,
        category="Легковоспламеняющиеся",
        label=1,
        stage="exact_first_image",
        key_hash="a" * 64,
        key_degree=2,
        corroboration="both_informative_first_images",
    )


def test_exp590_synthetic_plan_recombines_exactly_one_quarter_of_repeats() -> None:
    frame = pd.DataFrame(
        {
            "id": ["source", "donor", "other"],
            "category": ["Легковоспламеняющиеся"] * 3,
            "label": [1, 1, 0],
        }
    )
    oof = {"fold_ids": np.asarray([1, 1, 1], dtype=np.int8)}
    records = [0, 0, 0, 0, 0, 1, 2]
    review = {"decision": "GO"}
    manifest, audit = PLAN.build_fold_plan(
        frame, oof, records, [_pair("source", "donor")], holdout_fold=0, review_audit=review
    )
    assert len(manifest) == 1
    assert manifest[0]["source_id"] == "source"
    assert manifest[0]["donor_id"] == "donor"
    assert audit["eligible_repeated_occurrences"] == 4
    assert audit["recombined_occurrences"] == 1
    assert audit["recombined_fraction_of_repeats"] == 0.25
    assert audit["parent_record_multiset_sha256"] == audit["candidate_record_multiset_sha256"]
    assert audit["record_multiset_unchanged"] is True


def test_exp590_synthetic_blind_audit_all_positive_agreement_is_honestly_undefined_kappa(
    tmp_path: Path,
) -> None:
    sample = [_pair(f"left-{index}", f"right-{index}", f"c-{index}") for index in range(300)]
    rows = [
        {
            "audit_id": f"R{index:03d}",
            "left_id": pair.left_id,
            "right_id": pair.right_id,
            "stage": pair.stage,
            "reviewer_a_same_product": 1,
            "reviewer_b_same_product": 1,
        }
        for index, pair in enumerate(sample, 1)
    ]
    path = tmp_path / "reviews.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    audit = PLAN.evaluate_reviews(sample, path)
    assert audit["decision"] == "GO"
    assert audit["same_product_precision"] == 1.0
    assert audit["kappa_defined"] is False
    assert audit["cohen_kappa"] is None
    assert audit["raw_agreement"] == 1.0


def test_exp590_synthetic_blind_audit_fails_below_precision_gate(tmp_path: Path) -> None:
    sample = [_pair(f"left-{index}", f"right-{index}", f"c-{index}") for index in range(300)]
    rows = []
    for index, pair in enumerate(sample, 1):
        decision = 0 if index <= 7 else 1
        rows.append(
            {
                "audit_id": f"R{index:03d}",
                "left_id": pair.left_id,
                "right_id": pair.right_id,
                "reviewer_a_same_product": decision,
                "reviewer_b_same_product": decision,
            }
        )
    path = tmp_path / "reviews.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    audit = PLAN.evaluate_reviews(sample, path)
    assert audit["same_product_precision"] < 0.98
    assert audit["decision"] == "NO_GO"


def test_exp590_real_preflight_has_only_frozen_strong_pairs_and_is_blocked() -> None:
    summary = json.loads((EXPERIMENT / "analysis/preflight/preflight_summary.json").read_text())
    sample = pd.read_csv(EXPERIMENT / "analysis/preflight/blind_audit_pairs.csv", dtype=str)
    assert len(sample) == 300
    assert set(sample.stage).issubset(PLAN.ALLOWED_STAGES)
    assert not set(sample.stage).intersection(PLAN.FORBIDDEN_STAGES)
    assert sample.key_degree.astype(int).max() <= PLAN.MAX_KEY_DEGREE
    assert "category" not in sample and "label" not in sample
    assert summary["blind_audit"]["reviewed_pairs"] == 0
    assert summary["decision"] == "NO_GO"
    for fold in (0, 3):
        audit = json.loads(
            (EXPERIMENT / f"analysis/preflight/fold_{fold}/preflight_audit.json").read_text()
        )
        assert audit["eligible_pairs"] >= 200
        assert audit["eligible_components"] >= 50
        assert audit["forbidden_stage_pairs"] == 0
        assert audit["generic_provenance_pairs"] == 0
        assert audit["cross_category_pairs"] == 0
        assert audit["cross_label_pairs"] == 0
        assert audit["outer_validation_pairs"] == 0
        assert audit["sealed_holdout_pairs"] == 0
        assert audit["mixed_label_component_pairs"] == 0
        assert audit["recombined_fraction_of_repeats"] == 0.25
        assert audit["failures"] == ["blind_audit_not_approved"]


def test_exp590_exact_null_control_changes_no_route400_predictions() -> None:
    root = EXPERIMENT / "analysis/null_control"
    report = json.loads((root / "null_screen_control.json").read_text())
    arrays = np.load(root / "null_screen_control.npz", allow_pickle=False)
    assert report["null_control_passed"] is True
    assert report["evaluated_folds"] == [0, 3]
    assert np.array_equal(
        arrays["baseline_nested_predictions"], arrays["candidate_screen_nested_predictions"]
    )


def test_exp590_train_entrypoint_checks_no_go_before_parent_model_import() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT / "train_screen.py"),
            "--fold",
            "0",
            "--preflight-audit",
            str(EXPERIMENT / "analysis/preflight/fold_0/preflight_audit.json"),
            "--recombination-manifest",
            str(EXPERIMENT / "analysis/preflight/fold_0/recombination_manifest.jsonl"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "training is blocked before model import" in result.stderr
