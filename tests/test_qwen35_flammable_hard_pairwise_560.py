from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "560_qwen35_flammable_hard_pairwise"
SPEC = importlib.util.spec_from_file_location("pair_selector_560", EXPERIMENT / "pair_selector.py")
assert SPEC is not None and SPEC.loader is not None
SELECTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SELECTOR
SPEC.loader.exec_module(SELECTOR)


def _frozen_rows(fold: int) -> tuple[list[dict], dict]:
    analysis = EXPERIMENT / "analysis" / f"fold_{fold}"
    manifest = [
        json.loads(line) for line in (analysis / "pair_manifest.jsonl").read_text().splitlines()
    ]
    audit = json.loads((analysis / "pair_audit.json").read_text())
    return manifest, audit


def test_exp560_frozen_manifests_preserve_parent_and_exclude_unsafe_outer_rows() -> None:
    frame = pd.read_csv(ROOT / "research/data.csv", dtype={"id": str})
    oof = np.load(ROOT / "research/four-head-r2-extracted/four_head_oof.npz", allow_pickle=True)
    membership, _, exp530_audit = SELECTOR.load_frozen_exp530(0)
    safe = membership.set_index("id").safe_for_selection.to_dict()
    assert exp530_audit["selector_rows"] == 909
    for fold in (0, 3):
        manifest, audit = _frozen_rows(fold)
        assert audit["decision"] == "GO"
        assert audit["training_records"] == audit["ordered_training_records"] == 5390
        assert audit["parent_record_multiset_sha256"] == audit["candidate_record_multiset_sha256"]
        assert audit["multiplicity_unchanged"] is True
        assert audit["outer_validation_pairs"] == audit["unsafe_pairs"] == 0
        assert audit["realized_pairs"] == audit["pair_batches"] == 88
        for pair in manifest:
            positive, negative = pair["positive_index"], pair["negative_index"]
            assert (
                frame.iloc[positive].category == frame.iloc[negative].category == SELECTOR.FLAMMABLE
            )
            assert int(frame.iloc[positive].label) == 1
            assert int(frame.iloc[negative].label) == 0
            assert int(oof["fold_ids"][positive]) != fold
            assert int(oof["fold_ids"][negative]) != fold
            assert bool(safe[pair["positive_id"]]) is True
            assert bool(safe[pair["negative_id"]]) is True


def test_exp560_realized_manifest_obeys_occurrence_and_reuse_caps() -> None:
    for fold in (0, 3):
        manifest, audit = _frozen_rows(fold)
        positives = Counter(row["positive_index"] for row in manifest)
        negatives = Counter(row["negative_index"] for row in manifest)
        batches = Counter(row["batch_index"] for row in manifest)
        assert max(positives.values()) <= SELECTOR.MAX_PAIRS_PER_POSITIVE
        assert max(negatives.values()) <= SELECTOR.MAX_NEGATIVE_REUSE
        assert max(batches.values()) == 1
        assert audit["max_pairs_per_positive"] == 4
        assert audit["max_negative_reuse"] == 1
        assert all(0 <= row["positive_slot"] < 4 for row in manifest)
        assert all(0 <= row["negative_slot"] < 4 for row in manifest)


def test_exp560_donor_target_orientation_fails_closed_on_relabel(monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "id": ["positive", "negative", "filler-a", "filler-b"],
            "category": [SELECTOR.FLAMMABLE] * 4,
            "label": [0, 0, 0, 0],
        }
    )
    membership = pd.DataFrame(
        {
            "id": frame.id,
            "safe_for_selection": [True] * 4,
            "cue_mask": ["cue"] * 4,
            "connected_component": ["a", "b", "c", "d"],
        }
    )
    pairs = pd.DataFrame(
        {
            "pair_index": [0],
            "positive_id": ["positive"],
            "negative_id": ["negative"],
            "positive_cue_mask": ["cue"],
            "negative_cue_mask": ["cue"],
            "selection_scope": ["same_cue_mask"],
            "cosine_similarity": [0.9],
        }
    )
    audit = {"folds": {"0": {"file_sha256": "fixture"}}}
    monkeypatch.setattr(SELECTOR, "load_frozen_exp530", lambda fold: (membership, pairs, audit))
    with pytest.raises(ValueError, match="donor labels"):
        SELECTOR.build_pair_plan(
            frame,
            np.asarray([1, 1, 1, 1], dtype=np.int8),
            [0, 1, 2, 3],
            holdout_fold=0,
        )


def test_exp560_ordered_hash_is_diagnostic_not_a_strict_membership_gate() -> None:
    trainer = (EXPERIMENT / "trainer.py").read_text(encoding="utf-8")
    assert '"ordered_hash_is_membership_gate": False' in trainer
    strict_block = trainer.split("STRICT_AUDIT_KEYS = (", 1)[1].split(")", 1)[0]
    assert "ordered_records_sha256" not in strict_block
    for fold in (0, 3):
        _, audit = _frozen_rows(fold)
        assert audit["ordered_hash_is_membership_gate"] is False
        assert "ordered_records_sha256_diagnostic_only" in audit


def test_exp560_null_control_is_exact_and_keeps_route400_weights() -> None:
    analysis = EXPERIMENT / "analysis/null_control"
    report = json.loads((analysis / "null_screen_control.json").read_text())
    arrays = np.load(analysis / "null_screen_control.npz", allow_pickle=False)
    assert report["null_control_passed"] is True
    assert report["evaluated_folds"] == [0, 3]
    assert report["weights"][SELECTOR.FLAMMABLE]["qwen35"] == 0.75
    assert np.array_equal(
        arrays["baseline_nested_predictions"], arrays["candidate_screen_nested_predictions"]
    )


def test_exp560_audit_cli_is_dependency_light() -> None:
    result = subprocess.run(
        [sys.executable, str(EXPERIMENT / "local_run.py"), "audit", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--data" in result.stdout
