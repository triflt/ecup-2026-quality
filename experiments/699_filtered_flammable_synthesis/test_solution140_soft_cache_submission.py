from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_solution140_soft_cache_submission as builder
import soft_cache_runtime as runtime


def load_evaluator():
    path = ROOT / "evaluate_soft_cache_exploratory.py"
    spec = importlib.util.spec_from_file_location("exp699_soft_cache_evaluator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_run() -> str:
    return '''from src.model import compose_text, fingerprint, normalize

def main():
    frame = make_frame()
    predictions = np.zeros(len(frame), dtype=np.int8)
    for category, config in LORA_FUSION.items():
        mask = frame["category"].to_numpy() == category
        combined = unchanged_fusion(mask, config)
        predictions[mask] = (combined >= config["threshold"]).astype(np.int8)
    # The organizer confirmed that test labels come from the same ambiguous
    apply_unchanged_priors(predictions)
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_matches_frozen_evaluator_for_fusion_and_apply() -> None:
    evaluator = load_evaluator()
    channels = {
        "exact_text": [[0, 1, 2, 3], [], []],
        "normalized_text": [[0, 2, 1], [], []],
        "tfidf": [[1, 0, 2], [], []],
        "bm25": [[2, 0, 1], [], []],
        "image_exact": [[], [], []],
        "image_near": [[], [], []],
    }
    weights = {
        "exact_text": 2.0,
        "normalized_text": 1.5,
        "tfidf": 1.0,
        "bm25": 1.0,
        "image_exact": 2.0,
        "image_near": 1.0,
    }
    expected_fused, expected_evidence = evaluator.fuse_with_evidence(
        channels, weights, 3, runtime.CANDIDATE_TOP_K, runtime.RRF_OFFSET
    )
    actual_fused, actual_evidence = runtime.fuse_with_evidence(
        channels,
        weights,
        3,
        runtime.CANDIDATE_TOP_K,
        runtime.RRF_OFFSET,
        np.arange(4, dtype=np.int32),
    )
    assert actual_fused == expected_fused
    assert actual_evidence == expected_evidence

    labels = np.asarray([1, 1, 1, 0], dtype=np.int8)
    categories = np.asarray([runtime.FLAMMABLE, "БАД", "БАД"])
    baseline_scores = np.asarray([0.94, 0.25, 0.75], dtype=np.float64)
    baseline_predictions = np.asarray([0, 0, 1], dtype=np.int8)
    expected_scores, expected_predictions, expected_audit = evaluator.apply_soft_cache(
        labels,
        categories,
        baseline_scores,
        baseline_predictions,
        expected_fused,
        expected_evidence,
    )
    actual_scores, actual_predictions, actual_audit = runtime.apply_fused_cache(
        labels,
        categories,
        baseline_scores,
        baseline_predictions,
        actual_fused,
        actual_evidence,
    )
    np.testing.assert_array_equal(actual_predictions, expected_predictions)
    np.testing.assert_allclose(actual_scores, expected_scores, rtol=0.0, atol=0.0)
    assert len(actual_audit) == len(expected_audit) == 1
    for key in (
        "donors",
        "donor_positive_rate",
        "donor_agreement",
        "evidence_channels",
        "baseline_score",
        "candidate_score",
        "baseline_prediction",
        "candidate_prediction",
    ):
        assert actual_audit[0][key] == expected_audit[0][key]


def test_bad_rows_are_identity_and_only_flammable_can_change() -> None:
    labels = np.asarray([1, 1, 1], dtype=np.int8)
    categories = np.asarray(["БАД", runtime.FLAMMABLE])
    baseline_scores = np.asarray([0.1, 0.94], dtype=np.float64)
    baseline_predictions = np.asarray([0, 0], dtype=np.int8)
    fused = [[0, 1, 2], [0, 1, 2]]
    evidence = [
        {
            donor: {"rrf_score": 1.0, "channels": ["exact_text"]}
            for donor in row
        }
        for row in fused
    ]
    scores, predictions, audit = runtime.apply_fused_cache(
        labels,
        categories,
        baseline_scores,
        baseline_predictions,
        fused,
        evidence,
    )
    assert scores[0] == baseline_scores[0]
    assert predictions[0] == baseline_predictions[0]
    assert scores[1] != baseline_scores[1]
    assert [row["row_index"] for row in audit] == [1]


def test_fusion_ties_use_frozen_global_donor_order() -> None:
    channels = {
        "exact_text": [[0, 1]],
        "normalized_text": [[1, 0]],
    }
    fused, evidence = runtime.fuse_with_evidence(
        channels,
        {"exact_text": 1.0, "normalized_text": 1.0},
        1,
        2,
        runtime.RRF_OFFSET,
        np.asarray([20, 10], dtype=np.int32),
    )
    assert evidence[0][0]["rrf_score"] == evidence[0][1]["rrf_score"]
    assert fused == [[1, 0]]


def test_source_substitution_is_exact_and_pre_prior() -> None:
    original = source_run()
    patched = builder.patch_solution140_run(original)
    assert patched.count("from soft_cache_runtime import apply_runtime_cache") == 1
    assert patched.count("apply_runtime_cache(") == 1
    assert "combined = unchanged_fusion(mask, config)" in patched
    assert 'combined >= config["threshold"]' in patched
    assert patched.index("apply_runtime_cache(") < patched.index(
        "# The organizer confirmed"
    )
    assert patched.index("# The organizer confirmed") < patched.index(
        "apply_unchanged_priors"
    )

    with pytest.raises(ValueError, match="anchor mismatch"):
        builder.patch_solution140_run(original.replace("    predictions =", "    prediction ="))


def test_deterministic_package_and_only_declared_source_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.py").write_text(source_run(), encoding="utf-8")
    (source / "metadata.json").write_text('{"frozen":true}\n', encoding="utf-8")
    (source / "nested").mkdir()
    (source / "nested" / "member.bin").write_bytes(b"unchanged\x00member")
    cache_file = tmp_path / "cache.joblib"
    cache_file.write_bytes(b"deterministic-cache-fixture")
    fake_source_sha = sha256(source / "run.py")
    monkeypatch.setattr(builder, "EXPECTED_SOLUTION140_RUN_SHA256", fake_source_sha)

    outputs = []
    for suffix in ("a", "b"):
        destination = tmp_path / f"candidate-{suffix}"
        archive = tmp_path / f"candidate-{suffix}.zip"
        result = builder.build_package_from_verified_inputs(
            source=source,
            runtime_module=ROOT / "soft_cache_runtime.py",
            cache_file=cache_file,
            destination=destination,
            archive=archive,
            bindings={"frozen": "binding"},
        )
        outputs.append((destination, archive, result))

    assert outputs[0][2] == outputs[1][2]
    assert outputs[0][1].read_bytes() == outputs[1][1].read_bytes()
    assert outputs[0][2]["status"] == "ExploratoryReadyNotSubmitted"
    assert outputs[0][2]["donor_count"] == 4716
    for destination, archive, _ in outputs:
        assert (destination / "metadata.json").read_bytes() == (
            source / "metadata.json"
        ).read_bytes()
        assert (destination / "nested" / "member.bin").read_bytes() == (
            source / "nested" / "member.bin"
        ).read_bytes()
        manifest = json.loads(
            (destination / "exp699_candidate_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        declared_self = manifest.pop("self_sha256")
        assert declared_self == builder.canonical_sha256(manifest)
        with zipfile.ZipFile(archive) as package:
            assert all(
                member.date_time == builder.FIXED_ZIP_TIMESTAMP
                for member in package.infolist()
            )
            assert set(package.namelist()) == {
                "exp699_candidate_manifest.json",
                "metadata.json",
                "nested/member.bin",
                "run.py",
                "soft_cache.joblib",
                "soft_cache_runtime.py",
            }


def test_runtime_and_builder_are_bound_to_exact_donor_count() -> None:
    assert runtime.EXPECTED_DONORS == builder.EXPECTED_DONORS == 4716


def test_donor_alignment_excludes_every_non_flammable_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_paths: dict[int, Path] = {}
    component_rows = []
    flammable_ids = {"row-1", "row-4"}
    for fold in builder.FOLDS:
        category = runtime.FLAMMABLE if f"row-{fold}" in flammable_ids else "БАД"
        row = {
            "id": f"row-{fold}",
            "global_index": fold,
            "fold": fold,
            "category": category,
            "name": f"name {fold}",
            "description": f"description {fold}",
        }
        path = tmp_path / f"fold{fold}.jsonl"
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        runtime_paths[fold] = path
        component_rows.append(
            {
                "id": row["id"],
                "fold": fold,
                "category": category,
                "label": fold % 2,
            }
        )
    component_path = tmp_path / "components.jsonl"
    component_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in component_rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(builder, "EXPECTED_ROWS", 5)
    monkeypatch.setattr(builder, "EXPECTED_DONORS", 2)
    donors = builder.align_donors(runtime_paths, component_path)
    assert {row["id"] for row in donors} == flammable_ids
    assert [row["global_index"] for row in donors] == [1, 4]


def test_cache_serialization_is_byte_deterministic(tmp_path: Path) -> None:
    cache = {
        "schema": runtime.SCHEMA,
        "matrix": np.arange(12, dtype=np.float32).reshape(3, 4),
        "groups": {"a": [0, 2], "b": [1]},
    }
    first = tmp_path / "first.joblib"
    second = tmp_path / "second.joblib"
    runtime.dump_cache(cache, first)
    runtime.dump_cache(cache, second)
    assert first.read_bytes() == second.read_bytes()
