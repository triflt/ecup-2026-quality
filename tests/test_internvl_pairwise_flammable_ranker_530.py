from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/530_internvl_pairwise_flammable_ranker"


def _load(name: str, filename: str):
    contract_spec = importlib.util.spec_from_file_location(
        f"{name}_contract", EXPERIMENT / "contract.py"
    )
    assert contract_spec and contract_spec.loader
    contract = importlib.util.module_from_spec(contract_spec)
    contract_spec.loader.exec_module(contract)
    previous = sys.modules.get("contract")
    sys.modules["contract"] = contract
    try:
        spec = importlib.util.spec_from_file_location(name, EXPERIMENT / filename)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("contract", None)
        else:
            sys.modules["contract"] = previous


builder = _load("exp530_pair_builder", "build_pair_manifests.py")
objective = _load("exp530_pairwise_objective", "pairwise_objective.py")
evaluator = _load("exp530_evaluator", "evaluate_screen.py")
image_source = _load("exp530_image_source", "image_source.py")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png(mode: str = "L") -> bytes:
    stream = io.BytesIO()
    Image.new(mode, (12, 9), 127).save(stream, format="PNG")
    return stream.getvalue()


def _synthetic_selector() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for index in range(5):
        records.append(
            {
                "id": f"p{index}",
                "fold": 1,
                "safe_for_selection": True,
                "label": 1,
                "cue_mask": "100001" if index < 4 else "111111",
                "text": f"fuel refill family {index}",
            }
        )
    for index in range(8):
        records.append(
            {
                "id": f"n{index}",
                "fold": 2,
                "safe_for_selection": True,
                "label": 0,
                "cue_mask": "100001" if index < 6 else "000000",
                "text": f"fuel refill family negative {index}",
            }
        )
    records.extend(
        [
            {
                "id": "outer-positive",
                "fold": 0,
                "safe_for_selection": True,
                "label": 1,
                "cue_mask": "100001",
                "text": "must never enter donor pairs",
            },
            {
                "id": "unsafe-negative",
                "fold": 4,
                "safe_for_selection": False,
                "label": 0,
                "cue_mask": "100001",
                "text": "must never enter donor pairs",
            },
        ]
    )
    return pd.DataFrame(records)


def test_pair_builder_is_deterministic_donor_only_and_reuse_capped() -> None:
    selector = _synthetic_selector()
    first = builder.build_pairs(selector, outer_fold=0)
    second = builder.build_pairs(
        selector.sample(frac=1, random_state=17).reset_index(drop=True), outer_fold=0
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 5 * 4
    assert (first.positive_id.value_counts() == 4).all()
    assert first.negative_id.value_counts().max() <= 8
    assert "outer-positive" not in set(first.positive_id) | set(first.negative_id)
    assert "unsafe-negative" not in set(first.positive_id) | set(first.negative_id)
    assert set(first.selection_scope) == {"same_cue_mask", "global_fallback"}


def test_cyclic_batches_reach_exactly_96_locked_updates() -> None:
    batches = list(
        objective.cyclic_index_batches(
            length=336,
            batch_size=4,
            batches=96 * 4,
            seed=42,
        )
    )
    repeated = list(
        objective.cyclic_index_batches(
            length=336,
            batch_size=4,
            batches=96 * 4,
            seed=42,
        )
    )

    assert batches == repeated
    assert len(batches) == 384
    assert all(len(batch) == 4 and len(set(batch)) == 4 for batch in batches)


def test_manifest_images_download_in_parallel_decode_rgb_and_reuse_cache(
    tmp_path: Path,
) -> None:
    manifest = pd.DataFrame(
        {
            "id": ["a", "b", "unused"],
            "image_url": ["memory://a", "memory://b", "memory://unused"],
        }
    )
    calls: list[str] = []

    def opener(url: str, *, timeout: int):
        assert timeout == 60
        calls.append(url)
        return io.BytesIO(_png())

    store, report = image_source.prepare_manifest_images(
        manifest=manifest,
        expected_ids={"a", "b"},
        cache_dir=tmp_path / "cache",
        workers=2,
        opener=opener,
    )

    assert sorted(calls) == ["memory://a", "memory://b"]
    assert report["expected_images"] == 2
    assert report["download_requests"] == 2
    assert report["downloaded_images"] == 2
    assert report["decoded_rgb_images"] == 2
    assert report["expected_ids_sha256"] == report["prepared_ids_sha256"]
    assert report["failed_images"] == 0
    assert report["white_fallbacks"] == 0
    assert report["ready"] is True
    assert "memory://" not in json.dumps(report)
    assert store.rgb("a").mode == "RGB"

    def forbidden_opener(url: str, *, timeout: int):
        raise AssertionError(f"cache unexpectedly downloaded {url} at timeout {timeout}")

    _, cached_report = image_source.prepare_manifest_images(
        manifest=manifest,
        expected_ids={"a", "b"},
        cache_dir=tmp_path / "cache",
        workers=2,
        opener=forbidden_opener,
    )
    assert cached_report["cache_hits"] == 2
    assert cached_report["download_requests"] == 0


def test_manifest_image_retries_three_times_and_fails_without_fallback(
    tmp_path: Path,
) -> None:
    manifest = pd.DataFrame({"id": ["a"], "image_url": ["memory://a"]})
    attempts = 0

    def retry_opener(url: str, *, timeout: int):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError("transient")
        return io.BytesIO(_png("RGB"))

    _, report = image_source.prepare_manifest_images(
        manifest=manifest,
        expected_ids={"a"},
        cache_dir=tmp_path / "retry-cache",
        opener=retry_opener,
    )
    assert attempts == 3
    assert report["download_attempts"] == 3
    assert report["ready"] is True

    def corrupt_opener(url: str, *, timeout: int):
        return io.BytesIO(b"not-an-image")

    with pytest.raises(image_source.ImagePreparationError) as captured:
        image_source.prepare_manifest_images(
            manifest=manifest,
            expected_ids={"a"},
            cache_dir=tmp_path / "broken-cache",
            opener=corrupt_opener,
        )
    failure = captured.value.report
    assert failure["download_attempts"] == 3
    assert failure["failed_images"] == 1
    assert failure["decoded_rgb_images"] == 0
    assert failure["white_fallbacks"] == 0
    assert not list((tmp_path / "broken-cache").glob("*.jpg"))


def test_pairwise_logistic_loss_prefers_positive_margin_when_torch_available() -> None:
    if objective.torch is None:
        good = np.logaddexp(0.0, -np.asarray([2.0, 2.0])).mean()
        bad = np.logaddexp(0.0, -np.asarray([-2.0, -2.0])).mean()
        assert good < bad
        assert (
            "F.softplus(-(positive_scores - negative_scores)).mean()"
            in (EXPERIMENT / "pairwise_objective.py").read_text()
        )
        return
    torch = objective.torch
    positive = torch.tensor([2.0, 1.0], requires_grad=True)
    negative = torch.tensor([0.0, -1.0], requires_grad=True)
    good = objective.pairwise_logistic_loss(positive, negative)
    bad = objective.pairwise_logistic_loss(negative, positive)
    assert good < bad
    good.backward()
    assert (positive.grad < 0).all()
    assert (negative.grad > 0).all()


def test_donor_cdf_uses_right_closed_empirical_rank_and_rejects_nonfinite() -> None:
    ranks = evaluator.donor_cdf_rank(
        np.asarray([0.0, 1.0, 1.0, 3.0]), np.asarray([-1.0, 1.0, 2.0, 4.0])
    )
    np.testing.assert_allclose(ranks, [0.0, 0.75, 0.75, 1.0])
    with pytest.raises(ValueError, match="finite"):
        evaluator.donor_cdf_rank(np.asarray([0.0, np.nan]), np.asarray([1.0]))


def test_fold_score_loader_enforces_donor_outer_separation(tmp_path: Path) -> None:
    membership = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "fold": [0, 0, 1, 3],
            "safe_for_selection": [True, True, True, True],
        }
    )
    path = tmp_path / "scores.csv"
    pd.DataFrame(
        {
            "id": ["c", "d", "a", "b"],
            "fold": [1, 3, 0, 0],
            "row_role": ["donor", "donor", "outer", "outer"],
            "internvl_score": [0.0, 2.0, 1.0, 3.0],
        }
    ).to_csv(path, index=False)

    ids, ranks = evaluator.load_fold_rank(fold=0, predictions_path=path, membership=membership)
    assert ids.tolist() == ["a", "b"]
    np.testing.assert_allclose(ranks, [0.5, 1.0])
    broken = pd.read_csv(path)
    broken.loc[0, "row_role"] = "outer"
    broken.to_csv(path, index=False)
    with pytest.raises(ValueError, match="membership mismatch"):
        evaluator.load_fold_rank(fold=0, predictions_path=path, membership=membership)


def test_training_report_and_source_lock_exact_recipe(tmp_path: Path) -> None:
    audit = json.loads((EXPERIMENT / "analysis/pair_manifest_audit.json").read_text())
    report = {
        "experiment_id": "530",
        "status": "screen_fold_complete",
        "holdout_fold": 0,
        "optimizer_updates": 96,
        "seed": 42,
        "image_size": 448,
        "image_tiles": 1,
        "image_source": "manifest",
        "lora_rank": 16,
        "language_attention_only": True,
        "vision_frozen": True,
        "atomic_digit_tokens": True,
        "loss": "softplus(-(score_positive-score_negative))",
        "pair_manifest_sha256": audit["folds"]["0"]["file_sha256"],
        "selector_membership_sha256": audit["output_sha256"]["selector_membership"],
    }
    image_report = {
        "image_source": "manifest",
        "expected_images": 10,
        "cache_hits": 0,
        "download_requests": 10,
        "downloaded_images": 10,
        "download_attempts": 10,
        "configured_attempts_per_image": 3,
        "decoded_rgb_images": 10,
        "expected_ids_sha256": "a" * 64,
        "prepared_ids_sha256": "a" * 64,
        "failed_images": 0,
        "failure_error_types": {},
        "white_fallbacks": 0,
        "exactly_one_image_per_expected_id": True,
        "ready": True,
    }
    image_report_path = tmp_path / "image_download_report.json"
    image_report_path.write_text(json.dumps(image_report))
    report["image_download_report"] = image_report
    report["image_download_report_sha256"] = _sha256(image_report_path)
    path = tmp_path / "training_report.json"
    path.write_text(json.dumps(report))
    validated = evaluator.validate_training_report(path, fold=0, pair_audit=audit)
    assert validated["image_source"] == "manifest"
    report["optimizer_updates"] = 95
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="optimizer_updates"):
        evaluator.validate_training_report(path, fold=0, pair_audit=audit)

    source = (EXPERIMENT / "train_screen.py").read_text()
    assert 'VERIFIED_NATIVE_LOADER = ROOT / "research/internvl_native_backward_smoke.py"' in source
    assert 'name.startswith("language_model.")' in source
    assert "use_rslora=True" in source
    assert "row_field(row, 'name')" in source
    assert 'selector.set_index("id", drop=False)' in source
    assert "selector.loc[score_ids].reset_index(drop=True)" in source
    assert "OPTIMIZER_UPDATES * GRADIENT_ACCUMULATION" in source
    assert 'choices=("manifest",)' in source
    assert "--images-zip" not in source


def test_experiment_card_is_locked_and_not_launched() -> None:
    config = tomllib.loads((EXPERIMENT / "experiment.toml").read_text())
    metrics = json.loads((EXPERIMENT / "results/metrics.json").read_text())

    assert config["validation"]["screen_folds"] == [0, 3]
    assert config["training"]["optimizer_updates"] == 96
    assert config["training"]["lora_rank"] == 16
    assert config["training"]["image_size"] == 448
    assert config["training"]["image_source"] == "frozen_selector_manifest"
    assert config["training"]["image_download_attempts"] == 3
    assert config["training"]["image_fallback"] is False
    assert config["execution"]["allowed_image_sources"] == ["manifest"]
    assert config["inference"]["replacement_weight"] == pytest.approx(0.10)
    assert config["execution"]["launch_authorized"] is True
    assert metrics["launched"] is True
    assert metrics["screen_passed"] is False
    assert metrics["decision"] == "REJECT"

    null_control = json.loads((EXPERIMENT / "analysis/null_screen_control_audit.json").read_text())
    assert null_control["null_control_passed"] is True
    assert null_control["mean_screen_delta_macro_f1"] == 0.0
    assert null_control["bad_changed_predictions"] == 0
    assert [fold["baseline_macro_f1"] for fold in null_control["folds"]] == [
        pytest.approx(0.9312591699999442),
        pytest.approx(0.9166759475813936),
    ]
