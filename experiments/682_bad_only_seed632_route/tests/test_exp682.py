from __future__ import annotations

import ast
import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from contract import load_json, load_spec, sha256_file, verify_self_hash


def load_module(name: str, filename: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_spec_and_materialized_replay() -> None:
    spec = load_spec()
    assert spec["scientific_factor"].startswith("replace the frozen Qwen3.5")
    assert spec["candidate_route"] == {
        "БАД": "qwen35_seed632_logit",
        "Легковоспламеняющиеся": "qwen35_original_logit",
    }
    report = load_json(HERE / "results/replay_evaluation.json")
    verify_self_hash(report, "report_sha256")
    assert report["validation_signal_accepted"] is True
    assert report["gpu_launch_allowed"] is False
    assert all(report["gates"].values())
    metrics = report["metrics"]
    assert metrics["baseline_macro_f1"] == pytest.approx(0.9104830822677643)
    assert metrics["candidate_macro_f1"] == pytest.approx(0.9133075471136818)
    assert metrics["macro_delta"] == pytest.approx(0.0028244648459174737)
    assert metrics["winning_folds"] == 5
    assert list(metrics["folds"]) == ["0", "1", "2", "3", "4"]
    assert all(value["delta"] > 0 for value in metrics["folds"].values())
    assert metrics["categories"]["БАД"]["delta"] == pytest.approx(0.005648929691834836)
    assert metrics["categories"]["Легковоспламеняющиеся"]["delta"] == 0.0
    assert metrics["corrected"] == 72 and metrics["regressed"] == 19
    assert metrics["false_negatives_by_category"]["БАД"] == {
        "baseline": 236,
        "candidate": 201,
        "delta": -35,
    }
    assert metrics["false_negatives_by_category"]["Легковоспламеняющиеся"] == {
        "baseline": 20,
        "candidate": 20,
        "delta": 0,
    }
    bootstrap = metrics["component_bootstrap"]
    assert bootstrap["probability_delta_positive"] == 1.0
    assert bootstrap["delta_ci95"] == pytest.approx(
        [0.0014630942074745858, 0.004480938699248996]
    )
    identity = report["identity_checks"]
    assert all(value is True for key, value in identity.items() if key.endswith("array_equal"))
    assert identity["flammable_original_logit_bytes_sha256"] == identity[
        "flammable_routed_logit_bytes_sha256"
    ]


def test_replay_inputs_are_exactly_bound() -> None:
    report = load_json(HERE / "results/replay_evaluation.json")
    spec = load_spec()
    assert report["input_sha256"]["frozen_spec"] == sha256_file(HERE / "frozen_spec.json")
    assert report["input_sha256"]["replay_bundle"] == spec["frozen_inputs"][
        "replay_bundle_sha256"
    ]
    assert report["input_sha256"]["replay_contract"] == spec["frozen_inputs"][
        "replay_contract_file_sha256"
    ]
    assert report["input_sha256"]["accepted_632_report"] == spec["frozen_inputs"][
        "accepted_632_report_sha256"
    ]


def test_full_refit_and_launch_gates_bind_exact_development_runtime() -> None:
    refit = load_json(HERE / "results/full_refit_gate.json")
    verify_self_hash(refit, "gate_sha256")
    assert refit["stage"] == "post_policy_cpu_runtime_materialization"
    assert refit["development_rows"] == 11118
    assert refit["sealed_rows_read"] == 0 and refit["sealed_labels_read"] == 0
    assert refit["runtime_materialized"] is True
    assert refit["selected_id_multiset_sha256"].startswith("f4734df6")
    assert refit["optimizer_updates"] == 383
    assert refit["gpu_jobs_launched"] == 0
    launch = load_json(HERE / "results/launch_gate.json")
    verify_self_hash(launch, "gate_sha256")
    assert launch["validation_signal_accepted"] is True
    assert all(launch["gates"].values())
    assert launch["gpu_launch_allowed"] is True
    assert launch["preset_generation_allowed"] is True
    assert launch["package_allowed"] is False
    assert launch["public_submission_allowed"] is False
    assert launch["input_sha256"]["full_refit_gate"] == sha256_file(
        HERE / "results/full_refit_gate.json"
    )


def test_train_runtime_and_preset_design_use_exact_contract() -> None:
    train = load_module("_exp682_train", "train_full.py")
    assert sha256_file(HERE / "train_full.py") == load_spec()["full_refit"][
        "full_train_runner_sha256"
    ]
    runtime_dir = HERE / ".local/full_runtime_v2"
    rows, selected, audit = train._read_runtime(runtime_dir)
    assert len(rows) == 11118 and len(selected) == 6116
    assert audit["optimizer_updates"] == 383
    assert audit["sealed_rows_read"] == 0
    preset = load_module("_exp682_preset", "build_preset.py")
    args = Namespace(
        spec=HERE / "frozen_spec.json",
        launch_gate=HERE / "results/launch_gate.json",
        experiment_dir=HERE,
        upstream_dir=ROOT / "experiments/623_semantic_v3_multitask_span_head",
        parent=ROOT / "research/qwen3vl_lora_holdout.py",
        runtime_dir=runtime_dir,
        vendor_dir=HERE / ".local/vendor",
        model_mrid="huggingface-proxy/Qwen/Qwen3.5-4B/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        flavor="placeholder",
        image="placeholder",
        region="placeholder",
        time_limit="3h",
        output=HERE / ".local/compute/preset.yml",
    )
    rendered = preset.render_preset(args)
    assert "train_full.py" in rendered
    assert "--runtime-dir" in rendered
    assert "FULL_TRAIN" not in rendered


def test_policy_precedes_source_access_and_forbids_old_selector() -> None:
    builder = load_module("_exp682_runtime_builder", "build_full_runtime.py")
    policy = builder.verify_policy(
        policy_path=HERE / "post_validation_refit_policy.json",
        spec_path=HERE / "frozen_spec.json",
    )
    assert policy["selected_id_multiset_sha256"].startswith("f4734df6")
    assert policy["optimizer_updates"] == 383
    assert policy["old_12971_four_head_oof_allowed"] is False
    source = load_json(HERE / "runtime_source_manifest.json")
    verify_self_hash(source, "manifest_sha256")
    assert source["development_rows"] == 11118 and source["sealed_rows"] == 0


def test_champion_patch_preserves_original_pass_and_overwrites_only_bad() -> None:
    package = load_module("_exp682_package", "build_submission.py")
    champion_run = ROOT / "experiments/140_dual_lora_fusion/submission/run.py"
    assert sha256_file(champion_run) == load_spec()["frozen_inputs"]["champion_run_sha256"]
    patched = package.patch_champion_run(champion_run.read_text(encoding="utf-8"))
    ast.parse(patched)
    assert "adapter_qwen35_bad_seed632" in patched
    assert patched.index("qwen35_scores = compute_lora_scores") < patched.index(
        'bad_mask = frame["category"].to_numpy() == "БАД"'
    )
    assert 'qwen35_scores[bad_mask] = bad_scores' in patched
    assert "auxiliary_head" not in patched
    assert patched.count("QWEN35_BAD_ADAPTER_PATH") >= 3


def test_current_launch_gate_cannot_be_used_as_package_gate() -> None:
    package = load_module("_exp682_package_closed", "build_submission.py")
    args = Namespace(
        package_gate=HERE / "results/launch_gate.json",
        adapter_contract=HERE / "absent-adapter-contract.json",
        bad_adapter=HERE / "absent-adapter.zip",
        runtime_report=HERE / "absent-runtime.json",
    )
    with pytest.raises(ValueError, match="package gate schema mismatch"):
        package._verify_package_inputs(args, load_spec())
