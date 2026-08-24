from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from contract import (
    add_self_hash,
    load_json,
    load_spec,
    sha256_file,
    verify_self_hash,
)


def build_gate(
    *,
    spec_path: Path,
    replay_path: Path,
    refit_gate_path: Path,
    policy_path: Path,
    source_manifest_path: Path,
    runtime_audit_path: Path,
    runtime_builder_path: Path,
    trainer_path: Path,
) -> dict[str, Any]:
    spec = load_spec(spec_path)
    replay = load_json(replay_path)
    verify_self_hash(replay, "report_sha256")
    refit = load_json(refit_gate_path)
    verify_self_hash(refit, "gate_sha256")
    policy = load_json(policy_path)
    verify_self_hash(policy, "policy_sha256")
    source_manifest = load_json(source_manifest_path)
    verify_self_hash(source_manifest, "manifest_sha256")
    runtime = load_json(runtime_audit_path)
    verify_self_hash(runtime, "runtime_sha256")
    if replay.get("experiment_id") != "682" or replay.get("stage") != "five_fold_materialized_replay":
        raise ValueError("replay identity mismatch")
    if replay.get("input_sha256", {}).get("frozen_spec") != sha256_file(spec_path):
        raise ValueError("replay used another frozen spec")
    if refit.get("experiment_id") != "682" or refit.get("input_sha256", {}).get(
        "frozen_spec"
    ) != sha256_file(spec_path):
        raise ValueError("full-refit gate identity mismatch")
    replay_ok = replay.get("validation_signal_accepted") is True and all(
        replay.get("gates", {}).values()
    )
    exact_refit_inputs = {
        "frozen_spec": sha256_file(spec_path),
        "post_validation_policy": sha256_file(policy_path),
        "runtime_source_manifest": sha256_file(source_manifest_path),
        "full_runtime_builder": sha256_file(runtime_builder_path),
        "full_train_runner": sha256_file(trainer_path),
        "materialized_runtime_audit": sha256_file(runtime_audit_path),
    }
    if refit.get("input_sha256") != exact_refit_inputs:
        raise ValueError("full-refit gate does not bind the exact local inputs")
    frozen_refit = spec["full_refit"]
    spec_bound = (
        exact_refit_inputs["runtime_source_manifest"]
        == frozen_refit["runtime_source_manifest_file_sha256"]
        and exact_refit_inputs["full_runtime_builder"]
        == frozen_refit["full_runtime_builder_sha256"]
        and exact_refit_inputs["full_train_runner"] == frozen_refit["full_train_runner_sha256"]
        and refit.get("selected_id_multiset_sha256")
        == frozen_refit["selected_id_multiset_sha256"]
        and refit.get("optimizer_updates") == frozen_refit["optimizer_updates"]
    )
    refit_ok = (
        spec["full_refit"]["enabled"] is True
        and refit.get("policy_accepted") is True
        and refit.get("runtime_materialized") is True
        and refit.get("gpu_launch_allowed") is True
        and runtime.get("decision") == "GO"
        and runtime.get("sealed_rows_read") == 0
        and runtime.get("sealed_labels_read") == 0
        and runtime.get("sealed_rows_written") == 0
        and policy.get("public_feedback_used") is False
        and spec_bound
    )
    gates = {
        "five_fold_replay_accepted": replay_ok,
        "no_public_or_sealed_selection": replay.get("public_used_for_selection") is False
        and replay.get("sealed_rows_loaded") == 0,
        "flammable_route_byte_identical": all(
            replay.get("identity_checks", {}).get(key) is True
            for key in (
                "flammable_logit_array_equal",
                "flammable_probability_array_equal",
                "flammable_final_score_array_equal",
                "flammable_final_prediction_array_equal",
            )
        ),
        "accepted_post_validation_refit_policy": refit_ok,
        "exact_refit_sources_bound": spec_bound,
    }
    allowed = all(gates.values())
    return add_self_hash(
        {
            "schema_version": "exp682_launch_gate_v1",
            "experiment_id": "682",
            "input_sha256": {
                "frozen_spec": sha256_file(spec_path),
                "replay_report": sha256_file(replay_path),
                "full_refit_gate": sha256_file(refit_gate_path),
            },
            "gates": gates,
            "validation_signal_accepted": replay_ok,
            "gpu_launch_allowed": allowed,
            "preset_generation_allowed": allowed,
            "package_allowed": False,
            "public_submission_allowed": False,
            "decision": "GO_GPU" if allowed else "BLOCKED_NO_PROVABLE_FULL_REFIT",
        },
        "gate_sha256",
    )


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=here / "frozen_spec.json")
    parser.add_argument("--replay", type=Path, default=here / "results/replay_evaluation.json")
    parser.add_argument(
        "--full-refit-gate", type=Path, default=here / "results/full_refit_gate.json"
    )
    parser.add_argument("--policy", type=Path, default=here / "post_validation_refit_policy.json")
    parser.add_argument(
        "--source-manifest", type=Path, default=here / "runtime_source_manifest.json"
    )
    parser.add_argument(
        "--runtime-audit", type=Path, default=here / ".local/full_runtime_v2/runtime_audit.json"
    )
    parser.add_argument("--runtime-builder", type=Path, default=here / "build_full_runtime.py")
    parser.add_argument("--trainer", type=Path, default=here / "train_full.py")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_gate(
        spec_path=args.spec,
        replay_path=args.replay,
        refit_gate_path=args.full_refit_gate,
        policy_path=args.policy,
        source_manifest_path=args.source_manifest,
        runtime_audit_path=args.runtime_audit,
        runtime_builder_path=args.runtime_builder,
        trainer_path=args.trainer,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite launch gate")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0 if report["gpu_launch_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
