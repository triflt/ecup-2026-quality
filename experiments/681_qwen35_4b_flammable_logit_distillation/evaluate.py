from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT_PATH = ROOT / "experiments/680_qwen35_4b_flammable_only_hard_bce/evaluate.py"
VERIFIER_PATH = HERE / "verify_artifact.py"
CAUSAL_VERIFIER_PATH = ROOT / "experiments/680_qwen35_4b_flammable_only_hard_bce/verify_artifact.py"
FLAMMABLE = "Легковоспламеняющиеся"
RANKING_CONTROL_PREDICTION_SHA256 = {
    0: "5164909b7b56298f074e2d21a6e40f1ed53fd2d28d599609527715760f3468ae",
    1: "d7dff899b1af47858cd8855226f2688aad6756ed5472b175f0a7576891fe2ab1",
    2: "4ff669918b7bdcc9c1dddee458ce101f77d931c556171040e90794839a871211",
    3: "f73d1b06f33c74c40cce03afac1537134e8da251e5e101c14616459414d0ab6a",
    4: "a0649d7f25db993211c758aa8f9158ef71701eefa2dd2dc6938de7f94353a435",
}
# Only the preregistered screen controls exist. Full evaluation stays
# fail-closed until folds 1/2/4 are produced and their hashes are frozen.
CAUSAL_CONTROL_ARCHIVE_SHA256 = {
    0: "56a4475b0f7af880fa1766d8091df91c71e0263035ef08fdd1f7b96cd73f70e5",
    3: "fb90e45525e8f04cf78ac33955cd63caa7bb9b93ed484f8e085a85c3afcf402b",
}
CAUSAL_CONTROL_PREDICTION_SHA256 = {
    0: "3b9ce5f4178690efdb3a4c5079928f92141764dac7d1db77cd202c6636e54fbb",
    3: "13de94c324f02cdcbba8adc7826994d22e841bb9c1deb02a22a90fb94897c8de",
}


def load_parent():
    spec = importlib.util.spec_from_file_location("exp680_for_681", PARENT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen experiment-680 evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PARENT = load_parent()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = load_module(VERIFIER_PATH, "exp681_artifact_verifier")
CAUSAL_VERIFIER = load_module(CAUSAL_VERIFIER_PATH, "exp680_causal_verifier")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_prediction_paths(
    paths: list[Path],
    *,
    folds_scope: tuple[int, ...],
    expected_sha256: dict[int, str],
    label: str,
) -> list[Path]:
    by_fold: dict[int, Path] = {}
    for path in paths:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        folds = {int(row["fold"]) for row in rows}
        if len(folds) != 1:
            raise ValueError(f"{label} score file mixes folds")
        fold = folds.pop()
        if fold in by_fold or fold not in folds_scope:
            raise ValueError(f"{label} score fold mismatch or duplicate")
        if expected_sha256.get(fold) != sha256_file(path):
            raise ValueError(f"{label} score checksum is not frozen")
        by_fold[fold] = path
    if set(by_fold) != set(folds_scope):
        raise ValueError(f"{label} scores do not cover evaluation folds")
    return [by_fold[fold] for fold in folds_scope]


def stripped_training_payload(path: Path) -> str:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row.pop("teacher_score", None)
        rows.append(row)
    return json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def verify_causal_runtime_parity(candidate_runtime: Path, control_runtime: Path) -> None:
    candidate = json.loads((candidate_runtime / "runtime_audit.json").read_text(encoding="utf-8"))
    control = json.loads((control_runtime / "runtime_audit.json").read_text(encoding="utf-8"))
    parity_fields = (
        "outer_fold",
        "train_occurrences",
        "train_unique_ids",
        "validation_rows",
        "selected_multiset_sha256",
        "model_input_view_sha256",
        "grid_contract_sha256",
        "source_runtime_contract_sha256",
    )
    if any(candidate.get(field) != control.get(field) for field in parity_fields):
        raise ValueError("681 and causal 680 control runtime are not selection-identical")
    if candidate.get("output_sha256", {}).get("validation.jsonl") != control.get(
        "output_sha256", {}
    ).get("validation.jsonl"):
        raise ValueError("681 and causal 680 control validation differ")
    if stripped_training_payload(candidate_runtime / "train.jsonl") != stripped_training_payload(
        control_runtime / "train.jsonl"
    ):
        raise ValueError("681 changes training data beyond adding teacher_score")


def materialize_verified_candidates(
    args: argparse.Namespace,
    *,
    folds_scope: tuple[int, ...],
    directory: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    if len(args.candidate_archive) != len(folds_scope) or len(args.candidate_runtime_dir) != len(
        folds_scope
    ):
        raise ValueError("candidate archive/runtime count differs from stage")
    runtime_by_fold = {
        int(json.loads((path / "runtime_audit.json").read_text(encoding="utf-8"))["outer_fold"]): path
        for path in args.candidate_runtime_dir
    }
    if set(runtime_by_fold) != set(folds_scope):
        raise ValueError("candidate runtimes do not exactly cover evaluation folds")
    paths: dict[int, Path] = {}
    audits: dict[int, dict[str, Any]] = {}
    for archive in args.candidate_archive:
        with zipfile.ZipFile(archive) as handle:
            contract = json.loads(handle.read("output_contract.json"))
        fold = int(contract["outer_fold"])
        if fold in paths or fold not in folds_scope:
            raise ValueError("candidate archive fold mismatch or duplicate")
        audit = VERIFIER.verify(
            archive,
            fold=fold,
            runtime_dir=runtime_by_fold[fold],
            technical_smoke=False,
        )
        with zipfile.ZipFile(archive) as handle:
            payload = handle.read("predictions.jsonl")
        if hashlib.sha256(payload).hexdigest() != audit["predictions_sha256"]:
            raise ValueError("candidate predictions differ after artifact verification")
        path = directory / f"candidate_fold{fold}.jsonl"
        path.write_bytes(payload)
        paths[fold] = path
        audits[fold] = audit
    return [paths[fold] for fold in folds_scope], [audits[fold] for fold in folds_scope]


def materialize_verified_causal_controls(
    args: argparse.Namespace,
    *,
    folds_scope: tuple[int, ...],
    candidate_runtime_by_fold: dict[int, Path],
    directory: Path,
) -> tuple[list[Path], list[dict[str, Any]]]:
    if any(fold not in CAUSAL_CONTROL_ARCHIVE_SHA256 for fold in folds_scope):
        raise ValueError("full causal controls are not frozen; evaluation is closed")
    if len(args.causal_control_archive) != len(folds_scope) or len(
        args.causal_control_runtime_dir
    ) != len(folds_scope):
        raise ValueError("causal control archive/runtime count differs from stage")
    runtime_by_fold = {
        int(json.loads((path / "runtime_audit.json").read_text(encoding="utf-8"))["outer_fold"]): path
        for path in args.causal_control_runtime_dir
    }
    if set(runtime_by_fold) != set(folds_scope):
        raise ValueError("causal control runtimes do not cover evaluation folds")
    paths: dict[int, Path] = {}
    audits: dict[int, dict[str, Any]] = {}
    for archive in args.causal_control_archive:
        with zipfile.ZipFile(archive) as handle:
            contract = json.loads(handle.read("output_contract.json"))
        fold = int(contract["outer_fold"])
        if fold in paths or fold not in folds_scope:
            raise ValueError("causal control archive fold mismatch or duplicate")
        if sha256_file(archive) != CAUSAL_CONTROL_ARCHIVE_SHA256[fold]:
            raise ValueError("causal control archive checksum is not frozen")
        audit = CAUSAL_VERIFIER.verify(
            archive,
            fold=fold,
            runtime_validation=runtime_by_fold[fold] / "validation.jsonl",
        )
        if audit["predictions_sha256"] != CAUSAL_CONTROL_PREDICTION_SHA256[fold]:
            raise ValueError("causal control prediction checksum is not frozen")
        verify_causal_runtime_parity(candidate_runtime_by_fold[fold], runtime_by_fold[fold])
        with zipfile.ZipFile(archive) as handle:
            payload = handle.read("predictions.jsonl")
        path = directory / f"causal_control_fold{fold}.jsonl"
        path.write_bytes(payload)
        paths[fold] = path
        audits[fold] = audit
    return [paths[fold] for fold in folds_scope], [audits[fold] for fold in folds_scope]


def singleton_slice(args: argparse.Namespace) -> dict[str, Any]:
    folds_scope = PARENT.SCREEN_FOLDS if args.stage == "screen" else PARENT.FULL_FOLDS
    spec = PARENT.LEGACY.load_spec()
    PARENT.LEGACY.verify_source_recipe(spec)
    contract = PARENT.LEGACY.verify_replay_contract(
        path=args.replay_contract,
        bundle_path=args.bundle,
        registry_path=args.registry,
        spec=spec,
    )
    registry = PARENT.LEGACY._load_registry(args.registry, spec=spec)
    bundle = PARENT.LEGACY.load_replay_bundle(
        path=args.bundle, contract=contract, registry=registry
    )
    logits, _ = PARENT.load_scores(
        args.candidate_score, bundle=bundle, folds_scope=folds_scope
    )
    categories = bundle["categories"].astype(str)
    folds = bundle["folds"].astype(np.int8)
    components = bundle["semantic_components"].astype(str)
    labels = registry["label"].astype(np.int8).to_numpy()
    unique, counts = np.unique(components, return_counts=True)
    singleton_components = set(unique[counts == 1])
    mask = (
        np.isin(folds, folds_scope)
        & (categories == FLAMMABLE)
        & np.isin(components, list(singleton_components))
    )
    baseline_prob = PARENT.LEGACY.sigmoid(bundle["qwen35_original_logit"])
    candidate_prob = baseline_prob.copy()
    candidate_prob[categories == FLAMMABLE] = PARENT.LEGACY.sigmoid(
        logits[categories == FLAMMABLE]
    )
    baseline, _ = PARENT.LEGACY.route_predictions(
        bundle=bundle, qwen_probability=baseline_prob, route=spec["route"]
    )
    candidate, _ = PARENT.LEGACY.route_predictions(
        bundle=bundle, qwen_probability=candidate_prob, route=spec["route"]
    )
    corrected = int(np.sum(mask & (baseline != labels) & (candidate == labels)))
    regressed = int(np.sum(mask & (baseline == labels) & (candidate != labels)))
    before = PARENT.LEGACY.f1(labels[mask], baseline[mask])
    after = PARENT.LEGACY.f1(labels[mask], candidate[mask])
    return {
        "rows": int(mask.sum()),
        "baseline_flammable_f1": before,
        "candidate_flammable_f1": after,
        "flammable_f1_delta": after - before,
        "corrected": corrected,
        "regressed": regressed,
        "net_corrections": corrected - regressed,
    }


def routed_metrics_against_control(
    args: SimpleNamespace,
    *,
    folds_scope: tuple[int, ...],
) -> dict[str, Any]:
    """Compute production metrics with the supplied control logits as baseline.

    The parent evaluator uses its control logits only for AP and always routes
    the frozen 641 bundle logits as its production baseline.  That behavior is
    correct for experiment 680 but cannot measure the causal 681-vs-680 route.
    """

    spec = PARENT.LEGACY.load_spec()
    PARENT.LEGACY.verify_source_recipe(spec)
    contract = PARENT.LEGACY.verify_replay_contract(
        path=args.replay_contract,
        bundle_path=args.bundle,
        registry_path=args.registry,
        spec=spec,
    )
    registry = PARENT.LEGACY._load_registry(args.registry, spec=spec)
    bundle = PARENT.LEGACY.load_replay_bundle(
        path=args.bundle,
        contract=contract,
        registry=registry,
    )
    candidate_logits, _ = PARENT.load_scores(
        args.candidate_score,
        bundle=bundle,
        folds_scope=folds_scope,
    )
    control_logits, _ = PARENT.load_scores(
        args.control_score,
        bundle=bundle,
        folds_scope=folds_scope,
    )
    labels = registry["label"].astype(np.int8).to_numpy()
    categories = bundle["categories"].astype(str)
    folds = bundle["folds"].astype(np.int8)
    components = bundle["semantic_components"].astype(str)
    control_probability = PARENT.LEGACY.sigmoid(control_logits)
    candidate_probability = PARENT.LEGACY.sigmoid(candidate_logits)
    control_predictions, _ = PARENT.LEGACY.route_predictions(
        bundle=bundle,
        qwen_probability=control_probability,
        route=spec["route"],
    )
    candidate_predictions, _ = PARENT.LEGACY.route_predictions(
        bundle=bundle,
        qwen_probability=candidate_probability,
        route=spec["route"],
    )
    return PARENT.LEGACY._metrics(
        labels=labels,
        categories=categories,
        folds=folds,
        components=components,
        baseline=control_predictions,
        candidate=candidate_predictions,
        selected_folds=folds_scope,
        bootstrap=spec["bootstrap"] if args.stage == "full" else None,
    )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError("refusing to overwrite evaluation")
    folds_scope = PARENT.SCREEN_FOLDS if args.stage == "screen" else PARENT.FULL_FOLDS
    with tempfile.TemporaryDirectory(prefix="exp681_eval_") as directory:
        temp_dir = Path(directory)
        candidate_scores, candidate_audits = materialize_verified_candidates(
            args,
            folds_scope=folds_scope,
            directory=temp_dir,
        )
        candidate_runtime_by_fold = {
            int(
                json.loads((path / "runtime_audit.json").read_text(encoding="utf-8"))[
                    "outer_fold"
                ]
            ): path
            for path in args.candidate_runtime_dir
        }
        ranking_controls = frozen_prediction_paths(
            args.ranking_control_score,
            folds_scope=folds_scope,
            expected_sha256=RANKING_CONTROL_PREDICTION_SHA256,
            label="ranking control",
        )
        causal_controls, causal_control_audits = materialize_verified_causal_controls(
            args,
            folds_scope=folds_scope,
            candidate_runtime_by_fold=candidate_runtime_by_fold,
            directory=temp_dir,
        )
        parent_output = temp_dir / "production_control.json"
        parent_values = dict(vars(args))
        parent_values["output"] = parent_output
        parent_values["candidate_score"] = candidate_scores
        parent_values["control_score"] = ranking_controls
        parent_args = SimpleNamespace(**parent_values)
        result = PARENT.evaluate(parent_args)
        causal_output = temp_dir / "causal_control.json"
        causal_values = dict(parent_values)
        causal_values["output"] = causal_output
        causal_values["control_score"] = causal_controls
        causal_result = PARENT.evaluate(SimpleNamespace(**causal_values))
        causal_result["production_metrics"] = routed_metrics_against_control(
            SimpleNamespace(**causal_values),
            folds_scope=folds_scope,
        )
        singleton_args = SimpleNamespace(**parent_values)
        singleton = singleton_slice(singleton_args)
    result["experiment_id"] = "681"
    result["changed_factor"] = "hard_bce_to_fixed_hard_plus_teacher_soft_bce_on_680_specialist"
    result["temperature"] = 2.0
    result["soft_loss_weight"] = 0.5
    result["ordinary_oof_merge_used"] = False
    result["uses_27b_at_inference"] = False
    result["candidate_artifact_audits"] = candidate_audits
    result["causal_control_artifact_audits"] = causal_control_audits
    result["ranking_control_experiment_id"] = "641"
    result["routed_production_baseline"] = "frozen_semantic_v3_replay"
    result["causal_control_experiment_id"] = "680"
    result["causal_control_metrics"] = {
        "fold_flammable_average_precision": causal_result[
            "fold_flammable_average_precision"
        ],
        "mean_flammable_average_precision_delta": causal_result[
            "mean_flammable_average_precision_delta"
        ],
        "production_metrics": causal_result["production_metrics"],
    }
    result["semantic_singleton_flammable"] = singleton
    causal_metrics = causal_result["production_metrics"]
    causal_ap = [
        causal_result["fold_flammable_average_precision"][str(fold)]["delta"]
        for fold in folds_scope
    ]
    causal_ratio_ok = (
        causal_metrics["corrected"] > 0
        if causal_metrics["regressed"] == 0
        else causal_metrics["corrected_to_regressed"] >= 1.5
    )
    causal_common = {
        "causal_flammable_false_negatives_do_not_increase": causal_metrics[
            "false_negatives"
        ]["flammable"]["delta"]
        <= 0,
        "causal_corrected_to_regressed_at_least_1_5": causal_ratio_ok,
        "causal_flammable_f1_does_not_drop": causal_metrics["categories"][FLAMMABLE][
            "delta"
        ]
        >= 0.0,
    }
    if args.stage == "screen":
        causal_gates = causal_common | {
            "both_causal_folds_ap_positive": all(delta > 0 for delta in causal_ap),
            "mean_causal_ap_delta_at_least_0_003": float(np.mean(causal_ap)) >= 0.003,
            "both_causal_folds_macro_positive": causal_metrics["winning_folds"] == 2,
            "mean_causal_macro_delta_at_least_0_001": causal_metrics[
                "mean_fold_delta"
            ]
            >= 0.001,
        }
        result["ranking_and_routed_production_gates"] = dict(result["gates"])
        result["causal_distillation_gates"] = causal_gates
        result["gates"].update(causal_gates)
        result["passed"] = all(result["gates"].values())
        # Experiment 681 preregistered no relaxed screen continuation.  The
        # parent evaluator exposes an ablation screen for its own experiment;
        # carrying that decision forward here would silently weaken the gate.
        result["ablation_submission_eligible"] = False
        result["acceptance_tier"] = "primary" if result["passed"] else "no_go"
        result["decision"] = (
            "OPEN_CONFIRMATION" if result["passed"] else "REJECT_AT_SCREEN"
        )
    else:
        causal_gates = causal_common | {
            "at_least_four_of_five_causal_ap_wins": sum(delta > 0 for delta in causal_ap)
            >= 4,
            "mean_causal_ap_delta_at_least_0_003": float(np.mean(causal_ap)) >= 0.003,
            "all_causal_confirmation_folds_macro_positive": all(
                causal_metrics["folds"][str(fold)]["delta"] > 0 for fold in (1, 2, 4)
            ),
            "at_least_four_of_five_causal_macro_wins": causal_metrics["winning_folds"]
            >= 4,
            "mean_causal_macro_delta_at_least_0_003": causal_metrics[
                "mean_fold_delta"
            ]
            >= 0.003,
        }
        primary = result["gates"]
        fallback = result["ablation_gates"]
        primary["mean_macro_delta_at_least_0_006"] = (
            result["production_metrics"]["mean_fold_delta"] >= 0.006
        )
        primary["singleton_net_corrections_positive"] = singleton["net_corrections"] > 0
        fallback["mean_macro_delta_at_least_0_004"] = (
            result["production_metrics"]["mean_fold_delta"] >= 0.004
        )
        fallback["singleton_net_corrections_positive"] = singleton["net_corrections"] > 0
        result["ranking_and_routed_production_gates"] = dict(primary)
        result["causal_distillation_gates"] = causal_gates
        primary.update(causal_gates)
        fallback.update(causal_gates)
        result["passed"] = all(primary.values())
        result["ablation_submission_eligible"] = all(fallback.values())
        if result["passed"]:
            result["acceptance_tier"] = "primary"
            result["decision"] = "ACCEPT_FOR_REFIT"
        elif result["ablation_submission_eligible"]:
            result["acceptance_tier"] = "public_ablation"
            result["decision"] = "ACCEPT_FOR_REFIT"
        else:
            result["acceptance_tier"] = "no_go"
            result["decision"] = "REJECT_FULL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("screen", "full"), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--replay-contract", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--candidate-archive", type=Path, action="append", required=True)
    parser.add_argument("--candidate-runtime-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--ranking-control-score", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--causal-control-archive", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--causal-control-runtime-dir", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(evaluate(arguments), ensure_ascii=False, indent=2, sort_keys=True))
