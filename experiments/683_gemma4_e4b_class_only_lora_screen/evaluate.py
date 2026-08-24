from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from contract import EXPERIMENT_ID, canonical_sha256, sha256_file

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE.parent / "680_qwen35_4b_flammable_only_hard_bce" / "evaluate.py"


def load_base():
    spec = importlib.util.spec_from_file_location("exp680_for_683", BASE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen route evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base()
CONTROL_SCORE_SHA256 = {
    0: "5164909b7b56298f074e2d21a6e40f1ed53fd2d28d599609527715760f3468ae",
    3: "f73d1b06f33c74c40cce03afac1537134e8da251e5e101c14616459414d0ab6a",
}


def verify_prediction_provenance(
    *,
    candidate_scores: list[Path],
    candidate_acceptances: list[Path],
    control_scores: list[Path],
    folds_scope: tuple[int, ...],
) -> list[str]:
    if not (
        len(candidate_acceptances) == len(candidate_scores) == len(folds_scope)
        and len(control_scores) == len(folds_scope)
    ):
        raise ValueError("each fold requires candidate, acceptance and control files")
    audit_hashes = []
    for fold, score_path, audit_path, control_path in zip(
        folds_scope,
        candidate_scores,
        candidate_acceptances,
        control_scores,
        strict=True,
    ):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        digest = audit.pop("acceptance_sha256", None)
        if digest != canonical_sha256(audit):
            raise ValueError("candidate acceptance audit self-hash mismatch")
        if not (
            audit.get("experiment_id") == EXPERIMENT_ID
            and audit.get("fold") == fold
            and audit.get("technical_smoke") is False
            and audit.get("decision") == "ACCEPT_ARTIFACT"
            and audit.get("predictions_sha256") == sha256_file(score_path)
        ):
            raise ValueError("candidate prediction provenance mismatch")
        if sha256_file(control_path) != CONTROL_SCORE_SHA256[fold]:
            raise ValueError("frozen 641 control prediction checksum mismatch")
        audit_hashes.append(sha256_file(audit_path))
    return audit_hashes


def evaluate(args: argparse.Namespace) -> dict:
    folds_scope = BASE.SCREEN_FOLDS if args.stage == "screen" else BASE.FULL_FOLDS
    audit_hashes = verify_prediction_provenance(
        candidate_scores=args.candidate_score,
        candidate_acceptances=args.candidate_acceptance,
        control_scores=args.control_score,
        folds_scope=folds_scope,
    )
    temporary = args.output.with_suffix(".base680.tmp.json")
    if temporary.exists() or args.output.exists():
        raise FileExistsError("refusing to overwrite evaluation")
    base_args = argparse.Namespace(
        stage=args.stage,
        bundle=args.bundle,
        replay_contract=args.replay_contract,
        registry=args.registry,
        candidate_score=args.candidate_score,
        control_score=args.control_score,
        output=temporary,
    )
    result = BASE.evaluate(base_args)
    temporary.unlink()
    spec = BASE.LEGACY.load_spec()
    contract = BASE.LEGACY.verify_replay_contract(
        path=args.replay_contract,
        bundle_path=args.bundle,
        registry_path=args.registry,
        spec=spec,
    )
    registry = BASE.LEGACY._load_registry(args.registry, spec=spec)
    bundle = BASE.LEGACY.load_replay_bundle(
        path=args.bundle, contract=contract, registry=registry
    )
    candidate_logits, _ = BASE.load_scores(
        args.candidate_score, bundle=bundle, folds_scope=folds_scope
    )
    labels = registry["label"].astype("int8").to_numpy()
    categories = bundle["categories"].astype(str)
    folds = bundle["folds"].astype("int8")
    components = bundle["semantic_components"].astype(str)
    baseline_probability = BASE.LEGACY.sigmoid(bundle["qwen35_original_logit"])
    candidate_probability = baseline_probability.copy()
    candidate_probability[categories == BASE.FLAMMABLE] = BASE.LEGACY.sigmoid(
        candidate_logits[categories == BASE.FLAMMABLE]
    )
    baseline_prediction, _ = BASE.LEGACY.route_predictions(
        bundle=bundle, qwen_probability=baseline_probability, route=spec["route"]
    )
    candidate_prediction, _ = BASE.LEGACY.route_predictions(
        bundle=bundle, qwen_probability=candidate_probability, route=spec["route"]
    )
    component_sizes = Counter(components.tolist())
    selected = np.isin(folds, folds_scope) & np.array(
        [component_sizes[value] == 1 for value in components]
    )
    singleton_corrected = int(
        (selected & (baseline_prediction != labels) & (candidate_prediction == labels)).sum()
    )
    singleton_regressed = int(
        (selected & (baseline_prediction == labels) & (candidate_prediction != labels)).sum()
    )
    result["experiment_id"] = EXPERIMENT_ID
    result["control_experiment_id"] = "641"
    result["changed_factor"] = "qwen35_4b_to_gemma4_e4b"
    result["candidate_scope"] = "Gemma replaces Qwen3.5 only for flammable; BAD is byte-identical control"
    result["candidate_acceptance_audit_sha256"] = audit_hashes
    result["direct_component_control"] = "candidate raw logits are compared with fold-matched 641 logits by tie-aware flammable AP"
    result["singleton_families"] = {
        "rows": int(selected.sum()),
        "corrected": singleton_corrected,
        "regressed": singleton_regressed,
        "net": singleton_corrected - singleton_regressed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("screen", "full"), required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--replay-contract", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--candidate-score", type=Path, action="append", required=True)
    parser.add_argument("--candidate-acceptance", type=Path, action="append", required=True)
    parser.add_argument("--control-score", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), ensure_ascii=False, indent=2, sort_keys=True))
