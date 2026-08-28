from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

FLAMMABLE = "Легковоспламеняющиеся"
DEFAULT_SCREEN_FOLDS = (0, 3)
FULL_FOLDS = (0, 1, 2, 3, 4)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("exp699_eval_parent", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_spec(value: str) -> dict[str, Any]:
    parts = value.split("=", 5)
    if len(parts) != 6:
        raise ValueError("candidate must be NAME=ARCH=SOURCE=MODE=CAP=PATH")
    name, architecture, source, mode, cap, path = parts
    if not name or architecture not in {"qwen35_4b", "qwen3vl_2b"}:
        raise ValueError("invalid candidate name/architecture")
    return {
        "name": name,
        "architecture": architecture,
        "source": source,
        "mode": mode,
        "cap": int(cap),
        "path": Path(path),
    }


def _fold_members(root: Path, fold: int) -> tuple[Path, Path, Path]:
    files = [path for path in root.rglob("*") if path.is_file()]
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("artifact contains a symlink")
    predictions = [
        path
        for path in files
        if path.name == "predictions.jsonl" and path.parent.name == f"fold{fold}"
    ]
    contracts = [
        path
        for path in files
        if path.name == "output_contract.json" and path.parent.name == f"fold{fold}"
    ]
    adapters = [
        path
        for path in files
        if path.name == "adapter.zip" and path.parent.name == f"fold{fold}"
    ]
    if not (len(predictions) == len(contracts) == len(adapters) == 1):
        raise ValueError(f"fold{fold} artifact member multiplicity mismatch")
    return predictions[0], contracts[0], adapters[0]


def load_candidate(
    spec: dict[str, Any],
    ids: np.ndarray,
    folds: np.ndarray,
    categories: np.ndarray,
    global_indices: np.ndarray,
    evaluation_folds: tuple[int, ...] = DEFAULT_SCREEN_FOLDS,
) -> tuple[np.ndarray, dict[str, Any]]:
    root = spec["path"]
    if root.is_symlink() or not root.is_dir():
        raise ValueError("candidate artifact is not a regular directory")
    values = np.full(len(ids), np.nan, dtype=np.float32)
    audit = {"folds": {}}
    position_by_key = {
        (str(ids[index]), int(folds[index]), str(categories[index])): index
        for index in range(len(ids))
    }
    if len(position_by_key) != len(ids):
        raise ValueError("baseline row binding key is not unique")
    seen: set[int] = set()
    for fold in evaluation_folds:
        predictions_path, contract_path, adapter_path = _fold_members(root, fold)
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        payload = dict(contract)
        declared = payload.pop("contract_sha256", None)
        if declared != canonical_sha256(payload):
            raise ValueError("output contract self-hash mismatch")
        expected = {
            "experiment_id": "699",
            "architecture": spec["architecture"],
            "fold": fold,
            "source": spec["source"],
            "mode": spec["mode"],
            "cap": spec["cap"],
            "validation_labels_read": 0,
            "sealed_rows_used": 0,
            "public_rows_used": 0,
            "decision": "GO_EVALUATE",
        }
        if any(contract.get(key) != value for key, value in expected.items()):
            raise ValueError(f"fold{fold} output contract mismatch")
        if str(spec["mode"]).endswith("_append"):
            expected_steps = math.ceil(
                int(contract.get("train_occurrences", -1))
                / int(contract.get("effective_batch", -1))
            ) * int(contract.get("epochs", 1))
            if (
                contract.get("augmentation_arm") != "synth_append"
                or int(contract.get("synthetic_occurrences", -1))
                not in {5, 10, 19, 40, 160}
                or int(contract.get("optimizer_steps", -1)) != expected_steps
            ):
                raise ValueError(f"fold{fold} append contract mismatch")
        elif contract.get("optimizer_steps") != 306:
            raise ValueError(f"fold{fold} optimizer-step mismatch")
        loss_contract = (
            "assistant_suffix_lm"
            if spec["architecture"] == "qwen3vl_2b"
            else "binary_bce_last_token"
        )
        if contract.get("loss_contract") != loss_contract:
            raise ValueError("architecture loss contract mismatch")
        if contract.get("artifacts") != {
            "predictions.jsonl": sha256(predictions_path),
            "adapter.zip": sha256(adapter_path),
        }:
            raise ValueError("artifact checksum mismatch")
        rows = [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
        ]
        expected_positions = set(np.flatnonzero(folds == fold).tolist())
        local_positions: set[int] = set()
        seen_source_indices: set[int] = set()
        for row in rows:
            if "label" in row:
                raise ValueError("candidate predictions contain labels")
            source_index = int(row["global_index"])
            if source_index in seen_source_indices:
                raise ValueError("candidate source global_index is duplicated")
            seen_source_indices.add(source_index)
            key = (str(row["id"]), int(row["fold"]), str(row["category"]))
            index = position_by_key.get(key)
            if index is None or index in seen or index not in expected_positions:
                raise ValueError("candidate prediction coverage mismatch")
            if source_index != int(global_indices[index]):
                raise ValueError("candidate source global_index mismatch")
            score = float(row["score"])
            if not np.isfinite(score):
                raise ValueError("candidate score is non-finite")
            values[index] = score
            seen.add(index)
            local_positions.add(index)
        if local_positions != expected_positions:
            raise ValueError(f"fold{fold} prediction set mismatch")
        audit["folds"][str(fold)] = {
            "contract_sha256": declared,
            "contract_file_sha256": sha256(contract_path),
            "predictions_sha256": sha256(predictions_path),
            "adapter_sha256": sha256(adapter_path),
            "rows": len(rows),
            "runtime_minutes": contract["runtime_minutes"],
            "peak_gpu_bytes": contract["peak_gpu_bytes"],
        }
    screen = np.isin(folds, evaluation_folds)
    if not np.isfinite(values[screen]).all() or np.isfinite(values[~screen]).any():
        raise ValueError("candidate score scope mismatch")
    audit["rows"] = int(screen.sum())
    return values, audit


def load_source_split(
    runtime_map: Path,
    runtime_contract: Path,
    oof_ids: np.ndarray,
    oof_categories: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contract = json.loads(runtime_contract.read_text(encoding="utf-8"))
    payload = dict(contract)
    declared = payload.pop("self_sha256", None)
    if declared != canonical_sha256(payload):
        raise ValueError("runtime map contract self-hash mismatch")
    if (
        contract.get("schema") != "exp699_eval_runtime_map_v1"
        or contract.get("experiment_id") != "699"
        or contract.get("map_sha256") != sha256(runtime_map)
        or contract.get("labels_read") != 0
        or contract.get("sealed_rows") != 0
        or contract.get("public_used") is not False
        or set(contract.get("folds", {})) != {"0", "1", "2", "3", "4"}
    ):
        raise ValueError("runtime map contract mismatch")
    oof_position_by_key = {
        (str(oof_ids[index]), str(oof_categories[index])): index
        for index in range(len(oof_ids))
    }
    if len(oof_position_by_key) != len(oof_ids):
        raise ValueError("OOF id/category binding is not unique")
    source_rows = [
        json.loads(line) for line in runtime_map.read_text(encoding="utf-8").splitlines()
    ]
    if len(source_rows) != contract.get("rows"):
        raise ValueError("runtime map row count mismatch")
    if any(set(row) != {"global_index", "id", "fold", "category"} for row in source_rows):
        raise ValueError("runtime map row schema mismatch")
    global_indices = np.asarray(
        [int(row["global_index"]) for row in source_rows], dtype=np.int64
    )
    if not np.array_equal(global_indices, np.arange(len(source_rows), dtype=np.int64)):
        raise ValueError("source validation global_index coverage mismatch")
    keys = [(str(row["id"]), str(row["category"])) for row in source_rows]
    if len(set(keys)) != len(keys):
        raise ValueError("source validation id/category key is duplicated")
    try:
        oof_positions = np.asarray(
            [oof_position_by_key[key] for key in keys], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError("source validation row missing from frozen OOF registry") from error
    folds = np.asarray([int(row["fold"]) for row in source_rows], dtype=np.int8)
    return oof_positions, folds, global_indices


def load_semantic_families(
    runtime_root: Path,
    ids: np.ndarray,
    folds: np.ndarray,
    categories: np.ndarray,
    global_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    position_by_key = {
        (str(ids[index]), int(folds[index]), str(categories[index])): index
        for index in range(len(ids))
    }
    components = np.full(len(ids), "", dtype=object)
    for fold in range(5):
        path = runtime_root / f"fold{fold}" / "validation.jsonl"
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"fold{fold} source validation missing or symlinked")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            key = (str(row["id"]), fold, str(row["category"]))
            index = position_by_key.get(key)
            if index is None or components[index]:
                raise ValueError("semantic-family source binding mismatch")
            if int(row["global_index"]) != int(global_indices[index]):
                raise ValueError("semantic-family global-index mismatch")
            component = str(row.get("semantic_component") or "")
            if not component:
                raise ValueError("semantic-family component is empty")
            components[index] = component
    if np.any(components == ""):
        raise ValueError("semantic-family coverage mismatch")
    sizes = Counter(
        (str(categories[index]), str(components[index])) for index in range(len(ids))
    )
    row_sizes = np.asarray(
        [
            sizes[(str(categories[index]), str(components[index]))]
            for index in range(len(ids))
        ],
        dtype=np.int32,
    )
    return row_sizes, {
        "rows": len(ids),
        "singleton_rows": int(np.sum(row_sizes == 1)),
        "rare_le2_rows": int(np.sum(row_sizes <= 2)),
        "repeated_rows": int(np.sum(row_sizes > 1)),
    }


def screen_summary(
    evaluation,
    labels,
    categories,
    folds,
    predictions,
    scores=None,
    evaluation_folds: tuple[int, ...] = DEFAULT_SCREEN_FOLDS,
):
    screen = np.isin(folds, evaluation_folds)
    result = evaluation.summarize(
        labels[screen],
        categories[screen],
        folds[screen],
        predictions[screen],
        None if scores is None else scores[screen],
    )
    result["fold_macro_f1"] = {
        str(fold): evaluation.summarize(
            labels[folds == fold],
            categories[folds == fold],
            folds[folds == fold],
            predictions[folds == fold],
        )["macro_f1"]
        for fold in evaluation_folds
    }
    return result


def apply_frozen_selection(
    categories: np.ndarray,
    folds: np.ndarray,
    named_scores: dict[str, np.ndarray],
    selection: dict[str, dict[str, dict[str, Any]]],
) -> np.ndarray:
    predictions = np.zeros(len(categories), dtype=np.int8)
    expected_names = set(named_scores)
    for category in sorted(np.unique(categories)):
        for fold in sorted(np.unique(folds)):
            positions = (categories == category) & (folds == fold)
            if not positions.any():
                continue
            frozen = selection[category][str(int(fold))]
            weights = frozen["weights"]
            if set(weights) != expected_names:
                raise ValueError("frozen fusion component mismatch")
            scores = sum(
                float(weights[name]) * named_scores[name][positions]
                for name in sorted(expected_names)
            )
            predictions[positions] = (
                scores >= float(frozen["threshold"])
            ).astype(np.int8)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-module", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--baseline-qwen3vl", type=Path, required=True)
    parser.add_argument("--baseline-qwen35", type=Path, required=True)
    parser.add_argument("--runtime-map", type=Path, required=True)
    parser.add_argument("--runtime-map-contract", type=Path, required=True)
    parser.add_argument("--source-runtime-root", type=Path)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--fold", type=int, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluation_folds = tuple(
        sorted(args.fold if args.fold is not None else DEFAULT_SCREEN_FOLDS)
    )
    if evaluation_folds not in {(0,), (3,), DEFAULT_SCREEN_FOLDS, FULL_FOLDS}:
        raise ValueError("evaluation folds must be 0, 3, 0+3, or full fivefold")
    evaluation = load_module(args.evaluation_module)
    oof = np.load(args.oof, allow_pickle=True)
    oof_ids = oof["ids"].astype(str)
    oof_labels = oof["labels"].astype(np.int8)
    oof_categories = oof["categories"].astype(str)
    oof_folds = oof["fold_ids"].astype(np.int8)
    source_positions, folds, global_indices = load_source_split(
        args.runtime_map, args.runtime_map_contract, oof_ids, oof_categories
    )
    ids = oof_ids[source_positions]
    labels = oof_labels[source_positions]
    categories = oof_categories[source_positions]
    family_sizes = None
    family_audit = None
    if args.source_runtime_root is not None:
        family_sizes, family_audit = load_semantic_families(
            args.source_runtime_root, ids, folds, categories, global_indices
        )
    baseline = {
        "qwen3vl_2b": np.load(args.baseline_qwen3vl, allow_pickle=True),
        "qwen35_4b": np.load(args.baseline_qwen35, allow_pickle=True),
    }
    for source in baseline.values():
        if (
            not np.array_equal(source["ids"].astype(str), oof_ids)
            or not np.array_equal(source["labels"].astype(np.int8), oof_labels)
            or not np.array_equal(source["folds"].astype(np.int8), oof_folds)
            or not np.array_equal(source["categories"].astype(str), oof_categories)
        ):
            raise ValueError("baseline array binding mismatch")
    base_ranks = {
        architecture: source["lora_rank"].astype(np.float32)[source_positions]
        for architecture, source in baseline.items()
    }
    robust = baseline["qwen3vl_2b"]["base_rank"].astype(np.float32)[source_positions]
    baseline_predictions, baseline_selection = evaluation.nested_fusion(
        labels,
        categories,
        folds,
        {
            "robust_base": robust,
            "qwen3vl": base_ranks["qwen3vl_2b"],
            "qwen35": base_ranks["qwen35_4b"],
        },
    )
    baseline_summary = screen_summary(
        evaluation,
        labels,
        categories,
        folds,
        baseline_predictions,
        evaluation_folds=evaluation_folds,
    )
    candidates: dict[str, dict[str, Any]] = {}
    for raw_spec in args.candidate:
        spec = parse_spec(raw_spec)
        if spec["name"] in candidates:
            raise ValueError("duplicate candidate name")
        raw_scores, artifact_audit = load_candidate(
            spec,
            ids,
            folds,
            categories,
            global_indices,
            evaluation_folds=evaluation_folds,
        )
        hybrid = base_ranks[spec["architecture"]].copy()
        for fold in evaluation_folds:
            for category in sorted(np.unique(categories)):
                positions = np.flatnonzero((folds == fold) & (categories == category))
                hybrid[positions] = evaluation.rank01(raw_scores[positions])
        candidates[spec["name"]] = {
            **spec,
            "hybrid_rank": hybrid,
            "artifact_audit": artifact_audit,
        }

    scenarios: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, item in candidates.items():
        q3 = base_ranks["qwen3vl_2b"]
        q35 = base_ranks["qwen35_4b"]
        if item["architecture"] == "qwen3vl_2b":
            q3 = item["hybrid_rank"]
        else:
            q35 = item["hybrid_rank"]
        scenarios[name] = (q3, q35)
    q3_candidates = {
        name: item
        for name, item in candidates.items()
        if item["architecture"] == "qwen3vl_2b"
    }
    q35_candidates = {
        name: item
        for name, item in candidates.items()
        if item["architecture"] == "qwen35_4b"
    }
    for q3_name, q3 in q3_candidates.items():
        for q35_name, q35 in q35_candidates.items():
            scenarios[f"{q3_name}+{q35_name}"] = (
                q3["hybrid_rank"],
                q35["hybrid_rank"],
            )

    results = []
    screen = np.isin(folds, evaluation_folds)
    flammable = screen & (categories == FLAMMABLE)
    baseline_fn = baseline_summary["categories"][FLAMMABLE]["confusion"]["fn"]
    for name, (q3, q35) in scenarios.items():
        predictions = apply_frozen_selection(
            categories,
            folds,
            {"robust_base": robust, "qwen3vl": q3, "qwen35": q35},
            baseline_selection,
        )
        summary = screen_summary(
            evaluation,
            labels,
            categories,
            folds,
            predictions,
            evaluation_folds=evaluation_folds,
        )
        comparison = evaluation.compare(
            predictions[screen],
            baseline_predictions[screen],
            labels[screen],
            categories[screen],
        )
        family_diagnostics = None
        if family_sizes is not None:
            family_diagnostics = {}
            family_scopes = {
                "singleton": family_sizes == 1,
                "rare_le2": family_sizes <= 2,
                "repeated": family_sizes > 1,
            }
            for scope_name, family_scope in family_scopes.items():
                scope = screen & family_scope
                family_diagnostics[scope_name] = {
                    **evaluation.compare(
                        predictions[scope],
                        baseline_predictions[scope],
                        labels[scope],
                        categories[scope],
                    ),
                    "rows": int(np.sum(scope)),
                }
        fold_delta = {
            str(fold): summary["fold_macro_f1"][str(fold)]
            - baseline_summary["fold_macro_f1"][str(fold)]
            for fold in evaluation_folds
        }
        component_names = name.split("+")
        component_ap = {}
        for component_name in component_names:
            item = candidates[component_name]
            architecture = item["architecture"]
            component_ap[component_name] = {
                "baseline": float(
                    average_precision_score(
                        labels[flammable], base_ranks[architecture][flammable]
                    )
                ),
                "candidate": float(
                    average_precision_score(
                        labels[flammable], item["hybrid_rank"][flammable]
                    )
                ),
            }
            component_ap[component_name]["delta"] = (
                component_ap[component_name]["candidate"]
                - component_ap[component_name]["baseline"]
            )
        macro_delta = summary["macro_f1"] - baseline_summary["macro_f1"]
        baseline_flammable_f1 = baseline_summary["categories"][FLAMMABLE]["f1"]
        flammable_f1 = summary["categories"][FLAMMABLE]["f1"]
        flammable_fn = summary["categories"][FLAMMABLE]["confusion"]["fn"]
        fold_wins = sum(value > 0 for value in fold_delta.values())
        required_fold_wins = 4 if evaluation_folds == FULL_FOLDS else 1
        gate = bool(
            macro_delta > 0
            and fold_wins >= required_fold_wins
            and flammable_f1 >= baseline_flammable_f1
            and flammable_fn <= baseline_fn
            and comparison["corrections"] > comparison["regressions"]
        )
        results.append(
            {
                "scenario": name,
                "macro_f1": summary["macro_f1"],
                "macro_delta": macro_delta,
                "fold_macro_delta": fold_delta,
                "fold_wins": fold_wins,
                "required_fold_wins": required_fold_wins,
                "bad_f1": summary["categories"]["БАД"]["f1"],
                "flammable_f1": flammable_f1,
                "flammable_confusion": summary["categories"][FLAMMABLE]["confusion"],
                "component_ap": component_ap,
                "corrections_regressions": comparison,
                "semantic_families": family_diagnostics,
                "screen_pass": gate,
                "selection": baseline_selection,
                "selection_source": "frozen_baseline_140",
            }
        )
    results.sort(
        key=lambda item: (item["screen_pass"], item["macro_delta"]), reverse=True
    )
    report = {
        "schema": "exp699_gpu_evaluation_v3",
        "fusion_policy": "frozen_baseline_140_nested_oof_selection",
        "evaluation_folds": list(evaluation_folds),
        "baseline": {**baseline_summary, "selection": baseline_selection},
        "candidate_artifacts": {
            name: {
                "architecture": item["architecture"],
                "source": item["source"],
                "mode": item["mode"],
                "cap": item["cap"],
                "audit": item["artifact_audit"],
            }
            for name, item in candidates.items()
        },
        "semantic_family_audit": family_audit,
        "leaderboard": results,
        "public_used": False,
        "sealed_rows": 0,
    }
    report["self_sha256"] = canonical_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compact = [
        {
            key: value
            for key, value in item.items()
            if key not in {"selection"}
        }
        for item in results
    ]
    print("EXP699_GPU_SCREEN=" + json.dumps(compact, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
