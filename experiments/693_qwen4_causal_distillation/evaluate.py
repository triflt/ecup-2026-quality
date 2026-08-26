from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from exp691_consumer import read_jsonl, sha256_file, verify_routed_acceptance

FLAMMABLE = "Легковоспламеняющиеся"
BAD = "БАД"
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 42
LABEL_POLICY_ROWS = 25
LABEL_POLICY_SUBSTRINGS = ("топлив", "зажигал")
MIN_POOLED_MACRO_DELTA = 0.006
MIN_FLAMMABLE_F1_DELTA = 0.012
MIN_FOLD_WINS = 4
MIN_CORRECTIONS_REGRESSIONS_RATIO = 1.5
MIN_BOOTSTRAP_P_GAIN = 0.90
SCIENTIFIC_PASS_FIELDS = (
    "fold_wins_pass",
    "pooled_macro_delta_pass",
    "flammable_ap_delta_pass",
    "flammable_f1_delta_pass",
    "flammable_fn_nonincrease",
    "corrections_regressions_ratio_pass",
    "bootstrap_p_gain_pass",
    "singleton_strict_gain",
    "label_policy_fp_nonincrease",
    "evidence_slice_systematic_regression_guard",
    "bad_byte_identical",
)


def sha256_bytes(values: list[int]) -> str:
    return hashlib.sha256(bytes(values)).hexdigest()


def average_precision(labels: list[int], scores: list[float]) -> float:
    """Exact exp691/sklearn threshold-tie average precision."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be nonempty and aligned")
    positives = sum(int(value) for value in labels)
    if positives == 0:
        return 0.0
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    true_positives = 0
    total = 0.0
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        score = float(scores[order[cursor]])
        while end < len(order) and float(scores[order[end]]) == score:
            end += 1
        group_positives = sum(int(labels[order[index]]) for index in range(cursor, end))
        true_positives += group_positives
        total += (true_positives / end) * group_positives
        cursor = end
    return total / positives


def classification(labels: list[int], predictions: list[int]) -> dict[str, float | int]:
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions, strict=True))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions, strict=True))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "fp": fp, "fn": fn}


def is_label_policy_name(row: dict[str, Any]) -> bool:
    name = str(row.get("name", row.get("NAME", ""))).casefold()
    return all(token in name for token in LABEL_POLICY_SUBSTRINGS)


def label_policy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    local = [row for row in rows if is_label_policy_name(row)]
    return {
        "rows": len(local),
        "labels_zero": sum(int(row["label"]) == 0 for row in local),
        "fp": {
            arm: sum(int(row[arm]) == 1 for row in local)
            for arm in ("baseline", "control", "candidate")
        },
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bad_candidate = [row["candidate"] for row in rows if row["category"] == BAD]
    bad_baseline = [row["baseline"] for row in rows if row["category"] == BAD]
    if bad_candidate != bad_baseline:
        raise ValueError("BAD production replay is not byte-identical")
    categories: dict[str, Any] = {}
    for category in (BAD, FLAMMABLE):
        local = [row for row in rows if row["category"] == category]
        if not local:
            continue
        labels = [int(row["label"]) for row in local]
        categories[category] = {
            "control": classification(labels, [int(row["control"]) for row in local]),
            "candidate": classification(labels, [int(row["candidate"]) for row in local]),
        }
    has_macro = set(categories) == {BAD, FLAMMABLE}
    control_macro = (
        sum(categories[c]["control"]["f1"] for c in categories) / 2 if has_macro else None
    )
    candidate_macro = (
        sum(categories[c]["candidate"]["f1"] for c in categories) / 2 if has_macro else None
    )
    flammable = [row for row in rows if row["category"] == FLAMMABLE]
    tie_aware_ap = None
    if flammable:
        tie_aware_ap = {
            "control": average_precision(
                [int(row["label"]) for row in flammable],
                [float(row["control_score"]) for row in flammable],
            ),
            "candidate": average_precision(
                [int(row["label"]) for row in flammable],
                [float(row["candidate_score"]) for row in flammable],
            ),
        }
        tie_aware_ap["delta"] = tie_aware_ap["candidate"] - tie_aware_ap["control"]
    corrected = sum(
        row["control"] != row["label"] and row["candidate"] == row["label"] for row in rows
    )
    regressed = sum(
        row["control"] == row["label"] and row["candidate"] != row["label"] for row in rows
    )
    return {
        "tie_aware_ap": tie_aware_ap,
        "macro": {
            "control": control_macro,
            "candidate": candidate_macro,
            "delta": candidate_macro - control_macro if control_macro is not None else None,
        },
        "categories": categories,
        "corrected": corrected,
        "regressed": regressed,
        "corrections_regressions_ratio": corrected / regressed if regressed else "inf",
        "bad_exact": bad_candidate == bad_baseline,
        "bad_prediction_sha256": sha256_bytes(bad_candidate),
        "rows": len(rows),
    }


def bootstrap_probability(rows: list[dict[str, Any]]) -> float:
    by_category = {
        category: [row for row in rows if row["category"] == category]
        for category in (BAD, FLAMMABLE)
    }
    if any(not local for local in by_category.values()):
        raise ValueError("pooled bootstrap requires both categories")
    randomizer = random.Random(BOOTSTRAP_SEED)
    positive = 0
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = [
            row
            for category in (BAD, FLAMMABLE)
            for row in randomizer.choices(by_category[category], k=len(by_category[category]))
        ]
        positive += summarize(sampled)["macro"]["delta"] > 0.0
    return positive / BOOTSTRAP_REPLICATES


def evidence_slices(row: dict[str, Any], evidence_row: dict[str, Any]) -> set[str]:
    """Exact slice definitions frozen by experiment 692."""
    slices: set[str] = set()
    payload = evidence_row.get("evidence", {})
    relation = str(payload.get("relation", {}).get("value", "unknown"))
    relation_slices = {
        "device_only": "equipment_without_fuel",
        "sold_separately": "fuel_sold_separately",
        "included": "fuel_included_in_kit",
        "integrated_source": "integrated_burner_source",
    }
    if relation in relation_slices:
        slices.add(relation_slices[relation])
    if bool(payload.get("abstain", True)) or not bool(payload.get("grounded", False)):
        slices.add("ambiguous_evidence")
    sources = {
        str(payload.get(name, {}).get("source", "unknown"))
        for name in ("sold_object", "substance", "relation")
    }
    if "image:first" in sources:
        slices.add("image_required")
    if sources and sources <= {"text"}:
        slices.add("text_sufficient")
    text = f"{row.get('name', '')} {row.get('description', '')}".casefold()
    if any(token in text for token in ("розжиг", "зажиг", "спич", "огнив")):
        slices.add("ignition_products")
    if any(token in text for token in ("уголь", "угля", "древесн", "брикет")):
        slices.add("charcoal_fire_starting")
    if any(token in text for token in ("пиротех", "фейервер", "петард", "бенгал")):
        slices.add("pyrotechnics")
    return slices


def load_accepted_evidence(
    teacher_root: Path, acceptance_path: Path, acceptance_sha256: str
) -> dict[tuple[int, int], dict[str, Any]]:
    acceptance = verify_routed_acceptance(acceptance_path, expected_file_sha256=acceptance_sha256)
    output: dict[tuple[int, int], dict[str, Any]] = {}
    for fold, binding in enumerate(acceptance["artifact_bindings"]):
        path = teacher_root / f"fold{fold}" / "evidence.jsonl"
        if sha256_file(path) != binding["evidence_sha256"]:
            raise ValueError(f"fold{fold} evidence differs from exp692 acceptance")
        evidence_rows = read_jsonl(path)
        if len(evidence_rows) != binding["evidence_rows"]:
            raise ValueError(f"fold{fold} evidence row count differs from exp692 acceptance")
        for evidence_row in evidence_rows:
            key = (fold, int(evidence_row["global_index"]))
            if key in output:
                raise ValueError("duplicate accepted evidence identity")
            output[key] = evidence_row
    return output


def evaluate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report = summarize(rows)
    singleton = [
        row for row in rows if bool(row.get("singleton", int(row.get("component_size", 0)) == 1))
    ]
    singleton_summary = summarize(singleton) if singleton else None
    report["semantic_singletons"] = singleton_summary
    report["singleton_delta"] = (
        singleton_summary["corrected"] - singleton_summary["regressed"] if singleton_summary else 0
    )
    report["label_policy_slice"] = label_policy_summary(rows)
    report["evidence_slices"] = {
        name: summarize([row for row in rows if name in row.get("slices", [])])
        for name in sorted({name for row in rows for name in row.get("slices", [])})
    }
    return report


def verify_paired_arms_frozen(
    control_root: Path, candidate_root: Path
) -> dict[str, dict[str, list[float]]]:
    """Fail before opening any label registry unless both arms are complete."""
    resources: dict[str, dict[str, list[float]]] = {
        "control": {"runtime_minutes": [], "peak_gpu_memory_bytes": []},
        "candidate": {"runtime_minutes": [], "peak_gpu_memory_bytes": []},
    }
    for fold in range(5):
        for arm, root in (("control", control_root), ("candidate", candidate_root)):
            directory = root / f"fold{fold}"
            predictions = directory / "predictions.jsonl"
            contract_path = directory / "output_contract.json"
            if not predictions.is_file() or not contract_path.is_file():
                raise ValueError(f"paired arm is not frozen for fold {fold}")
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            if (
                int(contract.get("outer_fold", -1)) != fold
                or contract.get("decision") != "GO_EVALUATE"
                or contract.get("technical_smoke") is not False
            ):
                raise ValueError(f"paired arm contract is not evaluable for fold {fold}")
            for field in ("runtime_minutes", "peak_gpu_memory_bytes"):
                if field not in contract:
                    continue
                value = float(contract[field])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"paired arm {field} is invalid for fold {fold}")
                resources[arm][field].append(value)
    return resources


def build_label_registry(
    runtime_root: Path,
) -> tuple[dict[int, int], dict[int, tuple[str, str, str]]]:
    labels: dict[int, int] = {}
    identities: dict[int, tuple[str, str, str]] = {}
    for fold in range(5):
        for row in read_jsonl(runtime_root / f"fold{fold}" / "train.jsonl"):
            key = int(row["global_index"])
            label = int(row["label"])
            identity = (
                str(row["id"]),
                str(row["category"]),
                str(row.get("name", row.get("NAME", ""))),
            )
            if label not in (0, 1):
                raise ValueError("cross-fold registry contains a non-binary label")
            if key in labels and (labels[key] != label or identities[key] != identity):
                raise ValueError("cross-fold label registry conflict")
            labels[key] = label
            identities[key] = identity
    return labels, identities


def aligned_rows(
    runtime_root: Path,
    baseline_root: Path,
    control_root: Path,
    candidate_root: Path,
    *,
    resource_observations: dict[str, dict[str, list[float]]] | None = None,
) -> list[dict[str, Any]]:
    resources = verify_paired_arms_frozen(control_root, candidate_root)
    if resource_observations is not None:
        resource_observations.update(resources)
    labels, identities = build_label_registry(runtime_root)
    validation_by_fold = {
        fold: read_jsonl(runtime_root / f"fold{fold}" / "validation.jsonl") for fold in range(5)
    }
    if any("label" in row for rows in validation_by_fold.values() for row in rows):
        raise ValueError("outer validation runtime must remain label-free")
    component_counts = Counter(
        str(row.get("semantic_component", ""))
        for rows in validation_by_fold.values()
        for row in rows
    )
    output: list[dict[str, Any]] = []
    for fold in range(5):
        truth = {int(row["global_index"]): row for row in validation_by_fold[fold]}
        paths = (
            baseline_root / f"fold{fold}" / "predictions.jsonl",
            control_root / f"fold{fold}" / "predictions.jsonl",
            candidate_root / f"fold{fold}" / "predictions.jsonl",
        )
        streams = [{int(row["global_index"]): row for row in read_jsonl(path)} for path in paths]
        if any(set(stream) != set(truth) for stream in streams):
            raise ValueError(f"prediction alignment mismatch in fold {fold}")
        baseline, control, candidate = streams
        for key, source in truth.items():
            identity = (
                str(source["id"]),
                str(source["category"]),
                str(source.get("name", source.get("NAME", ""))),
            )
            if identities.get(key) != identity or key not in labels:
                raise ValueError("outer validation identity is absent from cross-fold registry")
            category = identity[1]
            baseline_prediction = int(baseline[key]["prediction"])
            component = str(source.get("semantic_component", ""))
            output.append(
                {
                    "global_index": key,
                    "id": identity[0],
                    "fold": fold,
                    "label": labels[key],
                    "category": category,
                    "name": identity[2],
                    "description": str(source.get("description", "")),
                    "singleton": bool(component) and component_counts[component] == 1,
                    "baseline": baseline_prediction,
                    "control": baseline_prediction
                    if category == BAD
                    else int(control[key]["prediction"]),
                    "candidate": baseline_prediction
                    if category == BAD
                    else int(candidate[key]["prediction"]),
                    "control_score": float(control[key]["score"]),
                    "candidate_score": float(candidate[key]["score"]),
                }
            )
    return output


def resource_summary(
    observations: dict[str, dict[str, list[float]]] | None,
    submission_limit_minutes: float | None,
) -> dict[str, Any]:
    runtime_by_arm = {
        arm: sum((observations or {}).get(arm, {}).get("runtime_minutes", []))
        for arm in ("control", "candidate")
    }
    runtime_counts = {
        arm: len((observations or {}).get(arm, {}).get("runtime_minutes", []))
        for arm in ("control", "candidate")
    }
    peak_values = {
        arm: (observations or {}).get(arm, {}).get("peak_gpu_memory_bytes", [])
        for arm in ("control", "candidate")
    }
    runtime_complete = all(count == 5 for count in runtime_counts.values())
    recorded_runtime = sum(runtime_by_arm.values()) if runtime_complete else None
    if submission_limit_minutes is not None and submission_limit_minutes <= 0:
        raise ValueError("submission limit must be positive")
    within_limit = (
        recorded_runtime <= submission_limit_minutes
        if recorded_runtime is not None and submission_limit_minutes is not None
        else None
    )
    if within_limit is None:
        status = "STAGE_PENDING"
    else:
        status = "PASS" if within_limit else "FAIL"
    return {
        "status": status,
        "runtime_contracts_complete": runtime_complete,
        "runtime_minutes_by_arm": runtime_by_arm,
        "runtime_contract_count_by_arm": runtime_counts,
        "recorded_paired_training_runtime_minutes": recorded_runtime,
        "submission_limit_minutes": submission_limit_minutes,
        "runtime_within_submission_limit": within_limit,
        "peak_gpu_memory_bytes_by_arm": {
            arm: max(values) if values else None for arm, values in peak_values.items()
        },
        "peak_gpu_memory_contract_count_by_arm": {
            arm: len(values) for arm, values in peak_values.items()
        },
    }


def scientific_gate_passes(gate: dict[str, Any]) -> bool:
    return all(bool(gate[field]) for field in SCIENTIFIC_PASS_FIELDS)


def build_report(
    rows: list[dict[str, Any]],
    *,
    resource_observations: dict[str, dict[str, list[float]]] | None = None,
    submission_limit_minutes: float | None = None,
) -> dict[str, Any]:
    policy = label_policy_summary(rows)
    if policy["rows"] != LABEL_POLICY_ROWS or policy["labels_zero"] != LABEL_POLICY_ROWS:
        raise ValueError("frozen NAME label-policy slice is not exactly 25/25 label=0")
    folds = {
        str(fold): evaluate_rows([row for row in rows if row["fold"] == fold]) for fold in range(5)
    }
    pooled = evaluate_rows(rows)
    pooled["bootstrap_p_gain_gt_zero"] = bootstrap_probability(rows)
    flammable = pooled["categories"][FLAMMABLE]
    flammable_f1_delta = flammable["candidate"]["f1"] - flammable["control"]["f1"]
    corrections_ratio = pooled["corrections_regressions_ratio"]
    corrections_ratio_pass = (
        corrections_ratio == "inf" or float(corrections_ratio) >= MIN_CORRECTIONS_REGRESSIONS_RATIO
    )
    evidence_slice_regression_failures = sorted(
        name
        for name, summary in pooled["evidence_slices"].items()
        if int(summary["regressed"]) > int(summary["corrected"])
    )
    gate = {
        "fold_wins": sum(
            report["macro"]["delta"] is not None and report["macro"]["delta"] > 0
            for report in folds.values()
        ),
        "fold_wins_required": MIN_FOLD_WINS,
        "fold_wins_pass": False,
        "pooled_macro_delta": pooled["macro"]["delta"],
        "pooled_macro_delta_minimum": MIN_POOLED_MACRO_DELTA,
        "pooled_macro_delta_pass": pooled["macro"]["delta"] >= MIN_POOLED_MACRO_DELTA,
        "flammable_ap_delta": pooled["tie_aware_ap"]["delta"],
        "flammable_ap_delta_pass": pooled["tie_aware_ap"]["delta"] > 0,
        "flammable_f1_delta": flammable_f1_delta,
        "flammable_f1_delta_minimum": MIN_FLAMMABLE_F1_DELTA,
        "flammable_f1_delta_pass": flammable_f1_delta >= MIN_FLAMMABLE_F1_DELTA,
        "flammable_fn_nonincrease": (flammable["candidate"]["fn"] <= flammable["control"]["fn"]),
        "corrections_regressions_ratio": corrections_ratio,
        "corrections_regressions_ratio_minimum": MIN_CORRECTIONS_REGRESSIONS_RATIO,
        "corrections_regressions_ratio_pass": corrections_ratio_pass,
        "bootstrap_p_gain_gt_zero": pooled["bootstrap_p_gain_gt_zero"],
        "bootstrap_p_gain_minimum": MIN_BOOTSTRAP_P_GAIN,
        "bootstrap_p_gain_pass": (pooled["bootstrap_p_gain_gt_zero"] >= MIN_BOOTSTRAP_P_GAIN),
        "singleton_delta": pooled["singleton_delta"],
        "singleton_strict_gain": pooled["singleton_delta"] > 0,
        "label_policy_fp_nonincrease": (policy["fp"]["candidate"] <= policy["fp"]["control"]),
        "evidence_slice_systematic_regression_guard": (not evidence_slice_regression_failures),
        "evidence_slice_regression_failures": evidence_slice_regression_failures,
        "bad_byte_identical": pooled["bad_exact"],
    }
    gate["fold_wins_pass"] = gate["fold_wins"] >= MIN_FOLD_WINS
    scientific_pass = scientific_gate_passes(gate)
    resources = resource_summary(resource_observations, submission_limit_minutes)
    if not scientific_pass:
        decision = "REJECT_CANDIDATE"
    elif resources["status"] == "STAGE_PENDING":
        decision = "STAGE_PENDING_RESOURCE_EVIDENCE"
    elif resources["status"] == "FAIL":
        decision = "REJECT_RUNTIME_LIMIT"
    else:
        decision = "ACCEPT_CANDIDATE"
    return {
        "folds": folds,
        "pooled": pooled,
        "gate": gate,
        "resources": resources,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--teacher-acceptance", type=Path, required=True)
    parser.add_argument("--teacher-acceptance-sha256", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--submission-limit-minutes", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite evaluation")
    resource_observations: dict[str, dict[str, list[float]]] = {}
    rows = aligned_rows(
        args.runtime_root,
        args.baseline_root,
        args.control_root,
        args.candidate_root,
        resource_observations=resource_observations,
    )
    accepted_evidence = load_accepted_evidence(
        args.teacher_root, args.teacher_acceptance, args.teacher_acceptance_sha256
    )
    for row in rows:
        if row["category"] != FLAMMABLE:
            row["slices"] = []
            continue
        evidence_row = accepted_evidence.get((int(row["fold"]), int(row["global_index"])))
        if evidence_row is None or str(evidence_row.get("id")) != str(row["id"]):
            raise ValueError("accepted exp692 evidence is missing an OOF flammable row")
        row["slices"] = sorted(evidence_slices(row, evidence_row))
    report = build_report(
        rows,
        resource_observations=resource_observations,
        submission_limit_minutes=args.submission_limit_minutes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
