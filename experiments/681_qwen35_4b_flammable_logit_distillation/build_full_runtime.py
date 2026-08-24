"""Build the post-validation full-data 4B distillation runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT_PATH = ROOT / "experiments/680_qwen35_4b_flammable_only_hard_bce/build_full_runtime.py"
PARENT_SHA256 = "30077e51269a0d385dbbdba11af993b7eb617e67394099f74956a251319afe1a"
FLAMMABLE = "Легковоспламеняющиеся"
EXPECTED_TRAIN_OCCURRENCES = 2590
EXPECTED_OOF_SHA256 = {
    0: "b74c50947ad2ea65e8ff77cdd0567aff61c8e6ddd2299320723c2eabea5c0b40",
    1: "16cacd7b6e986f9b661bee442b54d967fc254f80b8f529afc002868e289d98d1",
    2: "45ce7cb4c8896c203871262d8d12ddcf5d6bdc612a6a6d9b194653b618a3c142",
    3: "c731d3de64cd37fbc45461121d208ba5589981bb9a1971854098db2e6dc0cb81",
    4: "e970f01c41b3d35b1407abca7c2e37fb4e0a610d00f290ff55a83816df4c551c",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_parent() -> Any:
    if sha256_file(PARENT_PATH) != PARENT_SHA256:
        raise ValueError("frozen full-data parent builder checksum mismatch")
    spec = importlib.util.spec_from_file_location("exp680_full_for_681", PARENT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen full-data parent builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_full_gate(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not (
        report.get("experiment_id") == "681"
        and report.get("stage") == "full"
        and (report.get("passed") is True or report.get("ablation_submission_eligible") is True)
        and report.get("decision") == "ACCEPT_FOR_REFIT"
        and report.get("public_used") is False
        and report.get("sealed_rows") == 0
        and (all(report.get("gates", {}).values()) or all(report.get("ablation_gates", {}).values()))
        and report.get("gates", {}).get("bad_route_byte_identical") is True
        and set(report.get("fold_flammable_average_precision", {})) == {"0", "1", "2", "3", "4"}
        and report.get("ordinary_oof_merge_used") is False
        and report.get("uses_27b_at_inference") is False
    ):
        raise ValueError("full experiment-681 report does not authorize refit")
    return report


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_oof_scores(paths: list[Path], registry: Any) -> tuple[dict[str, float], dict[str, str]]:
    if len(paths) != 5:
        raise ValueError("exactly five frozen teacher OOF files are required")
    development = registry.loc[registry["split"].astype(str) == "development"].copy()
    development = development.sort_values(
        "id", key=lambda values: values.astype(str)
    ).reset_index(drop=True)
    scores: dict[str, float] = {}
    hashes: dict[str, str] = {}
    expected_fields = {"global_index", "id", "fold", "category", "prediction", "score"}
    for path in paths:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if not rows:
            raise ValueError("empty teacher OOF file")
        fold_values = {int(row["fold"]) for row in rows}
        if len(fold_values) != 1:
            raise ValueError("teacher OOF file mixes folds")
        fold = fold_values.pop()
        actual_sha = sha256_file(path)
        if actual_sha != EXPECTED_OOF_SHA256.get(fold):
            raise ValueError("teacher OOF prediction SHA mismatch")
        hashes[str(fold)] = actual_sha
        for row in rows:
            if set(row) != expected_fields:
                raise ValueError("teacher OOF prediction schema mismatch")
            index = int(row["global_index"])
            row_id = str(row["id"])
            if row_id in scores:
                raise ValueError("duplicate teacher OOF ID")
            score = float(row["score"])
            if not math.isfinite(score) or int(row["prediction"]) != int(score >= 0.0):
                raise ValueError("invalid teacher OOF score")
            if index < 0 or index >= len(development):
                raise ValueError("teacher OOF global index is outside development scope")
            expected = development.iloc[index]
            if (
                row_id != str(expected["id"])
                or int(row["fold"]) != int(expected["development_fold"])
                or str(row["category"]) != str(expected["category"])
                or str(expected["split"]) != "development"
            ):
                raise ValueError("teacher OOF row differs from immutable registry")
            scores[row_id] = score
    expected_ids = set(development["id"].astype(str))
    if set(hashes) != {"0", "1", "2", "3", "4"} or set(scores) != expected_ids:
        raise ValueError("teacher OOF predictions do not cover development exactly")
    return scores, hashes


def load_sealed_scores(
    score_path: Path,
    audit_path: Path,
    registry: Any,
    required_ids: set[str],
) -> tuple[dict[str, float], dict[str, str]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    body = dict(audit)
    digest = body.pop("contract_sha256", None)
    expected_common = {
        "experiment_id": "662",
        "mode": "sealed_holdout_label_free",
        "source_experiment_id": "654",
        "teacher_outer_fold": 0,
        "score_semantics": "raw_last_token_logit_1_minus_logit_0",
        "teacher_adapter_manifest_sha256": "910bfb3e6d4665eb431e1a840736f6595d155cad337a8a40856ac3bbd32c0cac",
        "labels_read": 0,
        "validation_labels_read": 0,
        "public_used": False,
        "decision": "ACCEPT_ARTIFACT",
    }
    if digest != canonical_sha256(body) or any(audit.get(k) != v for k, v in expected_common.items()):
        raise ValueError("sealed teacher audit contract mismatch")
    if int(audit.get("rows", -1)) != len(required_ids):
        raise ValueError("sealed teacher audit row count mismatch")
    if audit.get("teacher_scores_sha256") != sha256_file(score_path):
        raise ValueError("sealed teacher score SHA mismatch")
    rows = [json.loads(line) for line in score_path.read_text(encoding="utf-8").splitlines()]
    expected_fields = {"global_index", "id", "category", "split", "score"}
    result: dict[str, float] = {}
    for row in rows:
        if set(row) != expected_fields:
            raise ValueError("sealed teacher score schema mismatch")
        index = int(row["global_index"])
        row_id = str(row["id"])
        score = float(row["score"])
        expected = registry.iloc[index]
        if (
            row_id in result
            or row_id != str(expected["id"])
            or str(row["category"]) != str(expected["category"])
            or str(row["split"]) != "sealed_holdout"
            or str(expected["split"]) != "sealed_holdout"
            or not math.isfinite(score)
        ):
            raise ValueError("invalid sealed teacher score binding")
        result[row_id] = score
    if set(result) != required_ids:
        raise ValueError("sealed teacher scores do not cover required full-refit IDs")
    return result, {
        "audit_sha256": sha256_file(audit_path),
        "teacher_scores_sha256": sha256_file(score_path),
        "contract_sha256": str(digest),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    parent = load_parent()
    verify_full_gate(args.full_report)
    original_verify = parent.verify_full_gate
    parent.verify_full_gate = verify_full_gate
    try:
        parent_report = parent.build(args)
    finally:
        parent.verify_full_gate = original_verify
    if (
        parent_report.get("experiment_id") != "680"
        or int(parent_report.get("train_occurrences", -1)) != EXPECTED_TRAIN_OCCURRENCES
        or parent_report.get("decision") != "GO_SINGLE_REFIT"
    ):
        raise ValueError("unexpected parent full runtime")
    data = parent.pd.read_csv(args.data, dtype={"id": str})
    raw_registry = parent.pd.read_csv(
        args.registry, dtype={"id": str, "semantic_component": str}
    )
    registry = raw_registry.set_index("id").loc[data["id"]].reset_index()
    teacher_scores, prediction_hashes = load_oof_scores(args.teacher_oof, raw_registry)
    train_path = args.output_dir / "train.jsonl"
    validation_path = args.output_dir / "validation.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != EXPECTED_TRAIN_OCCURRENCES:
        raise ValueError("parent full runtime row count drifted")
    required_sealed_ids = {
        str(row["id"]) for row in rows if str(row["id"]) not in teacher_scores
    }
    if not required_sealed_ids:
        raise ValueError("expected a nonempty sealed teacher completion cohort")
    sealed_scores, sealed_manifest = load_sealed_scores(
        args.sealed_teacher_scores,
        args.sealed_teacher_audit,
        registry,
        required_sealed_ids,
    )
    for row in rows:
        index = int(row["global_index"])
        if row["category"] != FLAMMABLE or str(row["id"]) != str(registry.iloc[index]["id"]):
            raise ValueError("full runtime row differs from immutable registry")
        row_id = str(row["id"])
        row["teacher_score"] = (
            teacher_scores[row_id] if row_id in teacher_scores else sealed_scores[row_id]
        )
    parent.write_jsonl(train_path, rows)
    audit_path = args.output_dir / "runtime_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.pop("contract_sha256", None)
    audit.update(
        {
            "experiment_id": "681",
            "scope": "single_post_validation_flammable_distillation_full_competition_train_refit",
            "changed_factor": "hard_bce_to_fixed_hard_plus_teacher_soft_bce",
            "temperature": 2.0,
            "soft_loss_weight": 0.5,
            "teacher_experiment_id": "654",
            "teacher_target_scope": (
                "development_rowwise_oof_plus_label_free_sealed_teacher0"
            ),
            "teacher_score_semantics": "raw_last_token_logit_1_minus_logit_0",
            "teacher_oof_prediction_sha256": prediction_hashes,
            "sealed_teacher_completion": sealed_manifest,
            "sealed_teacher_unique_ids": len(required_sealed_ids),
            "ordinary_oof_merge_used_for_outer_cv": False,
            "uses_27b": True,
            "uses_27b_at_inference": False,
            "output_sha256": {
                "train.jsonl": parent.sha256_file(train_path),
                "validation.jsonl": parent.sha256_file(validation_path),
            },
        }
    )
    audit["contract_sha256"] = canonical_sha256(audit)
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--full-report", type=Path, required=True)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--registry", type=Path, required=True)
    result.add_argument("--selector", type=Path, required=True)
    result.add_argument("--image-manifest", type=Path, required=True)
    result.add_argument("--teacher-oof", type=Path, action="append", required=True)
    result.add_argument("--sealed-teacher-scores", type=Path, required=True)
    result.add_argument("--sealed-teacher-audit", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    parsed = parser().parse_args()
    print(json.dumps(build(parsed), ensure_ascii=False, indent=2, sort_keys=True))
