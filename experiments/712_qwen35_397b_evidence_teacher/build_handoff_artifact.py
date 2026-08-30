from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

from build_student_targets import explanation_only, rationale_then_label, student_scope_errors
from prompt import validate_output


ORIGIN_COMMIT = "88cdfe60dddfe4a864727e2bb9e16feb8e24f437"
EXPECTED_ROWS = 12_971
EXPECTED_DATA_SHA256 = "4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510"
EXPECTED_FOLDS_SHA256 = "03baaa25bd5a3aef6ad94e02067cccda114041f98d7a35a9e06330a425166e4d"
EXPECTED_OOF_SHA256 = "d78bb7df897e244e030fa388e1b9c770d15434cc33aa1f56b238c7d057646116"
MODES = ("rationale_then_label", "explanation_only")
ACCESSIBLE_EVIDENCE = {"title", "description", "absence", "image:0"}


class RejectRow(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(detail or code)
        self.code = code
        self.detail = detail


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_csv_by_id(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    order = [str(row["id"]) for row in rows]
    if len(order) != len(set(order)):
        raise ValueError(f"duplicate ids in {path}")
    return order, {str(row["id"]): row for row in rows}


def find_one(root: Path, name: str) -> Path:
    candidates = sorted(root.rglob(name))
    if len(candidates) != 1:
        raise ValueError(f"expected one {name} below {root}, got {candidates}")
    return candidates[0]


def read_teacher(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            item_id = str(record["id"])
            if item_id in output:
                raise ValueError(f"duplicate teacher id={item_id}")
            output[item_id] = record
    return output


def image_paths(root: Path, item_id: str) -> list[Path]:
    directory = root / item_id
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    ) if directory.is_dir() else []


def inspect_ensemble_image0(source: Path, image_root: Path) -> dict[str, Any]:
    source_sha = sha256_file(source)
    try:
        with Image.open(source) as opened:
            width, height = opened.size
            opened.verify()
    except Exception as error:
        raise RejectRow("IMAGE0_DECODE_ERROR", f"{type(error).__name__}: {error}") from error
    return {
        "path": str(source.relative_to(image_root)),
        "sha256": source_sha,
        "size_bytes": source.stat().st_size,
        "width": width,
        "height": height,
    }


def tree_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["size_bytes"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def quantiles(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {"min": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    def pick(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]
    return {"min": ordered[0], "p50": pick(0.50), "p95": pick(0.95), "max": ordered[-1]}


def coverage_report(
    *, data: dict[str, dict[str, str]], folds: dict[str, dict[str, str]], accepted: set[str]
) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    kept: Counter[str] = Counter()
    for item_id, row in data.items():
        fold = str(folds[item_id]["fold"])
        keys = (
            "all",
            f"fold:{fold}",
            f"cell:{row['category']}|{row['label']}",
            f"fold_cell:{fold}|{row['category']}|{row['label']}",
        )
        totals.update(keys)
        if item_id in accepted:
            kept.update(keys)
    return {
        key: {
            "accepted": kept[key],
            "total": total,
            "coverage": kept[key] / total if total else 0.0,
        }
        for key, total in sorted(totals.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--fold-manifest", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    args = parser.parse_args()

    rationale_contract = load_module(
        "handoff_rationale_contract", args.code_root / "713" / "target_contract.py"
    )
    explanation_contract = load_module(
        "handoff_explanation_contract", args.code_root / "714" / "target_contract.py"
    )

    if sha256_file(args.data) != EXPECTED_DATA_SHA256:
        raise ValueError("source data SHA mismatch")
    if sha256_file(args.folds) != EXPECTED_FOLDS_SHA256:
        raise ValueError("CV-5 folds SHA mismatch")
    if sha256_file(args.oof) != EXPECTED_OOF_SHA256:
        raise ValueError("historical OOF SHA mismatch")
    order, data = read_csv_by_id(args.data)
    fold_order, folds = read_csv_by_id(args.folds)
    if len(order) != EXPECTED_ROWS or order != fold_order:
        raise ValueError("data/CV-5 id order mismatch")
    for item_id in order:
        if (
            str(data[item_id]["category"]) != str(folds[item_id]["category"])
            or int(data[item_id]["label"]) != int(folds[item_id]["label"])
        ):
            raise ValueError(f"data/CV-5 row mismatch for id={item_id}")
    with np.load(args.oof, allow_pickle=True) as oof:
        if not np.array_equal(oof["ids"].astype(str), np.asarray(order, dtype=str)):
            raise ValueError("historical OOF id order mismatch")
        expected_fold_ids = np.asarray([int(folds[item_id]["fold"]) for item_id in order], dtype=np.int8)
        if not np.array_equal(oof["fold_ids"].astype(np.int8), expected_fold_ids):
            raise ValueError("historical OOF fold_ids mismatch")

    teacher_path = find_one(args.teacher_root, "explanation_synth_v5.jsonl")
    teacher_report_path = find_one(args.teacher_root, "explanation_synth_v5_report.json")
    teacher = read_teacher(teacher_path)
    if len(teacher) != EXPECTED_ROWS or set(teacher) != set(data):
        raise ValueError("teacher/data id set mismatch")
    teacher_report = json.loads(teacher_report_path.read_text(encoding="utf-8"))
    if not teacher_report.get("full_scope_complete") or teacher_report.get("invalid_rows") != 0:
        raise ValueError("teacher report is not terminal full-scope ACCEPT")

    args.output_root.mkdir(parents=True, exist_ok=True)
    validation_root = args.output_root / "validation"
    validation_root.mkdir(parents=True, exist_ok=True)
    copied_folds = validation_root / "folds.csv"
    copied_fold_manifest = validation_root / "manifest.json"
    copied_oof = validation_root / "four_head_oof.npz"
    shutil.copyfile(args.folds, copied_folds)
    shutil.copyfile(args.fold_manifest, copied_fold_manifest)
    shutil.copyfile(args.oof, copied_oof)
    mode_paths = {
        "rationale_then_label": args.output_root / "rationale_then_label_v5.jsonl",
        "explanation_only": args.output_root / "explanation_only_v5.jsonl",
    }
    mode_streams = {mode: path.open("w", encoding="utf-8") for mode, path in mode_paths.items()}
    row_manifest_path = args.output_root / "row_manifest.jsonl"
    accepted_path = args.output_root / "accepted_rows.jsonl"
    rejections_path = args.output_root / "rejections.jsonl"
    row_stream = row_manifest_path.open("w", encoding="utf-8")
    accepted_stream = accepted_path.open("w", encoding="utf-8")
    rejection_stream = rejections_path.open("w", encoding="utf-8")
    accepted_ids = {mode: set() for mode in MODES}
    target_chars = {mode: [] for mode in MODES}
    rejection_counts: Counter[str] = Counter()
    ensemble_images: dict[str, dict[str, Any]] = {}
    recovered_from_nee_ids: set[str] = set()

    try:
        for index, item_id in enumerate(order, 1):
            row = data[item_id]
            record = teacher[item_id]
            parsed = record.get("parsed_output")
            paths = image_paths(args.image_root, item_id)
            row_rejections: list[dict[str, str]] = []
            mode_targets: dict[str, str] = {}
            image0_sha: str | None = None
            image0_path: str | None = None
            image0_size: int | None = None
            image0_width: int | None = None
            image0_height: int | None = None
            try:
                if not paths:
                    raise RejectRow("IMAGE0_MISSING")
                if len(paths) != int(record.get("image_count", -1)):
                    raise RejectRow("IMAGE_COUNT_MISMATCH")
                source_image = inspect_ensemble_image0(paths[0], args.image_root)
                image0_sha = str(source_image["sha256"])
                image0_path = str(source_image["path"])
                image0_size = int(source_image["size_bytes"])
                image0_width = int(source_image["width"])
                image0_height = int(source_image["height"])
                teacher_hashes = list(record.get("image_sha256") or [])
                if not teacher_hashes or image0_sha != teacher_hashes[0]:
                    raise RejectRow("IMAGE0_TEACHER_SHA_MISMATCH")
                ensemble_images[item_id] = {
                    "id": item_id,
                    "path": image0_path,
                    "sha256": image0_sha,
                    "size_bytes": image0_size,
                    "width": image0_width,
                    "height": image0_height,
                }
                if not record.get("accepted") or not isinstance(parsed, dict):
                    raise RejectRow("TEACHER_NOT_ACCEPTED")
                label = int(row["label"])
                if int(parsed.get("label")) != label:
                    raise RejectRow("OUTPUT_LABEL_INTEGRITY_MISMATCH")
                validation = validate_output(
                    parsed,
                    category=str(row["category"]),
                    expected_label=label,
                    name=str(row["name"]),
                    description=str(row["description"]),
                    image_count=len(paths),
                )
                if validation.errors:
                    raise RejectRow(
                        "TEACHER_OUTPUT_CONTRACT_ERROR",
                        " | ".join(validation.errors),
                    )
                if parsed.get("reason") == "not_enough_evidence":
                    raise RejectRow("NOT_ENOUGH_EVIDENCE")
                evidence = parsed.get("evidence")
                source = str(evidence.get("source")) if isinstance(evidence, dict) else ""
                if source.startswith("image:") and source != "image:0":
                    raise RejectRow("SECONDARY_IMAGE_ONLY")
                if source not in ACCESSIBLE_EVIDENCE:
                    raise RejectRow("UNKNOWN_EVIDENCE_REFERENCE")
                scope_errors = student_scope_errors(parsed, max_student_image_index=0)
                if scope_errors:
                    raise RejectRow(scope_errors[0])
                mode_targets["rationale_then_label"] = rationale_then_label(
                    parsed, label, max_student_image_index=0
                )
                mode_targets["explanation_only"] = explanation_only(
                    parsed, label, max_student_image_index=0
                )
                for mode, target in mode_targets.items():
                    target_record = {
                        "id": item_id,
                        "label": label,
                        "target": target,
                    }
                    mode_streams[mode].write(compact_json(target_record) + "\n")
                    accepted_ids[mode].add(item_id)
                    target_chars[mode].append(len(target))
            except RejectRow as error:
                row_rejections.append({"code": error.code, "detail": error.detail})
            except (KeyError, TypeError, ValueError) as error:
                row_rejections.append({
                    "code": "TARGET_CONTRACT_ERROR",
                    "detail": f"{type(error).__name__}: {error}",
                })

            evidence = parsed.get("evidence") if isinstance(parsed, dict) else None
            evidence_source = str(evidence.get("source")) if isinstance(evidence, dict) else None
            evidence_value = str(evidence.get("value")) if isinstance(evidence, dict) else None
            eligibility = {mode: mode in mode_targets for mode in MODES}
            recovered_from_nee = (
                "student_scope_repair_nee_recovery_requested"
                in (record.get("normalization_actions") or [])
                and bool(mode_targets)
            )
            if recovered_from_nee:
                recovered_from_nee_ids.add(item_id)
            formatted_input_sha256: dict[str, str] = {}
            if mode_targets and image0_sha:
                contract_row = SimpleNamespace(
                    id=item_id,
                    category=str(row["category"]),
                    name=str(row["name"]),
                    description=str(row["description"]),
                )
                formatted_inputs = {
                    "rationale_then_label": rationale_contract.user_text(contract_row),
                    "explanation_only": explanation_contract.user_text(
                        contract_row, int(row["label"])
                    ),
                }
                formatted_input_sha256 = {
                    mode: sha256_bytes(compact_json({
                        "text": text,
                        "image0_sha256": image0_sha,
                    }).encode("utf-8"))
                    for mode, text in formatted_inputs.items()
                }
            student_input = {
                "id": item_id,
                "category": str(row["category"]),
                "name": str(row["name"]),
                "description": str(row["description"]),
                "image0_sha256": image0_sha,
            }
            row_record = {
                "id": item_id,
                "fold": int(folds[item_id]["fold"]),
                "category": str(row["category"]),
                "label": int(row["label"]),
                "image_count": len(paths),
                "ensemble_image0_sha256": image0_sha,
                "ensemble_image0_path": image0_path,
                "ensemble_image0_size_bytes": image0_size,
                "ensemble_image0_width": image0_width,
                "ensemble_image0_height": image0_height,
                "source_input_sha256": sha256_bytes(compact_json(student_input).encode("utf-8")),
                "input_sha256": formatted_input_sha256,
                "canonical_input_sha256": record.get("canonical_input_sha256"),
                "teacher_record_sha256": sha256_bytes(compact_json(record).encode("utf-8")),
                "evidence_source": evidence_source,
                "evidence_value_sha256": sha256_bytes(evidence_value.encode("utf-8"))
                if evidence_value is not None else None,
                "mode_eligibility": eligibility,
                "recovered_from_not_enough_evidence": recovered_from_nee,
                "target_sha256": {
                    mode: sha256_bytes(target.encode("utf-8")) for mode, target in mode_targets.items()
                },
                "rejection_codes": [item["code"] for item in row_rejections],
            }
            row_stream.write(compact_json(row_record) + "\n")
            if all(eligibility.values()):
                accepted_stream.write(compact_json(row_record) + "\n")
            for rejection in row_rejections:
                rejection_counts[rejection["code"]] += 1
                rejection_stream.write(compact_json({
                    "id": item_id,
                    "fold": int(folds[item_id]["fold"]),
                    "category": str(row["category"]),
                    "label": int(row["label"]),
                    "code": rejection["code"],
                    "detail": rejection["detail"],
                }) + "\n")
            if index % 500 == 0 or index == len(order):
                print(f"handoff_rows={index}/{len(order)}", flush=True)
    finally:
        for stream in mode_streams.values():
            stream.close()
        row_stream.close()
        accepted_stream.close()
        rejection_stream.close()

    support_intersection = set.intersection(*(accepted_ids[mode] for mode in MODES))
    exact_support_match = all(accepted_ids[mode] == support_intersection for mode in MODES)
    matched_digit_path = args.output_root / "matched_support_digit_v5.jsonl"
    with matched_digit_path.open("w", encoding="utf-8") as stream:
        for item_id in order:
            if item_id in support_intersection:
                stream.write(compact_json({
                    "id": item_id,
                    "label": int(data[item_id]["label"]),
                    "target": str(int(data[item_id]["label"])),
                }) + "\n")
    image_manifest_path = args.output_root / "ensemble_image0_manifest.jsonl"
    image_manifest_rows = [ensemble_images[item_id] for item_id in order if item_id in ensemble_images]
    with image_manifest_path.open("w", encoding="utf-8") as stream:
        for row in image_manifest_rows:
            stream.write(compact_json(row) + "\n")
    reports: dict[str, Any] = {}
    for mode in MODES:
        report = {
            "schema_version": "student_target_audit_v2",
            "mode": mode,
            "rows": EXPECTED_ROWS,
            "accepted": len(accepted_ids[mode]),
            "rejected": EXPECTED_ROWS - len(accepted_ids[mode]),
            "coverage": coverage_report(data=data, folds=folds, accepted=accepted_ids[mode]),
            "conservative_coverage_excluding_nee_recovery": coverage_report(
                data=data,
                folds=folds,
                accepted=accepted_ids[mode] - recovered_from_nee_ids,
            ),
            "accepted_recovered_from_not_enough_evidence": len(
                accepted_ids[mode] & recovered_from_nee_ids
            ),
            "target_chars": quantiles(target_chars[mode]),
            "target_tokens": "PENDING_EXACT_PROCESSOR_RECEIPT",
            "invalid_json": 0,
            "accepted_unsupported_evidence": 0,
            "accepted_unsupported_evidence_scope": (
                "automated source-access, exact-span and lexical scope checks only"
            ),
            "semantic_grounding_audit": "PENDING_USER_REVIEW_AND_BLIND_AUDIT",
            "accepted_secondary_only_evidence": 0,
            "accepted_student_scope_errors": 0,
            "rejected_unknown_evidence_refs": rejection_counts["UNKNOWN_EVIDENCE_REFERENCE"],
            "rejected_secondary_only_evidence": rejection_counts["SECONDARY_IMAGE_ONLY"],
            "target_truncation": "PENDING_EXACT_PROCESSOR_RECEIPT",
            "exact_support_intersection_rows": len(support_intersection),
            "exact_support_match_between_modes": exact_support_match,
            "targets_sha256": sha256_file(mode_paths[mode]),
        }
        report_path = args.output_root / f"{mode}_v5.audit.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        reports[mode] = {**report, "report_sha256": sha256_file(report_path)}

    fold_manifest = json.loads(args.fold_manifest.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "explanation_student_handoff_v2",
        "decision": "PENDING_EXACT_PROCESSOR_RECEIPT",
        "origin_commit": ORIGIN_COMMIT,
        "sealed": True,
        "public_submissions": 0,
        "label_contract": {
            "field": "label",
            "source": "competition train",
            "public_or_test_labels_used": 0,
            "teacher_output_label_is_integrity_echo_only": True,
        },
        "creation_scope": "competition train only; source-label-conditioned offline teacher",
        "source": {
            "dataset": "competition_train_v1",
            "dataset_sha256": sha256_file(args.data),
            "dataset_size_bytes": args.data.stat().st_size,
            "rows": len(order),
            "cv5_version": fold_manifest["evaluation_version"],
            "cv5_folds_sha256": sha256_file(args.folds),
            "cv5_folds_size_bytes": args.folds.stat().st_size,
            "cv5_manifest_sha256": sha256_file(args.fold_manifest),
            "cv5_manifest_size_bytes": args.fold_manifest.stat().st_size,
            "historical_oof_sha256": sha256_file(args.oof),
            "historical_oof_size_bytes": args.oof.stat().st_size,
            "teacher_output_sha256": sha256_file(teacher_path),
            "teacher_output_size_bytes": teacher_path.stat().st_size,
            "teacher_output_rows": len(teacher),
            "teacher_useful_explanations": teacher_report.get("useful_explanations"),
            "teacher_not_enough_evidence": teacher_report.get("not_enough_evidence"),
            "teacher_report_sha256": sha256_file(teacher_report_path),
            "teacher_report_size_bytes": teacher_report_path.stat().st_size,
            "teacher_model": teacher_report["model"],
            "teacher_revision": teacher_report["model_revision"],
            "teacher_generation_parameters": teacher_report["generation_parameters"],
            "teacher_prompt_sha256": teacher_report["prompt_py_sha256"],
            "teacher_generator_sha256": teacher_report["generator_py_sha256"],
            "failure_retry": {
                key: teacher_report.get(key)
                for key in (
                    "failure_retry_source_teacher_output_sha256",
                    "failure_retry_source_teacher_report_sha256",
                    "failure_retry_code_sha256",
                    "failure_retry_model",
                    "failure_retry_model_revision",
                    "failure_retry_workers",
                    "failure_retry_all_gallery_images",
                    "failure_retry_max_image_edge",
                    "failure_retry_pillow_version",
                    "failure_retry_requested",
                    "failure_retry_requested_transport",
                    "failure_retry_requested_nontransport_validation",
                    "failure_retry_source_useful_explanations",
                    "failure_retry_source_not_enough_evidence",
                    "failure_retry_source_intentional_not_enough_evidence",
                    "failure_retry_final_not_enough_evidence",
                    "failure_retry_final_useful",
                )
                if key in teacher_report
            },
            "student_scope_repair": {
                key: teacher_report.get(key)
                for key in (
                    "student_scope_repair_source_teacher_output_sha256",
                    "student_scope_repair_source_teacher_report_sha256",
                    "student_scope_repair_prompt_sha256",
                    "student_scope_repair_code_sha256",
                    "student_scope_repair_model",
                    "student_scope_repair_model_revision",
                    "student_scope_repair_generation_parameters",
                    "student_scope_repair_workers",
                    "student_scope_repair_requested",
                    "student_scope_repair_requested_nee_recovery",
                    "student_scope_repair_accepted",
                    "student_scope_repair_fallback",
                    "student_scope_repair_remained_not_enough_evidence",
                )
                if key in teacher_report
            },
        },
        "ensemble_image0_preprocessing": {
            "source": "unaltered first source image in the same gallery order as solution140",
            "source_files_reencoded_or_copied": False,
            "runtime_decode": "Pillow Image.open then convert RGB",
            "runtime_resize": "thumbnail 448x448 LANCZOS preserving aspect ratio",
            "runtime_serialization": "none; pass the in-memory PIL image directly to the Qwen3.5 processor",
            "reference": "experiments/140_dual_lora_fusion/submission/run.py::open_lora_image",
            "pillow_version": PILLOW_VERSION,
            "files": len(ensemble_images),
            "source_root_required_at_training_and_runtime": True,
            "manifest": image_manifest_path.name,
            "manifest_sha256": sha256_file(image_manifest_path),
            "tree_sha256": tree_sha256(image_manifest_rows),
            "total_size_bytes": sum(row["size_bytes"] for row in image_manifest_rows),
        },
        "outputs": {
            **{
                mode: {
                    "path": mode_paths[mode].name,
                    "sha256": sha256_file(mode_paths[mode]),
                    "size_bytes": mode_paths[mode].stat().st_size,
                    "rows": len(accepted_ids[mode]),
                }
                for mode in MODES
            },
            "row_manifest": {
                "path": row_manifest_path.name,
                "sha256": sha256_file(row_manifest_path),
                "size_bytes": row_manifest_path.stat().st_size,
                "rows": EXPECTED_ROWS,
            },
            "accepted_rows": {
                "path": accepted_path.name,
                "sha256": sha256_file(accepted_path),
                "size_bytes": accepted_path.stat().st_size,
                "rows": len(support_intersection),
            },
            "rejections": {
                "path": rejections_path.name,
                "sha256": sha256_file(rejections_path),
                "size_bytes": rejections_path.stat().st_size,
                "rows": sum(rejection_counts.values()),
                "reason_counts": dict(sorted(rejection_counts.items())),
            },
            "matched_support_digit_control": {
                "path": matched_digit_path.name,
                "sha256": sha256_file(matched_digit_path),
                "size_bytes": matched_digit_path.stat().st_size,
                "rows": len(support_intersection),
                "same_support_and_order_as_rationale": True,
            },
            "ensemble_image0_manifest": {
                "path": image_manifest_path.name,
                "sha256": sha256_file(image_manifest_path),
                "size_bytes": image_manifest_path.stat().st_size,
                "rows": len(image_manifest_rows),
            },
            "validation_folds": {
                "path": "validation/folds.csv",
                "sha256": sha256_file(copied_folds),
                "size_bytes": copied_folds.stat().st_size,
                "rows": EXPECTED_ROWS,
            },
            "validation_manifest": {
                "path": "validation/manifest.json",
                "sha256": sha256_file(copied_fold_manifest),
                "size_bytes": copied_fold_manifest.stat().st_size,
            },
            "historical_oof": {
                "path": "validation/four_head_oof.npz",
                "sha256": sha256_file(copied_oof),
                "size_bytes": copied_oof.stat().st_size,
                "rows": EXPECTED_ROWS,
                "ids_and_fold_ids_exact_cv5": True,
            },
        },
        "mode_audits": reports,
        "builder": {
            "path": "experiments/712_qwen35_397b_evidence_teacher/build_handoff_artifact.py",
            "sha256": sha256_file(Path(__file__)),
            "target_builder_sha256": sha256_file(Path(__file__).with_name("build_student_targets.py")),
            "rationale_contract_sha256": sha256_file(
                args.code_root / "713" / "target_contract.py"
            ),
            "explanation_contract_sha256": sha256_file(
                args.code_root / "714" / "target_contract.py"
            ),
        },
        "training_ownership": "downstream training owner after user-approved receipt",
        "training_blocked_until_receipt": True,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("HANDOFF_MANIFEST=" + compact_json(manifest), flush=True)


if __name__ == "__main__":
    main()
