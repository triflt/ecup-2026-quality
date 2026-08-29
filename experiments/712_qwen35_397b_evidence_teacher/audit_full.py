from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from build_student_targets import student_scope_errors
from prompt import validate_output


EXPECTED_ROWS = 12_971
EXPECTED_DATA_SHA256 = "4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_data(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        url = os.environ.get("DATA_CSV_URL")
        if not url:
            raise ValueError("DATA_CSV_URL is required")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    actual = sha256_file(path)
    if actual != EXPECTED_DATA_SHA256:
        raise ValueError(f"data SHA mismatch: {actual}")
    with path.open(encoding="utf-8", newline="") as stream:
        return {str(row["id"]): row for row in csv.DictReader(stream)}


def load_records(root: Path) -> tuple[Path, list[dict[str, Any]]]:
    candidates = sorted(root.rglob("explanation_synth_v5.jsonl"))
    if len(candidates) != 1:
        raise ValueError(f"expected one explanation_synth_v5.jsonl, got {candidates}")
    with candidates[0].open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    return candidates[0], records


def blind_rank(item_id: str, cell: tuple[str, int]) -> bytes:
    return hashlib.sha256(f"explanation-blind200-v1|{cell[0]}|{cell[1]}|{item_id}".encode()).digest()


def image_paths(root: Path, item_id: str) -> list[Path]:
    directory = root / item_id
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes),
        key=lambda path: (int(path.stem) if path.stem.isdigit() else 10**9, path.name),
    ) if directory.is_dir() else []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    data = ensure_data(args.data)
    teacher_path, records = load_records(args.teacher_root)
    if len(data) != EXPECTED_ROWS or len(records) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, data={len(data)} records={len(records)}")
    by_id = {str(record["id"]): record for record in records}
    if len(by_id) != EXPECTED_ROWS or set(by_id) != set(data):
        raise ValueError("teacher/data id set mismatch or duplicate teacher ids")

    reasons: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    normalizations: Counter[str] = Counter()
    cell_total: Counter[tuple[str, int]] = Counter()
    cell_useful: Counter[tuple[str, int]] = Counter()
    cell_useful_conservative: Counter[tuple[str, int]] = Counter()
    comments: Counter[str] = Counter()
    invalid: list[dict[str, Any]] = []
    student_scope_error_counts: Counter[str] = Counter()
    useful: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for item_id, row in data.items():
        record = by_id[item_id]
        parsed = record.get("parsed_output")
        validation = validate_output(
            parsed,
            category=str(row["category"]),
            expected_label=int(row["label"]),
            name=str(row["name"]),
            description=str(row["description"]),
            image_count=int(record["image_count"]),
        )
        scope_errors = list(student_scope_errors(parsed, max_student_image_index=0)) \
            if isinstance(parsed, dict) else []
        evidence = parsed.get("evidence") if isinstance(parsed, dict) else None
        source = str(evidence.get("source")) if isinstance(evidence, dict) else "none"
        if source.startswith("image:") and source != "image:0":
            scope_errors.append("SECONDARY_IMAGE_ONLY")
        student_scope_error_counts.update(scope_errors)
        row_errors = list(validation.errors) + scope_errors
        if row_errors or not record.get("accepted"):
            invalid.append({"id": item_id, "errors": row_errors})
        cell = (str(row["category"]), int(row["label"]))
        cell_total[cell] += 1
        reason = str(parsed.get("reason")) if isinstance(parsed, dict) else "INVALID"
        reasons[reason] += 1
        sources[source] += 1
        for action in record.get("normalization_actions") or []:
            normalizations[str(action).split(":", 1)[0]] += 1
        explanation = parsed.get("explanation") if isinstance(parsed, dict) else None
        if isinstance(explanation, str) and not row_errors and record.get("accepted"):
            recovered_from_nee = (
                "student_scope_repair_nee_recovery_requested"
                in (record.get("normalization_actions") or [])
            )
            cell_useful[cell] += 1
            if not recovered_from_nee:
                cell_useful_conservative[cell] += 1
            comments[" ".join(explanation.casefold().split())] += 1
            useful.setdefault(cell, []).append({
                "id": item_id,
                "category": cell[0],
                "name": str(row["name"]),
                "description": str(row["description"]),
                "image_count": int(record["image_count"]),
                "recovered_from_not_enough_evidence": recovered_from_nee,
                "output": {
                    "reason": parsed["reason"],
                    "evidence": parsed["evidence"],
                    "explanation": parsed["explanation"],
                },
            })

    blind: list[dict[str, Any]] = []
    for cell in sorted(cell_total):
        candidates = sorted(useful.get(cell, []), key=lambda row: blind_rank(row["id"], cell))
        if len(candidates) < 50:
            raise ValueError(f"fewer than 50 useful rows in cell={cell}")
        blind.extend(candidates[:50])
    blind.sort(
        key=lambda row: hashlib.sha256(
            f"explanation-blind200-order-v1|{row['id']}".encode()
        ).digest()
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    blind_images_root = args.output_root / "blind200_images"
    blind_image_files = 0
    for index, row in enumerate(blind, 1):
        row["audit_index"] = index
        item_id = str(row["id"])
        paths = image_paths(args.image_root, item_id)
        teacher_hashes = list(by_id[item_id].get("image_sha256") or [])
        actual_hashes = [sha256_file(path) for path in paths]
        if len(paths) != int(row["image_count"]):
            raise ValueError(
                f"blind image count mismatch for id={item_id}: "
                f"expected={row['image_count']} actual={len(paths)}"
            )
        if actual_hashes != teacher_hashes:
            raise ValueError(f"blind image SHA/order mismatch for id={item_id}")
        card_root = blind_images_root / f"{index:03d}"
        card_root.mkdir(parents=True, exist_ok=False)
        relative_files: list[str] = []
        for image_index, path in enumerate(paths):
            destination = card_root / f"{image_index:02d}{path.suffix.lower()}"
            shutil.copyfile(path, destination)
            relative_files.append(str(destination.relative_to(args.output_root)))
        row["image_files"] = relative_files
        row["image_sha256"] = actual_hashes
        blind_image_files += len(paths)
    coverage = {
        f"{cell[0]}|{cell[1]}": cell_useful[cell] / total
        for cell, total in sorted(cell_total.items())
    }
    conservative_coverage = {
        f"{cell[0]}|{cell[1]}": cell_useful_conservative[cell] / total
        for cell, total in sorted(cell_total.items())
    }
    duplicate_rows = sum(count - 1 for count in comments.values() if count > 1)
    report = {
        "schema_version": "explanation_full_audit_v1",
        "rows": len(records),
        "accepted_rows": len(records) - len(invalid),
        "invalid_rows": len(invalid),
        "first_invalid": invalid[:20],
        "useful_explanations": sum(cell_useful.values()),
        "useful_coverage": sum(cell_useful.values()) / len(records),
        "not_enough_evidence": reasons["not_enough_evidence"],
        "cell_useful_coverage": coverage,
        "cell_useful_coverage_min_085_passed": all(
            value >= 0.85 for value in coverage.values()
        ),
        "cell_useful_conservative_coverage_excluding_nee_recovery": (
            conservative_coverage
        ),
        "useful_recovered_from_not_enough_evidence": sum(cell_useful.values())
        - sum(cell_useful_conservative.values()),
        "reason_counts": dict(sorted(reasons.items())),
        "evidence_source_counts": dict(sorted(sources.items())),
        "normalization_action_counts": dict(sorted(normalizations.items())),
        "student_scope_error_counts": dict(sorted(student_scope_error_counts.items())),
        "unique_comments": len(comments),
        "duplicate_comment_rows": duplicate_rows,
        "blind_packet_rows": len(blind),
        "blind_image_files": blind_image_files,
        "blind_images_sha_verified_against_teacher": True,
        "teacher_sha256": sha256_file(teacher_path),
        "data_sha256": sha256_file(args.data),
    }
    report_path = args.output_root / "full_audit_report.json"
    packet_path = args.output_root / "blind200_packet.jsonl"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with packet_path.open("w", encoding="utf-8") as stream:
        for row in blind:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print("FULL_AUDIT_SUMMARY=" + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    print("BLIND_PACKET_SHA256=" + sha256_file(packet_path), flush=True)
    if invalid or report["useful_coverage"] < 0.85:
        raise SystemExit("full corpus failed structural or overall coverage gate")


if __name__ == "__main__":
    main()
