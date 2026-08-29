from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, __version__ as PILLOW_VERSION

from generate_all import index_images, load_rows, process_one, sha256_file, wait_for_api


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def record_sha256(record: dict[str, Any]) -> str:
    return hashlib.sha256(compact_json(record).encode("utf-8")).hexdigest()


def needs_retry(record: dict[str, Any]) -> bool:
    actions = set(record.get("normalization_actions") or [])
    if "fallback_not_enough_evidence_after_failed_validation" not in actions:
        return False
    attempts = list(record.get("generation_attempts") or [])
    return bool(attempts) and any(attempt.get("validation_errors") for attempt in attempts)


def has_transport_error(record: dict[str, Any]) -> bool:
    return any(
        str(error).startswith("transport_error:")
        for attempt in (record.get("generation_attempts") or [])
        for error in (attempt.get("validation_errors") or [])
    )


def resize_gallery(
    paths: list[Path], root: Path, item_id: str, max_edge: int
) -> list[Path]:
    output: list[Path] = []
    item_root = root / item_id
    item_root.mkdir(parents=True, exist_ok=False)
    for index, source in enumerate(paths):
        destination = item_root / f"{index}.jpg"
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            image.save(destination, format="JPEG", quality=92)
        output.append(destination)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-image-edge", type=int, default=768)
    parser.add_argument("--retry-image-root", type=Path, default=Path("/work/retry_images"))
    args = parser.parse_args()

    wait_for_api(args.api_base, 900)
    rows = load_rows(args.data)
    by_id = {str(row["id"]): row for row in rows}
    images = index_images(args.image_root, list(by_id))
    teacher_paths = sorted(args.teacher_root.rglob("explanation_synth_v5.jsonl"))
    report_paths = sorted(args.teacher_root.rglob("explanation_synth_v5_report.json"))
    if len(teacher_paths) != 1 or len(report_paths) != 1:
        raise ValueError("expected one full teacher corpus and report")
    source_report = json.loads(report_paths[0].read_text(encoding="utf-8"))
    if str(source_report.get("model_revision")) != args.model_revision:
        raise ValueError("retry model revision differs from source teacher revision")
    with teacher_paths[0].open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    ids = [str(record["id"]) for record in records]
    if len(records) != len(rows) or ids != [str(row["id"]) for row in rows]:
        raise ValueError("teacher/data order or scope mismatch")

    selected_source_records = [record for record in records if needs_retry(record)]
    selected = {str(record["id"]) for record in selected_source_records}
    requested_transport = sum(has_transport_error(record) for record in selected_source_records)
    source_not_enough_evidence = sum(
        (record.get("parsed_output") or {}).get("reason") == "not_enough_evidence"
        for record in records
    )
    transformed: dict[str, list[Path]] = {}
    for index, item_id in enumerate(sorted(selected), 1):
        transformed[item_id] = resize_gallery(
            images[item_id], args.retry_image_root, item_id, args.max_image_edge
        )
        if index % 100 == 0 or index == len(selected):
            print(f"retry_images={index}/{len(selected)}", flush=True)

    def task(record: dict[str, Any]) -> dict[str, Any]:
        item_id = str(record["id"])
        if item_id not in selected:
            return record
        retried = process_one(
            by_id[item_id], transformed[item_id], api_base=args.api_base,
            model=args.model, timeout=args.timeout,
        )
        output = dict(retried)
        output["image_sha256"] = list(record["image_sha256"])
        output["source_teacher_record_sha256"] = record_sha256(record)
        output["source_teacher_canonical_input_sha256"] = record.get("canonical_input_sha256")
        output["teacher_input_image_sha256"] = [
            sha256_file(path) for path in transformed[item_id]
        ]
        output["teacher_image_preprocessing"] = {
            "all_gallery_images": True,
            "decode": "Pillow convert RGB",
            "resize": f"thumbnail {args.max_image_edge}x{args.max_image_edge} LANCZOS",
            "serialization": "JPEG quality=92",
        }
        actions = list(output.get("normalization_actions") or [])
        actions.append(f"retry_failed_teacher_row_all_images_max_edge_{args.max_image_edge}")
        output["normalization_actions"] = actions
        return output

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        final_records = list(pool.map(task, records))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for index, record in enumerate(final_records, 1):
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if index % 500 == 0 or index == len(final_records):
                print(f"retry_written={index}/{len(final_records)}", flush=True)

    selected_records = [record for record in final_records if str(record["id"]) in selected]
    reasons = Counter(
        str((record.get("parsed_output") or {}).get("reason")) for record in selected_records
    )
    report = {
        **source_report,
        "failure_retry_source_teacher_output_sha256": sha256_file(teacher_paths[0]),
        "failure_retry_source_teacher_report_sha256": sha256_file(report_paths[0]),
        "failure_retry_code_sha256": sha256_file(Path(__file__)),
        "failure_retry_model": args.model,
        "failure_retry_model_revision": args.model_revision,
        "failure_retry_workers": args.workers,
        "failure_retry_all_gallery_images": True,
        "failure_retry_max_image_edge": args.max_image_edge,
        "failure_retry_pillow_version": PILLOW_VERSION,
        "failure_retry_requested": len(selected),
        "failure_retry_requested_transport": requested_transport,
        "failure_retry_requested_nontransport_validation": len(selected) - requested_transport,
        "failure_retry_source_useful_explanations": len(records)
        - source_not_enough_evidence,
        "failure_retry_source_not_enough_evidence": source_not_enough_evidence,
        "failure_retry_source_intentional_not_enough_evidence": (
            source_not_enough_evidence - len(selected)
        ),
        "failure_retry_final_not_enough_evidence": reasons["not_enough_evidence"],
        "failure_retry_final_useful": len(selected) - reasons["not_enough_evidence"],
        "derived_output_sha256": sha256_file(args.output),
        "derived_output_rows": len(final_records),
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("FAILURE_RETRY_REPORT=" + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
