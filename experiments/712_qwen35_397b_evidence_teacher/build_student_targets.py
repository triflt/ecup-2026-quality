from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def student_scope_errors(
    teacher: dict[str, Any], *, max_student_image_index: int
) -> tuple[str, ...]:
    evidence = teacher.get("evidence")
    explanation = str(teacher.get("explanation") or "")
    source = str(evidence.get("source")) if isinstance(evidence, dict) else ""
    errors: list[str] = []
    if max_student_image_index == 0 and re.search(
        r"\b(?:изображени(?:ях|ями)|фотографи(?:ях|ями))\b", explanation, re.IGNORECASE
    ):
        errors.append("SECONDARY_IMAGE_LANGUAGE")
    if source != "image:0" and re.search(
        r"(?:\bна\s+(?:упаковк\w*|этикетк\w*|изображени\w*|фотографи\w*)|"
        r"\b(?:видно|виден|видна|видны|изображено|изображена|изображены)\b)",
        explanation,
        re.IGNORECASE,
    ):
        errors.append("UNBOUND_VISUAL_CLAIM")
    if re.search(
        r"(?:\bсоответств\w*\s+(?:требован\w*|категори\w*)|"
        r"\bподтвержда\w*\s+(?:требован\w*|категори\w*)|"
        r"\bотнесени\w*\s+к\s+категори\w*|\bкритери\w*)",
        explanation,
        re.IGNORECASE,
    ):
        errors.append("CLASSIFIER_META_LANGUAGE")
    return tuple(errors)


def student_payload(teacher: dict[str, Any], *, max_student_image_index: int) -> dict[str, Any]:
    if teacher["reason"] == "not_enough_evidence":
        raise ValueError("teacher did not produce a supported explanation")
    evidence = teacher["evidence"]
    source = str(evidence["source"])
    if source.startswith("image:") and int(source.split(":", 1)[1]) > max_student_image_index:
        raise ValueError("teacher evidence refers to an image unavailable to the student")
    scope_errors = student_scope_errors(
        teacher, max_student_image_index=max_student_image_index
    )
    if scope_errors:
        raise ValueError(";".join(scope_errors))
    return {
        "reason": str(teacher["reason"]),
        "evidence": {"source": source, "value": str(evidence["value"])},
        "explanation": str(teacher["explanation"]),
    }


def rationale_then_label(
    teacher: dict[str, Any], label: int, *, max_student_image_index: int = 0
) -> str:
    payload = student_payload(teacher, max_student_image_index=max_student_image_index)
    return f"{compact_json(payload)}\nLABEL={int(label)}"


def explanation_only(
    teacher: dict[str, Any], label: int, *, max_student_image_index: int = 0
) -> str:
    if int(teacher["label"]) != int(label):
        raise ValueError("output label does not echo the source label")
    payload = student_payload(teacher, max_student_image_index=max_student_image_index)
    explanation = str(payload["explanation"]).strip()
    if not 50 <= len(explanation) <= 300:
        raise ValueError("teacher explanation must contain 50-300 characters")
    return explanation


def load_teacher(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("accepted"):
                output[str(record["id"])] = record["parsed_output"]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--mode", choices=("rationale_then_label", "explanation_only"), required=True)
    parser.add_argument("--max-student-image-index", type=int, default=0)
    args = parser.parse_args()

    teacher = load_teacher(args.teacher)
    accepted = 0
    rejected: list[dict[str, str]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.data.open(encoding="utf-8", newline="") as source, args.output.open("w", encoding="utf-8") as destination:
        for row in csv.DictReader(source):
            item_id = str(row["id"])
            try:
                parsed = teacher[item_id]
                label = int(row["label"])
                if int(parsed["label"]) != label:
                    raise ValueError("output label does not echo the source label")
                if args.mode == "rationale_then_label":
                    target = rationale_then_label(
                        parsed, label, max_student_image_index=args.max_student_image_index
                    )
                else:
                    target = explanation_only(
                        parsed, label, max_student_image_index=args.max_student_image_index
                    )
                record = {
                    "id": item_id,
                    "label": label,
                    "target": target,
                }
                destination.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                accepted += 1
            except (KeyError, TypeError, ValueError) as error:
                rejected.append({"id": item_id, "error": f"{type(error).__name__}: {error}"})
    rejection_counts = Counter(row["error"] for row in rejected)
    report = {
        "mode": args.mode,
        "data_rows": accepted + len(rejected),
        "accepted": accepted,
        "rejected": len(rejected),
        "coverage": accepted / (accepted + len(rejected)) if accepted + len(rejected) else 0.0,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "first_rejections": rejected[:20],
        "targets_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
