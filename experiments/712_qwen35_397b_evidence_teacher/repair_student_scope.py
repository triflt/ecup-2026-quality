from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from build_student_targets import student_scope_errors
from generate_all import (
    GENERATION_PARAMETERS,
    call_api,
    data_url,
    index_images,
    load_rows,
    parse_json_response,
    repair_near_exact_text_span,
    sha256_file,
    wait_for_api,
)
from prompt import ALLOWED_REASONS, validate_output


REPAIR_SYSTEM_PROMPT = """Ты исправляешь короткое объяснение для уже заданного поля label товарной карточки.

Student увидит только название, описание и первое изображение image:0. Поэтому:
1. Не меняй входной label и не спорь с ним. Если его нельзя честно объяснить доступными данными, верни not_enough_evidence.
2. evidence.source может быть только title, description, image:0 или absence. Для title/description value скопируй посимвольно как одну непрерывную подстроку.
3. Каждая фактическая деталь explanation должна следовать из выбранного evidence либо буквально дублироваться в названии/описании/image:0. Не используй остальные изображения и не пиши «на изображениях» или «на фотографиях».
4. Если evidence не image:0, не добавляй фразы «на упаковке», «на этикетке», «на изображении», «видно» и подобные визуальные утверждения.
5. explanation — конкретный русский комментарий 50–300 символов: назови продаваемый объект и решающий товарный факт. Не пиши о метке, классе, категории, критериях, правилах, соответствии требованиям или решении модели.
6. Заверши фактом о товаре. Не добавляй рассуждения сверх одного проверяемого основания.
7. Явно запрещены обороты «соответствует категории», «подтверждает категорию», «исключает отнесение к категории», «соответствует требованиям» и «не соответствует критерию». Вместо них просто напиши, что именно указано о продаваемом товаре.
8. Если исходные reason и evidence уже валидны и доступны, сохрани их: стилистическая ошибка explanation не является основанием для not_enough_evidence. Перепиши только explanation.
9. Верни только JSON ровно с полями label, reason, evidence, explanation. Для not_enough_evidence последние два поля null.
10. Если исходный reason — not_enough_evidence, сделай одну свежую попытку найти честное основание заданного label в названии, описании или image:0. Сохрани not_enough_evidence, если такого основания действительно нет.
"""


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_payload(
    row: dict[str, str], image0: Path, original: dict[str, Any], *, model: str
) -> dict[str, Any]:
    reasons = ", ".join(sorted(ALLOWED_REASONS[(str(row["category"]), int(row["label"]))]))
    text = f"""Категория: {row['category']}
label: {int(row['label'])}
Название:
<<<{row['name']}>>>
Описание:
<<<{row['description']}>>>
Доступно только первое изображение image:0.
Допустимые reason: {reasons}.

Исходный JSON, который нужно перепроверить или исправить для student-входа:
{json.dumps(original, ensure_ascii=False, sort_keys=True)}

Сформируй заново один полностью поддержанный JSON."""
    return {
        "model": model,
        **GENERATION_PARAMETERS,
        "messages": [
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url(image0)}},
                {"type": "text", "text": text},
            ]},
        ],
    }


def validate_student_output(parsed: Any, row: dict[str, str]) -> list[str]:
    validation = validate_output(
        parsed,
        category=str(row["category"]),
        expected_label=int(row["label"]),
        name=str(row["name"]),
        description=str(row["description"]),
        image_count=1,
    )
    errors = list(validation.errors)
    if isinstance(parsed, dict):
        errors.extend(student_scope_errors(parsed, max_student_image_index=0))
    return errors


def repair_one(
    row: dict[str, str], image0: Path, record: dict[str, Any], *, api_base: str,
    model: str, timeout: int,
) -> dict[str, Any]:
    original = record.get("parsed_output")
    if not isinstance(original, dict):
        return record
    recover_not_enough_evidence = original.get("reason") == "not_enough_evidence"
    source = str((original.get("evidence") or {}).get("source"))
    original_errors = (
        ["NOT_ENOUGH_EVIDENCE_RECOVERY"]
        if recover_not_enough_evidence
        else list(student_scope_errors(original, max_student_image_index=0))
    )
    if source.startswith("image:") and source != "image:0":
        original_errors.append("SECONDARY_IMAGE_ONLY")
    if not original_errors:
        return record

    started = time.monotonic()
    payload = request_payload(row, image0, original, model=model)
    attempts: list[dict[str, Any]] = []
    parsed: dict[str, Any] | None = None
    errors: list[str] = []
    for attempt in range(1, 3):
        raw = ""
        usage: dict[str, Any] = {}
        try:
            raw, usage = call_api(api_base, payload, timeout)
            parsed = parse_json_response(raw)
            errors = validate_student_output(parsed, row)
        except (urllib.error.URLError, TimeoutError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors = [f"{type(error).__name__}: {error}"]
        attempts.append({
            "attempt": attempt,
            "raw_output": raw,
            "validation_errors": list(errors),
            "usage": usage,
        })
        if not errors:
            break
        if isinstance(parsed, dict):
            parsed, action = repair_near_exact_text_span(
                parsed, name=str(row["name"]), description=str(row["description"])
            )
            if action is not None:
                errors = validate_student_output(parsed, row)
                if not errors:
                    attempts[-1]["normalization_action"] = action
                    break
        payload = dict(payload)
        payload["messages"] = list(payload["messages"]) + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": (
                "JSON всё ещё не прошёл проверку: " + "; ".join(errors)
                + ". Если указан CLASSIFIER_META_LANGUAGE, сохрани валидные reason и "
                  "evidence, убери все слова о категории, критерии, соответствии и "
                  "подтверждении; закончи буквальным фактом о товаре. Исправь только "
                  "эти ошибки. Если доступного evidence действительно нет, верни "
                  "not_enough_evidence. Только JSON."
            )},
        ]

    output = dict(record)
    image0_sha256 = sha256_file(image0)
    repair_input = {
        "category": str(row["category"]),
        "description": str(row["description"]),
        "id": str(row["id"]),
        "image0_sha256": image0_sha256,
        "label": int(row["label"]),
        "name": str(row["name"]),
        "original_output": original,
        "repair_prompt_sha256": hashlib.sha256(
            REPAIR_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
    }
    output["student_scope_repair"] = {
        "original_errors": sorted(set(original_errors)),
        "image0_sha256": image0_sha256,
        "canonical_input_sha256": hashlib.sha256(
            compact_json(repair_input).encode("utf-8")
        ).hexdigest(),
        "attempts": attempts,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }
    actions = list(output.get("normalization_actions") or [])
    if recover_not_enough_evidence:
        actions.append("student_scope_repair_nee_recovery_requested")
    if errors or not isinstance(parsed, dict):
        parsed = {
            "label": int(row["label"]),
            "reason": "not_enough_evidence",
            "evidence": None,
            "explanation": None,
        }
        actions.append("student_scope_repair_fallback_not_enough_evidence")
    elif parsed.get("reason") == "not_enough_evidence":
        actions.append("student_scope_repair_remained_not_enough_evidence")
    else:
        actions.append("student_scope_repair_accept")
    output["parsed_output"] = parsed
    output["accepted"] = True
    output["validation_errors"] = []
    output["normalization_actions"] = actions
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
    args = parser.parse_args()

    wait_for_api(args.api_base, 900)
    rows = load_rows(args.data)
    by_id = {str(row["id"]): row for row in rows}
    images = index_images(args.image_root, list(by_id))
    teacher_paths = sorted(args.teacher_root.rglob("explanation_synth_v5.jsonl"))
    report_paths = sorted(args.teacher_root.rglob("explanation_synth_v5_report.json"))
    if not teacher_paths:
        teacher_paths = sorted(args.teacher_root.rglob("teacher_smoke20_v5.jsonl"))
    if not report_paths:
        report_paths = sorted(args.teacher_root.rglob("smoke20_v5_report.json"))
    if len(teacher_paths) != 1 or len(report_paths) != 1:
        raise ValueError("expected one full teacher corpus and report")
    source_report = json.loads(report_paths[0].read_text(encoding="utf-8"))
    if str(source_report.get("model_revision")) != args.model_revision:
        raise ValueError("repair model revision differs from source teacher revision")
    with teacher_paths[0].open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    record_ids = [str(record["id"]) for record in records]
    if len(record_ids) != len(set(record_ids)) or not set(record_ids).issubset(by_id):
        raise ValueError("teacher/data scope mismatch")

    def task(record: dict[str, Any]) -> dict[str, Any]:
        item_id = str(record["id"])
        return repair_one(
            by_id[item_id], images[item_id][0], record,
            api_base=args.api_base, model=args.model, timeout=args.timeout,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        repaired = list(pool.map(task, records))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for index, record in enumerate(repaired, 1):
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            if index % 500 == 0 or index == len(repaired):
                print(f"scope_repair_written={index}/{len(repaired)}", flush=True)

    actions = Counter(
        action
        for record in repaired
        for action in (record.get("normalization_actions") or [])
    )
    final_nee = sum(
        (record.get("parsed_output") or {}).get("reason") == "not_enough_evidence"
        for record in repaired
    )
    report = {
        **source_report,
        "student_scope_repair_source_teacher_output_sha256": sha256_file(
            teacher_paths[0]
        ),
        "student_scope_repair_source_teacher_report_sha256": sha256_file(
            report_paths[0]
        ),
        "student_scope_repair_prompt_sha256": hashlib.sha256(
            REPAIR_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "student_scope_repair_code_sha256": sha256_file(Path(__file__)),
        "student_scope_repair_model": args.model,
        "student_scope_repair_model_revision": args.model_revision,
        "student_scope_repair_generation_parameters": GENERATION_PARAMETERS,
        "student_scope_repair_workers": args.workers,
        "student_scope_repair_requested": actions["student_scope_repair_accept"]
        + actions["student_scope_repair_fallback_not_enough_evidence"]
        + actions["student_scope_repair_remained_not_enough_evidence"],
        "student_scope_repair_requested_nee_recovery": actions[
            "student_scope_repair_nee_recovery_requested"
        ],
        "student_scope_repair_accepted": actions["student_scope_repair_accept"],
        "student_scope_repair_fallback": actions["student_scope_repair_fallback_not_enough_evidence"],
        "student_scope_repair_remained_not_enough_evidence": actions[
            "student_scope_repair_remained_not_enough_evidence"
        ],
        "not_enough_evidence": final_nee,
        "useful_explanations": len(repaired) - final_nee,
        "derived_output_sha256": sha256_file(args.output),
        "derived_output_rows": len(repaired),
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("STUDENT_SCOPE_REPAIR_REPORT=" + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
