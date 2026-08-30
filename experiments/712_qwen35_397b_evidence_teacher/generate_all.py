from __future__ import annotations

import argparse
import base64
import csv
import difflib
import hashlib
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from prompt import (
    MODEL_ID,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    parse_json_response,
    user_prompt,
    validate_output,
)


EXPECTED_ROWS = 12_971
EXPECTED_IMAGES = 49_456
MODEL_REVISION = "ea5b4f81096f3901c91dea97f81324302495781d"
GENERATION_PARAMETERS = {
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 384,
    "response_format": {"type": "json_object"},
    "chat_template_kwargs": {"enable_thinking": False},
}
MAX_VALIDATION_ATTEMPTS = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"id", "name", "description", "category"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"input CSV must contain {sorted(required)}")
    ids = [str(row["id"]).strip() for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate product ids")
    return rows


def index_images(root: Path, ids: list[str]) -> dict[str, list[Path]]:
    output: dict[str, list[Path]] = {}
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    for item_id in ids:
        directory = root / item_id
        paths = sorted(
            (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes),
            key=lambda path: (int(path.stem) if path.stem.isdigit() else 10**9, path.name),
        ) if directory.is_dir() else []
        if not paths:
            raise ValueError(f"no images for id={item_id} under {root}")
        output[item_id] = paths
    return output


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def request_payload(row: dict[str, str], images: list[Path], *, model: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": data_url(path)}} for path in images
    ]
    content.append({
        "type": "text",
        "text": user_prompt(
            category=str(row["category"]), label=int(row["label"]), name=str(row["name"]),
            description=str(row["description"]), image_count=len(images),
        ),
    })
    return {
        "model": model,
        **GENERATION_PARAMETERS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    }


def call_api(url: str, payload: dict[str, Any], timeout: int) -> tuple[str, dict[str, Any]]:
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        envelope = json.loads(response.read().decode("utf-8"))
    raw = envelope["choices"][0]["message"]["content"]
    return str(raw), envelope.get("usage", {})


def wait_for_api(url: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    health_url = url.rstrip("/") + "/health"
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            with urllib.request.urlopen(health_url, timeout=15) as response:
                if response.status == 200:
                    print(f"api_ready attempts={attempt}", flush=True)
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        if attempt % 4 == 0:
            print(f"waiting_for_api attempts={attempt}", flush=True)
        time.sleep(15)
    raise TimeoutError(f"API did not become healthy within {timeout}s: {health_url}")


def repair_near_exact_text_span(
    parsed: dict[str, Any], *, name: str, description: str
) -> tuple[dict[str, Any], str | None]:
    evidence = parsed.get("evidence")
    if not isinstance(evidence, dict):
        return parsed, None
    source = evidence.get("source")
    value = evidence.get("value")
    if source not in {"title", "description"} or not isinstance(value, str):
        return parsed, None
    source_text = name if source == "title" else description
    if value in source_text:
        return parsed, None
    candidates = [source_text] if source == "title" else [
        span.strip() for span in re.split(r"(?<=[.!?])(?:\s+|<br\s*/?>)+", source_text)
        if 5 <= len(span.strip()) <= 500
    ]
    if not candidates:
        return parsed, None
    replacement = max(candidates, key=lambda span: difflib.SequenceMatcher(None, value, span).ratio())
    similarity = difflib.SequenceMatcher(None, value, replacement).ratio()
    if similarity < 0.92:
        return parsed, None
    repaired = dict(parsed)
    repaired["evidence"] = dict(evidence)
    repaired["evidence"]["value"] = replacement
    return repaired, f"near_exact_{source}_span:{similarity:.6f}"


def process_one(
    row: dict[str, str], images: list[Path], *, api_base: str, model: str, timeout: int
) -> dict[str, Any]:
    started = time.monotonic()
    image_hashes = [sha256_file(path) for path in images]
    canonical_input = json.dumps(
        {
            "id": str(row["id"]),
            "category": str(row["category"]),
            "name": str(row["name"]),
            "description": str(row["description"]),
            "image_sha256": image_hashes,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = request_payload(row, images, model=model)
    raw = ""
    usage: dict[str, Any] = {}
    parsed: dict[str, Any] | None = None
    errors: list[str] = []
    generation_attempts: list[dict[str, Any]] = []
    normalization_actions: list[str] = []
    for validation_attempt in range(1, MAX_VALIDATION_ATTEMPTS + 1):
        transport_error: str | None = None
        for transport_attempt in range(1, 4):
            try:
                raw, usage = call_api(api_base, payload, timeout)
                transport_error = None
                break
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as error:
                transport_error = f"{type(error).__name__}: {error}"
                if transport_attempt < 3:
                    time.sleep(transport_attempt)
        parsed = None
        errors = []
        if transport_error is None:
            try:
                parsed = parse_json_response(raw)
                result = validate_output(
                    parsed, category=str(row["category"]), expected_label=int(row["label"]),
                    name=str(row["name"]), description=str(row["description"]),
                    image_count=len(images),
                )
                errors.extend(result.errors)
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                errors.append(f"parse_error: {type(error).__name__}: {error}")
        else:
            errors.append(f"transport_error: {transport_error}")
        generation_attempts.append({
            "attempt": validation_attempt,
            "raw_output": raw,
            "validation_errors": list(errors),
            "usage": usage,
        })
        if not errors or transport_error is not None or validation_attempt == MAX_VALIDATION_ATTEMPTS:
            break
        repair_instruction = (
            "Предыдущий JSON не прошёл автоматическую проверку:\n- "
            + "\n- ".join(errors)
            + "\nИсправь только эти ошибки, заново сверившись с исходной карточкой и всеми "
              "изображениями. Если gold нельзя обосновать без нарушения проверки, верни "
              "not_enough_evidence с evidence=null и explanation=null. Верни только исправленный JSON."
        )
        payload = dict(payload)
        payload["messages"] = list(payload["messages"]) + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": repair_instruction},
        ]
    if errors and isinstance(parsed, dict):
        parsed, span_action = repair_near_exact_text_span(
            parsed, name=str(row["name"]), description=str(row["description"])
        )
        if span_action is not None:
            normalization_actions.append(span_action)
            result = validate_output(
                parsed, category=str(row["category"]), expected_label=int(row["label"]),
                name=str(row["name"]), description=str(row["description"]),
                image_count=len(images),
            )
            errors = list(result.errors)
    if errors:
        normalization_actions.append("fallback_not_enough_evidence_after_failed_validation")
        parsed = {
            "label": int(row["label"]),
            "reason": "not_enough_evidence",
            "evidence": None,
            "explanation": None,
        }
        result = validate_output(
            parsed, category=str(row["category"]), expected_label=int(row["label"]),
            name=str(row["name"]), description=str(row["description"]),
            image_count=len(images),
        )
        errors = list(result.errors)
    return {
        "schema_version": "teacher_generation_record_v1",
        "id": str(row["id"]),
        "category": str(row["category"]),
        "image_count": len(images),
        "image_sha256": image_hashes,
        "canonical_input_sha256": hashlib.sha256(canonical_input).hexdigest(),
        "accepted": not errors,
        "validation_errors": errors,
        "raw_output": raw,
        "parsed_output": parsed,
        "usage": usage,
        "generation_attempts": generation_attempts,
        "normalization_actions": normalization_actions,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    output: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                output.add(str(json.loads(line)["id"]))
    return output


def stratified_smoke(rows: list[dict[str, str]], per_cell: int) -> list[dict[str, str]]:
    cells: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        cells.setdefault((str(row["category"]), int(row["label"])), []).append(row)
    expected = {
        ("БАД", 0), ("БАД", 1),
        ("Легковоспламеняющиеся", 0), ("Легковоспламеняющиеся", 1),
    }
    if set(cells) != expected:
        raise ValueError(f"unexpected category/label cells: {sorted(cells)}")
    selected: list[dict[str, str]] = []
    for cell in sorted(cells):
        ranked = sorted(
            cells[cell],
            key=lambda row: hashlib.sha256(
                f"explanation-synth-smoke-v1|{cell[0]}|{cell[1]}|{row['id']}".encode()
            ).digest(),
        )
        if len(ranked) < per_cell:
            raise ValueError(f"not enough rows in cell={cell}")
        selected.extend(ranked[:per_cell])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--api-base", default=os.environ.get("TEACHER_API_BASE"))
    parser.add_argument(
        "--model", default=os.environ.get("TEACHER_MODEL_ID", MODEL_ID.rsplit("/", 1)[-1])
    )
    parser.add_argument(
        "--model-revision", default=os.environ.get("TEACHER_MODEL_REVISION", MODEL_REVISION)
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--api-ready-timeout", type=int, default=7200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-per-cell", type=int)
    parser.add_argument("--allow-partial-scope", action="store_true")
    args = parser.parse_args()

    if not args.api_base:
        parser.error("--api-base or TEACHER_API_BASE is required")

    wait_for_api(args.api_base, args.api_ready_timeout)

    rows = load_rows(args.data)
    if len(rows) != EXPECTED_ROWS and not args.allow_partial_scope:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, got {len(rows)}")
    images = index_images(args.image_root, [str(row["id"]) for row in rows])
    image_count = sum(map(len, images.values()))
    if image_count != EXPECTED_IMAGES and not args.allow_partial_scope:
        raise ValueError(f"expected {EXPECTED_IMAGES} images, got {image_count}")

    selected_rows = (
        stratified_smoke(rows, args.sample_per_cell)
        if args.sample_per_cell is not None else rows
    )
    done = completed_ids(args.output)
    pending = [row for row in selected_rows if str(row["id"]) not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as destination, ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = pool.map(
            lambda row: process_one(
                row, images[str(row["id"])], api_base=args.api_base,
                model=args.model, timeout=args.timeout,
            ),
            pending,
        )
        for index, record in enumerate(iterator, 1):
            destination.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            destination.flush()
            if index % 100 == 0 or index == len(pending):
                print(f"written={index}/{len(pending)} id={record['id']} accepted={record['accepted']}", flush=True)

    records: list[dict[str, Any]] = []
    with args.output.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    report = {
        "schema_version": "teacher_generation_report_v1",
        "model": args.model,
        "model_revision": args.model_revision,
        "generation_parameters": GENERATION_PARAMETERS,
        "max_validation_attempts": MAX_VALIDATION_ATTEMPTS,
        "workers": args.workers,
        "request_timeout_seconds": args.timeout,
        "all_gallery_images": True,
        "data_sha256": sha256_file(args.data),
        "prompt_py_sha256": sha256_file(Path(__file__).with_name("prompt.py")),
        "generator_py_sha256": sha256_file(Path(__file__)),
        "prompt_schema_version": SCHEMA_VERSION,
        "scope_rows": len(rows),
        "scope_images": image_count,
        "selected_rows": len(selected_rows),
        "sample_per_cell": args.sample_per_cell,
        "written_rows": len(records),
        "accepted_rows": sum(bool(record["accepted"]) for record in records),
        "invalid_rows": sum(not bool(record["accepted"]) for record in records),
        "full_scope_complete": len(records) == len(rows),
        "output_sha256": sha256_file(args.output),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.limit is None and args.sample_per_cell is None and len(records) != len(rows):
        raise SystemExit("full pass is incomplete; resume before acceptance")


if __name__ == "__main__":
    main()
