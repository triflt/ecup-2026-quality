#!/usr/bin/env python3
"""Generate verdict-locked structured explanations with one local first image."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "frozen_spec.json").read_text(encoding="utf-8"))
EXPECTED_KEYS = {
    "verdict",
    "sold_object",
    "regulated_substance_or_marker",
    "relation",
    "evidence_source",
    "evidence_quote",
    "explanation",
}

RULES = {
    "БАД": (
        "Вердикт 'не бан' (1) допустим только при прямом указании, что продаваемый товар "
        "является БАД/dietary supplement. Спортивное питание, косметика или лекарственная "
        "форма без такой маркировки, а также отрицание статуса БАД — 'бан' (0)."
    ),
    "Легковоспламеняющиеся": (
        "Вердикт 'не бан' (1), если продаётся самостоятельный источник огня, газ, горючее "
        "вещество/топливо или такой предмет явно входит в комплект. Пустое устройство, "
        "упоминание совместимого топлива, покупаемого отдельно, либо горючего материала "
        "только как компонента — 'бан' (0)."
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if width * height > 262144:
        scale = math.sqrt(262144 / (width * height))
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        resized = image.resize(
            (max(28, int(width * scale)), max(28, int(height * scale))), resampling
        )
        image.close()
        image = resized
    return image


def make_prompt(row: SimpleNamespace) -> str:
    verdict_name = "не бан" if row.frozen_prediction == 1 else "бан"
    return (
        "Ты проверяешь уже ЗАМОРОЖЕННЫЙ вердикт и не имеешь права его менять. "
        "Нужна короткая проверяемая причина, а не цепочка рассуждений.\n"
        f"Категория: {row.category}\n"
        f"Название: {row.name}\n"
        f"Описание: {row.description}\n"
        f"Правило: {RULES[row.category]}\n"
        f"Замороженный вердикт: {row.frozen_prediction} ({verdict_name}).\n"
        "Определи: что именно продаётся; какое регулируемое вещество/маркер присутствует "
        "или упомянуто; отношение этого вещества/маркера к продаваемому объекту. "
        "relation строго одно из: sold_object, included, compatible_external, "
        "mentioned_only, negated, unknown. evidence_source строго одно из: name, "
        "description, image, none. Для name/description evidence_quote должен быть точной "
        "подстрокой соответствующего поля; для image — точной короткой надписью с изображения; "
        "для none — пустой строкой. Объяснение — 50–300 символов, только факты из карточки.\n"
        "Ответь только одним JSON-объектом без markdown и дополнительного текста с ключами: "
        'verdict, sold_object, regulated_substance_or_marker, relation, evidence_source, '
        'evidence_quote, explanation.'
    )


def validate_payload(raw: str, row: SimpleNamespace) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        payload = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None, ["invalid_json_or_extra_text"]
    if not isinstance(payload, dict) or set(payload) != EXPECTED_KEYS:
        return None, ["schema_keys_mismatch"]
    verdict = payload.get("verdict")
    if isinstance(verdict, bool) or verdict != row.frozen_prediction:
        errors.append("verdict_lock_failed")
    for key in ("sold_object", "regulated_substance_or_marker"):
        value = payload.get(key)
        if not isinstance(value, str) or not 1 <= len(value.strip()) <= 160 or value != value.strip():
            errors.append(f"invalid_{key}")
    relation = payload.get("relation")
    if relation not in SPEC["relation_values"]:
        errors.append("invalid_relation")
    source = payload.get("evidence_source")
    if source not in SPEC["evidence_source_values"]:
        errors.append("invalid_evidence_source")
    quote = payload.get("evidence_quote")
    if not isinstance(quote, str) or len(quote) > 200 or quote != quote.strip():
        errors.append("invalid_evidence_quote")
    elif source == "name" and (not quote or quote not in row.name):
        errors.append("name_quote_not_exact")
    elif source == "description" and (not quote or quote not in row.description):
        errors.append("description_quote_not_exact")
    elif source == "image" and not quote:
        errors.append("empty_image_quote")
    elif source == "none" and quote:
        errors.append("none_source_has_quote")
    explanation = payload.get("explanation")
    if (
        not isinstance(explanation, str)
        or not SPEC["explanation_min_chars"] <= len(explanation) <= SPEC["explanation_max_chars"]
        or explanation != explanation.strip()
    ):
        errors.append("invalid_explanation_length_or_whitespace")
    return payload, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--candidate-alias", choices=sorted(SPEC["candidate_models"]), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    expected_model = SPEC["candidate_models"][args.candidate_alias]
    if (args.model_id, args.model_revision) != (expected_model["model_id"], expected_model["revision"]):
        raise ValueError("candidate model identity differs from frozen spec")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        SimpleNamespace(**json.loads(line))
        for line in args.runtime.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != SPEC["rows"] or any(row.schema_version != "exp672_runtime_row_v1" for row in rows):
        raise ValueError("runtime scope or schema mismatch")

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_root, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_root,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()
    device_map = getattr(model, "hf_device_map", {})
    if any(str(device) in {"cpu", "disk"} for device in device_map.values()):
        raise RuntimeError(f"model was offloaded outside CUDA: {device_map}")
    input_device = model.device
    output_path = args.output_dir / "explanations.jsonl"
    started = time.monotonic()
    valid_rows = 0
    with output_path.open("x", encoding="utf-8") as output:
        for row in rows:
            image_path = args.images_dir / f"{row.id}.jpg"
            image = load_image(image_path)
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": make_prompt(row)},
                        ],
                    }
                ]
                batch = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                    enable_thinking=False,
                ).to(input_device)
                input_length = batch["input_ids"].shape[1]
                with torch.inference_mode():
                    generated = model.generate(
                        **batch,
                        do_sample=False,
                        max_new_tokens=320,
                    )
                raw = processor.batch_decode(
                    generated[:, input_length:], skip_special_tokens=True
                )[0].strip()
                payload, errors = validate_payload(raw, row)
                contract_valid = not errors
                valid_rows += int(contract_valid)
                output.write(
                    json.dumps(
                        {
                            "schema_version": "exp672_candidate_output_v1",
                            "global_index": row.global_index,
                            "id": row.id,
                            "semantic_component": row.semantic_component,
                            "category": row.category,
                            "frozen_prediction": row.frozen_prediction,
                            "candidate_alias": args.candidate_alias,
                            "raw_response": raw,
                            "parsed": payload,
                            "contract_valid": contract_valid,
                            "contract_errors": errors,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                output.flush()
            finally:
                image.close()
    elapsed = time.monotonic() - started
    report = {
        "schema_version": "exp672_generation_report_v1",
        "experiment_id": "672",
        "candidate_alias": args.candidate_alias,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "runtime_sha256": sha256_file(args.runtime),
        "output_sha256": sha256_file(output_path),
        "rows": len(rows),
        "contract_valid_rows": valid_rows,
        "elapsed_seconds": elapsed,
        "rows_per_second": len(rows) / elapsed if elapsed else 0.0,
        "thinking": False,
        "decoding": "greedy",
        "cuda_device_count": torch.cuda.device_count(),
        "hf_device_map": {str(key): str(value) for key, value in device_map.items()},
        "labels_read": 0,
        "sealed_rows_read": 0,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
