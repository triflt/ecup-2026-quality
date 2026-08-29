from __future__ import annotations

import html
import json
import re
from typing import Any


FORMAT_VERSION = "rationale_then_label_v2"
LABEL_LAST = re.compile(r"\nLABEL=([01])\Z")
PAYLOAD_FIELDS = {"reason", "evidence", "explanation"}

RULES = {
    "БАД": "Метка 1 только при прямой маркировке БАД/dietary supplement; иначе 0.",
    "Легковоспламеняющиеся": (
        "Метка 1 для продаваемого источника огня, горючего вещества/газа или "
        "явно включённого такого предмета; пустое оборудование и упоминание — 0."
    ),
}


def compact_text(value: Any, limit: int) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    return text[:head].rstrip() + " … " + text[-(limit - head):].lstrip()


def user_text(row: Any) -> str:
    return (
        f"Категория: {row.category}\nНазвание: {compact_text(row.name, 320)}\n"
        f"Описание: {compact_text(row.description, 1800)}\n"
        f"Правило: {RULES[str(row.category)]}\n"
        "Проанализируй карточку и выдай итоговое решение в формате, изученном "
        "при обучении. Не используй сведения вне карточки."
    )


def messages(row: Any, *, with_answer: bool, image: Any) -> list[dict[str, Any]]:
    value = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": user_text(row)},
    ]}]
    if with_answer:
        value.append({"role": "assistant", "content": [{"type": "text", "text": str(row.target)}]})
    return value


def parse_target(value: str) -> dict[str, Any]:
    text = str(value).strip()
    match = LABEL_LAST.search(text)
    if match is None:
        raise ValueError("atomic 0/1 label must be the final character")
    payload = json.loads(text[: match.start()])
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_FIELDS:
        raise ValueError("explanation payload has invalid fields")
    if not isinstance(payload["evidence"], dict) or set(payload["evidence"]) != {"source", "value"}:
        raise ValueError("evidence has invalid fields")
    if not isinstance(payload["explanation"], str) or not 50 <= len(payload["explanation"].strip()) <= 300:
        raise ValueError("explanation must contain 50-300 characters")
    payload["label"] = int(match.group(1))
    return payload
