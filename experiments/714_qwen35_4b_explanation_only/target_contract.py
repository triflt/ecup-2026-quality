from __future__ import annotations

import html
import re
from typing import Any


FORMAT_VERSION = "explanation_only_v3"

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


def user_text(row: Any, verdict: int) -> str:
    return (
        f"Категория: {row.category}\nНазвание: {compact_text(row.name, 320)}\n"
        f"Описание: {compact_text(row.description, 1800)}\n"
        f"Правило: {RULES[str(row.category)]}\nЗафиксированный verdict: {int(verdict)}.\n"
        "Не меняй verdict. Верни только конкретный комментарий длиной 50–300 "
        "символов без JSON, тегов и служебных пояснений."
    )


def messages(row: Any, *, verdict: int, with_answer: bool, image: Any) -> list[dict[str, Any]]:
    value = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": user_text(row, verdict)},
    ]}]
    if with_answer:
        value.append({"role": "assistant", "content": [{"type": "text", "text": str(row.target)}]})
    return value


def parse_target(value: str) -> dict[str, Any]:
    text = str(value).strip()
    if "LABEL=" in text or "<комментарий>" in text or "<вердикт>" in text:
        raise ValueError("explanation-only target must not contain labels or result tags")
    if text.startswith("{") or text.endswith("}"):
        raise ValueError("explanation-only target must be plain text, not JSON")
    if not 50 <= len(text) <= 300:
        raise ValueError("explanation must contain 50-300 characters")
    return {"explanation": text}
