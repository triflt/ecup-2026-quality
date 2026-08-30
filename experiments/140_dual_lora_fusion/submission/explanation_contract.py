from __future__ import annotations

import html
import re
from typing import Any

SPACE = re.compile(r"\s+")
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
SENTENCE_END = re.compile(r"[.!?…][\"»)]*\Z")
RESULT_PATTERN = re.compile(
    r"\A<комментарий>(?P<comment>[\s\S]{50,300})<вердикт>"
    r"(?P<verdict>бан|не бан)\Z"
)

RULES = {
    "БАД": "Метка 1 только при прямой маркировке БАД/dietary supplement; иначе 0.",
    "Легковоспламеняющиеся": (
        "Метка 1 для продаваемого источника огня, горючего вещества или газа; "
        "пустое оборудование и упоминание горючего — 0."
    ),
}

GENERIC_CONTRADICTIONS = {
    0: (
        r"\bвердикт\s*:?\s*не бан\b",
        r"\bне следует (?:блокировать|запрещать)\b",
        r"\bможно (?:допустить|разрешить) к продаже\b",
    ),
    1: (
        r"\bвердикт\s*:?\s*бан\b",
        r"\bследует (?:заблокировать|запретить)\b",
        r"\bнельзя допускать к продаже\b",
    ),
}

CATEGORY_CONTRADICTIONS = {
    "БАД": {
        0: (
            r"(?<!не )\bявляется бад\b",
            r"(?<!не )\bявляется биологически активной добавкой\b",
            r"(?<!не )\b(?:имеет|содержит) маркировку бад\b",
            r"\bмаркирован(?:а|о|ы)? как бад\b",
        ),
        1: (
            r"\bне является бад\b",
            r"\bне является биологически активной добавкой\b",
            r"\bне имеет маркировки бад\b",
            r"\bмаркировка бад отсутствует\b",
        ),
    },
    "Легковоспламеняющиеся": {
        0: (
            r"(?<!не )\bявляется легковоспламеняющимся товаром\b",
            r"(?<!не )\bсодержит (?:горючее вещество|топливо|горючий газ)\b",
            r"\bисточник открытого огня входит в комплект\b",
        ),
        1: (
            r"\bне является легковоспламеняющимся товаром\b",
            r"\bне содержит (?:горючего вещества|топлива|горючего газа)\b",
            r"\b(?:топливо|горючее вещество|горючий газ) не входит в комплект\b",
        ),
    },
}


def compact_text(value: Any, limit: int) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = SPACE.sub(" ", text).strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    return text[:head].rstrip() + " … " + text[-(limit - head):].lstrip()


def user_text(row: Any, verdict: int) -> str:
    return (
        f"Категория: {row.category}\nНазвание: {compact_text(row.name, 320)}\n"
        f"Описание: {compact_text(row.description, 1800)}\n"
        f"Правило: {RULES[str(row.category)]}\n"
        f"Зафиксированный verdict solution140: {int(verdict)}.\n"
        "Не меняй verdict. Верни только конкретный русский комментарий длиной "
        "50–300 символов без JSON, тегов и служебных пояснений."
    )


def messages(row: Any, *, verdict: int, image: Any) -> list[dict[str, Any]]:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text(row, verdict)},
        ],
    }]


def _clean(value: Any) -> str:
    text = SPACE.sub(" ", str(value or "")).strip().strip('"')
    return text.replace("<", " ").replace(">", " ").strip()


def explicit_verdict_mismatch(text: str, category: str, verdict: int) -> bool:
    if verdict not in {0, 1}:
        raise ValueError("verdict must be integer 0 or 1")
    value = SPACE.sub(" ", text).strip().lower()
    patterns = GENERIC_CONTRADICTIONS[verdict]
    patterns += CATEGORY_CONTRADICTIONS.get(category, {}).get(verdict, ())
    return any(re.search(pattern, value) is not None for pattern in patterns)


def _invalid_reason(text: str, category: str, verdict: int) -> str | None:
    if not 50 <= len(text) <= 300:
        return "length"
    if not CYRILLIC.search(text):
        return "non_russian"
    if not SENTENCE_END.search(text):
        return "incomplete_sentence"
    if explicit_verdict_mismatch(text, category, verdict):
        return "verdict_mismatch"
    return None


def normalize_generated_comment(
    raw: str,
    fallback: str,
    *,
    category: str,
    verdict: int,
) -> tuple[str, str]:
    source = str(raw or "").strip()
    malformed = (
        not source
        or "<" in source
        or ">" in source
        or source.startswith(("{", "["))
        or source.endswith(("}", "]"))
    )
    text = _clean(source)
    reason = "malformed" if malformed else _invalid_reason(text, category, verdict)
    if reason is None:
        return text, "generated_plain_text"
    safe_fallback = _clean(fallback)
    fallback_reason = _invalid_reason(safe_fallback, category, verdict)
    if fallback_reason is not None:
        raise ValueError(f"invalid category-aware fallback: {fallback_reason}")
    return safe_fallback, f"fallback_{reason}"


def verdict_from_prediction(prediction: int) -> str:
    if isinstance(prediction, bool) or int(prediction) not in {0, 1}:
        raise ValueError("prediction must be integer 0 or 1")
    return "не бан" if int(prediction) == 1 else "бан"


def format_result(comment: str, prediction: int) -> str:
    value = str(comment).strip()
    verdict = verdict_from_prediction(prediction)
    result = f"<комментарий>{value}<вердикт>{verdict}"
    match = RESULT_PATTERN.fullmatch(result)
    if match is None:
        raise ValueError("result does not satisfy the exact competition schema")
    parsed_prediction = 1 if match.group("verdict") == "не бан" else 0
    if parsed_prediction != int(prediction):
        raise AssertionError("formatter changed frozen solution140 prediction")
    return result
