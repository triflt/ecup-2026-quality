from __future__ import annotations

import re
from typing import Any


SPACE = re.compile(r"\s+")
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
SENTENCE_END = re.compile(r"[.!?…][\"»)]*\Z")

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


def _clean(value: Any) -> str:
    text = SPACE.sub(" ", str(value or "")).strip().strip('"')
    return text.replace("<", " ").replace(">", " ").strip()


def explicit_verdict_mismatch(text: str, category: str, verdict: int) -> bool:
    """Catch explicit, high-precision claims that oppose the frozen verdict."""
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
