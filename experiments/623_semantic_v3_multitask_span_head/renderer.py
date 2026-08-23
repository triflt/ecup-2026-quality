from __future__ import annotations

from protocol import CONCEPTS, canonical_text

TEMPLATES = {
    "OBJECT_OF_SALE": "Решение основано на том, что именно продаётся: «{quote}».",
    "COMPOSITION": "Решение основано на составе товара: «{quote}».",
    "COMPLETENESS": "Решение основано на комплектности товара: «{quote}».",
    "FUEL_OR_IGNITION": "Решение основано на указании топлива или источника огня: «{quote}».",
    "NEGATION": "Решение основано на явном отрицании: «{quote}».",
}


def render_explanation(
    *,
    name: object,
    description: object,
    char_start: int | None,
    char_end: int | None,
    concept: str | None,
) -> dict[str, str | None]:
    text, _ = canonical_text(name, description)
    if concept not in CONCEPTS or char_start is None or char_end is None:
        return {"evidence": "NO_EVIDENCE", "concept": None, "explanation": "NO_EVIDENCE"}
    if not 0 <= char_start < char_end <= len(text):
        return {"evidence": "NO_EVIDENCE", "concept": None, "explanation": "NO_EVIDENCE"}
    quote = text[char_start:char_end]
    if not quote.strip():
        return {"evidence": "NO_EVIDENCE", "concept": None, "explanation": "NO_EVIDENCE"}
    return {
        "evidence": quote,
        "concept": concept,
        "explanation": TEMPLATES[concept].format(quote=quote),
    }
