from __future__ import annotations

import json
import re
from typing import Any


SPACE = re.compile(r"\s+")


def _clean(value: Any) -> str:
    text = SPACE.sub(" ", str(value or "")).strip().strip('"')
    return text.replace("<", " ").replace(">", " ").strip()


def normalize_generated_comment(raw: str, fallback: str) -> tuple[str, str]:
    text = str(raw or "").strip()
    status = "generated_plain_text"
    if text.startswith("{") and text.endswith("}"):
        try:
            payload = json.loads(text)
            if isinstance(payload, dict) and isinstance(payload.get("explanation"), str):
                text = payload["explanation"]
                status = "recovered_json_explanation"
        except json.JSONDecodeError:
            pass
    text = _clean(text)
    if len(text) > 300:
        shortened = text[:300]
        text = shortened.rsplit(" ", 1)[0] if " " in shortened else shortened
        status += ":truncated"
    if 50 <= len(text) <= 300:
        return text, status
    safe_fallback = _clean(fallback)
    if not 50 <= len(safe_fallback) <= 300:
        raise ValueError("fallback comment must contain 50-300 characters")
    return safe_fallback, "fallback_invalid_generation"
