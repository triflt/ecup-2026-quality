from __future__ import annotations

import re
from typing import Any


RESULT_PATTERN = re.compile(
    r"\A<комментарий>(?P<comment>[\s\S]{50,300})<вердикт>(?P<verdict>бан|не бан)\Z"
)


def verdict_from_label(label: int) -> str:
    if isinstance(label, bool) or int(label) not in {0, 1}:
        raise ValueError("label must be integer 0 or 1")
    return "не бан" if int(label) == 1 else "бан"


def format_result(comment: str, label: int) -> str:
    value = str(comment).strip()
    if not 50 <= len(value) <= 300:
        raise ValueError("comment must contain 50-300 characters")
    if "<комментарий>" in value or "<вердикт>" in value:
        raise ValueError("comment must not contain result tags")
    result = f"<комментарий>{value}<вердикт>{verdict_from_label(label)}"
    parsed = parse_result(result)
    if parsed["label"] != int(label):
        raise AssertionError("formatted verdict changed label")
    return result


def parse_result(value: Any) -> dict[str, Any]:
    match = RESULT_PATTERN.fullmatch(str(value))
    if match is None:
        raise ValueError("result must use exact non-closing-tag competition format")
    verdict = match.group("verdict")
    return {
        "comment": match.group("comment"),
        "verdict": verdict,
        "label": 1 if verdict == "не бан" else 0,
    }
