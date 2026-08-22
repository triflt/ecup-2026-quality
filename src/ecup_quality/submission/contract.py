from __future__ import annotations

import re

import pandas as pd

OUTPUT_PATTERN = re.compile(r"^<комментарий>.{50,300}<вердикт>(?:бан|не бан)$", re.DOTALL)


def validate_submission_frame(frame: pd.DataFrame, expected_ids: list[str] | None = None) -> None:
    if list(frame.columns) != ["id", "answer"]:
        raise ValueError(f"expected columns ['id', 'answer'], got {list(frame.columns)!r}")
    ids = frame["id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("submission contains duplicate ids")
    if expected_ids is not None and ids != [str(value) for value in expected_ids]:
        raise ValueError("submission ids or row order do not match input")
    invalid = [index for index, answer in enumerate(frame["answer"].astype(str)) if not OUTPUT_PATTERN.fullmatch(answer)]
    if invalid:
        raise ValueError(f"invalid answer format in rows: {invalid[:20]}")
