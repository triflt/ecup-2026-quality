import numpy as np
import pandas as pd

from ecup_quality.data.text import canonicalize, compose_text
from ecup_quality.submission.contract import validate_submission_frame
from ecup_quality.validation.metrics import binary_f1, rank01


def test_text_normalization_is_deterministic() -> None:
    assert compose_text("Ёлка 10 мл", "<b>Описание</b>") == "елка 10 мл\nелка 10 мл\nописание"
    assert canonicalize("10.5 мл", mask_digits=True) == "# мл"


def test_metrics() -> None:
    assert binary_f1([1, 1, 0], [1, 0, 0]) == 2 / 3
    assert np.array_equal(rank01([2, 1]), np.asarray([1, 0], dtype=np.float32))


def test_submission_contract() -> None:
    answer = "<комментарий>" + "корректное объяснение " * 4 + "<вердикт>бан"
    validate_submission_frame(pd.DataFrame({"id": ["1"], "answer": [answer]}), ["1"])
