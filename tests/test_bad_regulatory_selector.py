from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "experiments/320_bad_regulatory_evidence/artifacts/bad_regulatory_selector_v1.csv.gz"
)
METRICS = ROOT / "experiments/320_bad_regulatory_evidence/results/metrics.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_frozen_selector_is_label_blind_and_registered() -> None:
    frame = pd.read_csv(ARTIFACT, dtype={"id": str})
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    selector = metrics["selector"]

    assert len(frame) == selector["rows"] == 6210
    assert not frame["id"].duplicated().any()
    assert selector["selection_uses_labels"] is False
    assert not [
        column
        for column in frame.columns
        if "label" in column.lower() or "prediction" in column.lower()
    ]
    assert int(frame["safe_for_selection"].sum()) == selector["safe_rows"] == 5439
    assert file_sha256(ARTIFACT) == selector["sha256"]


def test_selector_only_contains_predeclared_bad_regulatory_features() -> None:
    frame = pd.read_csv(ARTIFACT, dtype={"id": str})
    allowed_prefixes = ("name_", "description_", "count_", "sports_x_")
    identity = {
        "id",
        "fold",
        "group_hash",
        "connected_component",
        "safe_for_selection",
        "locked_score",
        "locked_uncertainty",
    }
    cue_flags = {
        "sports_nutrition",
        "explicit_bad",
        "medicine_language",
        "dosage_form",
        "food_beverage",
        "pet_veterinary",
        "not_a_drug",
    }
    unexpected = [
        column
        for column in frame.columns
        if column not in identity
        and column not in cue_flags
        and not column.startswith(allowed_prefixes)
    ]
    assert unexpected == []
