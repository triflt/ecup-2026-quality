"""Strict, label-blind checks for the public 621 experiment scaffold."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

SAFE_MANIFEST_COLUMNS = (
    "id",
    "category",
    "semantic_component",
    "component_size",
    "split",
    "development_fold",
)
FORBIDDEN_NAMES = frozenset(
    {"label", "labels", "gold", "target", "targets", "y", "sealed", "sealed_ids"}
)
PUBLIC_FORBIDDEN_TOKENS = (
    "private://",
    "internal://",
    "/private-runtime/",
    "${private_runtime_root}",
    "/users/",
    "/home/",
)


def _normalise(name: object) -> str:
    return str(name).strip().lower()


def read_label_blind_manifest(path: Path) -> pd.DataFrame:
    """Read only the frozen membership columns; never load supervision columns."""

    header = pd.read_csv(path, nrows=0)
    columns = list(header.columns)
    forbidden = sorted({_normalise(column) for column in columns} & FORBIDDEN_NAMES)
    if forbidden:
        raise ValueError(f"manifest exposes forbidden supervision columns: {forbidden}")
    missing = [column for column in SAFE_MANIFEST_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"manifest lacks safe columns: {missing}")
    frame = pd.read_csv(path, usecols=list(SAFE_MANIFEST_COLUMNS), dtype={"id": str})
    # The returned frame is development-only.  A sealed membership row is not
    # needed for this preflight and must not flow into any downstream artifact.
    return frame.loc[frame["split"].eq("development")].reset_index(drop=True)


def audit_label_blind_manifest(frame: pd.DataFrame) -> dict[str, Any]:
    """Audit fold/family structure without looking at labels or predictions."""

    if tuple(frame.columns) != SAFE_MANIFEST_COLUMNS:
        raise ValueError("safe manifest schema/order mismatch")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("manifest IDs must be non-null and unique")
    folds = frame["development_fold"]
    if not folds.isin([-1, 0, 1, 2, 3, 4]).all():
        raise ValueError("unexpected development fold")
    development = frame.loc[frame["split"].eq("development")]
    if development.empty:
        raise ValueError("manifest contains no development rows")
    cross_fold = development.groupby("semantic_component")["development_fold"].nunique()
    if (cross_fold > 1).any():
        raise ValueError("semantic components cross development folds")
    return {
        "status": "passed",
        "rows": len(frame),
        "development_rows": len(development),
        "development_folds": sorted(
            int(value) for value in development["development_fold"].unique()
        ),
        "labels_loaded": False,
        "sealed_rows_loaded": 0,
    }


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(key)
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


def validate_public_metadata(payload: Any) -> None:
    """Reject private paths and infrastructure references before publication."""

    text = "\n".join(_walk_strings(payload)).lower()
    found = [token for token in PUBLIC_FORBIDDEN_TOKENS if token in text]
    if found:
        raise ValueError(
            f"public metadata contains forbidden private token(s): {sorted(set(found))}"
        )


def load_and_audit_manifest(path: Path) -> dict[str, Any]:
    frame = read_label_blind_manifest(path)
    return audit_label_blind_manifest(frame)


def write_preflight_report(output: Path, report: dict[str, Any]) -> None:
    validate_public_metadata(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
