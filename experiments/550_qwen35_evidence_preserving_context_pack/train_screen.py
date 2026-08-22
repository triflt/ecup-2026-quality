from __future__ import annotations

"""Locked two-fold wrapper changing only Qwen3.5 description packing."""

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from context_pack import DESCRIPTION_BUDGET, pack_description
from contract import (
    AUDIT_ROWS_SHA256,
    AUDIT_SHA256,
    AUDIT_VERSION,
    EXTRACTOR_SHA256,
    PARENT_ENVIRONMENT,
    SCREEN_FOLDS,
    VOCABULARY_SHA256,
)
from parent_recipe import configure_parent_environment, load_parent_module

EXPERIMENT_DIR = Path(__file__).resolve().parent
AUDIT_PATH = EXPERIMENT_DIR / "analysis/context_pack_audit.json"
AUDIT_ROWS_PATH = EXPERIMENT_DIR / "analysis/context_pack_rows.csv.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_audit() -> tuple[dict[str, object], pd.DataFrame]:
    if sha256(AUDIT_PATH) != AUDIT_SHA256 or sha256(AUDIT_ROWS_PATH) != AUDIT_ROWS_SHA256:
        raise ValueError("frozen context-pack audit checksum mismatch")
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    if (
        audit.get("audit_version") != AUDIT_VERSION
        or audit.get("status") != "GO"
        or audit.get("labels_read") is not False
        or audit.get("folds_read") is not False
        or not all(audit.get("gates", {}).values())
    ):
        raise ValueError("frozen context-pack audit did not authorize the screen")
    rows = pd.read_csv(AUDIT_ROWS_PATH, compression="gzip", dtype={"id": str})
    if rows.id.duplicated().any() or len(rows) != int(audit["rows"]):
        raise ValueError("frozen context-pack row audit identity mismatch")
    return audit, rows.set_index("id")


def row_field(row: object, name: str) -> object:
    if isinstance(row, pd.Series):
        return row[name]
    return getattr(row, name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=SCREEN_FOLDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    configure_parent_environment(fold=args.fold, output_dir=args.output_dir, environment=os.environ)
    audit, expected = load_frozen_audit()
    parent = load_parent_module()
    context_report_path = args.output_dir / "context_pack_runtime_report.json"
    parent_report_path = args.output_dir / "lora_holdout_report.json"
    candidate_targets = (
        context_report_path,
        parent_report_path,
        args.output_dir / "lora_holdout_predictions.csv",
        args.output_dir / "adapter",
        args.output_dir / "adapter.zip",
    )
    if any(path.exists() for path in candidate_targets):
        raise FileExistsError("refusing to overwrite existing screen outputs")

    cache: dict[str, str] = {}
    invocations = 0

    def evidence_preserving_user_text(row) -> str:
        nonlocal invocations
        item_id = str(row_field(row, "id"))
        if item_id not in cache:
            if item_id not in expected.index:
                raise ValueError("runtime row is absent from the frozen packing audit")
            packed = pack_description(
                row_id=item_id,
                category=str(row_field(row, "category")),
                name=str(row_field(row, "name")),
                description=str(row_field(row, "description")),
            )
            expected_row = expected.loc[item_id]
            if (
                not packed.packable
                or len(packed.text) > DESCRIPTION_BUDGET
                or packed.text_sha256 != str(expected_row.text_sha256)
                or packed.provenance_sha256 != str(expected_row.provenance_sha256)
            ):
                raise ValueError("runtime context pack differs from the frozen audit")
            cache[item_id] = packed.text
        invocations += 1
        return (
            f"Категория: {row.category}\n"
            f"Название: {parent.compact_text(row.name, 320)}\n"
            f"Описание: {cache[item_id]}\n"
            f"Правило: {parent.RULES[row.category]}\n"
            "Определи правильность категории. Ответь только одной цифрой: 1 или 0."
        )

    parent.user_text = evidence_preserving_user_text
    parent.main()
    if not parent_report_path.is_file():
        raise RuntimeError("parent training did not produce its holdout report")
    parent_report = json.loads(parent_report_path.read_text(encoding="utf-8"))
    train_records = int(parent_report.get("train_records", -1))
    expected_updates = (train_records + 3) // 4
    expected_updates = (expected_updates + 3) // 4
    if train_records != 5390 or expected_updates != 337:
        raise ValueError("runtime parent selector/step count differs from the frozen screen")
    runtime_report = {
        "experiment_id": "550",
        "status": "screen_fold_complete",
        "holdout_fold": args.fold,
        "single_changed_factor": "description packing only",
        "description_budget": DESCRIPTION_BUDGET,
        "packing_audit_version": AUDIT_VERSION,
        "packing_audit_sha256": sha256(AUDIT_PATH),
        "packing_rows_sha256": sha256(AUDIT_ROWS_PATH),
        "extractor_sha256": EXTRACTOR_SHA256,
        "vocabulary_sha256": VOCABULARY_SHA256,
        "selection_uses_labels": False,
        "selection_uses_folds": False,
        "prediction_source": "constant union of frozen_prediction=0 and 1",
        "unique_runtime_rows_packed": len(cache),
        "runtime_pack_invocations": invocations,
        "parent_train_records": train_records,
        "parent_optimizer_updates": expected_updates,
        "parent_environment": PARENT_ENVIRONMENT,
        "parent_holdout_report_sha256": sha256(parent_report_path),
        "audit_go_metrics": {
            "long_evidence_rows": audit["long_evidence_rows"],
            "strict_improvement_fraction": audit["strict_improvement_fraction"],
            "regressed": audit["long_evidence_rows_regressed"],
        },
    }
    context_report_path.write_text(
        json.dumps(runtime_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(runtime_report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
