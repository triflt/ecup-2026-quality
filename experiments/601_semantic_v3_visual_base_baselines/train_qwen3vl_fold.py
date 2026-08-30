from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
from collections.abc import MutableMapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from semantic_v3_contract import (
    DEVELOPMENT_FOLDS,
    EXPECTED_DEVELOPMENT_ROWS,
    EXPECTED_QWEN_PARENT_SHA256,
    QWEN_FIRST_IMAGE_MAX_EDGE,
    QWEN_FIRST_IMAGE_MAX_PIXELS,
    QWEN_SEED,
    ROOT,
    VALIDATION_VERSION,
    require_development_fold,
)

PARENT_RUNNER = ROOT / "research/qwen3vl_lora_holdout.py"
LOCKED_ENVIRONMENT = {
    "SEED": str(QWEN_SEED),
    "TRAINING_MODE": "hard",
    "MODEL_CLASS": "image_text",
    "DESCRIPTION_LIMIT": "1800",
    "QWEN3VL_FIRST_IMAGE_MAX_EDGE": str(QWEN_FIRST_IMAGE_MAX_EDGE),
    "QWEN3VL_FIRST_IMAGE_MAX_PIXELS": str(QWEN_FIRST_IMAGE_MAX_PIXELS),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


LOCKED_DISABLED_ENVIRONMENT = {
    "FULL_TRAIN": ("", "0", "false"),
    "SOFT_TARGETS": ("",),
    "DOWNSAMPLE_MODE": ("",),
    "MAX_SLICE_NUMS": ("", "0"),
    "LAST_LOGIT_ONLY": ("", "0", "false"),
    "USE_CHAT_BATCH": ("", "0", "false"),
    "LINEAR_ONLY_TARGETS": ("", "0", "false"),
}


def runtime_paths(runtime_dir: Path) -> dict[str, Path]:
    return {
        "data": runtime_dir / "development_data.csv",
        "folds": runtime_dir / "development_folds.csv",
        "manifest": runtime_dir / "development_first_image_manifest.tsv.gz",
        "oof": runtime_dir / "qwen_parent_oof.npz",
        "audit": runtime_dir / "zero_sealed_runtime_audit.json",
    }


def verify_runtime_inputs(runtime_dir: Path) -> dict[str, object]:
    paths = runtime_paths(runtime_dir)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"runtime inputs missing: {missing}")
    audit = json.loads(paths["audit"].read_text())
    if audit.get("development_rows") != EXPECTED_DEVELOPMENT_ROWS:
        raise ValueError("development runtime row count mismatch")
    if audit.get("sealed_ids_in_any_runtime_output") != 0:
        raise ValueError("runtime audit does not prove zero sealed ids")
    data = pd.read_csv(paths["data"], dtype={"id": str})
    folds = pd.read_csv(paths["folds"], dtype={"id": str})
    if len(data) != EXPECTED_DEVELOPMENT_ROWS:
        raise ValueError("development data row count mismatch")
    if not np.array_equal(data["id"].to_numpy(), folds["id"].to_numpy()):
        raise ValueError("runtime data/folds id mismatch")
    if set(folds["development_fold"].astype(int)) != set(DEVELOPMENT_FOLDS):
        raise ValueError("runtime development fold set mismatch")
    if (folds["split"].astype(str) != "development").any():
        raise ValueError("runtime fold manifest contains a non-development row")
    oof = np.load(paths["oof"], allow_pickle=False)
    if not np.array_equal(oof["ids"].astype(str), data["id"].to_numpy()):
        raise ValueError("runtime OOF id mismatch")
    return audit


def configure_environment(
    *,
    fold: int,
    runtime_dir: Path,
    output_dir: Path,
    environment: MutableMapping[str, str],
) -> MutableMapping[str, str]:
    require_development_fold(fold)
    for key, expected in LOCKED_ENVIRONMENT.items():
        current = environment.get(key)
        if current not in (None, "", expected):
            raise ValueError(f"{key} must remain at parent value {expected}")
        environment[key] = expected
    enabled = [
        key
        for key, disabled in LOCKED_DISABLED_ENVIRONMENT.items()
        if environment.get(key, "").strip().lower() not in disabled
    ]
    if enabled:
        raise ValueError(f"non-parent switches are forbidden: {sorted(enabled)}")
    paths = runtime_paths(runtime_dir)
    environment.update(
        {
            "HOLDOUT_FOLD": str(fold),
            "ECUP_DATA": str(paths["data"]),
            "ECUP_OOF": str(paths["oof"]),
            "ECUP_MANIFEST": str(paths["manifest"]),
            "ECUP_OUTPUT_DIR": str(output_dir),
        }
    )
    return environment


def annotate_and_verify_output(
    *,
    fold: int,
    runtime_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    paths = runtime_paths(runtime_dir)
    data = pd.read_csv(paths["data"], dtype={"id": str})
    folds = pd.read_csv(paths["folds"], dtype={"id": str})
    expected = folds["development_fold"].to_numpy(np.int8) == fold
    predictions_path = output_dir / "lora_holdout_predictions.csv"
    report_path = output_dir / "lora_holdout_report.json"
    predictions = pd.read_csv(predictions_path, dtype={"id": str})
    if not np.array_equal(predictions["id"].to_numpy(), data.loc[expected, "id"]):
        raise ValueError("Qwen holdout id/order mismatch")
    if not np.all(predictions["fold"].to_numpy(np.int8) == fold):
        raise ValueError("Qwen output contains another fold")
    report = json.loads(report_path.read_text())
    report.update(
        {
            "validation_version": VALIDATION_VERSION,
            "development_only": True,
            "development_rows": len(data),
            "sealed_rows_seen_by_train_selector_threshold_eval": 0,
            "first_image_max_edge": QWEN_FIRST_IMAGE_MAX_EDGE,
            "first_image_max_pixels": QWEN_FIRST_IMAGE_MAX_PIXELS,
            "inference_image_index": 0,
            "inference_images_per_row": 1,
            "inference_passes": 1,
            "parent_recipe": "experiment_110_exact",
        }
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=DEVELOPMENT_FOLDS)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    verify_runtime_inputs(args.runtime_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite non-empty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_environment(
        fold=args.fold,
        runtime_dir=args.runtime_dir,
        output_dir=args.output_dir,
        environment=os.environ,
    )
    if not PARENT_RUNNER.is_file():
        raise FileNotFoundError(PARENT_RUNNER)
    if sha256(PARENT_RUNNER) != EXPECTED_QWEN_PARENT_SHA256:
        raise ValueError("Qwen3-VL parent runner checksum mismatch")
    sys.argv = [str(PARENT_RUNNER)]
    runpy.run_path(str(PARENT_RUNNER), run_name="__main__")
    report = annotate_and_verify_output(
        fold=args.fold, runtime_dir=args.runtime_dir, output_dir=args.output_dir
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
