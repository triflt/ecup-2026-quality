from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from contract import (
    INFERENCE_VIEW_POLICY,
    SCREEN_FOLDS,
    SEED,
    TRAINING_VIEW_POLICY,
)
from parent_selector import select_parent_training_records
from recombination_plan import canonical_sha256, file_sha256

ROOT = Path(__file__).resolve().parents[2]
PARENT_RUNNER = ROOT / "research/qwen3vl_lora_holdout.py"
LOCKED_ENVIRONMENT = {
    "SEED": "42",
    "TRAINING_MODE": "hard",
    "MODEL_CLASS": "image_text",
    "DESCRIPTION_LIMIT": "1800",
    "QWEN3VL_FIRST_IMAGE_MAX_EDGE": "448",
    "QWEN3VL_FIRST_IMAGE_MAX_PIXELS": "262144",
}
LOCKED_DISABLED = {
    "FULL_TRAIN": ("", "0", "false"),
    "SOFT_TARGETS": ("",),
    "DOWNSAMPLE_MODE": ("",),
    "MAX_SLICE_NUMS": ("", "0"),
    "LAST_LOGIT_ONLY": ("", "0", "false"),
    "USE_CHAT_BATCH": ("", "0", "false"),
    "LINEAR_ONLY_TARGETS": ("", "0", "false"),
}


def configure_environment(fold: int) -> None:
    if fold not in SCREEN_FOLDS:
        raise ValueError("only folds 0 and 3 are allowed")
    for key, expected in LOCKED_ENVIRONMENT.items():
        current = os.environ.get(key)
        if current not in (None, "", expected):
            raise ValueError(f"{key} differs from exact parent")
        os.environ[key] = expected
    enabled = [
        key
        for key, disabled in LOCKED_DISABLED.items()
        if os.environ.get(key, "").strip().lower() not in disabled
    ]
    if enabled:
        raise ValueError(f"non-parent switches are forbidden: {sorted(enabled)}")
    os.environ["HOLDOUT_FOLD"] = str(fold)


def load_authorized_plan(audit_path: Path, manifest_path: Path, fold: int):
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("decision") != "GO":
        raise ValueError("preflight is NO_GO; training is blocked before model import")
    if audit.get("holdout_fold") != fold:
        raise ValueError("preflight fold mismatch")
    manifest = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    if canonical_sha256(manifest) != audit.get("manifest_sha256"):
        raise ValueError("recombination manifest checksum mismatch")
    if len(manifest) != audit.get("recombined_occurrences"):
        raise ValueError("recombination manifest count mismatch")
    return audit, manifest


def _load_parent():
    spec = importlib.util.spec_from_file_location("_exp590_parent_qwen3vl", PARENT_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load exact Qwen3-VL parent")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RecombinationRuntime:
    def __init__(self, parent, audit: dict[str, Any], manifest: list[dict[str, Any]]) -> None:
        self.parent = parent
        self.audit = audit
        self.by_position = {int(row["occurrence_position"]): row for row in manifest}
        if len(self.by_position) != len(manifest):
            raise ValueError("duplicate transformed occurrence positions")
        self.training = True
        self.position = 0
        self.replacements = 0
        self.original_select = parent.select_training
        self.original_open_images = parent.open_images
        self.original_validation = parent.validation_scores

    def select_training(self, frame, oof):
        records = self.original_select(frame, oof)
        light = select_parent_training_records(
            frame, oof, seed=SEED, holdout_fold=self.parent.HOLDOUT_FOLD
        )
        if records != light:
            raise ValueError("exact runtime parent selector differs from lightweight selector")
        multiset = canonical_sha256(sorted(Counter(records).items()))
        if multiset != self.audit["parent_record_multiset_sha256"]:
            raise ValueError("runtime parent multiset differs from frozen preflight")
        folds = oof["fold_ids"].astype(np.int8)
        for plan in self.by_position.values():
            source, donor = int(plan["source_index"]), int(plan["donor_index"])
            if str(frame.iloc[source].id) != str(plan["source_id"]):
                raise ValueError("runtime source id/index differs from frozen manifest")
            if str(frame.iloc[donor].id) != str(plan["donor_id"]):
                raise ValueError("runtime donor id/index differs from frozen manifest")
            if frame.iloc[source].category != frame.iloc[donor].category:
                raise ValueError("runtime donor category mismatch")
            if int(frame.iloc[source].label) != int(frame.iloc[donor].label):
                raise ValueError("runtime donor label mismatch")
            if (
                folds[source] == self.parent.HOLDOUT_FOLD
                or folds[donor] == self.parent.HOLDOUT_FOLD
            ):
                raise ValueError("outer-validation row entered runtime recombination")
        return records

    def open_images(self, rows):
        if not self.training:
            return self.original_open_images(rows)
        from PIL import Image

        images = []
        for row in rows:
            plan = self.by_position.get(self.position)
            item_id = str(row.id)
            if plan is None:
                image_id = item_id
            else:
                if item_id != str(plan["source_id"]) or int(row.name) != int(plan["source_index"]):
                    raise ValueError("runtime training occurrence differs from frozen manifest")
                image_id = str(plan["donor_id"])
                if str(row.category) != str(plan["category"]) or int(row.label) != int(
                    plan["label"]
                ):
                    raise ValueError("runtime source target differs from frozen manifest")
                self.replacements += 1
            images.append(Image.open(self.parent.IMAGE_DIR / f"{image_id}.jpg").convert("RGB"))
            self.position += 1
        return images

    def validation_scores(self, *args, **kwargs):
        self.training = False
        return self.original_validation(*args, **kwargs)

    def report(self) -> dict[str, Any]:
        if self.position != self.audit["parent_training_records"]:
            raise RuntimeError("runtime training occurrence count changed")
        if self.replacements != self.audit["recombined_occurrences"]:
            raise RuntimeError("runtime replacement count differs from frozen manifest")
        return {
            "experiment_id": "590",
            "training_view_policy": TRAINING_VIEW_POLICY,
            "inference_view_policy": INFERENCE_VIEW_POLICY,
            "inference_image_index": 0,
            "inference_image_count": 1,
            "inference_passes": 1,
            "first_image_max_edge": 448,
            "first_image_max_pixels": 262144,
            "recombined_occurrences": self.replacements,
            "eligible_repeated_occurrences": self.audit["eligible_repeated_occurrences"],
            "recombined_fraction_of_repeats": self.audit["recombined_fraction_of_repeats"],
            "manifest_sha256": self.audit["manifest_sha256"],
            "preflight_decision": self.audit["decision"],
            "record_multiset_unchanged": True,
            "steps_unchanged": True,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed exp590 fold trainer.")
    parser.add_argument("--fold", required=True, type=int, choices=SCREEN_FOLDS)
    parser.add_argument("--preflight-audit", required=True, type=Path)
    parser.add_argument("--recombination-manifest", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit, manifest = load_authorized_plan(
        args.preflight_audit.resolve(), args.recombination_manifest.resolve(), args.fold
    )
    configure_environment(args.fold)
    parent = _load_parent()
    if file_sha256(parent.DATA.resolve()) != audit["data_sha256"]:
        raise ValueError("runtime data differs from frozen preflight")
    if file_sha256(parent.OOF.resolve()) != audit["oof_sha256"]:
        raise ValueError("runtime OOF differs from frozen preflight")
    if parent.FIRST_IMAGE_MAX_EDGE != 448 or parent.FIRST_IMAGE_MAX_PIXELS != 262144:
        raise ValueError("parent first-image inference contract mismatch")
    if parent.BATCH_SIZE != 4 or parent.GRAD_ACCUM != 4 or parent.EPOCHS != 1:
        raise ValueError("parent step recipe mismatch")
    runtime = RecombinationRuntime(parent, audit, manifest)
    parent.select_training = runtime.select_training
    parent.open_images = runtime.open_images
    parent.validation_scores = runtime.validation_scores
    existing = [
        parent.OUTPUT / name
        for name in ("lora_holdout_predictions.csv", "lora_holdout_report.json", "adapter.zip")
        if (parent.OUTPUT / name).exists()
    ]
    if existing:
        raise FileExistsError("refusing to overwrite training outputs")
    parent.main()
    report_path = parent.OUTPUT / "lora_holdout_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(runtime.report())
    report["preflight_audit_sha256"] = file_sha256(args.preflight_audit.resolve())
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
