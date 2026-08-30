from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoProcessor

import qwen35_bad_family_diverse_positives_lora as base


ALPHA = float(os.environ["ALPHA_260"])
SOURCE_190 = Path(os.environ.get("SOURCE_190", "/work/input/source190"))
SOURCE_260 = Path(os.environ.get("SOURCE_260", "/work/input/source260"))
MERGED = Path(os.environ.get("MERGED_ADAPTERS", "/work/soup_adapters"))
OUTPUT = Path(os.environ.get("ECUP_OUTPUT_DIR", "/work/output"))
MERGER = Path(os.environ.get("SOUP_MERGER", "/work/code/build_soup_adapter.py"))
FLAMMABLE = "Легковоспламеняющиеся"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_adapters() -> list[dict[str, object]]:
    reports = []
    for fold in range(5):
        destination = MERGED / f"fold_{fold}"
        subprocess.run(
            [
                sys.executable, str(MERGER),
                "--adapter-190", str(SOURCE_190 / f"fold_{fold}.zip"),
                "--adapter-260", str(SOURCE_260 / f"fold_{fold}.zip"),
                "--alpha-260", str(ALPHA),
                "--output", str(destination),
            ],
            check=True,
        )
        reports.append(json.loads((destination / "merge_report.json").read_text()))
    return reports


def main() -> None:
    if ALPHA not in {0.25, 0.5, 0.75}:
        raise ValueError(f"alpha outside frozen grid: {ALPHA}")
    torch.manual_seed(base.SEED)
    np.random.seed(base.SEED)
    base.install_peft()
    from peft import PeftModel

    merge_reports = build_adapters()
    frame = base.load_training_frame()
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["category"] = frame["category"].astype(str)
    oof = np.load(base.OOF, allow_pickle=True)
    ids = frame["id"].astype(str).to_numpy()
    if not np.array_equal(ids, oof["ids"].astype(str)):
        raise ValueError("OOF id mismatch")
    folds = oof["fold_ids"].astype(np.int8)
    categories = frame["category"].astype(str).to_numpy()
    labels = frame["label"].to_numpy(dtype=np.int8)
    flammable_positions = np.flatnonzero(categories == FLAMMABLE)

    urls = base.load_urls()
    failures = base.predownload(ids[flammable_positions].tolist(), urls)
    if failures:
        raise ValueError(f"image download failures: {len(failures)}")
    print(json.dumps({
        "experiment": 430,
        "alpha_260": ALPHA,
        "validation_flammable": len(flammable_positions),
        "download_failures": len(failures),
    }), flush=True)

    processor_kwargs = {"local_files_only": True, "trust_remote_code": True}
    if base.MODEL_CLASS == "image_text":
        processor_kwargs.update(min_pixels=4 * 28 * 28, max_pixels=262144)
    processor = AutoProcessor.from_pretrained(base.MODEL, **processor_kwargs)
    processor.tokenizer.padding_side = "left"
    token_zero_ids = processor.tokenizer.encode("0", add_special_tokens=False)
    token_one_ids = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(token_zero_ids) != 1 or len(token_one_ids) != 1:
        raise ValueError("digit tokens are not atomic")
    token_zero, token_one = token_zero_ids[0], token_one_ids[0]

    loader = AutoModelForMultimodalLM if base.MODEL_CLASS == "multimodal" else AutoModelForImageTextToText
    foundation = loader.from_pretrained(
        base.MODEL,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
    model = None
    scores = np.full(len(frame), np.nan, dtype=np.float32)
    fold_reports = []
    for fold in range(5):
        adapter_name = f"fold_{fold}"
        adapter = MERGED / adapter_name
        if model is None:
            model = PeftModel.from_pretrained(foundation, adapter, adapter_name=adapter_name)
        else:
            model.load_adapter(adapter, adapter_name=adapter_name)
        model.set_adapter(adapter_name)
        positions = np.flatnonzero((folds == fold) & (categories == FLAMMABLE))
        local_scores = base.validation_scores(
            model, processor, frame, positions, token_zero, token_one
        )
        scores[positions] = local_scores
        diagnostic_f1, diagnostic_threshold = base.best_threshold(labels[positions], local_scores)
        fold_reports.append({
            "fold": fold,
            "rows": len(positions),
            "standalone_flammable_f1": diagnostic_f1,
            "standalone_flammable_threshold": diagnostic_threshold,
        })
        print(json.dumps(fold_reports[-1]), flush=True)

    if np.isnan(scores[flammable_positions]).any():
        raise ValueError("missing flammable scores")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prediction_path = OUTPUT / "soup_oof_predictions.csv"
    pd.DataFrame({
        "id": ids[flammable_positions],
        "category": categories[flammable_positions],
        "label": labels[flammable_positions],
        "fold": folds[flammable_positions],
        "lora_score": scores[flammable_positions],
    }).to_csv(prediction_path, index=False)
    report = {
        "experiment_id": "430",
        "alpha_260": ALPHA,
        "method": "one-pass exact rank-32 LoRA-delta interpolation",
        "prediction_rows": len(flammable_positions),
        "predictions_sha256": sha256(prediction_path),
        "download_failures": len(failures),
        "folds": fold_reports,
        "merge_reports": merge_reports,
    }
    (OUTPUT / "soup_oof_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
