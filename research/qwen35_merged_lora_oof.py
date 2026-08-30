from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import qwen3vl_lora_holdout as common


ADAPTER_SOURCES = Path(os.environ.get("ECUP_ADAPTER_SOURCES", "/work/adapters"))
ADAPTER_ROOT = Path(os.environ.get("ECUP_ADAPTER_ROOT", "/work/merged_adapters"))
OUTPUT = Path(os.environ.get("ECUP_OUTPUT_DIR", "/work/output"))
MERGE_SCRIPT = Path(os.environ.get("ECUP_MERGE_SCRIPT", "/work/code/merge_lora_cat.py"))


def materialize_adapters() -> None:
    for fold in range(5):
        sources = []
        for seed in ("seed42", "seed31415"):
            candidates = sorted((ADAPTER_SOURCES / seed / f"fold_{fold}").rglob("adapter.zip"))
            if len(candidates) != 1:
                raise ValueError(
                    f"expected one adapter.zip for {seed} fold {fold}, got {candidates}"
                )
            sources.append(candidates[0])
        subprocess.run(
            [
                sys.executable,
                str(MERGE_SCRIPT),
                "--adapter-a",
                str(sources[0]),
                "--adapter-b",
                str(sources[1]),
                "--output",
                str(ADAPTER_ROOT / f"fold_{fold}"),
            ],
            check=True,
        )


def main() -> None:
    torch.manual_seed(common.SEED)
    np.random.seed(common.SEED)
    common.install_peft()
    from peft import PeftModel

    materialize_adapters()
    frame = common.load_training_frame()
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["category"] = frame["category"].astype(str)
    oof = np.load(common.OOF, allow_pickle=True)
    ids = frame["id"].astype(str).to_numpy()
    if not np.array_equal(ids, oof["ids"].astype(str)):
        raise ValueError("OOF id mismatch")

    urls = common.load_urls()
    failures = common.predownload(ids.tolist(), urls)
    processor_kwargs = {"local_files_only": True, "trust_remote_code": True}
    if common.MODEL_CLASS == "image_text":
        processor_kwargs.update(min_pixels=4 * 28 * 28, max_pixels=262144)
    processor = common.AutoProcessor.from_pretrained(common.MODEL, **processor_kwargs)
    processor.tokenizer.padding_side = "left"
    token_zero_ids = processor.tokenizer.encode("0", add_special_tokens=False)
    token_one_ids = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(token_zero_ids) != 1 or len(token_one_ids) != 1:
        raise ValueError(f"digit tokens are not atomic: {token_zero_ids}, {token_one_ids}")
    token_zero, token_one = token_zero_ids[0], token_one_ids[0]

    loader = (
        common.AutoModelForMultimodalLM
        if common.MODEL_CLASS == "multimodal"
        else common.AutoModelForImageTextToText
    )
    base = loader.from_pretrained(
        common.MODEL,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")

    fold_ids = oof["fold_ids"].astype(np.int8)
    labels = frame["label"].to_numpy(dtype=np.int8)
    categories = frame["category"].astype(str).to_numpy()
    all_scores = np.full(len(frame), np.nan, dtype=np.float32)
    model = None
    fold_reports = []
    for fold in range(5):
        adapter = ADAPTER_ROOT / f"fold_{fold}"
        adapter_name = f"fold_{fold}"
        if model is None:
            model = PeftModel.from_pretrained(base, adapter, adapter_name=adapter_name)
        else:
            model.load_adapter(adapter, adapter_name=adapter_name)
        model.set_adapter(adapter_name)
        model.eval()
        positions = np.flatnonzero(fold_ids == fold)
        scores = common.validation_scores(
            model, processor, frame, positions, token_zero, token_one
        )
        all_scores[positions] = scores
        category_report = {}
        for category in sorted(frame["category"].unique()):
            local = categories[positions] == category
            value, threshold = common.best_threshold(labels[positions][local], scores[local])
            category_report[category] = {
                "rows": int(local.sum()),
                "f1": value,
                "threshold": threshold,
            }
        fold_reports.append({"fold": fold, "categories": category_report})
        print(json.dumps(fold_reports[-1], ensure_ascii=False), flush=True)

    if np.isnan(all_scores).any():
        raise ValueError("some OOF positions were not scored")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": ids,
            "category": categories,
            "label": labels,
            "fold": fold_ids,
            "lora_score": all_scores,
        }
    ).to_csv(OUTPUT / "merged_rank32_oof_predictions.csv", index=False)
    report = {
        "method": "one-pass rank-32 concatenation of two rank-16 LoRA deltas",
        "rows": len(frame),
        "download_failures": len(failures),
        "folds": fold_reports,
    }
    (OUTPUT / "merged_rank32_oof_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
