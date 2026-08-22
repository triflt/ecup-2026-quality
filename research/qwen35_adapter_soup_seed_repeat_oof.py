from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import qwen35_bad_family_diverse_positives_lora as base
import torch
from transformers import AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoProcessor

OUTPUT = Path(os.environ.get("ECUP_OUTPUT_DIR", "/work/output"))
ORIGINAL = Path("/work/input/original_seed_adapter.zip")
SPECIALIST = Path(os.environ["SPECIALIST_ADAPTER"])
MERGED = Path("/work/soup_adapter")
MERGER = Path("/work/code/build_soup_adapter.py")
FLAMMABLE = "Легковоспламеняющиеся"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    fold = int(os.environ["HOLDOUT_FOLD"])
    if fold not in {0, 3} or base.SEED != 31415:
        raise ValueError("frozen fold/seed mismatch")
    subprocess.run([
        sys.executable, str(MERGER),
        "--adapter-190", str(ORIGINAL),
        "--adapter-260", str(SPECIALIST),
        "--alpha-260", "0.5",
        "--output", str(MERGED),
    ], check=True)

    torch.manual_seed(base.SEED)
    np.random.seed(base.SEED)
    base.install_peft()
    from peft import PeftModel

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
    positions = np.flatnonzero((folds == fold) & (categories == FLAMMABLE))

    processor_kwargs = {"local_files_only": True, "trust_remote_code": True}
    if base.MODEL_CLASS == "image_text":
        processor_kwargs.update(min_pixels=4 * 28 * 28, max_pixels=262144)
    processor = AutoProcessor.from_pretrained(base.MODEL, **processor_kwargs)
    processor.tokenizer.padding_side = "left"
    token_zero_ids = processor.tokenizer.encode("0", add_special_tokens=False)
    token_one_ids = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(token_zero_ids) != 1 or len(token_one_ids) != 1:
        raise ValueError("digit tokens are not atomic")

    loader = AutoModelForMultimodalLM if base.MODEL_CLASS == "multimodal" else AutoModelForImageTextToText
    foundation = loader.from_pretrained(
        base.MODEL,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
    model = PeftModel.from_pretrained(foundation, MERGED)
    scores = base.validation_scores(
        model, processor, frame, positions, token_zero_ids[0], token_one_ids[0]
    )
    if not np.isfinite(scores).all():
        raise ValueError("non-finite soup scores")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    prediction_path = OUTPUT / "soup_holdout_predictions.csv"
    pd.DataFrame({
        "id": ids[positions],
        "category": categories[positions],
        "label": labels[positions],
        "fold": folds[positions],
        "lora_score": scores,
    }).to_csv(prediction_path, index=False)
    report = {
        "experiment_id": "470",
        "fold": fold,
        "seed": base.SEED,
        "alpha_specialist": 0.5,
        "rows": len(positions),
        "predictions_sha256": sha256(prediction_path),
        "original_adapter_sha256": sha256(ORIGINAL),
        "specialist_adapter_sha256": sha256(SPECIALIST),
        "merge_report": json.loads((MERGED / "merge_report.json").read_text()),
    }
    (OUTPUT / "soup_holdout_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
