from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import qwen35_bad_family_diverse_positives_lora as base
import torch
from transformers import AutoModelForImageTextToText, AutoModelForMultimodalLM, AutoProcessor

OUTPUT = Path(os.environ.get("ECUP_OUTPUT_DIR", "/work/output"))
ORIGINAL_ZIP = Path("/work/input/original_adapter.zip")
SPECIALIST_ZIP = Path("/work/input/specialist_adapter.zip")
MERGER = Path("/work/code/build_blockwise_soup_adapter.py")
MERGED = Path("/work/fisher_soup_adapter")
FLAMMABLE = "Легковоспламеняющиеся"
BLOCK_PATTERN = re.compile(r"^(.*)\.lora_[AB]\.[^.]+\.weight$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unpack(source: Path, destination: Path) -> Path:
    with zipfile.ZipFile(source) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"adapter CRC failure: {bad}")
        archive.extractall(destination)
    return destination


def stable_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def family_representatives(frame: pd.DataFrame, positions: np.ndarray) -> list[int]:
    selected: dict[str, int] = {}
    for position in positions:
        key = base.family_key(frame.iloc[int(position)])
        selected.setdefault(key, int(position))
    return sorted(selected.values(), key=lambda pos: stable_key(base.family_key(frame.iloc[pos])))


def choose_calibration(frame: pd.DataFrame, oof, fold: int) -> tuple[list[int], list[int], dict]:
    categories = frame.category.astype(str).to_numpy()
    labels = frame.label.to_numpy(dtype=np.int8)
    folds = oof["fold_ids"].astype(np.int8)
    train = folds != fold
    uncertainty = np.abs(base.fused_oof_scores(oof) - base.oof_thresholds(oof, categories))

    broad: list[int] = []
    for category in sorted(set(categories)):
        positives = family_representatives(
            frame, np.flatnonzero(train & (categories == category) & (labels == 1))
        )[:128]
        negatives = family_representatives(
            frame, np.flatnonzero(train & (categories == category) & (labels == 0))
        )
        negatives = sorted(
            negatives, key=lambda pos: (float(uncertainty[pos]), stable_key(str(pos)))
        )[:64]
        broad.extend(positives)
        broad.extend(negatives)

    rare_positive = family_representatives(
        frame, np.flatnonzero(train & (categories == FLAMMABLE) & (labels == 1))
    )
    rare_negative = family_representatives(
        frame, np.flatnonzero(train & (categories == FLAMMABLE) & (labels == 0))
    )
    rare_negative = sorted(
        rare_negative, key=lambda pos: (float(uncertainty[pos]), stable_key(str(pos)))
    )[: len(rare_positive)]
    rare = rare_positive + rare_negative
    return (
        broad,
        rare,
        {
            "holdout_fold": fold,
            "broad_rows": len(broad),
            "rare_rows": len(rare),
            "rare_positive_rows": len(rare_positive),
            "rare_negative_rows": len(rare_negative),
            "family_cap": 1,
            "uses_holdout_labels": False,
        },
    )


def projection_type(block: str) -> str:
    return next(
        (name for name in ("q_proj", "k_proj", "v_proj", "o_proj") if name in block), "other"
    )


def normalize_energy(values: dict[str, float]) -> dict[str, float]:
    result = {}
    for kind in sorted({projection_type(key) for key in values}):
        keys = [key for key in values if projection_type(key) == kind]
        positive = [values[key] for key in keys if values[key] > 0]
        scale = float(np.median(positive)) if positive else 1.0
        for key in keys:
            result[key] = float(values[key] / max(scale, 1e-30))
    return result


def compute_energy(
    adapter: Path, positions: list[int], frame, processor, tag: str
) -> dict[str, float]:
    base.install_peft()
    from peft import PeftModel

    loader = (
        AutoModelForMultimodalLM
        if base.MODEL_CLASS == "multimodal"
        else AutoModelForImageTextToText
    )
    foundation = loader.from_pretrained(
        base.MODEL,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
    model = PeftModel.from_pretrained(foundation, adapter, is_trainable=True)
    model.config.use_cache = False
    model.train()
    trainable = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("adapter has no trainable parameters")
    energy: dict[str, float] = {}
    counts: dict[str, int] = {}
    batch_size = 2
    for start in range(0, len(positions), batch_size):
        rows = [frame.iloc[position] for position in positions[start : start + batch_size]]
        batch = base.training_batch(processor, rows)
        batch = {key: value.to("cuda") for key, value in batch.items()}
        model.zero_grad(set_to_none=True)
        model(**batch).loss.backward()
        for name, parameter in trainable:
            match = BLOCK_PATTERN.match(name)
            if match is None or parameter.grad is None:
                continue
            block = match.group(1)
            energy[block] = energy.get(block, 0.0) + float(
                parameter.grad.float().square().mean().cpu()
            )
            counts[block] = counts.get(block, 0) + 1
        done = min(start + batch_size, len(positions))
        if done % 40 < batch_size or done == len(positions):
            print(json.dumps({"energy": tag, "done": done, "rows": len(positions)}), flush=True)
    if not energy:
        raise ValueError("no LoRA gradient energies were collected")
    result = {key: energy[key] / counts[key] for key in energy}
    del model, foundation
    gc.collect()
    torch.cuda.empty_cache()
    return normalize_energy(result)


def main() -> None:
    fold = int(os.environ["HOLDOUT_FOLD"])
    if fold not in {0, 3} or base.SEED != 31415:
        raise ValueError("frozen fold/seed mismatch")
    torch.manual_seed(base.SEED)
    np.random.seed(base.SEED)

    frame = base.load_training_frame()
    frame["name"] = frame.name.fillna("").astype(str)
    frame["description"] = frame.description.fillna("").astype(str)
    frame["category"] = frame.category.astype(str)
    oof = np.load(base.OOF, allow_pickle=True)
    ids = frame.id.astype(str).to_numpy()
    if not np.array_equal(ids, oof["ids"].astype(str)):
        raise ValueError("OOF id mismatch")
    folds = oof["fold_ids"].astype(np.int8)
    categories = frame.category.astype(str).to_numpy()
    labels = frame.label.to_numpy(dtype=np.int8)
    broad, rare, selection = choose_calibration(frame, oof, fold)
    holdout = np.flatnonzero((folds == fold) & (categories == FLAMMABLE))
    needed = sorted(set(ids[broad]) | set(ids[rare]) | set(ids[holdout]))
    failures = base.predownload(needed, base.load_urls())
    if failures:
        raise ValueError(f"image download failures: {len(failures)}")

    processor = AutoProcessor.from_pretrained(
        base.MODEL, local_files_only=True, trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    zero = processor.tokenizer.encode("0", add_special_tokens=False)
    one = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero) != 1 or len(one) != 1:
        raise ValueError("digit tokens are not atomic")

    with (
        tempfile.TemporaryDirectory() as original_dir,
        tempfile.TemporaryDirectory() as specialist_dir,
    ):
        original = unpack(ORIGINAL_ZIP, Path(original_dir))
        specialist = unpack(SPECIALIST_ZIP, Path(specialist_dir))
        original_energy = compute_energy(original, broad, frame, processor, "broad_original")
        specialist_energy = compute_energy(specialist, rare, frame, processor, "rare_specialist")
        if set(original_energy) != set(specialist_energy):
            raise ValueError("Fisher block mismatch")
        alphas = {}
        for block in sorted(original_energy):
            left, right = original_energy[block], specialist_energy[block]
            raw = right / max(left + right, 1e-30)
            alphas[block.replace(".lora_A", "").replace(".lora_B", "")] = float(
                np.clip(raw, 0.10, 0.90)
            )
        if len(alphas) != len(original_energy):
            raise ValueError("duplicate normalized Fisher block")
        fisher_report = {
            "experiment_id": "480",
            "fold": fold,
            "seed": base.SEED,
            "selection": selection,
            "normalization": "median within q/k/v/o projection type and checkpoint",
            "lambda_specialist": 1.0,
            "alpha_clip": [0.10, 0.90],
            "original_energy": original_energy,
            "specialist_energy": specialist_energy,
            "block_alphas": alphas,
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        fisher_path = OUTPUT / "fisher_block_alphas.json"
        fisher_path.write_text(json.dumps(fisher_report, ensure_ascii=False, indent=2) + "\n")
        subprocess.run(
            [
                sys.executable,
                str(MERGER),
                "--adapter-original",
                str(ORIGINAL_ZIP),
                "--adapter-specialist",
                str(SPECIALIST_ZIP),
                "--block-alphas",
                str(fisher_path),
                "--output",
                str(MERGED),
            ],
            check=True,
        )

    loader = (
        AutoModelForMultimodalLM
        if base.MODEL_CLASS == "multimodal"
        else AutoModelForImageTextToText
    )
    foundation = loader.from_pretrained(
        base.MODEL,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
    from peft import PeftModel

    model = PeftModel.from_pretrained(foundation, MERGED)
    scores = base.validation_scores(model, processor, frame, holdout, zero[0], one[0])
    prediction_path = OUTPUT / "fisher_holdout_predictions.csv"
    pd.DataFrame(
        {
            "id": ids[holdout],
            "category": categories[holdout],
            "label": labels[holdout],
            "fold": folds[holdout],
            "lora_score": scores,
        }
    ).to_csv(prediction_path, index=False)
    report = {
        "experiment_id": "480",
        "fold": fold,
        "seed": base.SEED,
        "rows": len(holdout),
        "predictions_sha256": sha256(prediction_path),
        "original_adapter_sha256": sha256(ORIGINAL_ZIP),
        "specialist_adapter_sha256": sha256(SPECIALIST_ZIP),
        "fisher_report_sha256": sha256(fisher_path),
        "merge_report": json.loads((MERGED / "merge_report.json").read_text()),
    }
    (OUTPUT / "fisher_holdout_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    archive = Path(shutil.make_archive(f"/work/fisher_blockwise_fold_{fold}", "zip", OUTPUT))
    shutil.move(str(archive), OUTPUT / f"output_bundle_fold_{fold}.zip")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
