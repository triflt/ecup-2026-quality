from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
TARGETS = Path(os.environ.get("ECUP_TEACHER_TARGETS", "/work/input/rationale_then_label.jsonl"))
IMAGE0_MANIFEST = Path(os.environ.get(
    "ECUP_IMAGE0_MANIFEST", "/work/handoff/ensemble_image0_manifest.jsonl"
))
SOURCE_IMAGES = Path(os.environ.get("ECUP_SOURCE_IMAGES", "/work/input/images"))
OBJECTIVE = os.environ.get("EXP713_OBJECTIVE", "rationale_then_label")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = load_module("exp713_contract", Path(__file__).with_name("target_contract.py"))
parent = load_module("exp713_parent", ROOT / "research" / "qwen3vl_lora_holdout.py")
original_load_training_frame = parent.load_training_frame
original_select_training = parent.select_training
original_training_batch = parent.training_batch
generated_records: list[dict[str, object]] = []
available_target_ids: set[str] = set()
bound_images: dict[str, dict[str, object]] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bound_images() -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    with IMAGE0_MANIFEST.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            item_id = str(row["id"])
            if item_id in output:
                raise ValueError(f"duplicate image0 manifest id={item_id}")
            output[item_id] = row
    return output


def bound_urls() -> dict[str, str]:
    bound_images.clear()
    bound_images.update(load_bound_images())
    return {item_id: "manifest-bound-image0" for item_id in bound_images}


def verify_bound_images(ids, urls):
    del urls
    failures = []
    for item_id in ids:
        item_id = str(item_id)
        row = bound_images.get(item_id)
        path = SOURCE_IMAGES / str(row["path"]) if row is not None else SOURCE_IMAGES
        if row is None or not path.is_file():
            failures.append((item_id, "missing source ensemble image0"))
            continue
        if path.stat().st_size != int(row["size_bytes"]) or sha256_file(path) != row["sha256"]:
            failures.append((item_id, "source ensemble image0 SHA/size mismatch"))
    if failures:
        raise ValueError(f"source ensemble image0 verification failed: {failures[:20]}")
    print(json.dumps({"source_ensemble_image0_verified": len(ids)}), flush=True)
    return []


def open_ensemble_images(rows):
    images = []
    for row in rows:
        binding = bound_images[str(row.id)]
        with Image.open(SOURCE_IMAGES / str(binding["path"])) as opened:
            image = opened.convert("RGB")
            image.thumbnail((448, 448), Image.Resampling.LANCZOS)
            image.load()
        images.append(image)
    return images


def use_vendored_peft() -> None:
    """Use the manifest-bound PEFT wheel contents mounted by the remote compute preset."""
    package = parent.VENDOR / "peft" / "__init__.py"
    if not package.is_file():
        raise FileNotFoundError(f"vendored PEFT is missing: {package}")
    sys.path.insert(0, str(parent.VENDOR))


def load_targets(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            item_id = str(row["id"])
            label = int(row["label"])
            if OBJECTIVE == "rationale_then_label":
                parsed = contract.parse_target(row["target"])
                if int(parsed["label"]) != label:
                    raise ValueError(f"target label mismatch for id={item_id}")
            elif str(row["target"]) != str(label):
                raise ValueError(f"matched digit target mismatch for id={item_id}")
            output[item_id] = str(row["target"])
    return output


def load_training_frame():
    frame = original_load_training_frame()
    targets = load_targets(TARGETS)
    available_target_ids.clear()
    available_target_ids.update(targets)
    ids = frame["id"].astype(str)
    frame["target"] = ids.map(targets)
    for row in frame.loc[frame["target"].notna()].itertuples(index=False):
        if OBJECTIVE == "rationale_then_label":
            parsed = contract.parse_target(row.target)
            if int(parsed["label"]) != int(row.label):
                raise ValueError(f"source/target label mismatch for id={row.id}")
        elif str(row.target) != str(int(row.label)):
            raise ValueError(f"source/matched digit mismatch for id={row.id}")
    return frame


def select_training(frame, oof):
    records = original_select_training(frame, oof)
    filtered = [index for index in records if str(frame.iloc[index].id) in available_target_ids]
    coverage = len(filtered) / len(records) if records else 0.0
    if coverage < 0.85:
        raise ValueError(f"supported rationale train-record coverage below 85%: {coverage:.6f}")
    print(json.dumps({
        "selected_train_records_before_target_filter": len(records),
        "selected_train_records_after_target_filter": len(filtered),
        "supported_train_record_coverage": coverage,
    }), flush=True)
    return filtered


def messages(row, with_answer=False, image=None):
    image = image if image is not None else str(parent.IMAGE_DIR / f"{row.id}.jpg")
    return contract.messages(row, with_answer=with_answer, image=image)


def training_batch(processor, rows):
    batch = original_training_batch(processor, rows)
    labels = batch["labels"]
    for row_index, row in enumerate(rows):
        supervised = labels[row_index][labels[row_index] != -100].tolist()
        decoded = processor.tokenizer.decode(supervised, skip_special_tokens=True).strip()
        expected = (
            str(row.target).strip()
            if OBJECTIVE == "rationale_then_label"
            else str(int(row.label))
        )
        if decoded != expected:
            raise ValueError(
                f"assistant target was truncated or altered for id={row.id}: "
                f"decoded_chars={len(decoded)} expected_chars={len(expected)}"
            )
    return batch


@parent.torch.inference_mode()
def validation_scores(model, processor, frame, positions, token_zero, token_one):
    del token_zero, token_one
    model.eval()
    generated_records.clear()
    scores: list[float] = []
    for start in range(0, len(positions), 2):
        local = positions[start:start + 2]
        rows = [frame.iloc[index] for index in local]
        images = parent.open_images(rows)
        batch = parent.chat_batch(
            processor,
            [messages(row, False, image) for row, image in zip(rows, images)],
            True,
        )
        for image in images:
            image.close()
        batch = {key: value.to("cuda") for key, value in batch.items()}
        prompt_length = batch["input_ids"].shape[1]
        generated = model.generate(
            **batch,
            max_new_tokens=384,
            do_sample=False,
            use_cache=True,
        )
        texts = processor.tokenizer.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True
        )
        for row, text in zip(rows, texts, strict=True):
            try:
                if OBJECTIVE == "rationale_then_label":
                    parsed = contract.parse_target(text)
                    label = int(parsed["label"])
                else:
                    value = str(text).strip()
                    if value not in {"0", "1"}:
                        raise ValueError("digit control must generate exactly 0 or 1")
                    label = int(value)
                valid = True
                error = None
            except (TypeError, ValueError) as exception:
                label = -1
                valid = False
                error = f"{type(exception).__name__}: {exception}"
            scores.append(8.0 if label == 1 else -8.0 if label == 0 else 0.0)
            generated_records.append({
                "id": str(row.id),
                "category": str(row.category),
                "label": int(row.label),
                "generated_label": label,
                "format_valid": valid,
                "error": error,
                "generation": text,
            })
        print(f"generated_validation={len(scores)}/{len(positions)}", flush=True)
    return parent.np.asarray(scores, dtype=parent.np.float32)


def write_generation_report() -> None:
    parent.OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = "label_last" if OBJECTIVE == "rationale_then_label" else "matched_digit"
    output = parent.OUTPUT / f"{stem}_generations.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        for row in generated_records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    valid = [row for row in generated_records if row["format_valid"]]
    correct = [row for row in valid if row["generated_label"] == row["label"]]
    category_metrics: dict[str, dict[str, object]] = {}
    category_f1: list[float] = []
    for category in sorted({str(row["category"]) for row in generated_records}):
        rows = [row for row in generated_records if row["category"] == category]
        valid_rows = [row for row in rows if row["format_valid"]]
        strict_predictions = [
            int(row["generated_label"])
            if row["format_valid"] else 1 - int(row["label"])
            for row in rows
        ]
        pairs = list(zip(strict_predictions, rows))
        tp = sum(prediction == 1 and row["label"] == 1 for prediction, row in pairs)
        fp = sum(prediction == 1 and row["label"] == 0 for prediction, row in pairs)
        fn = sum(prediction == 0 and row["label"] == 1 for prediction, row in pairs)
        tn = sum(prediction == 0 and row["label"] == 0 for prediction, row in pairs)
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0.0
        category_f1.append(f1)
        strict_correct = sum(
            row["format_valid"] and row["generated_label"] == row["label"]
            for row in rows
        )
        category_metrics[category] = {
            "rows": len(rows),
            "format_valid": len(valid_rows),
            "format_valid_rate": len(valid_rows) / len(rows) if rows else 0.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "f1_invalid_is_wrong": f1,
            "strict_accuracy_invalid_is_wrong": strict_correct / len(rows) if rows else 0.0,
        }
    report = {
        "schema_version": f"{OBJECTIVE}_generation_report_v1",
        "objective": OBJECTIVE,
        "rows": len(generated_records),
        "format_valid": len(valid),
        "format_valid_rate": len(valid) / len(generated_records) if generated_records else 0.0,
        "exact_accuracy_on_valid": len(correct) / len(valid) if valid else 0.0,
        "strict_accuracy_invalid_is_wrong": len(correct) / len(generated_records)
        if generated_records else 0.0,
        "macro_f1_invalid_is_wrong": sum(category_f1) / len(category_f1)
        if category_f1 else 0.0,
        "categories": category_metrics,
        "decision_rule": (
            "Parse terminal LABEL directly" if OBJECTIVE == "rationale_then_label"
            else "Parse the exact generated digit directly"
        ) + "; no fitted threshold.",
        "note": (
            "Primary causal comparison uses greedy generated decisions from the same "
            "prompt, support, order and generation budget in both arms."
        ),
    }
    (parent.OUTPUT / f"{stem}_generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main() -> None:
    if OBJECTIVE not in {"rationale_then_label", "matched_support_digit"}:
        raise ValueError(f"unsupported EXP713_OBJECTIVE={OBJECTIVE!r}")
    if parent.SOFT_TARGETS is not None:
        raise ValueError("soft targets are forbidden for the label-last experiment")
    if parent.FULL_TRAIN:
        raise ValueError("full train is blocked until folds 0/3 pass")
    if parent.HOLDOUT_FOLD not in {0, 3}:
        raise ValueError("only preregistered folds 0 and 3 are allowed")
    if (parent.BATCH_SIZE, parent.GRAD_ACCUM) != (4, 4):
        raise ValueError("unexpected parent batch contract")
    parent.BATCH_SIZE = 2
    parent.GRAD_ACCUM = 8
    parent.MAX_LENGTH = 2304
    parent.install_peft = use_vendored_peft
    parent.load_urls = bound_urls
    parent.predownload = verify_bound_images
    parent.open_images = open_ensemble_images
    parent.load_training_frame = load_training_frame
    parent.select_training = select_training
    parent.training_batch = training_batch
    parent.messages = messages
    parent.validation_scores = validation_scores
    parent.main()
    write_generation_report()
    if OBJECTIVE == "matched_support_digit":
        parent.OUTPUT.mkdir(parents=True, exist_ok=True)
        (parent.OUTPUT / "matched_support_control.json").write_text(
            json.dumps({
                "schema_version": "matched_support_digit_control_v1",
                "objective": OBJECTIVE,
                "targets_used_only_as_support_mask": True,
                "note": "Exact candidate support mask and parent digit-only assistant target.",
            }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
