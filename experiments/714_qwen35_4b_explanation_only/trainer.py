from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
TARGETS = Path(os.environ.get("ECUP_TEACHER_TARGETS", "/work/input/explanation_only.jsonl"))
IMAGE0_MANIFEST = Path(os.environ.get(
    "ECUP_IMAGE0_MANIFEST", "/work/handoff/ensemble_image0_manifest.jsonl"
))
SOURCE_IMAGES = Path(os.environ.get("ECUP_SOURCE_IMAGES", "/work/input/images"))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = load_module("exp714_contract", Path(__file__).with_name("target_contract.py"))
parent = load_module("exp714_parent", ROOT / "research" / "qwen3vl_lora_holdout.py")
original_load_training_frame = parent.load_training_frame
original_select_training = parent.select_training
original_training_batch = parent.training_batch
generated_records: list[dict[str, object]] = []
available_target_ids: set[str] = set()
bound_images: dict[str, dict[str, object]] = {}


def static_fallback(category: str, verdict: int) -> str:
    if category == "БАД":
        return (
            "Текст и первое изображение подтверждают маркировку товара как "
            "биологически активной добавки."
            if verdict
            else "Текст и первое изображение не подтверждают обязательную маркировку "
                 "товара как биологически активной добавки."
        )
    return (
        "Текст и первое изображение подтверждают наличие самостоятельного "
        "горючего товара или источника воспламенения."
        if verdict
        else "Текст и первое изображение не подтверждают наличие самостоятельного "
             "горючего товара или источника воспламенения."
    )


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


def load_targets(path: Path) -> dict[str, tuple[int, str]]:
    output: dict[str, tuple[int, str]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            item_id = str(row["id"])
            contract.parse_target(row["target"])
            label = int(row["label"])
            output[item_id] = (label, str(row["target"]))
    return output


def load_training_frame():
    frame = original_load_training_frame()
    targets = load_targets(TARGETS)
    available_target_ids.clear()
    available_target_ids.update(targets)
    ids = frame["id"].astype(str)
    for item_id, label in zip(ids, frame["label"].astype(int), strict=True):
        if item_id in targets and targets[item_id][0] != int(label):
            raise ValueError(f"target/source label mismatch for id={item_id}")
    frame["verdict"] = frame["label"].astype(int)
    frame["target"] = ids.map({key: value[1] for key, value in targets.items()})
    return frame


def select_training(frame, oof):
    records = original_select_training(frame, oof)
    filtered = [index for index in records if str(frame.iloc[index].id) in available_target_ids]
    coverage = len(filtered) / len(records) if records else 0.0
    if coverage < 0.85:
        raise ValueError(f"supported explanation train-record coverage below 85%: {coverage:.6f}")
    print(json.dumps({
        "selected_train_records_before_target_filter": len(records),
        "selected_train_records_after_target_filter": len(filtered),
        "supported_train_record_coverage": coverage,
    }), flush=True)
    return filtered


def messages(row, with_answer=False, image=None, verdict=None):
    image = image if image is not None else str(parent.IMAGE_DIR / f"{row.id}.jpg")
    conditioning_verdict = int(row.verdict) if verdict is None else int(verdict)
    return contract.messages(
        row, verdict=conditioning_verdict, with_answer=with_answer, image=image
    )


def training_batch(processor, rows):
    batch = original_training_batch(processor, rows)
    labels = batch["labels"]
    for row_index, row in enumerate(rows):
        supervised = labels[row_index][labels[row_index] != -100].tolist()
        decoded = processor.tokenizer.decode(supervised, skip_special_tokens=True).strip()
        expected = str(row.target).strip()
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
    oof = parent.np.load(parent.OOF, allow_pickle=True)
    oof_ids = oof["ids"].astype(str)
    frame_ids = frame["id"].astype(str).to_numpy()
    if not parent.np.array_equal(oof_ids, frame_ids):
        raise ValueError("OOF id mismatch in explanation-only screen")
    frozen_scores = parent.fused_oof_scores(oof)
    frozen_thresholds = parent.oof_thresholds(
        oof, frame["category"].astype(str).to_numpy()
    )
    frozen_verdicts = (frozen_scores >= frozen_thresholds).astype(parent.np.int8)
    verdict_by_id = dict(zip(frame_ids, frozen_verdicts, strict=True))
    scores: list[float] = []
    for start in range(0, len(positions), 2):
        local = positions[start:start + 2]
        rows = [frame.iloc[index] for index in local]
        images = parent.open_images(rows)
        batch = parent.chat_batch(
            processor,
            [
                messages(
                    row,
                    False,
                    image,
                    verdict=verdict_by_id[str(row.id)],
                )
                for row, image in zip(rows, images)
            ],
            True,
        )
        for image in images:
            image.close()
        batch = {key: value.to("cuda") for key, value in batch.items()}
        prompt_length = batch["input_ids"].shape[1]
        generated = model.generate(
            **batch,
            max_new_tokens=192,
            do_sample=False,
            use_cache=True,
        )
        texts = processor.tokenizer.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True
        )
        for row, text in zip(rows, texts, strict=True):
            frozen_verdict = int(verdict_by_id[str(row.id)])
            use_static_fallback = frozen_verdict != int(row.label)
            if use_static_fallback:
                text = static_fallback(str(row.category), frozen_verdict)
            try:
                parsed = contract.parse_target(text)
                valid = True
                error = None
                comment_chars = len(parsed["explanation"].strip())
            except (TypeError, ValueError) as exception:
                valid = False
                error = f"{type(exception).__name__}: {exception}"
                comment_chars = None
            scores.append(8.0 if frozen_verdict == 1 else -8.0)
            generated_records.append({
                "id": str(row.id),
                "category": str(row.category),
                "label": int(row.label),
                "verdict": frozen_verdict,
                "conditioning_source": "frozen_solution140_oof_verdict",
                "explanation_source": "static_fallback_oof_disagreement"
                if use_static_fallback else "generated",
                "format_valid": valid,
                "comment_chars": comment_chars,
                "error": error,
                "generation": text,
            })
        print(f"generated_explanations={len(scores)}/{len(positions)}", flush=True)
    return parent.np.asarray(scores, dtype=parent.np.float32)


def write_generation_report() -> None:
    parent.OUTPUT.mkdir(parents=True, exist_ok=True)
    output = parent.OUTPUT / "explanation_generations.jsonl"
    with output.open("w", encoding="utf-8") as stream:
        for row in generated_records:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    valid = sum(bool(row["format_valid"]) for row in generated_records)
    generated = [row for row in generated_records if row["explanation_source"] == "generated"]
    generated_valid = sum(bool(row["format_valid"]) for row in generated)
    fallback_rows = len(generated_records) - len(generated)
    report = {
        "schema_version": "explanation_only_generation_report_v1",
        "rows": len(generated_records),
        "format_valid": valid,
        "format_valid_rate": valid / len(generated_records) if generated_records else 0.0,
        "generated_rows": len(generated),
        "generated_format_valid": generated_valid,
        "generated_format_valid_rate": generated_valid / len(generated) if generated else 0.0,
        "static_fallback_oof_disagreement_rows": fallback_rows,
        "classifier_scores_changed": 0,
        "training_conditioning_source": "label",
        "validation_conditioning_source": "frozen_solution140_oof_verdict",
        "note": (
            "The explanation branch receives the immutable frozen OOF verdict and cannot "
            "alter classifier scores. Gold-conditioned rationales are never presented as "
            "explanations of the opposite OOF verdict; those audit rows use a static fallback."
        ),
    }
    (parent.OUTPUT / "explanation_generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main() -> None:
    if parent.SOFT_TARGETS is not None:
        raise ValueError("soft targets are forbidden for explanation-only")
    if parent.FULL_TRAIN:
        raise ValueError("full train is blocked until blind explanation audit passes")
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


if __name__ == "__main__":
    main()
