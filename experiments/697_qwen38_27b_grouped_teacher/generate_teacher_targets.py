from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

from atomic_publish import atomic_output_directory
from run_fold import MODEL, batch_inputs, open_image, scores

FLAMMABLE = "Легковоспламеняющиеся"
RESUME_SCHEMA = "exp697_teacher_target_resume_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def unique_consumed_rows(train_rows: list[dict]) -> list[dict]:
    """Return one inference row per ID whose teacher score is consumed."""
    selected: dict[str, dict] = {}
    for row in train_rows:
        if str(row["category"]) != FLAMMABLE:
            continue
        row_id = str(row["id"])
        previous = selected.get(row_id)
        if previous is not None:
            identity = ("image_path", "category", "label", "fold")
            if any(previous[key] != row[key] for key in identity):
                raise ValueError("repeated teacher-target ID has conflicting identity")
            continue
        selected[row_id] = row
    if not selected:
        raise ValueError("teacher-target runtime has no consumed flammable rows")
    return list(selected.values())


def target_records(
    train_rows: list[dict], fold: int, score_by_id: dict[str, float]
) -> list[dict]:
    records = []
    for occurrence_index, item in enumerate(train_rows):
        consumed = str(item["category"]) == FLAMMABLE
        score = score_by_id[str(item["id"])] if consumed else 0.0
        if not math.isfinite(score):
            raise ValueError("teacher score is non-finite")
        records.append(
            {
                "occurrence_index": occurrence_index,
                "id": str(item["id"]),
                "outer_fold": fold,
                "source_fold": int(item["fold"]),
                "category": str(item["category"]),
                "label": int(item["label"]),
                "score": float(score),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate outer-safe train-occurrence scores from a frozen fold teacher."
    )
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite target output")

    runtime_audit_path = args.runtime_dir / "runtime_audit.json"
    output_contract_path = args.adapter_dir.parent / "output_contract.json"
    runtime_audit = json.loads(runtime_audit_path.read_text())
    output_contract = json.loads(output_contract_path.read_text())
    if runtime_audit.get("experiment_id") != "697" or runtime_audit.get("fold") != args.fold:
        raise ValueError("runtime identity mismatch")
    if output_contract.get("experiment_id") != "697" or output_contract.get("fold") != args.fold:
        raise ValueError("adapter output identity mismatch")
    if output_contract.get("technical_smoke"):
        raise ValueError("technical-smoke adapter cannot produce teacher targets")

    train_path = args.runtime_dir / "train.jsonl"
    train_rows = read_jsonl(train_path)
    if len(train_rows) != int(runtime_audit["train_occurrences"]):
        raise ValueError("train occurrence count differs from runtime audit")
    if any(int(row["fold"]) == args.fold for row in train_rows):
        raise ValueError("outer validation occurrence entered teacher targets")

    inference_rows = unique_consumed_rows(train_rows)
    inference_ids_sha256 = hashlib.sha256(
        "\n".join(str(row["id"]) for row in inference_rows).encode("utf-8")
    ).hexdigest()
    adapter_model_path = args.adapter_dir / "adapter_model.safetensors"
    resume_path = args.output_dir.parent / f".{args.output_dir.name}.resume.json"
    resume_contract = {
        "schema_version": RESUME_SCHEMA,
        "experiment_id": "697",
        "fold": args.fold,
        "runtime_audit_sha256": sha256(runtime_audit_path),
        "train_runtime_sha256": sha256(train_path),
        "adapter_output_contract_sha256": sha256(output_contract_path),
        "adapter_model_sha256": sha256(adapter_model_path),
        "inference_scope": "unique_consumed_flammable_ids_only",
        "inference_ids_sha256": inference_ids_sha256,
    }
    resumed = False
    score_values: list[float] = []
    if resume_path.is_file():
        resume = json.loads(resume_path.read_text(encoding="utf-8"))
        if resume.get("contract") != resume_contract:
            raise ValueError("teacher-target resume contract mismatch")
        score_values = [float(value) for value in resume.get("scores", [])]
        if len(score_values) > len(inference_rows) or any(
            not math.isfinite(value) for value in score_values
        ):
            raise ValueError("teacher-target resume progress is invalid")
        resumed = True

    started = time.monotonic()
    final_batch_size = args.batch_size
    if len(score_values) < len(inference_rows):
        processor = AutoProcessor.from_pretrained(
            MODEL, local_files_only=True, trust_remote_code=True
        )
        processor.tokenizer.padding_side = "left"
        zero = processor.tokenizer.encode("0", add_special_tokens=False)
        one = processor.tokenizer.encode("1", add_special_tokens=False)
        if len(zero) != 1 or len(one) != 1:
            raise ValueError("0/1 are not atomic tokens")

        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL,
            dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=True,
            attn_implementation="eager",
            device_map="balanced",
            max_memory={index: "70GiB" for index in range(torch.cuda.device_count())},
        )
        model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False)
        model.eval()
        model.config.use_cache = True

        offset = len(score_values)
        with torch.inference_mode():
            while offset < len(inference_rows):
                local = inference_rows[offset : offset + final_batch_size]
                rows = [SimpleNamespace(**item) for item in local]
                images = [open_image(item) for item in local]
                try:
                    values = scores(
                        model,
                        batch_inputs(processor, rows, images),
                        zero[0],
                        one[0],
                    )
                except torch.cuda.OutOfMemoryError:
                    if final_batch_size == 1:
                        raise
                    final_batch_size = max(1, final_batch_size // 2)
                    torch.cuda.empty_cache()
                    print(
                        json.dumps(
                            {
                                "target_batch_fallback": final_batch_size,
                                "retry_offset": offset,
                            }
                        ),
                        flush=True,
                    )
                    continue
                finally:
                    for image in images:
                        image.close()
                score_values.extend(float(value) for value in values.float().cpu())
                offset += len(local)
                if offset % 200 == 0 or offset == len(inference_rows):
                    atomic_write_json(
                        {
                            "contract": resume_contract,
                            "scores": score_values,
                        },
                        resume_path,
                    )
                    print(
                        json.dumps(
                            {
                                "fold": args.fold,
                                "processed": offset,
                                "rows": len(inference_rows),
                                "elapsed_min": round(
                                    (time.monotonic() - started) / 60, 2
                                ),
                            }
                        ),
                        flush=True,
                    )

    score_by_id = {
        str(row["id"]): value
        for row, value in zip(inference_rows, score_values, strict=True)
    }
    records = target_records(train_rows, args.fold, score_by_id)
    with atomic_output_directory(args.output_dir) as staging:
        target_path = staging / "teacher_targets.jsonl"
        with target_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )

        report = {
            "schema_version": "exp697_teacher_targets_v1",
            "experiment_id": "697",
            "fold": args.fold,
            "model": str(MODEL),
            "outer_safe_validation_excluded": True,
            "target_scope": "outer_train_occurrences",
            "targets_are_in_sample_within_outer_train": True,
            "teacher_frozen_before_target_generation": True,
            "atomic_directory_publish": True,
            "rows": len(train_rows),
            "unique_ids": len({row["id"] for row in train_rows}),
            "runtime_audit_sha256": sha256(runtime_audit_path),
            "adapter_output_contract_sha256": sha256(output_contract_path),
            "adapter_model_sha256": sha256(adapter_model_path),
            "train_runtime_sha256": sha256(train_path),
            "teacher_targets_sha256": sha256(target_path),
            "inference_scope": "unique_consumed_flammable_ids_only",
            "teacher_signal_consumed_categories": [FLAMMABLE],
            "neutral_score_for_unconsumed_categories": 0.0,
            "inference_occurrences": sum(
                str(row["category"]) == FLAMMABLE for row in train_rows
            ),
            "inference_unique_ids": len(inference_rows),
            "inference_ids_sha256": inference_ids_sha256,
            "deduplicated_by_id": True,
            "batch_size_requested": args.batch_size,
            "batch_size_final": final_batch_size,
            "resumed_from_checkpoint": resumed,
            "runtime_minutes": (time.monotonic() - started) / 60,
        }
        (staging / "teacher_target_contract.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    resume_path.unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
