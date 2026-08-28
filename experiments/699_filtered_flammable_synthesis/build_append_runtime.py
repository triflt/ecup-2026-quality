from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any

from build_synth_runtime import (
    FLAMMABLE,
    _select_synthetic,
    _validate_parent,
    canonical_sha256,
    sha256_file,
    write_jsonl,
)


def _text_only_extra(
    row: dict[str, Any],
    *,
    index: int,
    source: str,
    repeat_index: int,
    synthetic_repeat: int,
) -> dict[str, Any]:
    candidate_id = str(row["candidate_id"])
    name = str(row["name"])
    description = str(row["description"])
    semantic_component = hashlib.sha256(
        f"exp699:append:{source}:{candidate_id}".encode()
    ).hexdigest()
    row_id = f"synth-append-{source}-{candidate_id}-r{repeat_index}"
    result = {
        "category": FLAMMABLE,
        "description": description,
        "fold": -1,
        "global_index": -(index + 1),
        "id": row_id,
        "label": int(row["label"]),
        "name": name,
        "occurrence_index": repeat_index,
        "ocr_images": [],
        "semantic_component": semantic_component,
        # Both arms are deliberately text-only for exact modality parity.
        "synthetic": True,
        "augmentation_arm": "synth_append",
        "image_policy": "text_only",
    }
    result["synthetic_candidate_id"] = candidate_id
    result["synthetic_source"] = source
    result["synthetic_repeat_index"] = repeat_index
    result["synthetic_repeat"] = synthetic_repeat
    return result


def build_append_runtime(
    *,
    parent_dir: Path,
    ranked_path: Path,
    output_dir: Path,
    fold: int,
    source: str,
    cap: int,
    mode: str,
    epochs: int = 1,
    synthetic_repeat: int = 1,
) -> dict[str, Any]:
    allowed = {
        ("v2", "positive_only", 5, 1),
        ("v2", "positive_only", 10, 1),
        ("v2", "positive_only", 10, 4),
        ("v2", "positive_only", 19, 1),
        ("v2", "balanced", 10, 1),
        ("v1", "positive_only", 40, 1),
        ("both", "balanced", 80, 1),
    }
    if (source, mode, cap, synthetic_repeat) not in allowed:
        raise ValueError("variant is outside the frozen append screen")
    if isinstance(epochs, bool) or epochs not in {1, 2}:
        raise ValueError("epochs must be 1 or 2")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty runtime directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, parent_audit = _validate_parent(parent_dir, fold)
    selected, ranked_sha = _select_synthetic(
        ranked_path,
        fold=fold,
        source=source,
        mode=mode,
        cap=cap,
    )
    extras = []
    for candidate_index, row in enumerate(selected):
        for repeat_index in range(synthetic_repeat):
            extras.append(
                _text_only_extra(
                    row,
                    index=candidate_index * synthetic_repeat + repeat_index,
                    source=source,
                    repeat_index=repeat_index,
                    synthetic_repeat=synthetic_repeat,
                )
            )
    output_train = [dict(row) for row in train] + extras
    if output_train[: len(train)] != train:
        raise RuntimeError("original real training multiset changed")
    if len(output_train) != len(train) + len(extras):
        raise RuntimeError("append occurrence count mismatch")
    before_bad = Counter(
        (str(row["id"]), int(row["label"]))
        for row in train
        if row["category"] == "БАД"
    )
    after_bad = Counter(
        (str(row["id"]), int(row["label"]))
        for row in output_train
        if row["category"] == "БАД"
    )
    if before_bad != after_bad:
        raise RuntimeError("BAD training rows changed")
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    write_jsonl(train_path, output_train)
    write_jsonl(validation_path, validation)
    effective_batch = 16
    audit: dict[str, Any] = {
        "schema_version": 2,
        "experiment_id": "699",
        "stage": "paired_append_screen",
        "outer_fold": fold,
        "parent_experiment_id": "641",
        "parent_runtime_contract_sha256": parent_audit["contract_sha256"],
        "ranked_manifest_sha256": ranked_sha,
        "source": source,
        "mode": f"{mode}_append",
        "cap": cap,
        "augmentation_arm": "synth_append",
        "image_policy": "all_real_unchanged_plus_text_only_extra",
        "original_real_occurrences": len(train),
        "train_occurrences": len(output_train),
        "validation_rows": len(validation),
        "synthetic_unique": len(selected),
        "synthetic_repeat": synthetic_repeat,
        "synthetic_occurrences": len(extras),
        "appended_occurrences": len(extras),
        "appended_positive_occurrences": sum(int(row["label"]) == 1 for row in extras),
        "appended_negative_occurrences": sum(int(row["label"]) == 0 for row in extras),
        "removed_real_occurrences": 0,
        "original_real_rows_changed": 0,
        "bad_rows_changed": 0,
        "validation_rows_changed": 0,
        "validation_labels_written": 0,
        "sealed_rows_written": 0,
        "public_rows_used": 0,
        "effective_batch": effective_batch,
        "epochs": epochs,
        "expected_optimizer_steps": (
            math.ceil(len(output_train) / effective_batch) * epochs
        ),
        "selected_synth_ids_sha256": canonical_sha256(
            [str(row["candidate_id"]) for row in selected]
        ),
        "output_sha256": {
            "train.jsonl": sha256_file(train_path),
            "validation.jsonl": sha256_file(validation_path),
        },
        "decision": "GO_GPU_APPEND_SCREEN",
    }
    audit["contract_sha256"] = canonical_sha256(audit)
    (output_dir / "runtime_audit.json").write_text(
        __import__("json").dumps(audit, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--ranked", dest="ranked_path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--source", choices=("v1", "v2", "both"), required=True)
    parser.add_argument("--mode", choices=("positive_only", "balanced"), required=True)
    parser.add_argument("--cap", type=int, choices=(5, 10, 19, 40, 80), required=True)
    parser.add_argument("--epochs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--synthetic-repeat", type=int, choices=(1, 4), default=1)
    args = parser.parse_args()
    print(
        __import__("json").dumps(
            build_append_runtime(**vars(args)),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
