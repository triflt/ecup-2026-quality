from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import qwen3vl_lora_holdout as base

OUTPUT = Path("/work/output/selector_records.json")


def digest_lines(values: list[str]) -> str:
    payload = ("\n".join(values) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    frame = base.load_training_frame()
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["description"] = frame["description"].fillna("").astype(str)
    frame["category"] = frame["category"].astype(str)
    ids = frame["id"].astype(str).to_numpy()
    oof = np.load(base.OOF, allow_pickle=True)
    if not np.array_equal(ids, oof["ids"].astype(str)):
        raise ValueError("OOF id mismatch")
    result: dict[str, object] = {
        "experiment_id": "440",
        "purpose": "Freeze the exact experiment-110 selector output in the private compute platform runtime",
        "seed": base.SEED,
        "training_mode": base.TRAINING_MODE,
        "stages": {},
    }
    for fold in (0, 3):
        base.HOLDOUT_FOLD = fold
        base.FULL_TRAIN = False
        records = [int(value) for value in base.select_training(frame, oof)]
        record_ids = ids[records].astype(str).tolist()
        result["stages"][f"fold_{fold}"] = {
            "records": len(records),
            "unique_rows": len(set(records)),
            "ordered_id_sha256": digest_lines(record_ids),
            "sorted_id_sha256": digest_lines(sorted(record_ids)),
            "record_indices": records,
        }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: {field: value for field, value in stage.items() if field != "record_indices"}
                for key, stage in result["stages"].items()
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
