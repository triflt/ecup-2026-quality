from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

FLAMMABLE = "Легковоспламеняющиеся"
EXPECTED = {
    0: {
        "source_contract": "38802115365cef7e3a0c1a82abc5efc5ce92a41e0046f1ddad4ab9e02647c568",
        "train": "c157aaf80fd287f1153b322adb506af94bc8d2d1b534286c47bc6fb15ed3493e",
        "validation": "109807781797f51ec8c1d98d0385aef2e9edd0d89183c4a44aab76e2a26f56bc",
        "multiset": "646b18ad1c6ef1cae3e2272ff7bc45d45229f44618830d722476b82795d57766",
    },
    3: {
        "source_contract": "e08a51c2db16163953c45841f3dd1e7b30b293a7265b2bbd8084d1479c20ea36",
        "train": "05e8ca8f84379fb9c43d153c8746a0c532d17b140f99c7f1901d3f5aa0b3abca",
        "validation": "bbbbae3f4ddb38bf3af238cd53dd8fc1b851b04598101ab7b57809e73c7a2257",
        "multiset": "8252ba8a8f2eb6179647612256a5ac768c543242d0ba1cc7df1b55554d9783cf",
    },
    1: {
        "source_contract": "3d62eed9817bbbdb0ff4d55dd511904b4fa1dfef5db44deddb104dda0c60f576",
        "train": "62d3cdac1a6d798a2917312f03f49ab4a399be8e9b867499a2ea41525ead9373",
        "validation": "f6111b8977957e93469c033980853512dc865bfeebc6a9256b7fb082338895cc",
        "multiset": "88d563943c964e380a2900a89fb9d50e24339085906df33307544ab421cf656d",
    },
    2: {
        "source_contract": "321b5e5165110fc729598956d121208ab14aee12af38e8f0b3ab81ddbd52f5e8",
        "train": "683458ebbbb917da4be09598061fcbea0e02b5cc0281ea67a5e6b93341c79eed",
        "validation": "47b59ecd0a0eb50136052f24883ba07a2af75ddae5eb989fcf2a8dc1bf889a44",
        "multiset": "528aa1b692a8a3381313a8464ed97e3f3a4c71b124401f146417111371cedd5c",
    },
    4: {
        "source_contract": "22cad9c7a1510a73b6ec606826839328329a9c0469f2fb00c45fd24ed9dcf33f",
        "train": "c9b1a0edce6efa0972565bddc2b98fd3376d5d7800c5d3649d1a550bf3279b72",
        "validation": "1d1e089269041217aa7f197ce6a79ca6fcd849becc124780fe6511b92bdeca0c",
        "multiset": "8226e018529311ec5773d83c64ba132168f33264d5128ff47516bb5c4e2c93e0",
    },
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(source: Path, output: Path, fold: int) -> dict[str, Any]:
    if fold not in EXPECTED:
        raise ValueError("fold must be one of the five frozen outer folds")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty runtime")
    audit = json.loads((source / "runtime_audit.json").read_text(encoding="utf-8"))
    source_hash = audit.pop("contract_sha256", None)
    if source_hash != canonical_sha256(audit) or source_hash != EXPECTED[fold]["source_contract"]:
        raise ValueError("source runtime contract mismatch")
    if audit.get("experiment_id") != "641" or audit.get("outer_fold") != fold:
        raise ValueError("unexpected source runtime")
    train = [row for row in read_jsonl(source / "train.jsonl") if row["category"] == FLAMMABLE]
    validation = [
        row for row in read_jsonl(source / "validation.jsonl") if row["category"] == FLAMMABLE
    ]
    expected_validation_rows = 944 if fold == 2 else 943
    if len(train) != 2280 or len(validation) != expected_validation_rows:
        raise ValueError("filtered runtime row count mismatch")
    if Counter(int(row["label"]) for row in train) != Counter({0: 1600, 1: 680}):
        raise ValueError("filtered class multiset mismatch")
    if any("label" in row for row in validation):
        raise ValueError("validation supervision is forbidden")
    multiset = canonical_sha256(sorted(Counter(str(row["id"]) for row in train).items()))
    if multiset != EXPECTED[fold]["multiset"]:
        raise ValueError("filtered selector multiset mismatch")
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "validation.jsonl", validation)
    if sha256_file(output / "train.jsonl") != EXPECTED[fold]["train"]:
        raise ValueError("literal filtered train checksum mismatch")
    if sha256_file(output / "validation.jsonl") != EXPECTED[fold]["validation"]:
        raise ValueError("literal filtered validation checksum mismatch")
    view_rows = {str(row["id"]): row for row in [*train, *validation]}
    view = [
        {key: row[key] for key in ("global_index", "id", "fold", "category", "name", "description", "image_url")}
        for row in sorted(view_rows.values(), key=lambda item: int(item["global_index"]))
    ]
    report = dict(audit)
    report.update(
        {
            "experiment_id": "641",
            "derived_experiment_id": "680",
            "changed_factor": "remove_bad_training_occurrences",
            "source_runtime_contract_sha256": source_hash,
            "train_occurrences": len(train),
            "train_unique_ids": len({str(row["id"]) for row in train}),
            "validation_rows": len(validation),
            "model_input_view_sha256": canonical_sha256(view),
            "selected_multiset_sha256": multiset,
            "output_sha256": {
                "train.jsonl": sha256_file(output / "train.jsonl"),
                "validation.jsonl": sha256_file(output / "validation.jsonl"),
            },
            "bad_train_occurrences": 0,
            "flammable_train_occurrences": len(train),
            "decision": "GO",
        }
    )
    report["contract_sha256"] = canonical_sha256(report)
    (output / "runtime_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.fold), ensure_ascii=False, indent=2))
