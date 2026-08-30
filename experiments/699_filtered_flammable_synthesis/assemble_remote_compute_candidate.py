from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


EXPECTED_FOLDS = set(range(5))
ARTIFACTS = ("predictions.jsonl", "adapter.zip")
COMMON_FIELDS = (
    "experiment_id",
    "architecture",
    "model_id",
    "model_revision",
    "source",
    "mode",
    "cap",
    "image_policy",
    "seed",
    "epochs",
    "learning_rate",
    "micro_batch",
    "gradient_accumulation",
    "effective_batch",
    "loss_contract",
    "threshold",
    "threshold_tuned",
    "packages",
)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_fold_source(raw: str) -> tuple[int, Path]:
    fold_text, separator, path_text = raw.partition("=")
    if not separator:
        raise ValueError("fold source must be FOLD=PATH")
    fold = int(fold_text)
    if fold not in EXPECTED_FOLDS:
        raise ValueError(f"invalid fold: {fold}")
    return fold, Path(path_text)


def verify_source(fold: int, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"fold{fold} source must be a regular directory")
    contract_path = root / "output_contract.json"
    if not contract_path.is_file() or contract_path.is_symlink():
        raise ValueError(f"fold{fold} contract missing or symlinked")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    payload = dict(contract)
    declared = payload.pop("contract_sha256", None)
    if declared != canonical_sha256(payload):
        raise ValueError(f"fold{fold} contract self-hash mismatch")
    if (
        contract.get("fold") != fold
        or contract.get("experiment_id") != "699"
        or contract.get("decision") != "GO_EVALUATE"
        or contract.get("validation_labels_read") != 0
        or contract.get("sealed_rows_used") != 0
        or contract.get("public_rows_used") != 0
    ):
        raise ValueError(f"fold{fold} contract gate mismatch")
    if str(contract.get("mode", "")).endswith("_append"):
        expected_steps = math.ceil(
            int(contract.get("train_occurrences", -1))
            / int(contract.get("effective_batch", -1))
        ) * int(contract.get("epochs", 1))
        if (
            contract.get("augmentation_arm") != "synth_append"
            or int(contract.get("synthetic_occurrences", -1)) not in {5, 10, 19, 40, 160}
            or int(contract.get("optimizer_steps", -1)) != expected_steps
        ):
            raise ValueError(f"fold{fold} append contract gate mismatch")
    elif contract.get("optimizer_steps") != 306:
        raise ValueError(f"fold{fold} optimizer-step gate mismatch")
    files = {name: root / name for name in ARTIFACTS}
    if any(not path.is_file() or path.is_symlink() for path in files.values()):
        raise ValueError(f"fold{fold} artifact missing or symlinked")
    hashes = {name: sha256_file(path) for name, path in files.items()}
    if contract.get("artifacts") != hashes:
        raise ValueError(f"fold{fold} artifact checksum mismatch")
    predictions = [
        json.loads(line)
        for line in files["predictions.jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    if (
        len(predictions) != contract.get("validation_rows")
        or any("label" in row for row in predictions)
        or any(not math.isfinite(float(row["score"])) for row in predictions)
    ):
        raise ValueError(f"fold{fold} prediction contract mismatch")
    audit = {
        "contract_file_sha256": sha256_file(contract_path),
        "contract_sha256": declared,
        "predictions_sha256": hashes["predictions.jsonl"],
        "adapter_sha256": hashes["adapter.zip"],
        "rows": len(predictions),
        "runtime_minutes": contract["runtime_minutes"],
        "peak_gpu_bytes": contract["peak_gpu_bytes"],
    }
    return contract, audit


def assemble(sources: dict[int, Path], destination: Path) -> dict[str, Any]:
    if set(sources) != EXPECTED_FOLDS:
        raise ValueError("exact folds0..4 are required")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty destination")
    contracts: dict[int, dict[str, Any]] = {}
    audits: dict[str, dict[str, Any]] = {}
    for fold in sorted(sources):
        contract, audit = verify_source(fold, sources[fold])
        contracts[fold] = contract
        audits[str(fold)] = audit
    baseline = contracts[0]
    for fold, contract in contracts.items():
        mismatch = [key for key in COMMON_FIELDS if contract.get(key) != baseline.get(key)]
        if mismatch:
            raise ValueError(f"fold{fold} common contract mismatch: {mismatch}")
    destination.mkdir(parents=True, exist_ok=True)
    for fold, source in sorted(sources.items()):
        target = destination / f"fold{fold}"
        target.mkdir()
        for name in (*ARTIFACTS, "output_contract.json"):
            os.link(source / name, target / name)
    manifest: dict[str, Any] = {
        "schema": "exp699_remote_compute_full5_candidate_v1",
        "experiment_id": "699",
        "folds": sorted(sources),
        "common_contract": {key: baseline[key] for key in COMMON_FIELDS},
        "optimizer_steps_by_fold": {
            str(fold): int(contract["optimizer_steps"])
            for fold, contract in sorted(contracts.items())
        },
        "artifacts": audits,
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_rows_used": 0,
        "decision": "GO_FULL5_EVALUATE",
    }
    manifest["self_sha256"] = canonical_sha256(manifest)
    (destination / "candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-source", action="append", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    pairs = [parse_fold_source(raw) for raw in args.fold_source]
    sources = dict(pairs)
    if len(sources) != len(pairs):
        raise ValueError("duplicate fold source")
    print(json.dumps(assemble(sources, args.destination), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
