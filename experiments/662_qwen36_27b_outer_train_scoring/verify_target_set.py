from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

FOLDS = tuple(range(5))
EXPECTED_ROWS = {0: 4892, 1: 4894, 2: 4892, 3: 4892, 4: 4894}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifact_verifier():
    path = Path(__file__).with_name("verify_artifact.py")
    spec = importlib.util.spec_from_file_location("verify_artifact_662", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load artifact verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_target_set(
    fold_dirs: dict[int, Path],
    runtime_root: Path,
    *,
    expected_rows: dict[int, int] | None = None,
) -> dict[str, Any]:
    expected_rows = EXPECTED_ROWS if expected_rows is None else expected_rows
    if set(fold_dirs) != set(FOLDS):
        raise ValueError("exactly folds 0..4 are required")
    if set(expected_rows) != set(FOLDS):
        raise ValueError("expected row contract must cover folds 0..4")
    resolved_dirs = [path.resolve() for path in fold_dirs.values()]
    if len(set(resolved_dirs)) != len(FOLDS):
        raise ValueError("fold artifact directories must be distinct")

    verifier = load_artifact_verifier()
    rows_by_fold: dict[str, int] = {}
    runtime_contracts: dict[str, str] = {}
    source_train_hashes: dict[str, str] = {}
    adapter_manifest_hashes: dict[str, str] = {}
    artifact_hashes: dict[str, str] = {}
    score_hashes: dict[str, str] = {}
    acceptance_hashes: dict[str, str] = {}

    for fold in FOLDS:
        artifact_dir = fold_dirs[fold]
        runtime_dir = runtime_root / f"fold{fold}_full"
        archive = artifact_dir / "extracted" / "teacher_outer_train_scores.zip"
        persisted_audit_path = artifact_dir / "acceptance_audit.json"
        if not archive.is_file() or not persisted_audit_path.is_file():
            raise ValueError(f"fold {fold} acceptance files are missing")
        observed = verifier.verify(archive, runtime_dir)
        persisted = json.loads(persisted_audit_path.read_text(encoding="utf-8"))
        if persisted != observed:
            raise ValueError(f"fold {fold} persisted acceptance audit mismatch")
        if observed.get("outer_fold") != fold or observed.get("mode") != "full":
            raise ValueError(f"fold {fold} identity or mode mismatch")
        if observed.get("rows") != expected_rows[fold]:
            raise ValueError(f"fold {fold} row count mismatch")
        if observed.get("validation_rows_read") != 0 or observed.get("validation_labels_read") != 0:
            raise ValueError(f"fold {fold} validation isolation mismatch")

        runtime = json.loads((runtime_dir / "runtime_audit.json").read_text(encoding="utf-8"))
        key = str(fold)
        rows_by_fold[key] = observed["rows"]
        runtime_contracts[key] = observed["runtime_contract_sha256"]
        source_train_hashes[key] = runtime["source_train_sha256"]
        adapter_manifest_hashes[key] = runtime["teacher_adapter_manifest_sha256"]
        artifact_hashes[key] = observed["archive_sha256"]
        score_hashes[key] = observed["teacher_scores_sha256"]
        acceptance_hashes[key] = sha256_file(persisted_audit_path)

    if len(set(runtime_contracts.values())) != len(FOLDS):
        raise ValueError("runtime contracts must be distinct by outer fold")
    if len(set(source_train_hashes.values())) != len(FOLDS):
        raise ValueError("source train payloads must be distinct by outer fold")

    return {
        "schema_version": 1,
        "experiment_id": "662",
        "source_experiment_id": "654",
        "folds": list(FOLDS),
        "rows_by_fold": rows_by_fold,
        "total_occurrences": sum(rows_by_fold.values()),
        "runtime_contract_sha256": runtime_contracts,
        "source_train_sha256": source_train_hashes,
        "teacher_adapter_manifest_sha256": adapter_manifest_hashes,
        "artifact_sha256": artifact_hashes,
        "teacher_scores_sha256": score_hashes,
        "acceptance_audit_sha256": acceptance_hashes,
        "validation_rows_read": 0,
        "validation_labels_read": 0,
        "labels_read": 0,
        "sealed_rows": 0,
        "public_used": False,
        "score_semantics": "raw_last_token_logit_1_minus_logit_0",
        "submission_eligible": False,
        "decision": "ACCEPT_FULL_TARGET_SET",
    }


def parse_fold_dir(value: str) -> tuple[int, Path]:
    fold_text, separator, path_text = value.partition("=")
    if not separator or not fold_text.isdigit():
        raise argparse.ArgumentTypeError("fold dir must use FOLD=PATH")
    fold = int(fold_text)
    if fold not in FOLDS or not path_text:
        raise argparse.ArgumentTypeError("fold dir must use fold 0..4 and a path")
    return fold, Path(path_text)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--fold-dir", action="append", type=parse_fold_dir, required=True)
    result.add_argument("--runtime-root", type=Path, required=True)
    result.add_argument("--output", type=Path)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    fold_dirs = dict(args.fold_dir)
    if len(fold_dirs) != len(args.fold_dir):
        raise ValueError("duplicate fold-dir argument")
    payload = verify_target_set(fold_dirs, args.runtime_root)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite aggregate acceptance output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
