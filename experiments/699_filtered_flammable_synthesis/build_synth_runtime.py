from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

FLAMMABLE = "Легковоспламеняющиеся"


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path.name}:{line_number}") from error
            if not isinstance(row, dict):
                raise TypeError(f"non-object at {path.name}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _validate_parent(parent_dir: Path, fold: int) -> tuple[list[dict], list[dict], dict]:
    train_path = parent_dir / "train.jsonl"
    validation_path = parent_dir / "validation.jsonl"
    audit_path = parent_dir / "runtime_audit.json"
    for path in (train_path, validation_path, audit_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"parent runtime member missing: {path.name}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    payload = dict(audit)
    declared = payload.pop("contract_sha256", None)
    if declared != canonical_sha256(payload):
        raise ValueError("parent runtime audit self-hash mismatch")
    if int(audit.get("outer_fold", -1)) != fold:
        raise ValueError("parent runtime fold mismatch")
    expected = {
        "train.jsonl": sha256_file(train_path),
        "validation.jsonl": sha256_file(validation_path),
    }
    if audit.get("output_sha256") != expected:
        raise ValueError("parent runtime payload checksum mismatch")
    train = read_jsonl(train_path)
    validation = read_jsonl(validation_path)
    expected_train = int(audit.get("train_occurrences", -1))
    expected_validation = int(audit.get("validation_rows", -1))
    if expected_train not in {4892, 4894} or len(train) != expected_train:
        raise ValueError(f"unexpected parent train occurrences: {len(train)}")
    if expected_validation not in {2223, 2224} or len(validation) != expected_validation:
        raise ValueError(f"unexpected parent validation rows: {len(validation)}")
    if any("label" in row for row in validation):
        raise ValueError("parent validation contains labels")
    return train, validation, audit


def _select_synthetic(
    ranked_path: Path, *, fold: int, source: str, mode: str, cap: int
) -> tuple[list[dict], str]:
    rows = read_jsonl(ranked_path)
    if not rows or any(int(row.get("selector_fold", -1)) != fold for row in rows):
        raise ValueError("ranked synthetic manifest fold mismatch")
    if mode not in {"positive_only", "balanced"}:
        raise ValueError("mode must be positive_only or balanced")
    limits = {0: cap if mode == "balanced" else 0, 1: cap}
    selected = []
    for label in (0, 1):
        eligible = sorted(
            (
                row
                for row in rows
                if (source == "both" or row.get("source") == source)
                and row.get("category") == FLAMMABLE
                and type(row.get("label")) is int
                and int(row["label"]) == label
            ),
            key=lambda row: (int(row["label_rank"]), str(row["candidate_id"])),
        )[: limits[label]]
        if len(eligible) != limits[label]:
            raise ValueError(
                f"insufficient candidates for fold={fold} source={source} "
                f"label={label}: {len(eligible)} < {limits[label]}"
            )
        selected.extend(eligible)
    candidate_ids = [str(row["candidate_id"]) for row in selected]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("duplicate selected synthetic candidate")
    return selected, sha256_file(ranked_path)


def _replacement_positions(train: list[dict], *, label: int, count: int) -> list[int]:
    if count == 0:
        return []
    eligible = [
        index
        for index, row in enumerate(train)
        if row.get("category") == FLAMMABLE and int(row.get("label", -1)) == label
    ]
    # Remove at most one occurrence per real item before taking a second one.
    # This keeps the replacement dispersed across semantic families.
    eligible.sort(
        key=lambda index: (
            int(train[index].get("occurrence_index", 0)),
            str(train[index].get("semantic_component", "")),
            str(train[index]["id"]),
            index,
        )
    )
    chosen = []
    seen_ids: set[str] = set()
    for index in eligible:
        row_id = str(train[index]["id"])
        if row_id in seen_ids:
            continue
        chosen.append(index)
        seen_ids.add(row_id)
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError("not enough real flammable occurrences to replace")
    if len({str(train[index]["id"]) for index in chosen}) != count:
        raise ValueError("replacement unexpectedly removes multiple occurrences of one item")
    return chosen


def build_runtime(
    *,
    parent_dir: Path,
    ranked_path: Path,
    output_dir: Path,
    fold: int,
    source: str,
    mode: str,
    cap: int,
) -> dict[str, Any]:
    allowed = {
        ("v1", "positive_only", 40),
        ("v2", "positive_only", 19),
        ("both", "positive_only", 20),
        ("both", "balanced", 80),
    }
    if (source, mode, cap) not in allowed:
        raise ValueError("variant is outside the frozen first GPU wave")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty runtime directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, parent_audit = _validate_parent(parent_dir, fold)
    selected, ranked_sha = _select_synthetic(
        ranked_path, fold=fold, source=source, mode=mode, cap=cap
    )
    selected_by_label = {
        label: [row for row in selected if int(row["label"]) == label]
        for label in (0, 1)
    }
    positions_by_label = {
        label: _replacement_positions(
            train, label=label, count=len(selected_by_label[label])
        )
        for label in (0, 1)
    }
    replacements = sorted(
        (
            (position, candidate)
            for label in (0, 1)
            for position, candidate in zip(
                positions_by_label[label], selected_by_label[label], strict=True
            )
        ),
        key=lambda item: item[0],
    )
    output_train = [dict(row) for row in train]
    replaced_ids = []
    for replacement_index, (position, candidate) in enumerate(
        replacements
    ):
        original = train[position]
        replaced_ids.append(str(original["id"]))
        candidate_id = str(candidate["candidate_id"])
        output_train[position] = {
            "category": FLAMMABLE,
            "description": str(candidate["description"]),
            "fold": -1,
            "global_index": -(replacement_index + 1),
            "id": f"synth-{source}-{candidate_id}",
            "label": int(candidate["label"]),
            "name": str(candidate["name"]),
            "occurrence_index": 0,
            "ocr_images": [],
            "semantic_component": hashlib.sha256(
                f"exp699:{source}:{candidate_id}".encode()
            ).hexdigest(),
            "synthetic": True,
            "synthetic_candidate_id": candidate_id,
            "synthetic_source": source,
            "image_policy": "text_only",
        }
    if len(output_train) != len(train):
        raise RuntimeError("training occurrence count changed")
    before = Counter((row["category"], int(row["label"])) for row in train)
    after = Counter((row["category"], int(row["label"])) for row in output_train)
    if before != after:
        raise RuntimeError("category/label support changed")
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    write_jsonl(train_path, output_train)
    write_jsonl(validation_path, validation)
    audit: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "699",
        "outer_fold": fold,
        "parent_experiment_id": "641",
        "parent_runtime_contract_sha256": parent_audit["contract_sha256"],
        "ranked_manifest_sha256": ranked_sha,
        "source": source,
        "mode": mode,
        "cap": cap,
        "image_policy": "text_only",
        "train_occurrences": len(output_train),
        "validation_rows": len(validation),
        "synthetic_occurrences": len(selected),
        "synthetic_unique": len(selected),
        "synthetic_by_label": {
            str(label): len(selected_by_label[label]) for label in (0, 1)
        },
        "replaced_real_occurrences": len(selected),
        "replaced_real_unique": len(set(replaced_ids)),
        "bad_rows_changed": 0,
        "validation_rows_changed": 0,
        "validation_labels_written": 0,
        "sealed_rows_written": 0,
        "public_rows_used": 0,
        "output_sha256": {
            "train.jsonl": sha256_file(train_path),
            "validation.jsonl": sha256_file(validation_path),
        },
        "decision": "GO_GPU_SCREEN",
    }
    audit["contract_sha256"] = canonical_sha256(audit)
    (output_dir / "runtime_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--ranked", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--source", choices=("v1", "v2", "both"), required=True)
    parser.add_argument("--mode", choices=("positive_only", "balanced"), required=True)
    parser.add_argument("--cap", type=int, required=True)
    args = parser.parse_args()
    result = build_runtime(
        parent_dir=args.parent_dir,
        ranked_path=args.ranked,
        output_dir=args.output_dir,
        fold=args.fold,
        source=args.source,
        mode=args.mode,
        cap=args.cap,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
