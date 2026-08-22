from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def normalize(value: object) -> str:
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    value = unicodedata.normalize("NFKC", value).lower().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def group_hash(name: str, description: str) -> str:
    payload = f"{normalize(name)}\n{normalize(name)}\n{normalize(description)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--oof-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("validation/grouped_text_v1/folds.csv"))
    parser.add_argument("--basket-output", type=Path, default=Path("validation/grouped_text_v1/basket.csv"))
    parser.add_argument("--manifest-output", type=Path, default=Path("validation/grouped_text_v1/manifest.json"))
    parser.add_argument("--basket-fold", type=int, default=4)
    parser.add_argument("--dataset-version", default="competition_train_v1")
    parser.add_argument("--evaluation-version", default="grouped_text_v1")
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing immutable version.")
    args = parser.parse_args()

    targets = (args.output, args.basket_output, args.manifest_output)
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.force:
        raise SystemExit(
            "refusing to overwrite versioned validation files; choose a new version/path "
            f"or pass --force explicitly: {existing}"
        )

    with args.oof_csv.open(encoding="utf-8", newline="") as stream:
        fold_by_id = {str(row["id"]): int(row["fold"]) for row in csv.DictReader(stream)}
    with args.data.open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    source_ids = [str(row["id"]) for row in source_rows]
    if set(source_ids) != set(fold_by_id) or len(source_ids) != len(fold_by_id):
        raise ValueError("OOF ids do not match source data ids")

    columns = ["id", "category", "label", "fold", "group_hash"]
    output_rows = []
    for row in source_rows:
        output_rows.append({
            "id": str(row["id"]),
            "category": row["category"],
            "label": int(row["label"]),
            "fold": fold_by_id[str(row["id"])],
            "group_hash": group_hash(row.get("name", ""), row.get("description", "")),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output_rows)
    basket_rows = [row for row in output_rows if row["fold"] == args.basket_fold]
    with args.basket_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(basket_rows)

    fold_rows = Counter(row["fold"] for row in output_rows)
    fold_positives = Counter(row["fold"] for row in output_rows if row["label"] == 1)
    manifest = {
        "evaluation_version": args.evaluation_version,
        "dataset_version": args.dataset_version,
        "protocol": "frozen category-specific StratifiedGroupKFold by normalized full text",
        "n_splits": 5,
        "random_state": 42,
        "fold_source": "historical OOF cache",
        "data_rows": len(output_rows),
        "basket_fold": args.basket_fold,
        "basket_rows": len(basket_rows),
        "fold_rows": {str(key): fold_rows[key] for key in sorted(fold_rows)},
        "fold_positives": {str(key): fold_positives[key] for key in sorted(fold_rows)},
        "source_data_sha256": sha256(args.data),
        "folds_sha256": sha256(args.output),
        "basket_sha256": sha256(args.basket_output),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
