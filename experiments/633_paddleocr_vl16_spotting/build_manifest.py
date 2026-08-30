from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expand_manifest(source: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    with gzip.open(source, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["id", "image_urls"]:
            raise ValueError(f"unexpected manifest columns: {reader.fieldnames}")
        for row in reader:
            item_id = str(row["id"])
            urls = json.loads(row["image_urls"])
            if not isinstance(urls, list) or not urls:
                raise ValueError(f"empty or invalid image list for id={item_id}")
            for image_index, url in enumerate(urls):
                key = (item_id, image_index)
                if key in seen:
                    raise ValueError(f"duplicate image key: {key}")
                if not isinstance(url, str) or not url.startswith("https://"):
                    raise ValueError(f"invalid image URL for id={item_id} image={image_index}")
                seen.add(key)
                output.append({"id": item_id, "image_index": image_index, "url": url})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite an existing manifest or report")
    rows = expand_manifest(args.source)
    item_count = len({str(row["id"]) for row in rows})
    if item_count != 12971 or len(rows) != 49456:
        raise ValueError(f"unexpected scope: items={item_count}, images={len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": 1,
        "source_sha256": sha256_file(args.source),
        "output_sha256": sha256_file(args.output),
        "items": item_count,
        "images": len(rows),
        "labels_read": 0,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
