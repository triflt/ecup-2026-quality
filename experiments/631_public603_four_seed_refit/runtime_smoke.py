from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

OUTPUT_RE = re.compile(r"^<комментарий>(.{50,300})<вердикт>(бан|не бан)$", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True, type=Path)
    parser.add_argument("--data-parts", required=True, type=Path)
    parser.add_argument("--image-manifest", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rows-per-category", default=300, type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts = sorted(args.data_parts.glob("data.csv.gz.part-*"))
    frame = pd.read_csv(io.BytesIO(gzip.decompress(b"".join(p.read_bytes() for p in parts))))
    sample = pd.concat(
        [
            group.sample(n=min(args.rows_per_category, len(group)), random_state=20260821)
            for _, group in frame.groupby("category", sort=True)
        ],
        ignore_index=True,
    ).sort_values("id").reset_index(drop=True)
    test_path = args.runtime_dir / "test.csv"
    sample[["id", "name", "description", "category"]].to_csv(test_path, index=False)
    selected = set(sample["id"].astype(str))
    items: list[tuple[str, int, str]] = []
    with gzip.open(args.image_manifest, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if str(row["id"]) in selected:
                for index, url in enumerate(json.loads(row["image_urls"])):
                    items.append((str(row["id"]), index, url))

    def download(item: tuple[str, int, str]) -> None:
        item_id, index, url = item
        directory = args.runtime_dir / "images" / item_id
        directory.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            (directory / f"{index}.jpg").write_bytes(response.read())

    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(download, items))
    started = time.monotonic()
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(args.submission / "run.py"),
            "--test_data_path",
            str(test_path),
            "--output_path",
            str(args.output_dir / "submission.csv"),
        ],
        cwd=args.submission,
        check=True,
    )
    elapsed = time.monotonic() - started
    result = pd.read_csv(args.output_dir / "submission.csv", dtype={"id": str})
    schema_valid = (
        len(result) == len(sample)
        and result["id"].nunique() == len(sample)
        and result["result"].map(lambda value: bool(OUTPUT_RE.fullmatch(str(value)))).all()
    )
    report = {
        "rows": len(sample),
        "images": len(items),
        "qwen35_passes": 4,
        "inference_seconds": elapsed,
        "seconds_per_row": elapsed / len(sample),
        "projected_public_minutes": elapsed / len(sample) * 1600 / 60,
        "projected_private_minutes": elapsed / len(sample) * 3800 / 60,
        "schema_valid": bool(schema_valid),
    }
    (args.output_dir / "runtime_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0 if schema_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
