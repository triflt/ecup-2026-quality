from __future__ import annotations

import csv
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


submission = Path("/work/submission")
runtime = Path("/work/runtime")
output = Path("/work/output")
runtime.mkdir(parents=True, exist_ok=True)
output.mkdir(parents=True, exist_ok=True)
shutil.unpack_archive("/work/qwen3vl-lora-artifacts/adapter.zip", submission / "adapter_qwen3vl", "zip")
shutil.unpack_archive("/work/qwen35-lora-artifacts/adapter.zip", submission / "adapter_qwen35", "zip")
shutil.unpack_archive("/work/peft-artifacts/peft-0.20.0.zip", submission / "vendor", "zip")

data_path = runtime / "all.csv"
urllib.request.urlretrieve(os.environ["DATA_URL"], data_path)
frame = pd.read_csv(data_path)
sample = pd.concat([
    group.sample(n=min(300, len(group)), random_state=20260821)
    for _, group in frame.groupby("category", sort=True)
], ignore_index=True).sort_values("id").reset_index(drop=True)
test_path = runtime / "test.csv"
sample[["id", "name", "description", "category"]].to_csv(test_path, index=False)

selected = set(sample.id.astype(str))
items = []
with gzip.open("/work/input/multi_image_manifest.tsv.gz", "rt", encoding="utf-8", newline="") as stream:
    for row in csv.DictReader(stream, delimiter="\t"):
        item_id = str(row["id"])
        if item_id in selected:
            for image_index, url in enumerate(json.loads(row["image_urls"])):
                items.append((item_id, image_index, url))


def download(item):
    item_id, image_index, url = item
    folder = runtime / "images" / item_id
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{image_index}.jpg"
    with urllib.request.urlopen(url, timeout=60) as response:
        destination.write_bytes(response.read())
    return destination


with ThreadPoolExecutor(max_workers=32) as pool:
    futures = [pool.submit(download, item) for item in items]
    for index, future in enumerate(as_completed(futures), 1):
        future.result()
        if index % 500 == 0 or index == len(futures):
            print(f"downloaded={index}/{len(futures)}", flush=True)

started = time.monotonic()
subprocess.run(
    [
        sys.executable,
        "-u",
        str(submission / "run.py"),
        "--test_data_path",
        str(test_path),
        "--output_path",
        str(output / "submission.csv"),
    ],
    cwd=submission,
    check=True,
)
elapsed = time.monotonic() - started
result = pd.read_csv(output / "submission.csv")
pattern = re.compile(r"^<комментарий>(.{50,300})<вердикт>(бан|не бан)$", re.S)
if len(result) != len(sample) or result.id.astype(str).nunique() != len(sample):
    raise ValueError("row/id mismatch")
if not result.result.map(lambda value: bool(pattern.fullmatch(str(value)))).all():
    raise ValueError("output regex mismatch")
report = {
    "rows": len(sample),
    "images": len(items),
    "inference_seconds": elapsed,
    "seconds_per_row": elapsed / len(sample),
    "projected_public_5400_minutes": elapsed / len(sample) * 5400 / 60,
    "projected_private_3800_minutes": elapsed / len(sample) * 3800 / 60,
    "schema_valid": True,
}
(output / "runtime_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2), flush=True)
