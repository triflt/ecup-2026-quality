from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


MODEL = Path(os.environ.get("MODEL_PATH", "/hf_models"))
INPUT = Path("/work/input/attribute_probe_manifest.csv.gz")
IMAGE_DIR = Path("/work/images")
OUTPUT = Path("/work/output")
VENDOR = Path("/work/vendor/internvl")
IMAGE_SIZE = 448
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
DESCRIPTION_LIMIT = int(os.environ.get("DESCRIPTION_LIMIT", "1400"))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CONTIGUOUS_PATTERN = re.compile(r"(?<![01])([01]{6})(?![01])")
SEPARATED_PATTERN = re.compile(r"(?<![01])([01](?:[\s,;|]+[01]){5})(?![01])")


def extract_digits(response: str) -> tuple[str, str]:
    contiguous = CONTIGUOUS_PATTERN.search(response)
    if contiguous:
        return contiguous.group(1), "contiguous"
    separated = SEPARATED_PATTERN.search(response)
    if separated:
        return "".join(re.findall(r"[01]", separated.group(1))), "separated"
    return "", "invalid"


def install_dependencies() -> None:
    VENDOR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(VENDOR),
            "--no-cache-dir",
            "--no-deps",
            "einops==0.8.1",
            "timm==1.0.22",
        ],
        check=True,
    )
    sys.path.insert(0, str(VENDOR))


def compact(value: object, limit: int) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(value) <= limit:
        return value
    head = int(limit * 0.7)
    return value[:head].rstrip() + " … " + value[-(limit - head) :].lstrip()


def download(row: object) -> tuple[str, bool, str]:
    destination = IMAGE_DIR / f"{row.id}.jpg"
    last_error = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(row.image_url, timeout=60) as response:
                payload = response.read()
            image = Image.open(io.BytesIO(payload)).convert("RGB")
            image.save(destination, format="JPEG", quality=94)
            return str(row.id), True, ""
        except Exception as error:
            last_error = error
    Image.new("RGB", (64, 64), "white").save(destination, format="JPEG")
    return str(row.id), False, str(last_error)


def prepare_images(frame: pd.DataFrame) -> dict[str, str]:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(download, row) for row in frame.itertuples(index=False)]
        for index, future in enumerate(as_completed(futures), 1):
            item_id, ok, error = future.result()
            if not ok:
                failures[item_id] = error
            if index % 200 == 0 or index == len(futures):
                print(
                    json.dumps(
                        {"downloaded": index, "total": len(futures), "failures": len(failures)}
                    ),
                    flush=True,
                )
    return failures


TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def pixel_values(rows: list[object]) -> torch.Tensor:
    values = []
    for row in rows:
        with Image.open(IMAGE_DIR / f"{row.id}.jpg") as image:
            values.append(TRANSFORM(image.convert("RGB")))
    return torch.stack(values).to(device="cuda", dtype=torch.bfloat16)


def question(row: object) -> str:
    return (
        "<image>\n"
        "Товар проверяется для категории «Легковоспламеняющиеся».\n"
        f"Название: {compact(row.name, 320)}\n"
        f"Описание: {compact(row.description, DESCRIPTION_LIMIT)}\n"
        "Ответ начни со слова КОД и верни после него ровно шесть цифр 0 или 1 "
        "без пробелов и пояснений, например: КОД 010100. Порядок:\n"
        "1) продаётся горючее топливо или газ; "
        "2) это пустое оборудование или пустая ёмкость; "
        "3) продаётся самостоятельный источник огня; "
        "4) топливо или газ явно входит в комплект; "
        "5) продаётся самостоятельный горючий материал: свеча, уголь или растопка; "
        "6) явно сказано, что топлива/газа нет или оно не входит в комплект.\n"
        "Если признак подтверждён текстом или изображением — 1, иначе 0."
    )


def main() -> None:
    install_dependencies()
    frame = pd.read_csv(INPUT, compression="gzip", dtype={"id": str})
    if frame.id.duplicated().any():
        raise ValueError("duplicate ids in attribute manifest")
    failures = prepare_images(frame)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    model = AutoModel.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
        use_flash_attn=False,
    ).eval().to("cuda")
    # The first run showed that InternVL may emit one digit and EOS even when the
    # prompt requests six. Requiring six generated tokens changes only response
    # serialization, not the selected rows or the semantic questions.
    generation = {"min_new_tokens": 6, "max_new_tokens": 20, "do_sample": False}
    records = []
    started = time.monotonic()
    for start in range(0, len(frame), BATCH_SIZE):
        rows = list(frame.iloc[start : start + BATCH_SIZE].itertuples(index=False))
        responses = model.batch_chat(
            tokenizer,
            pixel_values(rows),
            [question(row) for row in rows],
            generation,
            num_patches_list=[1] * len(rows),
        )
        for row, response in zip(rows, responses):
            digits, parse_mode = extract_digits(response)
            record = {
                "id": str(row.id),
                "fold": int(row.fold),
                "valid": bool(digits),
                "image_download_ok": str(row.id) not in failures,
                "digits": digits,
                "parse_mode": parse_mode,
                "raw_response": response,
            }
            for index, name in enumerate(
                [
                    "fuel_or_gas",
                    "empty_equipment",
                    "ignition_source",
                    "fuel_included",
                    "combustible_material",
                    "absence_or_negation",
                ]
            ):
                record[name] = int(digits[index]) if digits else -1
            records.append(record)
        if len(records) % 80 < len(rows) or len(records) == len(frame):
            print(
                json.dumps(
                    {
                        "predicted": len(records),
                        "total": len(frame),
                        "valid": sum(record["valid"] for record in records),
                        "elapsed_min": (time.monotonic() - started) / 60,
                        "max_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    }
                ),
                flush=True,
            )
    result = pd.DataFrame(records)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT / "attribute_predictions.csv", index=False)
    report = {
        "model": "OpenGVLab/InternVL3_5-2B",
        "format": "native GitHub-format AutoModel/batch_chat",
        "rows": int(len(result)),
        "valid_rows": int(result.valid.sum()),
        "valid_rate": float(result.valid.mean()),
        "download_failures": int(len(failures)),
        "batch_size": BATCH_SIZE,
        "image_tiles": 1,
        "runtime_minutes": (time.monotonic() - started) / 60,
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }
    (OUTPUT / "inference_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
