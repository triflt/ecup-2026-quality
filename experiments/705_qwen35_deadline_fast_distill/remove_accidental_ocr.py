from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path


OCR_MEMBER = "src/ocr_stage.py"
RUN_MEMBER = "run.py"

ACCIDENTAL_BLOCK = '''    image_paths = []
    for row in frame.itertuples(index=False):
        folder = images_root / normalize_id(row.id)
        paths = []
        if folder.is_dir():
            paths = [
                str(path)
                for path in sorted(folder.iterdir())
                if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
            ]
        image_paths.append(paths)
    frame["image_paths"] = image_paths

    # OCR (PP-OCRv5 mobile) over all product images; matches training-time cache.
    from src.ocr_stage import OcrStage
    ocr_stage = OcrStage()
    items = [
        {"id": normalize_id(row.id), "image_paths": paths}
        for row, paths in zip(frame.itertuples(index=False), image_paths)
    ]
    ocr_map = ocr_stage.run(items)
    ocr_stage.close()
    print(f"ocr_done rows={len(ocr_map)} nonempty={sum(v != '' for v in ocr_map.values())}", flush=True)

    texts = []
    for row in frame.itertuples(index=False):
        ocr = ocr_map.get(normalize_id(row.id), "")[:700]
        texts.append(
            f"Declared category: {row.category}\\n"
            f"Product name: {row.name}\\n"
            f"Description: {row.description}\\n"
            f"Текст с изображениями товара (OCR): {ocr}\\n"
            f"{CATEGORY_HINTS.get(str(row.category), '')}"
        )
    frame["text"] = texts
'''

TRUE_140_BLOCK = '''    image_paths, texts = [], []
    for row in frame.itertuples(index=False):
        folder = images_root / normalize_id(row.id)
        paths = []
        if folder.is_dir():
            paths = [
                str(path)
                for path in sorted(folder.iterdir())
                if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
            ]
        image_paths.append(paths)
        texts.append(
            f"Declared category: {row.category}\\n"
            f"Product name: {row.name}\\n"
            f"Description: {row.description}\\n"
            f"{CATEGORY_HINTS.get(str(row.category), '')}"
        )
    frame["text"] = texts
    frame["image_paths"] = image_paths
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite output")

    with zipfile.ZipFile(args.input) as source:
        if source.testzip() is not None:
            raise ValueError("input ZIP integrity failure")
        names = source.namelist()
        if names.count(RUN_MEMBER) != 1 or names.count(OCR_MEMBER) != 1:
            raise ValueError("expected exactly one run.py and accidental OCR member")
        original_run = source.read(RUN_MEMBER).decode("utf-8")
        if original_run.count(ACCIDENTAL_BLOCK) != 1:
            raise ValueError("run.py does not contain the exact accidental OCR block")
        corrected_run = original_run.replace(ACCIDENTAL_BLOCK, TRUE_140_BLOCK)
        lowered = corrected_run.lower()
        if "ocr_stage" in lowered or "paddleocr" in lowered or "ocr_done" in lowered:
            raise ValueError("corrected run.py still references OCR runtime")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with zipfile.ZipFile(temporary, "w", allowZip64=True) as destination:
                for info in source.infolist():
                    if info.filename == OCR_MEMBER:
                        continue
                    payload = corrected_run.encode("utf-8") if info.filename == RUN_MEMBER else source.read(info)
                    destination.writestr(info, payload)
            temporary.replace(args.output)
        finally:
            if temporary.exists():
                temporary.unlink()

    with zipfile.ZipFile(args.output) as result:
        if result.testzip() is not None:
            raise ValueError("output ZIP integrity failure")
        if OCR_MEMBER in result.namelist():
            raise ValueError("OCR member survived correction")
        if result.read(RUN_MEMBER).decode("utf-8") != corrected_run:
            raise ValueError("corrected run.py payload mismatch")

    report = {
        "schema_version": "exp705_remove_accidental_ocr_v1",
        "source_sha256": sha256(args.input),
        "output_sha256": sha256(args.output),
        "removed_member": OCR_MEMBER,
        "run_py_restored_to_true_exp140_prepare_frame": True,
        "ocr_runtime_references_absent": True,
        "zip_integrity": "PASS",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
