"""Build SFT datasets std/noocr/ocrlong from split.json + ocr_cache.parquet."""
import os
import sys
import json
import pandas as pd
from pathlib import Path

SUBMIT_DIR = '/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/submit'
if SUBMIT_DIR not in sys.path:
    sys.path.insert(0, SUBMIT_DIR)

from src.gemma_stage import GemmaStage, _load_image, RULES, SYS
from src.ocr_stage import image_paths_for

WORK_DIR = '/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC'
DATA_DIR = os.path.join(WORK_DIR, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'data.csv')
IMAGES_DIR = os.path.join(DATA_DIR, 'images', 'images')
SPLIT_PATH = os.path.join(WORK_DIR, 'split.json')
OCR_CACHE_PATH = os.path.join(WORK_DIR, 'ocr_cache.parquet')


def verdict_from_label(label: int) -> str:
    # Correct mapping verified in data report: 1 -> бан, 0 -> не бан
    return 'бан' if label == 1 else 'не бан'


def _messages_std(item: dict, n_images: int = 5) -> list:
    from src.gemma_stage import DESC_MAX, OCR_MAX
    desc = str(item.get("description") or "")[:DESC_MAX]
    ocr = str(item.get("ocr_text") or "")[:OCR_MAX] or "[текст на изображениях не обнаружен]"
    text = (f"Категория: {item['category']}\n"
            f"Название: {item['name']}\n"
            f"Описание: {desc}\n"
            f"Текст с изображениями товара (OCR): {ocr}")
    content = [{"type": "image", "image": p} for p in item["image_paths"][:n_images]]
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": SYS.format(category=item["category"], rules=RULES[item["category"]])},
        {"role": "user", "content": content},
    ]


def _messages_noocr(item: dict, n_images: int = 5) -> list:
    from src.gemma_stage import DESC_MAX
    desc = str(item.get("description") or "")[:DESC_MAX]
    text = (f"Категория: {item['category']}\n"
            f"Название: {item['name']}\n"
            f"Описание: {desc}")
    content = [{"type": "image", "image": p} for p in item["image_paths"][:n_images]]
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": SYS.format(category=item["category"], rules=RULES[item["category"]])},
        {"role": "user", "content": content},
    ]


def _messages_ocrlong(item: dict, n_images: int = 5) -> list:
    from src.gemma_stage import DESC_MAX
    OCR_MAX = 1500
    desc = str(item.get("description") or "")[:DESC_MAX]
    ocr = str(item.get("ocr_text") or "")[:OCR_MAX] or "[текст на изображениях не обнаружен]"
    text = (f"Категория: {item['category']}\n"
            f"Название: {item['name']}\n"
            f"Описание: {desc}\n"
            f"Текст с изображениями товара (OCR): {ocr}")
    content = [{"type": "image", "image": p} for p in item["image_paths"][:n_images]]
    content.append({"type": "text", "text": text})
    return [
        {"role": "system", "content": SYS.format(category=item["category"], rules=RULES[item["category"]])},
        {"role": "user", "content": content},
    ]


BUILDERS = {
    'std': _messages_std,
    'noocr': _messages_noocr,
    'ocrlong': _messages_ocrlong,
}


def main():
    df = pd.read_csv(CSV_PATH)
    ocr_df = pd.read_parquet(OCR_CACHE_PATH)
    ocr_map = dict(zip(ocr_df['id'].astype(int), ocr_df['ocr_text_full']))

    with open(SPLIT_PATH, 'r', encoding='utf-8') as f:
        split = json.load(f)

    for split_name, ids in split.items():
        sub = df[df['id'].isin(ids)].copy()
        print(f'Building {split_name}: {len(sub)} items')

        for ds_name, builder in BUILDERS.items():
            out_file = os.path.join(WORK_DIR, f'sft_{ds_name}_{split_name}.jsonl')
            rows = []
            for _, row in sub.iterrows():
                item_id = int(row['id'])
                image_paths = image_paths_for(item_id, IMAGES_DIR)[:5]
                ocr_text = ocr_map.get(item_id, '')
                item = {
                    'id': item_id,
                    'category': row['category'],
                    'name': row['name'],
                    'description': row['description'],
                    'image_paths': image_paths,
                    'ocr_text': ocr_text,
                }
                msgs = builder(item, n_images=5)
                target = verdict_from_label(int(row['label']))
                rows.append({
                    'id': item_id,
                    'category': row['category'],
                    'image_paths': image_paths,
                    'prompt_messages': msgs,
                    'target': target,
                })
            with open(out_file, 'w', encoding='utf-8') as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(f'  {ds_name}: {len(rows)} -> {out_file}')


if __name__ == '__main__':
    main()
