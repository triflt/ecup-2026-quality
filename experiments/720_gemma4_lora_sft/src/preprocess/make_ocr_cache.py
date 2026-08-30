"""Build OCR cache for all items in split.json using src/ocr_stage.py from submit.
Run in parallel: for k in 0..7: python make_ocr_cache.py --shard k --num-shards 8
"""
import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path

# Ensure submit src is importable
SUBMIT_DIR = '/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/submit'
if SUBMIT_DIR not in sys.path:
    sys.path.insert(0, SUBMIT_DIR)

from src.ocr_stage import OcrStage, image_paths_for

WORK_DIR = '/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC'
DATA_DIR = os.path.join(WORK_DIR, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'data.csv')
IMAGES_DIR = os.path.join(DATA_DIR, 'images', 'images')  # nested due to unzip
SPLIT_PATH = os.path.join(WORK_DIR, 'split.json')
OUT_DIR = os.path.join(WORK_DIR, 'ocr_cache_parts')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=8)
    ap.add_argument('--batch', type=int, default=512)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    with open(SPLIT_PATH, 'r', encoding='utf-8') as f:
        split = json.load(f)

    all_ids = set(split['train'] + split['val'])
    df = df[df['id'].isin(all_ids)].copy()

    # shard
    df = df.sort_values('id').reset_index(drop=True)
    df = df[df.index % args.num_shards == args.shard]

    out_path = os.path.join(OUT_DIR, f'ocr_cache_shard{args.shard}.parquet')
    if os.path.exists(out_path):
        print(f'[shard {args.shard}] already exists: {out_path}', flush=True)
        return

    items = []
    for _, row in df.iterrows():
        paths = image_paths_for(row['id'], IMAGES_DIR)[:5]
        items.append({'id': int(row['id']), 'image_paths': paths})

    tmp_dir = os.path.join(OUT_DIR, f'tmp_shard{args.shard}')
    ocr = OcrStage(batch=args.batch, tmp_dir=tmp_dir)
    results = ocr.run(items)
    ocr.close()

    rows = []
    for it in items:
        item_id = it['id']
        text_full = str(results.get(item_id, ''))
        rows.append({
            'id': item_id,
            'n_images': len(it['image_paths']),
            'ocr_text_full': text_full,
            'ocr_chars': len(text_full),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    print(f'[shard {args.shard}] done: {len(out_df)} items -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
