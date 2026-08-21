"""Evaluate Qwen3-VL-Embedding-2B (zero-shot or fine-tuned) on QC val."""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

WORK_DIR = '/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC'
EMBEDDER_DIR = '/home/jovyan/shares/SR008.fs2/litvinov/models/Qwen3-VL-Embedding-2B'
SUBMIT_DIR = '/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/submit'


def label_for_class(category: str, raw_label: int) -> int:
    cat_idx = 0 if category == 'БАД' else 1
    lbl = 1 if raw_label == 1 else 0
    return cat_idx * 2 + lbl


def load_items(split_name: str):
    df = pd.read_csv(os.path.join(WORK_DIR, 'data', 'data.csv'))
    ocr = pd.read_parquet(os.path.join(WORK_DIR, 'ocr_cache.parquet'))
    ocr_map = dict(zip(ocr['id'].astype(int), ocr['ocr_text_full']))
    with open(os.path.join(WORK_DIR, 'split.json'), 'r', encoding='utf-8') as f:
        split = json.load(f)
    df = df[df['id'].isin(split[split_name])]
    rows = []
    for _, row in df.iterrows():
        rows.append({
            'id': int(row['id']),
            'category': row['category'],
            'label': int(row['label']),
            'name': row['name'],
            'description': row['description'],
            'ocr_text': ocr_map.get(int(row['id']), ''),
        })
    return rows


def build_text(item):
    sys.path.insert(0, SUBMIT_DIR)
    from src.gemma_stage import DESC_MAX, OCR_MAX
    desc = str(item.get('description') or '')[:DESC_MAX]
    ocr = str(item.get('ocr_text') or '')[:OCR_MAX]
    text = (f"Категория: {item['category']}\n"
            f"Название: {item['name']}\n"
            f"Описание: {desc}\n"
            f"Текст с изображениями товара (OCR): {ocr}")
    return text


def pool_last_token(hidden, attention_mask):
    last_idx = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    emb = hidden[batch_idx, last_idx, :]
    return F.normalize(emb, p=2, dim=1)


def embed_items(model, processor, items, images_dir, batch_size=8, device='cuda', n_images=5):
    sys.path.insert(0, SUBMIT_DIR)
    from src.ocr_stage import image_paths_for
    from PIL import Image
    embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            images_batch = []
            prompts = []
            for it in batch:
                images = [Image.open(p).convert('RGB') for p in image_paths_for(it['id'], images_dir)[:n_images]]
                text = build_text(it)
                messages = [
                    {'role': 'system', 'content': 'Represent the user\'s input.'},
                    {'role': 'user', 'content': [{'type': 'image', 'image': im} for im in images] + [{'type': 'text', 'text': text}]},
                ]
                prompts.append(processor.apply_chat_template(messages, tokenize=False, add_vision_id=False))
                images_batch.append(images)
            inputs = processor(text=prompts, images=images_batch, return_tensors='pt', padding=True, truncation=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True)
            emb = pool_last_token(outputs.hidden_states[-1], inputs['attention_mask'])
            embs.append(emb.cpu().float().numpy())
            del inputs, outputs, emb
            torch.cuda.empty_cache()
            print(f'[embed] {i+len(batch)}/{len(items)}', flush=True)
    return np.concatenate(embs, axis=0)


def evaluate(train_embs, train_items, val_embs, val_items):
    train_y = {cat: np.array([1 if it['label'] == 1 else 0 for it in train_items if it['category'] == cat]) for cat in ['БАД', 'Легковоспламеняющиеся']}
    val_y = {cat: np.array([1 if it['label'] == 1 else 0 for it in val_items if it['category'] == cat]) for cat in ['БАД', 'Легковоспламеняющиеся']}

    results = {}
    p_ban_records = []

    for cat in ['БАД', 'Легковоспламеняющиеся']:
        idx_tr = [i for i, it in enumerate(train_items) if it['category'] == cat]
        idx_val = [i for i, it in enumerate(val_items) if it['category'] == cat]
        X_tr = train_embs[idx_tr]
        y_tr = train_y[cat]
        X_val = val_embs[idx_val]
        y_val = val_y[cat]

        knn = KNeighborsClassifier(n_neighbors=min(15, len(X_tr)), weights='distance', metric='cosine')
        knn.fit(X_tr, y_tr)
        pred_knn = knn.predict(X_val)

        logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
        logreg.fit(X_tr, y_tr)
        pred_lr = logreg.predict(X_val)
        prob_lr = logreg.predict_proba(X_val)[:, 1]

        p_knn = knn.predict_proba(X_val)[:, 1] if hasattr(knn, 'predict_proba') else None

        p, r, f, _ = precision_recall_fscore_support(y_val, pred_lr, labels=[0, 1], zero_division=0)
        results[cat] = {
            'n_val': len(y_val),
            'f1_ban_lr': float(f[1]),
            'f1_nonban_lr': float(f[0]),
            'knn_f1_ban': float(precision_recall_fscore_support(y_val, pred_knn, labels=[0,1], zero_division=0)[2][1]),
        }

        for j, orig_idx in enumerate(idx_val):
            item = val_items[orig_idx]
            p_ban_records.append({
                'id': item['id'],
                'category': cat,
                'target': 'бан' if item['label'] == 1 else 'не бан',
                'p_ban_lr': float(prob_lr[j]),
                'pred_lr': 'бан' if pred_lr[j] == 1 else 'не бан',
                'p_ban_knn': float(p_knn[j]) if p_knn is not None else None,
                'pred_knn': 'бан' if pred_knn[j] == 1 else 'не бан',
            })

    f1s = [results[cat]['f1_ban_lr'] for cat in results]
    results['macro_f1_lr'] = float(np.mean(f1s))
    f1s_knn = [results[cat]['knn_f1_ban'] for cat in results]
    results['macro_f1_knn'] = float(np.mean(f1s_knn))
    return results, pd.DataFrame(p_ban_records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp_id', type=str, default=None)
    ap.add_argument('--adapter', type=str, default=None)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--n_images', type=int, default=5)
    args = ap.parse_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda'

    os.makedirs(args.out, exist_ok=True)

    processor = AutoProcessor.from_pretrained(EMBEDDER_DIR, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(EMBEDDER_DIR, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)

    train_items = load_items('train')
    val_items = load_items('val')
    images_dir = Path(WORK_DIR) / 'data' / 'images' / 'images'

    train_embs = embed_items(model, processor, train_items, images_dir, batch_size=args.batch_size, device=device, n_images=args.n_images)
    val_embs = embed_items(model, processor, val_items, images_dir, batch_size=args.batch_size, device=device, n_images=args.n_images)

    results, preds_df = evaluate(train_embs, train_items, val_embs, val_items)

    np.save(os.path.join(args.out, 'train_embs.npy'), train_embs)
    np.save(os.path.join(args.out, 'val_embs.npy'), val_embs)
    preds_df.to_csv(os.path.join(args.out, 'embed_predictions.csv'), index=False)

    with open(os.path.join(args.out, 'metrics_embed.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f'[eval_embedder] macro_f1_lr={results["macro_f1_lr"]:.4f}, macro_f1_knn={results["macro_f1_knn"]:.4f}')
    print(f'[eval_embedder] saved to {args.out}')


if __name__ == '__main__':
    main()
