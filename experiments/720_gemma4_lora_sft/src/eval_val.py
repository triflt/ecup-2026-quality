"""Eval harness for Gemma LoRA on local val set."""
import os
import sys
# vLLM env vars BEFORE any imports
os.environ.setdefault("VLLM_PLUGINS", "")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
os.environ.setdefault("HOST_IP", "127.0.0.1")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import json
import argparse
import math
import pandas as pd
from pathlib import Path

SUBMIT_DIR = '/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/submit'
if SUBMIT_DIR not in sys.path:
    sys.path.insert(0, SUBMIT_DIR)

WORK_DIR = '/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC'
DATA_DIR = os.path.join(WORK_DIR, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'data.csv')
IMAGES_DIR = os.path.join(DATA_DIR, 'images', 'images')
SPLIT_PATH = os.path.join(WORK_DIR, 'split.json')


def load_items(dataset: str):
    with open(SPLIT_PATH, 'r', encoding='utf-8') as f:
        split = json.load(f)
    df = pd.read_csv(CSV_PATH)
    val_df = df[df['id'].isin(split['val'])].copy()

    # load OCR cache
    ocr_df = pd.read_parquet(os.path.join(WORK_DIR, 'ocr_cache.parquet'))
    ocr_map = dict(zip(ocr_df['id'].astype(int), ocr_df['ocr_text_full']))

    from src.ocr_stage import image_paths_for
    items = []
    for _, row in val_df.iterrows():
        item_id = int(row['id'])
        image_paths = image_paths_for(item_id, IMAGES_DIR)[:5]
        items.append({
            'id': item_id,
            'category': row['category'],
            'name': row['name'],
            'description': row['description'],
            'image_paths': image_paths,
            'ocr_text': ocr_map.get(item_id, ''),
            'target': 'бан' if int(row['label']) == 1 else 'не бан',
        })
    return items


def compute_metrics(df: pd.DataFrame) -> dict:
    from sklearn.metrics import precision_recall_fscore_support
    results = {}
    cats = sorted(df['category'].unique())
    f1s = []
    for cat in cats:
        sub = df[df['category'] == cat]
        y_true = (sub['target'] == 'бан').astype(int)
        y_pred = (sub['verdict'] == 'бан').astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        results[cat] = {
            'precision_ban': float(p[1]),
            'recall_ban': float(r[1]),
            'f1_ban': float(f[1]),
            'precision_nonban': float(p[0]),
            'recall_nonban': float(r[0]),
            'f1_nonban': float(f[0]),
            'n': len(sub),
        }
        f1s.append(float(f[1]))
    results['macro_f1'] = float(sum(f1s) / len(f1s))
    results['parse_fail_rate'] = float((~df['parse_ok']).mean())
    results['n_parse_fail'] = int((~df['parse_ok']).sum())
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lora', type=str, default=None)
    ap.add_argument('--dataset', type=str, default='std', choices=['std', 'noocr', 'ocrlong'])
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--max_lora_rank', type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # choose GemmaStage variant
    if args.dataset == 'noocr':
        import importlib.util
        spec = importlib.util.spec_from_file_location('gemma_stage_noocr', os.path.join(WORK_DIR, 'gemma_stage_noocr.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        GemmaStage = mod.GemmaStage
        parse_verdict = mod.parse_verdict
    else:
        from src.gemma_stage import GemmaStage, parse_verdict

    items = load_items(args.dataset)
    print(f'[eval] {len(items)} val items, dataset={args.dataset}, lora={args.lora}')

    gemma = GemmaStage(n_images=5, gpu_mem=0.85, lora=args.lora)
    # patch max_lora_rank if needed
    if args.lora:
        gemma.llm.lora_config.max_lora_rank = max(args.max_lora_rank, gemma.llm.lora_config.max_lora_rank)

    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    sp = SamplingParams(temperature=0, max_tokens=8, logprobs=20)
    lora_req = LoRARequest('adapter', 1, args.lora) if args.lora else None

    verdicts, raws, p_ban_list = [], [], []
    tokenizer = gemma.llm.get_tokenizer()
    ban_tok = tokenizer.encode('бан', add_special_tokens=False)
    neban_tok = tokenizer.encode('не', add_special_tokens=False)
    ban_id = ban_tok[0] if ban_tok else None
    neban_id = neban_tok[0] if neban_tok else None

    for i in range(0, len(items), args.batch):
        chunk = items[i:i + args.batch]
        msgs = [gemma._messages(it) for it in chunk]
        outs = gemma.llm.chat(msgs, sp, lora_request=lora_req,
                              chat_template_kwargs={'enable_thinking': False})
        for o in outs:
            raw = o.outputs[0].text
            v, ok = parse_verdict(raw)
            verdicts.append(v)
            raws.append(raw)
            # p_ban from first token logprobs
            logprobs = o.outputs[0].logprobs
            tok0 = logprobs[0] if logprobs else {}
            scores = {tok_id: float(lp.logprob) for tok_id, lp in tok0.items()}
            exp_scores = {k: math.exp(v) for k, v in scores.items()}
            p_ban = 0.5
            if ban_id in exp_scores or neban_id in exp_scores:
                total = exp_scores.get(ban_id, 0) + exp_scores.get(neban_id, 0)
                if total > 0:
                    p_ban = exp_scores.get(ban_id, 0) / total
            p_ban_list.append(p_ban)
        print(f'[eval] {i + len(chunk)}/{len(items)}', flush=True)
    gemma.close()

    rows = []
    for it, v, raw, p_ban in zip(items, verdicts, raws, p_ban_list):
        parsed, ok = parse_verdict(v)
        rows.append({
            'id': it['id'],
            'category': it['category'],
            'verdict': v,
            'p_ban': p_ban,
            'target': it['target'],
            'raw': raw,
            'parse_ok': ok,
        })

    pred_df = pd.DataFrame(rows)
    pred_path = os.path.join(args.out, 'val_predictions.csv')
    pred_df.to_csv(pred_path, index=False)

    # errors top50: wrong predictions sorted by confidence margin desc
    wrong = pred_df[pred_df['verdict'] != pred_df['target']].copy()
    wrong['conf_margin'] = (wrong['p_ban'] - 0.5).abs()
    wrong = wrong.sort_values('conf_margin', ascending=False).head(50)
    wrong.to_csv(os.path.join(args.out, 'errors_top50.csv'), index=False)

    metrics = compute_metrics(pred_df)
    metrics['n_items'] = len(pred_df)

    with open(os.path.join(args.out, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f'[eval] macro_f1={metrics["macro_f1"]:.4f}, parse_fail={metrics["parse_fail_rate"]:.4f}')
    print(f'[eval] saved to {args.out}')


if __name__ == '__main__':
    main()
