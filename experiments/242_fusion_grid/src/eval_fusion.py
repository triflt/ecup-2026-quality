"""Track C: simple fusion of track A and track B probabilities."""
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def load_pred_csv(path, p_col='p_ban'):
    df = pd.read_csv(path)
    return dict(zip(df['id'].astype(int), df[p_col].astype(float))), df


def compute_metrics(df):
    cats = sorted(df['category'].unique())
    results = {}
    f1s = []
    for cat in cats:
        sub = df[df['category'] == cat]
        y_true = (sub['target'] == 'бан').astype(int)
        y_pred = (sub['verdict'] == 'бан').astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        results[cat] = {'f1_ban': float(f[1]), 'f1_nonban': float(f[0])}
        f1s.append(float(f[1]))
    results['macro_f1'] = float(np.mean(f1s))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a_csv', type=str, required=True)
    ap.add_argument('--b_csv', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--w', type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    a_map, a_df = load_pred_csv(args.a_csv, 'p_ban')
    b_map, b_df = load_pred_csv(args.b_csv, 'p_ban_lr')

    common_ids = set(a_map.keys()) & set(b_map.keys())
    rows = []
    for item_id in sorted(common_ids):
        p_fused = args.w * a_map[item_id] + (1 - args.w) * b_map[item_id]
        verdict = 'бан' if p_fused >= 0.5 else 'не бан'
        a_row = a_df[a_df['id'] == item_id].iloc[0]
        rows.append({
            'id': item_id,
            'category': a_row['category'],
            'p_ban': p_fused,
            'verdict': verdict,
            'target': a_row['target'],
        })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(os.path.join(args.out, 'fusion_predictions.csv'), index=False)

    metrics = compute_metrics(pred_df)
    metrics['w'] = args.w
    with open(os.path.join(args.out, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f'[fusion] w={args.w} macro_f1={metrics["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
