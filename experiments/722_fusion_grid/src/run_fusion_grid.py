"""Grid search over fusion weight and threshold for Track A + Track B predictions."""
import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def load_a(path):
    df = pd.read_csv(path)
    return df[['id', 'category', 'target', 'p_ban']].copy()


def load_b(path):
    df = pd.read_csv(path)
    return df[['id', 'category', 'target', 'p_ban_lr']].rename(columns={'p_ban_lr': 'p_ban'}).copy()


def evaluate(df, w, t):
    df = df.copy()
    df['p_fused'] = w * df['p_ban_a'] + (1 - w) * df['p_ban_b']
    df['verdict'] = np.where(df['p_fused'] >= t, 'бан', 'не бан')
    cats = sorted(df['category'].unique())
    f1s = []
    per_cat = {}
    for cat in cats:
        sub = df[df['category'] == cat]
        y_true = (sub['target'] == 'бан').astype(int)
        y_pred = (sub['verdict'] == 'бан').astype(int)
        _, _, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
        per_cat[cat] = {'f1_ban': float(f[1]), 'f1_nonban': float(f[0])}
        f1s.append(float(f[1]))
    macro_f1 = float(np.mean(f1s))
    return macro_f1, per_cat, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a_csv', type=str, required=True)
    ap.add_argument('--b_csv', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    a = load_a(args.a_csv)
    b = load_b(args.b_csv)
    merged = pd.merge(a, b, on=['id', 'category', 'target'], suffixes=('_a', '_b'))

    best = {'macro_f1': -1.0}
    records = []
    for w in np.linspace(0, 1, 21):
        for t in np.linspace(0.3, 0.7, 9):
            macro, per_cat, _ = evaluate(merged, w, t)
            records.append({'w': float(w), 't': float(t), 'macro_f1': macro})
            if macro > best['macro_f1']:
                best = {'w': float(w), 't': float(t), 'macro_f1': macro, 'per_cat': per_cat}

    pd.DataFrame(records).to_csv(os.path.join(args.out, 'grid.csv'), index=False)
    macro, _, pred = evaluate(merged, best['w'], best['t'])
    pred.to_csv(os.path.join(args.out, 'fusion_predictions.csv'), index=False)

    with open(os.path.join(args.out, 'best.json'), 'w', encoding='utf-8') as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print(f'[fusion] best w={best["w"]:.2f} t={best["t"]:.2f} macro_f1={best["macro_f1"]:.4f}')
    print(f'[fusion] saved to {args.out}')


if __name__ == '__main__':
    main()
