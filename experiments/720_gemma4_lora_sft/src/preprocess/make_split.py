import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

WORK_DIR = '/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC'
CSV_PATH = os.path.join(WORK_DIR, 'data', 'data.csv')
OUT_PATH = os.path.join(WORK_DIR, 'split.json')

df = pd.read_csv(CSV_PATH)

df['strat'] = df['category'].astype(str) + '_' + df['label'].astype(str)

train_idx, val_idx = train_test_split(
    df.index,
    test_size=0.10,
    random_state=42,
    stratify=df['strat']
)

train_ids = df.loc[train_idx, 'id'].astype(int).tolist()
val_ids = df.loc[val_idx, 'id'].astype(int).tolist()

split = {"train": train_ids, "val": val_ids}

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(split, f, ensure_ascii=False, indent=2)

print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
print(f"Saved to {OUT_PATH}")

# print distribution
for name, idx in [("train", train_idx), ("val", val_idx)]:
    sub = df.loc[idx]
    print(f"\n{name}:")
    print(pd.crosstab(sub['category'], sub['label']))
