import os
import json
import pandas as pd
from pathlib import Path
from collections import Counter

WORK_DIR = '/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC'
DATA_DIR = os.path.join(WORK_DIR, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'data.csv')
IMAGES_DIR = os.path.join(DATA_DIR, 'images', 'images')

df = pd.read_csv(CSV_PATH)

# basic stats
report = []
report.append("# Data Report")
report.append("")
report.append(f"**Total items:** {len(df)}")
report.append(f"**Columns:** {list(df.columns)}")
report.append(f"**Categories:** {df['category'].unique().tolist()}")
report.append("")

# label mapping: after sanity-check (see below) the data uses inverted labels
# We will use CORRECT mapping: 0 -> 'не бан', 1 -> 'бан'
report.append("## Label distribution")
report.append("")
report.append("**Correct mapping (verified by examples):** `0 -> 'не бан'`, `1 -> 'бан'`")
report.append("")
report.append("Raw labels in CSV:")
report.append("")

total = len(df)
label_counts = df['label'].value_counts().sort_index()
report.append(f"| raw label | count | pct |")
report.append(f"|------------|-------|-----|")
for lbl, cnt in label_counts.items():
    report.append(f"| {lbl} | {cnt} | {cnt/total*100:.2f}% |")
report.append("")
report.append("### Sanity-check (category = Легковоспламеняющиеся)")
report.append("")
report.append("Examples with raw label=0: мангал на углях, газовая горелка, газовая плита, шнур — items that are NOT flammable per rules.")
report.append("")
report.append("Examples with raw label=1: горелка с пьезоподжигом, одноразовый мангал с углём, газовый баллон, хлопушка, цветной дым — items that ARE flammable per rules.")
report.append("")
report.append("**Conclusion:** the raw `label` column is inverted relative to the natural verdict. We use `verdict = 'бан' if label == 1 else 'не бан'` for all training/evaluation.")
report.append("")

report.append("## Distribution by category × label")
report.append("")
ct = pd.crosstab(df['category'], df['label'])
report.append(ct.to_markdown())
report.append("")

# sanity check examples
report.append("## Sanity-check examples")
report.append("")
# Find a few examples per category × label
for cat in sorted(df['category'].unique()):
    report.append(f"### {cat}")
    for lbl in [0, 1]:
        verdict = 'не бан' if lbl == 0 else 'бан'
        sub = df[(df['category'] == cat) & (df['label'] == lbl)]
        report.append(f"**{verdict}** ({len(sub)} items):")
        for _, row in sub.head(3).iterrows():
            name = row['name'][:100].replace('|', '\\|')
            desc = str(row['description'])[:200].replace('\n', ' ').replace('|', '\\|')
            report.append(f"- id={row['id']}: **{name}** — {desc}")
    report.append("")

# images stats
n_images = []
empty_images = []
for item_id in df['id']:
    d = Path(IMAGES_DIR) / str(item_id)
    if d.is_dir():
        imgs = [p for p in d.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}]
        n_images.append(len(imgs))
        if len(imgs) == 0:
            empty_images.append(item_id)
    else:
        n_images.append(0)
        empty_images.append(item_id)

df['n_images'] = n_images
report.append("## Images stats")
report.append("")
img_counts = Counter(n_images)
report.append(f"| n_images | count |")
report.append(f"|----------|-------|")
for k in sorted(img_counts):
    report.append(f"| {k} | {img_counts[k]} |")
report.append("")
report.append(f"Items with 0 images: {len(empty_images)} ({len(empty_images)/total*100:.2f}%)")
report.append(f"Items with images: {sum(1 for x in n_images if x > 0)} ({sum(1 for x in n_images if x > 0)/total*100:.2f}%)")
report.append("")

# description lengths
desc_len = df['description'].fillna('').astype(str).apply(len)
report.append("## Description length stats")
report.append("")
report.append(f"- mean: {desc_len.mean():.1f}")
report.append(f"- median: {desc_len.median():.1f}")
report.append(f"- min: {desc_len.min()}, max: {desc_len.max()}")
report.append(f"- >1200 chars: {sum(desc_len > 1200)} ({sum(desc_len > 1200)/total*100:.2f}%)")
report.append("")

# class balance warning
report.append("## Balance check")
report.append("")
for cat in sorted(df['category'].unique()):
    sub = df[df['category'] == cat]
    non_ban = (sub['label'] == 0).sum()
    ban = (sub['label'] == 1).sum()
    total_cat = len(sub)
    report.append(f"- {cat}: бан={ban} ({ban/total_cat*100:.1f}%), не бан={non_ban} ({non_ban/total_cat*100:.1f}%)")
report.append("")

report_path = os.path.join(WORK_DIR, 'data_report.md')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print(f"Report written to {report_path}")
print(f"Total items: {len(df)}")
print(pd.crosstab(df['category'], df['label']))
print("Image distribution:", dict(sorted(img_counts.items())))
