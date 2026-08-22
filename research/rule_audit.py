from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


DATA = Path("research/oof-cache-extracted/hard_cases.csv")
OUTPUT = Path("research/rule-audit.json")

BAD_EXPLICIT = re.compile(
    r"(?iu)(?:\bбад\b|биологически\s+активн(?:ая|ой|ые|ых)?\s+добавк|"
    r"dietary\s+supplement|food\s+supplement|пищевая\s+добавка)"
)
SPORTS = re.compile(
    r"(?iu)(?:спортивн(?:ое|ого|ому)?\s+питан|спортпит|\bbcaas?\b|\bbcaa\b|"
    r"\bпротеин|гейнер|предтрен|креатин|л[-\s]?карнитин|l[-\s]?carnitine|"
    r"аминокислот)"
)
FLAMMABLE_EXPLICIT = re.compile(
    r"(?iu)(?:\bзажигалк|\bспич(?:к|еч)|сух(?:ое|ого)\s+горюч|"
    r"жидкост\S*\s+для\s+розжиг|газов\S*\s+(?:баллон|картридж|цангов)|"
    r"\bпропан|\bбутан|\bкеросин|\bбензин|легковоспламен|горюч(?:ее|ая)\s+(?:веществ|газ)|"
    r"топливн\S*\s+(?:брикет|таблет|смесь))"
)
EMPTY_EQUIPMENT = re.compile(
    r"(?iu)(?:\bмангал|\bгрил|газов\S*\s+плит|горелк(?:а|и|у)|печ(?:ь|ка)|"
    r"насадк\S*\s+(?:на|для)\s+баллон)"
)
INCLUDED_FUEL = re.compile(
    r"(?iu)(?:в\s+комплект\S*\s+(?:зажигалк|спич|баллон|топлив|угол)|"
    r"комплект\S*\s+с\s+(?:зажигалк|спич|баллон|топлив|угл)|"
    r"заправлен|с\s+газом|с\s+угл(?:ем|ём))"
)


def f1(labels, predictions):
    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def match(series: pd.Series, pattern: re.Pattern) -> pd.Series:
    return series.map(lambda value: bool(pattern.search(value)))


def describe_mask(labels, mask):
    count = int(mask.sum())
    positives = int(labels[mask].sum())
    return {
        "count": count,
        "positive": positives,
        "precision_if_forced_positive": positives / count if count else None,
    }


def main():
    frame = pd.read_csv(DATA)
    text = (
        frame["name"].fillna("").astype(str)
        + "\n"
        + frame["description"].fillna("").astype(str)
    )
    labels = frame["label"].astype(int)
    baseline = frame["prediction"].astype(int)
    report = {}

    bad_mask = frame["category"] == "БАД"
    bad_text = text[bad_mask]
    bad_labels = labels[bad_mask]
    bad_base = baseline[bad_mask]
    explicit = match(bad_text, BAD_EXPLICIT)
    sports = match(bad_text, SPORTS)
    bad_variants = {"baseline": bad_base.copy()}
    positive_override = bad_base.copy()
    positive_override[explicit] = 1
    bad_variants["force_explicit_positive"] = positive_override
    sports_override = bad_base.copy()
    sports_override[sports & ~explicit] = 0
    bad_variants["force_sports_negative_without_explicit"] = sports_override
    both = positive_override.copy()
    both[sports & ~explicit] = 0
    bad_variants["both"] = both
    report["БАД"] = {
        "signals": {
            "explicit": describe_mask(bad_labels, explicit),
            "sports": describe_mask(bad_labels, sports),
            "sports_without_explicit": describe_mask(bad_labels, sports & ~explicit),
        },
        "f1": {name: f1(bad_labels, pred) for name, pred in bad_variants.items()},
        "changed": {name: int((pred != bad_base).sum()) for name, pred in bad_variants.items()},
    }

    flame_mask = frame["category"] == "Легковоспламеняющиеся"
    flame_text = text[flame_mask]
    flame_labels = labels[flame_mask]
    flame_base = baseline[flame_mask]
    explicit_flame = match(flame_text, FLAMMABLE_EXPLICIT)
    equipment = match(flame_text, EMPTY_EQUIPMENT)
    included = match(flame_text, INCLUDED_FUEL)
    flame_variants = {"baseline": flame_base.copy()}
    positive = flame_base.copy()
    positive[explicit_flame & (~equipment | included)] = 1
    flame_variants["force_explicit_positive_unless_empty_equipment"] = positive
    negative = flame_base.copy()
    negative[equipment & ~included] = 0
    flame_variants["force_empty_equipment_negative"] = negative
    report["Легковоспламеняющиеся"] = {
        "signals": {
            "explicit": describe_mask(flame_labels, explicit_flame),
            "equipment": describe_mask(flame_labels, equipment),
            "included_fuel": describe_mask(flame_labels, included),
            "positive_rule": describe_mask(flame_labels, explicit_flame & (~equipment | included)),
        },
        "f1": {name: f1(flame_labels, pred) for name, pred in flame_variants.items()},
        "changed": {name: int((pred != flame_base).sum()) for name, pred in flame_variants.items()},
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
