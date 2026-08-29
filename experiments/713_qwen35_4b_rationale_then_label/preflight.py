from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from target_contract import FORMAT_VERSION, parse_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("rationale_then_label", "matched_support_digit"),
        default="rationale_then_label",
    )
    parser.add_argument("--folds", type=Path, help="optional JSON id->fold mapping")
    parser.add_argument("--min-coverage", type=float, default=0.85)
    args = parser.parse_args()

    with args.data.open(encoding="utf-8", newline="") as stream:
        data = {
            str(row["id"]): {"label": int(row["label"]), "category": str(row["category"])}
            for row in csv.DictReader(stream)
        }
    targets = {}
    with args.targets.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            item_id = str(row["id"])
            parsed = parse_target(row["target"]) if args.mode == "rationale_then_label" else None
            if item_id not in data:
                raise ValueError(f"target id is absent from data: {item_id}")
            if item_id in targets:
                raise ValueError(f"duplicate target id={item_id}")
            label = int(row["label"])
            if (
                (parsed is not None and int(parsed["label"]) != label)
                or label != data[item_id]["label"]
                or (args.mode == "matched_support_digit" and str(row["target"]) != str(label))
            ):
                raise ValueError(f"label mismatch for id={item_id}")
            targets[item_id] = parsed if parsed is not None else {"label": label}
    missing = sorted(set(data) - set(targets))
    extra = sorted(set(targets) - set(data))
    cell_total = Counter((row["category"], row["label"]) for row in data.values())
    cell_valid = Counter((data[item_id]["category"], data[item_id]["label"]) for item_id in targets)
    cell_coverage = {
        f"{category}|{label}": cell_valid[(category, label)] / total
        for (category, label), total in sorted(cell_total.items())
    }
    coverage = len(targets) / len(data) if data else 0.0
    report = {
        "format_version": FORMAT_VERSION if args.mode == "rationale_then_label" else "matched_support_digit_v1",
        "mode": args.mode,
        "data_rows": len(data),
        "valid_targets": len(targets),
        "missing_targets": len(missing),
        "extra_targets": len(extra),
        "coverage": coverage,
        "cell_coverage": cell_coverage,
        "min_coverage": args.min_coverage,
        "first_missing": missing[:20],
        "cell_coverage_gate_pass": all(
            value >= args.min_coverage for value in cell_coverage.values()
        ),
        "gate_pass": not extra and coverage >= args.min_coverage,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
