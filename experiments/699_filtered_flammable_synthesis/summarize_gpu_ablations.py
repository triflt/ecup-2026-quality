from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "exp699_gpu_evaluation_v3"
SUMMARY_SCHEMA = "exp699_gpu_ablation_summary_v1"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def load_report(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("report must be a regular file")
    report = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(report)
    declared = payload.pop("self_sha256", None)
    if declared != canonical_sha256(payload):
        raise ValueError("evaluation report self-hash mismatch")
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("only frozen evaluator v3 reports are accepted")
    if report.get("fusion_policy") != "frozen_baseline_140_nested_oof_selection":
        raise ValueError("evaluation fusion policy mismatch")
    if report.get("public_used") is not False or report.get("sealed_rows") != 0:
        raise ValueError("Public or sealed rows are forbidden")
    folds = report.get("evaluation_folds")
    if not isinstance(folds, list) or not folds or any(fold not in range(5) for fold in folds):
        raise ValueError("invalid evaluation folds")
    if len(folds) != len(set(folds)) or folds != sorted(folds):
        raise ValueError("evaluation folds must be unique and sorted")
    leaderboard = report.get("leaderboard")
    if not isinstance(leaderboard, list) or not leaderboard:
        raise ValueError("empty evaluation leaderboard")
    return report


def _family(entry: dict[str, Any], name: str) -> tuple[int, int]:
    value = entry.get("semantic_families", {}).get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing semantic family: {name}")
    return int(value["corrections"]), int(value["regressions"])


def flatten(path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    folds = [int(fold) for fold in report["evaluation_folds"]]
    rows: list[dict[str, Any]] = []
    for entry in report["leaderboard"]:
        scenario = str(entry["scenario"])
        component = entry.get("component_ap", {}).get(scenario)
        if not isinstance(component, dict):
            raise ValueError("scenario AP component missing")
        confusion = entry["flammable_confusion"]
        changes = entry["corrections_regressions"]
        singleton = _family(entry, "singleton")
        rare = _family(entry, "rare_le2")
        repeated = _family(entry, "repeated")
        rows.append(
            {
                "report": path.name,
                "scenario": scenario,
                "folds": folds,
                "macro_f1": float(entry["macro_f1"]),
                "macro_delta": float(entry["macro_delta"]),
                "fold_macro_delta": {
                    str(key): float(value)
                    for key, value in sorted(entry["fold_macro_delta"].items())
                },
                "fold_wins": int(entry["fold_wins"]),
                "bad_f1": float(entry["bad_f1"]),
                "flammable_f1": float(entry["flammable_f1"]),
                "flammable_ap_delta": float(component["delta"]),
                "tp": int(confusion["tp"]),
                "fp": int(confusion["fp"]),
                "fn": int(confusion["fn"]),
                "tn": int(confusion["tn"]),
                "corrections": int(changes["corrections"]),
                "regressions": int(changes["regressions"]),
                "singleton_corrections": singleton[0],
                "singleton_regressions": singleton[1],
                "rare_corrections": rare[0],
                "rare_regressions": rare[1],
                "repeated_corrections": repeated[0],
                "repeated_regressions": repeated[1],
                "screen_pass": bool(entry["screen_pass"]),
            }
        )
    return rows


def summarize(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one report is required")
    rows = [row for path in paths for row in flatten(path, load_report(path))]
    identities = [(row["scenario"], tuple(row["folds"])) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate scenario/fold evaluation")
    rows.sort(key=lambda row: (-row["macro_delta"], row["scenario"], row["folds"]))
    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "evaluator_schema": REPORT_SCHEMA,
        "fusion_policy": "frozen_baseline_140_nested_oof_selection",
        "rows": rows,
        "public_used": False,
        "decision": "ABLATION_SUMMARY_ONLY",
    }
    summary["self_sha256"] = canonical_sha256(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.report)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
