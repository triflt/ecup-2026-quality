from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "exp699_summarizer", ROOT / "summarize_gpu_ablations.py"
)
assert SPEC is not None and SPEC.loader is not None
summarizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarizer)


def report() -> dict:
    entry = {
        "scenario": "q35_v2p10",
        "macro_f1": 0.91,
        "macro_delta": 0.02,
        "fold_macro_delta": {"0": 0.01, "3": 0.03},
        "fold_wins": 2,
        "bad_f1": 0.96,
        "flammable_f1": 0.86,
        "flammable_confusion": {"tp": 30, "fp": 5, "fn": 4, "tn": 900},
        "component_ap": {
            "q35_v2p10": {"baseline": 0.85, "candidate": 0.87, "delta": 0.02}
        },
        "corrections_regressions": {"corrections": 12, "regressions": 3},
        "semantic_families": {
            "singleton": {"corrections": 3, "regressions": 1},
            "rare_le2": {"corrections": 5, "regressions": 2},
            "repeated": {"corrections": 7, "regressions": 1},
        },
        "screen_pass": True,
    }
    value = {
        "schema": "exp699_gpu_evaluation_v3",
        "evaluation_folds": [0, 3],
        "fusion_policy": "frozen_baseline_140_nested_oof_selection",
        "leaderboard": [entry],
        "public_used": False,
        "sealed_rows": 0,
    }
    value["self_sha256"] = summarizer.canonical_sha256(value)
    return value


def write_report(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_summarizes_frozen_v3_report(tmp_path: Path) -> None:
    path = tmp_path / "screen.json"
    write_report(path, report())
    summary = summarizer.summarize([path])
    assert summary["schema"] == "exp699_gpu_ablation_summary_v1"
    assert summary["rows"][0]["macro_delta"] == 0.02
    assert summary["rows"][0]["flammable_ap_delta"] == 0.02
    assert summary["rows"][0]["fn"] == 4
    assert summary["rows"][0]["rare_corrections"] == 5
    payload = dict(summary)
    declared = payload.pop("self_sha256")
    assert declared == summarizer.canonical_sha256(payload)


def test_rejects_v2_and_tampered_reports(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    value = report()
    value["schema"] = "exp699_gpu_evaluation_v2"
    write_report(path, value)
    try:
        summarizer.summarize([path])
    except ValueError as error:
        assert "self-hash" in str(error)
    else:
        raise AssertionError("tampered report was accepted")
    value["self_sha256"] = summarizer.canonical_sha256(
        {key: item for key, item in value.items() if key != "self_sha256"}
    )
    write_report(path, value)
    try:
        summarizer.summarize([path])
    except ValueError as error:
        assert "v3" in str(error)
    else:
        raise AssertionError("v2 report was accepted")


def test_rejects_duplicate_scenario_fold_pair(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_report(first, report())
    write_report(second, report())
    try:
        summarizer.summarize([first, second])
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate evaluation was accepted")
