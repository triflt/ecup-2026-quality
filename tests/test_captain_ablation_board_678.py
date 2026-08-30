from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/678_captain_ablation_board/slice_full_candidate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("experiment_678_slices", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SLICES = load_module()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_full_candidate_slices_are_diagnostic_and_exactly_bound(tmp_path: Path) -> None:
    registry_path = tmp_path / "folds.csv"
    fields = [
        "id",
        "category",
        "label",
        "semantic_component",
        "component_size",
        "split",
        "development_fold",
    ]
    registry_rows = []
    baseline_paths, large_paths = [], []
    flammable_labels = [1, 0, 1, 0, 1]
    baseline_flammable_scores = [-1.0, -1.0, 1.0, 1.0, -1.0]
    large_flammable_scores = [3.0, -1.0, 1.0, -3.0, 3.0]
    for fold in range(5):
        bad_id, flammable_id = str(2 * fold), str(2 * fold + 1)
        registry_rows.extend(
            [
                {
                    "id": bad_id,
                    "category": "БАД",
                    "label": 1,
                    "semantic_component": f"bad-{fold}",
                    "component_size": 1,
                    "split": "development",
                    "development_fold": fold,
                },
                {
                    "id": flammable_id,
                    "category": "Легковоспламеняющиеся",
                    "label": flammable_labels[fold],
                    "semantic_component": "mixed" if fold in {0, 1} else f"flammable-{fold}",
                    "component_size": 2 if fold in {0, 1} else 1,
                    "split": "development",
                    "development_fold": fold,
                },
            ]
        )
        baseline_path, large_path = tmp_path / f"small{fold}.jsonl", tmp_path / f"large{fold}.jsonl"
        common_bad = {
            "global_index": 2 * fold,
            "id": bad_id,
            "fold": fold,
            "category": "БАД",
            "score": 1.0,
            "prediction": 1,
        }
        baseline_flammable = {
            "global_index": 2 * fold + 1,
            "id": flammable_id,
            "fold": fold,
            "category": "Легковоспламеняющиеся",
            "score": baseline_flammable_scores[fold],
            "prediction": int(baseline_flammable_scores[fold] >= 0.0),
        }
        large_flammable = dict(baseline_flammable)
        large_flammable["score"] = large_flammable_scores[fold]
        large_flammable["prediction"] = int(large_flammable_scores[fold] >= 0.0)
        write_jsonl(baseline_path, [common_bad, baseline_flammable])
        write_jsonl(large_path, [common_bad, large_flammable])
        baseline_paths.append(baseline_path)
        large_paths.append(large_path)
    with registry_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(registry_rows)

    ocr_path = tmp_path / "ocr.jsonl"
    write_jsonl(
        ocr_path,
        [
            {
                "id": str(index),
                "image_index": 0,
                "status": "OCR_AVAILABLE" if index < 8 else "OCR_UNAVAILABLE",
            }
            for index in range(10)
        ],
    )
    result = SLICES.evaluate(
        registry_path=registry_path,
        baseline_paths=baseline_paths,
        large_paths=large_paths,
        ocr_availability_path=ocr_path,
        output_path=tmp_path / "slices.json",
    )
    assert result["decision"] == "DIAGNOSTIC_ONLY_NO_SELECTION"
    assert result["threshold_tuned"] is False
    assert result["public_used"] is False
    assert result["slices"]["all"]["rows"] == 10
    assert result["slices"]["all"]["corrected"] == 3
    assert result["slices"]["all"]["regressed"] == 0
    assert result["slices"]["semantic_repeated"]["rows"] == 2
    assert result["slices"]["mixed_label_component"]["rows"] == 2
    assert result["slices"]["ocr_all_unavailable"]["rows"] == 2
    payload = dict(result)
    digest = payload.pop("contract_sha256")
    assert digest == SLICES.canonical_sha256(payload)
