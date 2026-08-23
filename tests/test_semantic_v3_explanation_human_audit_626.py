import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/626_semantic_v3_explanation_human_audit"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load_module("exp626_builder", EXP / "build_human_audit.py")
evaluator = load_module("exp626_evaluator", EXP / "evaluate_human_audit.py")


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fixture(tmp_path: Path):
    features, membership, predictions = [], [], []
    for index in range(820):
        row_id = str(index)
        name = f"Товар {index}"
        card = f"Название: {name}\nОписание: описание"
        span = name
        start = card.index(span)
        features.append({"id": row_id, "category": "БАД", "name": name, "description": "описание"})
        membership.append({
            "id": row_id, "category": "БАД", "semantic_component": f"component-{index}",
            "component_size": "1", "split": "development", "development_fold": str(index % 5),
        })
        predictions.append({
            "id": row_id, "category": "БАД", "fold": str(index % 5), "verdict": "1",
            "evidence": span, "concept": "OBJECT_OF_SALE",
            "explanation": f"Решение основано на товаре: «{span}».",
            "char_start": str(start), "char_end": str(start + len(span)),
        })
    paths = {name: tmp_path / f"{name}.csv" for name in ("features", "membership", "predictions")}
    write_csv(paths["features"], ["id", "category", "name", "description"], features)
    write_csv(paths["membership"], [
        "id", "category", "semantic_component", "component_size", "split", "development_fold"
    ], membership)
    write_csv(paths["predictions"], [
        "id", "category", "fold", "verdict", "evidence", "concept", "explanation", "char_start", "char_end"
    ], predictions)
    exp490 = tmp_path / "exp490.csv"
    exp622 = tmp_path / "exp622.csv"
    prior = tmp_path / "prior.json"
    write_csv(exp490, ["row_id"], [{"row_id": str(i)} for i in range(200)])
    write_csv(exp622, ["row_id"], [{"row_id": str(i)} for i in range(200, 500)])
    prior.write_text(json.dumps([{"row_id": "819"}]), encoding="utf-8")
    private = tmp_path / ".local"
    return paths, exp490, exp622, prior, private


def build_packet(tmp_path: Path):
    paths, exp490, exp622, prior, private = fixture(tmp_path)
    packet, manifest = private / "packet.csv", private / "manifest.json"
    result = builder.build(
        features_path=paths["features"], membership_path=paths["membership"],
        prediction_paths=[paths["predictions"]], exp490_manifest=exp490,
        exp622_manifest=exp622, prior_manifests=[prior], packet_path=packet,
        manifest_path=manifest,
    )
    return packet, manifest, result


def completed_copy(packet: Path, *, strict_passes: int = 300, critical_index: int | None = None) -> Path:
    with packet.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = list(reader.fieldnames or []), [dict(row) for row in reader]
    for index, row in enumerate(rows):
        passed = index < strict_passes and index != critical_index
        row.update({
            "evidence_relevant": "1" if passed else "0",
            "object_of_sale_correct": "1" if passed else "0",
            "negation_correct": "NA", "composition_or_completeness_correct": "NA",
            "verdict_consistent": "1" if passed else "0",
            "unsupported_fact": "1" if index == critical_index else "0",
            "strict_pass": "1" if passed else "0",
            "critical_unsupported": "1" if index == critical_index else "0",
            "scope_or_negation_failure": "0",
            "review_notes": "" if passed else "Проверка не пройдена",
        })
    path = packet.parent / "completed.csv"
    write_csv(path, columns, rows)
    return path


def test_builder_freezes_fresh_unique_development_sample_with_empty_ratings(tmp_path: Path) -> None:
    packet, manifest, result = build_packet(tmp_path)
    assert result["sample_rows"] == 300
    assert result["unique_semantic_components"] == 300
    assert result["overlap_with_excluded_ids"] == 0
    assert result["human_ratings_present"] is False
    assert result["sealed_rows_read"] == 0
    with packet.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(not row[field] for row in rows for field in builder.HUMAN_COLUMNS)
    assert len({row["sample_sha256"] for row in rows}) == 1
    assert json.loads(manifest.read_text())["sample_sha256"] == rows[0]["sample_sha256"]


def test_evaluator_accepts_only_completed_untampered_copy(tmp_path: Path) -> None:
    packet, manifest, _ = build_packet(tmp_path)
    completed = completed_copy(packet)
    result = evaluator.evaluate(
        frozen_packet=packet, completed_copy=completed, frozen_manifest=manifest,
        output=packet.parent / "result.json",
    )
    assert result["decision"] == "GO"
    assert result["strict_pass"] == 300
    assert result["sealed_rows_read"] == 0


def test_completed_failing_audit_is_rejected(tmp_path: Path) -> None:
    packet, manifest, _ = build_packet(tmp_path)
    completed = completed_copy(packet, strict_passes=281)
    result = evaluator.evaluate(
        frozen_packet=packet, completed_copy=completed, frozen_manifest=manifest,
        output=packet.parent / "result.json",
    )
    assert result["decision"] == "NO_GO"
    assert result["status"] == "rejected_by_gate"


def test_evaluator_rejects_immutable_tampering_and_incomplete_ratings(tmp_path: Path) -> None:
    packet, manifest, _ = build_packet(tmp_path)
    completed = completed_copy(packet)
    with completed.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = list(reader.fieldnames or []), [dict(row) for row in reader]
    rows[0]["explanation"] = "Подменено"
    write_csv(completed, columns, rows)
    with pytest.raises(ValueError, match="immutable"):
        evaluator.evaluate(
            frozen_packet=packet, completed_copy=completed, frozen_manifest=manifest,
            output=packet.parent / "result.json",
        )


def test_builder_rejects_sealed_membership(tmp_path: Path) -> None:
    paths, exp490, exp622, prior, private = fixture(tmp_path)
    columns, rows = builder.read_csv(paths["membership"])
    rows[700]["split"] = "sealed"
    write_csv(paths["membership"], columns, rows)
    with pytest.raises(ValueError, match="Sealed/non-development"):
        builder.build(
            features_path=paths["features"], membership_path=paths["membership"],
            prediction_paths=[paths["predictions"]], exp490_manifest=exp490,
            exp622_manifest=exp622, prior_manifests=[prior],
            packet_path=private / "packet.csv", manifest_path=private / "manifest.json",
        )
