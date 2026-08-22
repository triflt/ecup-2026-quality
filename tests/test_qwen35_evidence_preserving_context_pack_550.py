from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tomllib
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/550_qwen35_evidence_preserving_context_pack"


def _load(name: str, filename: str, aliases: dict[str, object] | None = None):
    previous = {alias: sys.modules.get(alias) for alias in aliases or {}}
    try:
        for alias, module in (aliases or {}).items():
            sys.modules[alias] = module
        spec = importlib.util.spec_from_file_location(name, EXPERIMENT / filename)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for alias, module in previous.items():
            if module is None:
                sys.modules.pop(alias, None)
            else:
                sys.modules[alias] = module


context = _load("exp550_context_pack", "context_pack.py")
contract = _load("exp550_contract", "contract.py")
parent_recipe = _load("exp550_parent_recipe", "parent_recipe.py", {"contract": contract})
evaluator = _load("exp550_evaluator", "evaluate_screen.py", {"contract": contract})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _long_description() -> str:
    head = "Начальные технические сведения " + "общая информация " * 90
    evidence = "В комплект входит газовый баллон."
    tail = " Условия эксплуатации " + "дополнительные сведения " * 55
    return head + evidence + tail


def test_candidate_retains_middle_evidence_with_exact_provenance() -> None:
    description = _long_description()
    packed = context.pack_description(
        row_id="x",
        category="Легковоспламеняющиеся",
        name="Набор",
        description=description,
    )
    repeated = context.pack_description(
        row_id="x",
        category="Легковоспламеняющиеся",
        name="Набор",
        description=description,
    )
    surface = context.evidence.surface_text(description).text
    baseline = context.baseline_positions(description)

    assert packed == repeated
    assert packed.packable is True
    assert len(packed.text) <= 1800
    assert packed.text == "".join(surface[position] for position in packed.source_positions)
    assert len(set(packed.source_positions)) == len(packed.source_positions)
    raw_map = context.evidence.surface_text(description).raw_intervals
    assert packed.raw_intervals == tuple(raw_map[position] for position in packed.source_positions)
    assert context.retained_evidence(baseline, packed.evidence_spans) == 0
    assert context.retained_evidence(packed.source_positions, packed.evidence_spans) == 1
    span = packed.evidence_spans[0]
    assert all(
        position in set(packed.source_positions)
        for position in range(span.sentence_start, span.sentence_end)
    )


def test_conflict_is_excluded_and_short_description_is_unchanged() -> None:
    description = "В комплект входит газовый баллон. Газовый баллон не входит в комплект."
    packed = context.pack_description(
        row_id="conflict",
        category="Легковоспламеняющиеся",
        name="Набор",
        description=description,
    )

    assert packed.evidence_spans == ()
    assert "conflicting_evidence" in str(packed.blocked_reason)
    assert packed.text == context.evidence.surface_text(description).text
    assert packed.source_positions == tuple(range(len(packed.text)))
    assert len(packed.raw_intervals) == len(packed.text)


def test_packing_selection_ignores_label_and_fold_columns() -> None:
    source = pd.DataFrame(
        {
            "id": ["x"],
            "category": ["Легковоспламеняющиеся"],
            "name": ["Набор"],
            "description": [_long_description()],
            "label": [0],
            "fold": [0],
        }
    )
    changed = source.assign(label=1, fold=4)
    first = context.pack_description(
        row_id=source.loc[0, "id"],
        category=source.loc[0, "category"],
        name=source.loc[0, "name"],
        description=source.loc[0, "description"],
    )
    second = context.pack_description(
        row_id=changed.loc[0, "id"],
        category=changed.loc[0, "category"],
        name=changed.loc[0, "name"],
        description=changed.loc[0, "description"],
    )

    assert first == second


def test_checked_in_full_audit_passes_every_predeclared_gate() -> None:
    audit_path = EXPERIMENT / "analysis/context_pack_audit.json"
    rows_path = EXPERIMENT / "analysis/context_pack_rows.csv.gz"
    audit = json.loads(audit_path.read_text())

    assert _sha256(audit_path) == contract.AUDIT_SHA256
    assert _sha256(rows_path) == contract.AUDIT_ROWS_SHA256
    assert audit["status"] == "GO"
    assert audit["labels_read"] is False
    assert audit["folds_read"] is False
    assert audit["columns_read"] == ["id", "category", "name", "description"]
    assert audit["long_evidence_rows"] == 2059
    assert audit["long_evidence_rows_strictly_improved"] == 390
    assert audit["strict_improvement_fraction"] == pytest.approx(0.1894123360854784)
    assert audit["long_evidence_rows_regressed"] == 0
    assert all(audit["gates"].values())
    assert rows_path.read_bytes()[4:8] == b"\x00\x00\x00\x00"
    rows = pd.read_csv(rows_path, compression="gzip")
    assert rows.provenance_ok.all()
    assert rows.deterministic.all()
    assert rows.budget_ok.all()
    assert rows.complete_evidence_sentences.all()


def test_parent_environment_is_locked_to_folds_zero_and_three(tmp_path: Path) -> None:
    environment: dict[str, str] = {}
    parent_recipe.configure_parent_environment(fold=0, output_dir=tmp_path, environment=environment)
    assert environment["HOLDOUT_FOLD"] == "0"
    assert environment["FULL_TRAIN"] == "0"
    assert environment["DESCRIPTION_LIMIT"] == "1800"
    assert environment["FAMILY_DIVERSE_BAD_POSITIVES"] == "1"
    assert environment["FAMILY_DIVERSE_FLAMMABLE_NEGATIVES"] == "0"
    with pytest.raises(ValueError, match="predeclared"):
        parent_recipe.configure_parent_environment(fold=1, output_dir=tmp_path, environment={})
    with pytest.raises(ValueError, match="must remain"):
        parent_recipe.configure_parent_environment(
            fold=3,
            output_dir=tmp_path,
            environment={"DESCRIPTION_LIMIT": "2000"},
        )


def test_context_report_proves_parent_recipe_and_pack_audit(tmp_path: Path) -> None:
    parent_report_path = tmp_path / "lora_holdout_report.json"
    parent_report = {"holdout_fold": 0, "train_records": 5390, "download_failures": 0}
    parent_report_path.write_text(json.dumps(parent_report))
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "use_rslora": True,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            }
        )
    )
    report = {
        "experiment_id": "550",
        "status": "screen_fold_complete",
        "holdout_fold": 0,
        "single_changed_factor": "description packing only",
        "description_budget": 1800,
        "packing_audit_version": contract.AUDIT_VERSION,
        "packing_audit_sha256": contract.AUDIT_SHA256,
        "packing_rows_sha256": contract.AUDIT_ROWS_SHA256,
        "extractor_sha256": contract.EXTRACTOR_SHA256,
        "vocabulary_sha256": contract.VOCABULARY_SHA256,
        "selection_uses_labels": False,
        "selection_uses_folds": False,
        "prediction_source": "constant union of frozen_prediction=0 and 1",
        "unique_runtime_rows_packed": 10,
        "parent_environment": contract.PARENT_ENVIRONMENT,
        "parent_train_records": 5390,
        "parent_optimizer_updates": 337,
        "parent_holdout_report_sha256": _sha256(parent_report_path),
    }
    report_path = tmp_path / "context_pack_runtime_report.json"
    report_path.write_text(json.dumps(report))

    validated = evaluator.validate_context_report(report_path, fold=0)
    assert validated["description_budget"] == 1800
    report["description_budget"] = 1801
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="description_budget"):
        evaluator.validate_context_report(report_path, fold=0)


def test_null_control_and_experiment_card_are_frozen_not_run() -> None:
    config = tomllib.loads((EXPERIMENT / "experiment.toml").read_text())
    metrics = json.loads((EXPERIMENT / "results/metrics.json").read_text())
    null = json.loads((EXPERIMENT / "analysis/null_screen_control_audit.json").read_text())

    assert config["validation"]["screen_folds"] == [0, 3]
    assert config["packing"]["description_budget"] == 1800
    assert config["packing"]["uses_gold_label"] is False
    assert config["packing"]["uses_model_prediction"] is False
    assert config["packing"]["allows_invented_chars"] is False
    assert config["execution"]["launch_authorized"] is False
    assert metrics["launched"] is False
    assert null["null_control_passed"] is True
    assert null["mean_screen_delta_macro_f1"] == 0.0
    assert null["bad_changed_predictions"] == 0
