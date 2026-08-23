import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUGS = {
    624: "semantic_v3_route_integration",
    625: "semantic_v3_independent_seed",
    626: "semantic_v3_explanation_human_audit",
    627: "semantic_v3_final_recipe_freeze",
    628: "semantic_v3_sealed_holdout_once",
    629: "semantic_v3_full_data_refit",
    630: "semantic_v3_production_submission",
}


def load_contract(experiment_id: int) -> tuple[dict, dict]:
    directory = ROOT / "experiments" / f"{experiment_id}_{SLUGS[experiment_id]}"
    with (directory / "experiment.toml").open("rb") as handle:
        config = tomllib.load(handle)
    metrics = json.loads((directory / "results" / "metrics.json").read_text())
    return config, metrics


ALLOWED_STATUSES = {
    "blocked_by_dependency",
    "confirmed_external_blocker",
    "running",
    "accepted",
    "rejected_by_gate",
    "skipped_by_gate",
    "completed_fallback",
}


def test_all_future_stages_follow_honest_status_invariants() -> None:
    for experiment_id in SLUGS:
        config, metrics = load_contract(experiment_id)
        assert config["id"] == experiment_id
        assert config["status"] in ALLOWED_STATUSES
        assert metrics["experiment_id"] == str(experiment_id)
        assert metrics["status"] == config["status"]
        if config["status"] in {"blocked_by_dependency", "confirmed_external_blocker"}:
            assert metrics["metrics"] == {}
        if config["status"] == "skipped_by_gate":
            assert metrics.get("artifacts", []) == []
            assert metrics.get("sha256") is None


def test_dependency_chain_and_sealed_contract() -> None:
    exp624, _ = load_contract(624)
    assert exp624["dependency"]["requires_any_accepted"] == [621, 622, 623]
    assert exp624["dependency"]["fallback_experiment"] == 603
    assert exp624["dependency"]["all_upstreams_must_be_terminal_before_fallback"] is True
    assert exp624["integration"]["fallback_must_match_experiment_603_manifest"] is True

    exp625, _ = load_contract(625)
    assert exp625["parent_experiment"] == 624
    assert exp625["gate"]["minimum_winning_folds"] == 4
    assert exp625["gate"]["minimum_macro_delta"] == 0.003
    assert exp625["reproduction"]["requires_experiment_624_recipe_manifest_sha256"] is True

    exp626, _ = load_contract(626)
    assert exp626["audit"]["minimum_sample_size"] >= 300
    assert exp626["audit"]["synthetic_human_ratings_allowed"] is False
    assert exp626["audit"]["exclude_experiment_622_rows"] is True
    assert exp626["gate"]["minimum_strict_pass"] >= 282
    assert exp626["gate"]["maximum_critical_unsupported"] == 0
    assert exp626["gate"]["completed_is_not_accepted"] is True

    exp627, _ = load_contract(627)
    assert exp627["freeze"]["exact_recipe_count"] == 1
    assert exp627["freeze"]["post_freeze_recipe_changes_allowed"] is False
    assert exp627["freeze"]["manifest_verifier_required"] is True
    assert exp627["sealed_acceptance"]["policy_frozen_before_reveal"] is True

    exp628, metrics628 = load_contract(628)
    assert exp628["sealed_protocol"]["maximum_evaluations"] == 1
    assert exp628["sealed_protocol"]["post_evaluation_tuning_allowed"] is False
    assert exp628["sealed_protocol"]["second_evaluation_must_fail_closed"] is True
    assert exp628["sealed_protocol"]["public_champion_fallback_experiment"] == 400
    assert exp628["sealed_protocol"]["local_safe_fallback_experiment"] == 603
    assert metrics628["sealed_holdout_used"] is False
    assert metrics628["sealed_evaluation_count"] == 0

    exp629, metrics629 = load_contract(629)
    assert exp629["parent_experiment"] == 628
    assert exp629["uses_sealed_holdout"] is False
    assert exp629["training"]["fits_per_component"] == 1
    assert metrics629["artifacts"] == []

    exp630, metrics630 = load_contract(630)
    assert exp630["parent_experiment"] == 629
    assert exp630["uses_sealed_holdout"] is False
    assert exp630["assembly"]["requires_successful_parent"] is True
    assert exp630["assembly"]["exact_expected_id_set_required"] is True
    assert exp630["assembly"]["exact_substring_offsets_required"] is True
    assert exp630["assembly"]["reasoning_chain"] == [
        "card",
        "exact_substring_or_NO_EVIDENCE",
        "closed_concept",
        "verdict",
        "short_explanation_with_quote",
    ]
    assert metrics630["artifacts"] == []
    assert metrics630["sha256"] is None
