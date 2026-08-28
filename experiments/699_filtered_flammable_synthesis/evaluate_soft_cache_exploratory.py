from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
RECIPE_SELF_SHA256 = "dbc0d666d4e9e1ce93f46034c393fcc67fe082732c6e8bd5ccf786cfb5914ff9"
SOURCE_MANIFEST_SELF_SHA256 = (
    "b06f3e3d98637239b227ae946ca690c2c65dc72addfd71fe8fb8801bd24767a0"
)
FLAMMABLE_THRESHOLD = 0.953912615776062
CACHE_ALPHA = 0.25
CACHE_TOP_K = 5
MINIMUM_DONORS = 3
MINIMUM_AGREEMENT = 0.8
MINIMUM_CHANNELS = 2
DEPLOYABLE_CHANNELS = (
    "exact_text",
    "normalized_text",
    "tfidf",
    "bm25",
    "image_exact",
    "image_near",
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fuse_with_evidence(
    channel_lists: dict[str, list[list[int]]],
    channel_weights: dict[str, float],
    row_count: int,
    top_k: int,
    rrf_offset: int,
) -> tuple[list[list[int]], list[dict[int, dict[str, Any]]]]:
    fused: list[list[int]] = []
    evidence: list[dict[int, dict[str, Any]]] = []
    for query in range(row_count):
        scores: dict[int, float] = defaultdict(float)
        channels: dict[int, set[str]] = defaultdict(set)
        best_rank: dict[int, int] = {}
        for channel, rows in channel_lists.items():
            weight = float(channel_weights[channel])
            for rank, donor in enumerate(rows[query], 1):
                donor = int(donor)
                scores[donor] += weight / (rrf_offset + rank)
                channels[donor].add(channel)
                best_rank[donor] = min(best_rank.get(donor, rank), rank)
        ordered = sorted(
            scores,
            key=lambda donor: (
                -scores[donor],
                -len(channels[donor]),
                best_rank[donor],
                donor,
            ),
        )[:top_k]
        fused.append(ordered)
        evidence.append(
            {
                donor: {
                    "rrf_score": float(scores[donor]),
                    "channels": sorted(channels[donor]),
                }
                for donor in ordered
            }
        )
    return fused, evidence


def apply_soft_cache(
    labels: np.ndarray,
    categories: np.ndarray,
    baseline_scores: np.ndarray,
    baseline_predictions: np.ndarray,
    fused: list[list[int]],
    evidence: list[dict[int, dict[str, Any]]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    candidate_scores = baseline_scores.copy()
    candidate_predictions = baseline_predictions.copy()
    audit: list[dict[str, Any]] = []
    for query in np.flatnonzero(categories == "Легковоспламеняющиеся"):
        donors = fused[query][:CACHE_TOP_K]
        if len(donors) < MINIMUM_DONORS:
            continue
        weights = np.asarray(
            [float(evidence[query][donor]["rrf_score"]) for donor in donors],
            dtype=np.float64,
        )
        donor_labels = labels[donors].astype(np.float64)
        positive_rate = float(np.average(donor_labels, weights=weights))
        agreement = max(positive_rate, 1.0 - positive_rate)
        evidence_channels = sorted(
            {
                channel
                for donor in donors
                for channel in evidence[query][donor]["channels"]
            }
        )
        exact_text_present = "exact_text" in evidence_channels
        gate = agreement >= MINIMUM_AGREEMENT and (
            len(evidence_channels) >= MINIMUM_CHANNELS or exact_text_present
        )
        if not gate:
            continue
        old_score = float(baseline_scores[query])
        new_score = (1.0 - CACHE_ALPHA) * old_score + CACHE_ALPHA * positive_rate
        candidate_scores[query] = new_score
        candidate_predictions[query] = int(new_score >= FLAMMABLE_THRESHOLD)
        audit.append(
            {
                "global_index": int(query),
                "donors": [int(value) for value in donors],
                "donor_positive_rate": positive_rate,
                "donor_agreement": agreement,
                "evidence_channels": evidence_channels,
                "baseline_score": old_score,
                "candidate_score": new_score,
                "baseline_prediction": int(baseline_predictions[query]),
                "candidate_prediction": int(candidate_predictions[query]),
            }
        )
    return candidate_scores, candidate_predictions, audit


def comparison(
    labels: np.ndarray,
    categories: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    changed = mask & (baseline != candidate)
    corrected = changed & (candidate == labels)
    regressed = changed & (candidate != labels)
    result = {
        "rows": int(mask.sum()),
        "changed": int(changed.sum()),
        "corrections": int(corrected.sum()),
        "regressions": int(regressed.sum()),
        "net": int(corrected.sum() - regressed.sum()),
        "by_category": {},
    }
    for category in ("БАД", "Легковоспламеняющиеся"):
        local = categories == category
        result["by_category"][category] = {
            "changed": int((changed & local).sum()),
            "corrections": int((corrected & local).sum()),
            "regressions": int((regressed & local).sum()),
        }
    return result


def validate_source_manifest(args: argparse.Namespace, oracle) -> tuple[dict[str, Any], str]:
    manifest = json.loads(args.source_manifest.read_text())
    manifest_self = oracle.verify_self_hash(manifest)
    if (
        manifest_self != SOURCE_MANIFEST_SELF_SHA256
        or manifest.get("schema") != "exp699_retrieval_source_manifest_v1"
        or manifest.get("decision")
        != "ACCEPT_INPUTS_FOR_DIAGNOSTIC_AND_PACKAGE_PARITY_BUILD"
        or manifest.get("public_used") is not False
    ):
        raise ValueError("source manifest contract mismatch")

    source_manifest = json.loads(args.solution140_source_manifest.read_text())
    source_manifest_self = oracle.verify_existing_self_hash(source_manifest)
    source = manifest["solution140_source"]
    actual_source = {
        "run_sha256": oracle.sha256_file(args.solution140_source / "run.py"),
        "metadata_sha256": oracle.sha256_file(
            args.solution140_source / "metadata.json"
        ),
        "annotator_prior_sha256": oracle.sha256_file(
            args.solution140_source / "annotator_prior.json.gz"
        ),
        "manifest_file_sha256": oracle.sha256_file(
            args.solution140_source_manifest
        ),
        "manifest_self_sha256": source_manifest_self,
    }
    if source != actual_source:
        raise ValueError("solution140 source binding mismatch")
    if (
        source_manifest.get("schema") != "exp699_solution140_source_manifest_v1"
        or source_manifest.get("decision") != "ACCEPT_SOURCE"
        or source_manifest.get("files", {}).get("run.py") != source["run_sha256"]
    ):
        raise ValueError("solution140 source manifest mismatch")

    component = manifest["component_outputs"]
    if component != {
        "file_sha256": oracle.sha256_file(args.component_outputs),
        "audit_file_sha256": oracle.sha256_file(args.component_audit),
        "audit_self_sha256": oracle.verify_existing_self_hash(
            json.loads(args.component_audit.read_text())
        ),
        "rows": sum(1 for _ in args.component_outputs.open()),
    }:
        raise ValueError("component source binding mismatch")
    topology = json.loads(args.topology.read_text())
    if manifest["topology"] != {
        "file_sha256": oracle.sha256_file(args.topology),
        "self_sha256": oracle.verify_self_hash(topology),
        "module_sha256": oracle.sha256_file(args.topology_module),
    }:
        raise ValueError("topology source binding mismatch")
    runtime_paths = dict(oracle.parse_fold_path(value) for value in args.fold_runtime)
    if manifest["fold_runtime_sha256"] != {
        str(fold): oracle.sha256_file(path) for fold, path in runtime_paths.items()
    }:
        raise ValueError("fold runtime binding mismatch")
    expected_packages = {
        "python": ".".join(__import__("sys").version.split()[0].split(".")),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "pillow": importlib.metadata.version("Pillow"),
    }
    if manifest["runtime_packages"] != expected_packages:
        raise ValueError("runtime package binding mismatch")
    return manifest, manifest_self


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--topology-module", type=Path, required=True)
    parser.add_argument("--component-outputs", type=Path, required=True)
    parser.add_argument("--component-audit", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--solution140-source", type=Path, required=True)
    parser.add_argument("--solution140-source-manifest", type=Path, required=True)
    parser.add_argument("--fold-runtime", action="append", required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--image-resolved-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite exploratory evaluation")

    oracle = load_module(
        ROOT / "evaluate_retrieval_oracle.py", "exp699_oracle_for_soft_cache"
    )
    source_manifest, source_manifest_self = validate_source_manifest(args, oracle)
    preregister = json.loads(args.preregister.read_text())
    if (
        oracle.verify_self_hash(preregister) != RECIPE_SELF_SHA256
        or preregister.get("schema") != "exp699_retrieval_exploratory_preregister_v3"
    ):
        raise ValueError("frozen recipe contract mismatch")
    recipe = preregister["frozen_recipe"]
    expected = {
        "cache_top_k": CACHE_TOP_K,
        "minimum_unique_donors": MINIMUM_DONORS,
        "minimum_donor_label_agreement": MINIMUM_AGREEMENT,
        "minimum_evidence_channels": MINIMUM_CHANNELS,
        "cache_alpha": CACHE_ALPHA,
        "threshold": FLAMMABLE_THRESHOLD,
    }
    for key, value in expected.items():
        if recipe.get(key) != value:
            raise ValueError(f"frozen recipe mismatch: {key}")

    topology = json.loads(args.topology.read_text())
    topology_self = oracle.verify_self_hash(topology)
    topology_module = load_module(args.topology_module, "exp699_topology_for_cache")
    runtime_paths = dict(oracle.parse_fold_path(value) for value in args.fold_runtime)
    runtime_rows = oracle.load_runtime_rows(runtime_paths)
    component_rows = oracle.load_component_outputs(args.component_outputs)
    registry, runtime_rows, component_rows = oracle.align_sources(
        topology, runtime_rows, component_rows
    )
    component_audit = json.loads(args.component_audit.read_text())
    component_audit_self = oracle.verify_existing_self_hash(component_audit)
    if (
        component_audit.get("schema") != "exp699_component_output_audit_v1"
        or component_audit.get("public_used") is not False
        or type(component_audit.get("gpu_used")) is not int
        or component_audit.get("gpu_used") != 0
        or type(component_audit.get("local_downloads")) is not int
        or component_audit.get("local_downloads") != 0
        or component_audit.get("rows") != len(component_rows)
    ):
        raise ValueError("component-output audit contract mismatch")

    row_count = len(registry)
    ids = np.asarray([str(row["id"]) for row in registry], dtype=str)
    folds = np.asarray([int(row["fold"]) for row in registry], dtype=np.int8)
    categories = np.asarray([str(row["category"]) for row in registry], dtype=str)
    texts = [topology_module.row_text(row, mask_digits=False) for row in runtime_rows]
    normalized_texts = [
        topology_module.row_text(row, mask_digits=True) for row in runtime_rows
    ]
    channel_lists = {name: [[] for _ in range(row_count)] for name in DEPLOYABLE_CHANNELS}
    channel_lists["exact_text"] = oracle.group_candidates(
        texts, folds, categories, oracle.TOP_K
    )
    channel_lists["normalized_text"] = oracle.group_candidates(
        normalized_texts, folds, categories, oracle.TOP_K
    )
    oracle.add_fold_sparse_channels(
        channel_lists, texts, folds, categories, oracle.TOP_K
    )
    image_audit = oracle.add_image_channels(
        channel_lists,
        ids,
        folds,
        categories,
        args.image_cache,
        args.image_resolved_root,
        topology_module,
        oracle.TOP_K,
    )
    channel_weights = {
        "exact_text": 2.0,
        "normalized_text": 1.5,
        "tfidf": 1.0,
        "bm25": 1.0,
        "image_exact": 2.0,
        "image_near": 1.0,
    }
    fused, evidence = fuse_with_evidence(
        channel_lists,
        channel_weights,
        row_count,
        oracle.TOP_K,
        oracle.RRF_OFFSET,
    )

    # Labels are accessed only after the frozen candidate lists exist.
    labels = np.asarray([int(row["label"]) for row in component_rows], dtype=np.int8)
    baseline_scores = np.asarray(
        [
            float(row["ensemble"]["production_fixed_baseline_before_prior"]["score"])
            for row in component_rows
        ],
        dtype=np.float64,
    )
    baseline_before = np.asarray(
        [
            int(row["ensemble"]["production_fixed_baseline_before_prior"]["prediction"])
            for row in component_rows
        ],
        dtype=np.int8,
    )
    baseline_after = np.asarray(
        [int(row["ensemble"]["production_fixed_baseline_after_prior"])
         for row in component_rows],
        dtype=np.int8,
    )
    flammable = categories == "Легковоспламеняющиеся"
    if not np.array_equal(
        baseline_before[flammable],
        (baseline_scores[flammable] >= FLAMMABLE_THRESHOLD).astype(np.int8),
    ):
        raise ValueError("frozen flammable threshold replay mismatch")
    replay_after, baseline_prior_audit = oracle.apply_frozen_annotator_prior(
        component_rows, baseline_before
    )
    if not np.array_equal(replay_after, baseline_after):
        raise ValueError("frozen production prior replay mismatch")

    candidate_scores, candidate_before, cache_audit = apply_soft_cache(
        labels,
        categories,
        baseline_scores,
        baseline_before,
        fused,
        evidence,
    )
    if not np.array_equal(candidate_before[categories == "БАД"], baseline_before[categories == "БАД"]):
        raise ValueError("BAD route drifted before prior")
    candidate_after, candidate_prior_audit = oracle.apply_frozen_annotator_prior(
        component_rows, candidate_before
    )
    if not np.array_equal(candidate_after[categories == "БАД"], baseline_after[categories == "БАД"]):
        raise ValueError("BAD route drifted after prior")

    family_counts = Counter(
        (str(row["category"]), str(row["semantic_component"]))
        for row in component_rows
    )
    family_sizes = np.asarray(
        [
            family_counts[(str(row["category"]), str(row["semantic_component"]))]
            for row in component_rows
        ],
        dtype=np.int32,
    )
    recurrence = np.asarray(
        [str(row["regime"]) == "recurrence_cross_fold" for row in registry]
    )
    masks = {
        "mixed_all": np.ones(row_count, dtype=bool),
        "recurrence": recurrence,
        "novel_family": ~recurrence,
        "singleton": family_sizes == 1,
        "rare_le2": family_sizes <= 2,
        "repeated": family_sizes > 2,
        "flammable_fp": flammable & (labels == 0) & (baseline_after == 1),
        "flammable_fn": flammable & (labels == 1) & (baseline_after == 0),
    }
    comparisons = {
        name: {
            "before_prior": comparison(
                labels, categories, baseline_before, candidate_before, mask
            ),
            "after_prior": comparison(
                labels, categories, baseline_after, candidate_after, mask
            ),
        }
        for name, mask in masks.items()
    }
    changed_after = np.flatnonzero(candidate_after != baseline_after)
    audit_by_index = {int(row["global_index"]): row for row in cache_audit}
    changed_decisions = []
    for index in changed_after:
        cache = audit_by_index.get(int(index))
        changed_decisions.append(
            {
                "global_index": int(index),
                "id": str(ids[index]),
                "fold": int(folds[index]),
                "category": str(categories[index]),
                "label": int(labels[index]),
                "baseline_before": int(baseline_before[index]),
                "candidate_before": int(candidate_before[index]),
                "baseline_after": int(baseline_after[index]),
                "candidate_after": int(candidate_after[index]),
                "baseline_score": float(baseline_scores[index]),
                "candidate_score": float(candidate_scores[index]),
                "donor_positive_rate": (
                    None if cache is None else float(cache["donor_positive_rate"])
                ),
                "evidence_channels": (
                    [] if cache is None else list(cache["evidence_channels"])
                ),
            }
        )

    baseline_metrics = oracle.metric_summary(labels, categories, baseline_after)
    candidate_metrics = oracle.metric_summary(labels, categories, candidate_after)
    payload = {
        "schema": "exp699_soft_cache_exploratory_oof_v1",
        "experiment": 699,
        "decision": (
            "EXPLORATORY_READY_FOR_PACKAGE"
            if len(changed_decisions) > 0
            else "NO_GO_FACTUALLY_ZERO_FINAL_CHANGE"
        ),
        "status": "diagnostic_not_validated_champion",
        "public_used": False,
        "sealed_rows": 0,
        "candidate_generation_labels_read": 0,
        "labels_entered_after_candidate_lists_frozen": True,
        "preregister_self_sha256": RECIPE_SELF_SHA256,
        "recipe": preregister["frozen_recipe"],
        "rows": row_count,
        "cache_gate_rows": len(cache_audit),
        "final_changed_rows": len(changed_decisions),
        "baseline_after_prior": baseline_metrics,
        "candidate_after_prior": candidate_metrics,
        "macro_delta": candidate_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "comparisons": comparisons,
        "changed_decisions": changed_decisions,
        "baseline_prior_audit": baseline_prior_audit,
        "candidate_prior_audit": candidate_prior_audit,
        "image_audit": image_audit,
        "source_bindings": {
            "source_manifest_file_sha256": oracle.sha256_file(args.source_manifest),
            "source_manifest_self_sha256": source_manifest_self,
            "solution140_source_manifest_file_sha256": oracle.sha256_file(
                args.solution140_source_manifest
            ),
            "solution140_source_manifest_self_sha256": source_manifest[
                "solution140_source"
            ]["manifest_self_sha256"],
            "solution140_run_sha256": source_manifest["solution140_source"][
                "run_sha256"
            ],
            "evaluator_sha256": oracle.sha256_file(Path(__file__)),
            "preregister_file_sha256": oracle.sha256_file(args.preregister),
            "topology_file_sha256": oracle.sha256_file(args.topology),
            "topology_self_sha256": topology_self,
            "topology_module_sha256": oracle.sha256_file(args.topology_module),
            "component_outputs_sha256": oracle.sha256_file(args.component_outputs),
            "component_audit_file_sha256": oracle.sha256_file(args.component_audit),
            "component_audit_self_sha256": component_audit_self,
            "fold_runtime_sha256": {
                str(fold): oracle.sha256_file(path)
                for fold, path in runtime_paths.items()
            },
        },
        "packaging_authorized_even_if_macro_negative": True,
        "ods_submit_authorized": False,
    }
    file_sha, self_sha = oracle.write_self_hashed(args.output, payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "file_sha256": file_sha,
                "self_sha256": self_sha,
                "cache_gate_rows": len(cache_audit),
                "final_changed_rows": len(changed_decisions),
                "macro_delta": payload["macro_delta"],
                "corrections": comparisons["mixed_all"]["after_prior"]["corrections"],
                "regressions": comparisons["mixed_all"]["after_prior"]["regressions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
