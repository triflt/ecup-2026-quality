from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
PAIR_PATH = ROOT / "evaluate_retrieval_pair_verifier.py"
MAX_HARD_PER_CLASS = 4
SCREEN_FOLDS = (0, 3)
PREREGISTER_SELF_SHA256 = "0762ff0c76064f0ab2d7e842ad9d324aeaaf83d567f035f35a6c3f030670776b"


def load_pair_module():
    spec = importlib.util.spec_from_file_location("exp699_pair_preflight_base", PAIR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen pair-verifier module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_balanced_hard_pairs(
    query: int,
    donors: list[int],
    labels: np.ndarray,
    *,
    maximum: int = MAX_HARD_PER_CLASS,
) -> list[tuple[int, int]]:
    same = [donor for donor in donors if labels[donor] == labels[query]]
    opposite = [donor for donor in donors if labels[donor] != labels[query]]
    count = min(maximum, len(same), len(opposite))
    if count == 0:
        return []
    selected = [(donor, 1) for donor in same[:count]]
    selected.extend((donor, 0) for donor in opposite[:count])
    return selected


def prior_arrays(component_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    source_codes = {"none": 0, "exact": 1, "name": 2}
    sources: list[int] = []
    values: list[int] = []
    for row in component_rows:
        prior = row["ensemble"]["annotator_prior"]
        source = str(prior["source"])
        if source not in source_codes:
            raise ValueError("unsupported frozen prior source")
        value = prior["value"]
        if source == "none":
            if value is not None:
                raise ValueError("none prior carries a value")
            values.append(-1)
        else:
            if value not in {0, 1}:
                raise ValueError("invalid frozen prior value")
            values.append(int(value))
        sources.append(source_codes[source])
    return np.asarray(sources, dtype=np.int8), np.asarray(values, dtype=np.int8)


def v1_harmful_five(path: Path, pair) -> np.ndarray:
    report = json.loads(path.read_text())
    pair.verify_self_hash(report)
    if report.get("schema") != "exp699_soft_cache_exploratory_oof_v1":
        raise ValueError("soft-cache-v1 report schema mismatch")
    harmful = sorted(
        int(row["global_index"])
        for row in report.get("changed_decisions", [])
        if int(row["fold"]) in SCREEN_FOLDS
        if int(row["baseline_after"]) == int(row["label"])
        and int(row["candidate_after"]) != int(row["label"])
    )
    if len(harmful) != 5:
        raise ValueError("soft-cache-v1 screen cohort must have exactly five harmful rows")
    return np.asarray(harmful, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--topology-module", type=Path, required=True)
    parser.add_argument("--component-outputs", type=Path, required=True)
    parser.add_argument("--component-audit", type=Path, required=True)
    parser.add_argument("--fold-runtime", action="append", required=True)
    parser.add_argument("--multimodal-embeddings", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--image-resolved-root", type=Path, required=True)
    parser.add_argument("--v1-report", type=Path, required=True)
    parser.add_argument("--pair-verifier-terminal", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_npz.exists() or args.output_report.exists():
        raise FileExistsError("refusing to overwrite learned-pair preflight")

    pair = load_pair_module()
    runtime_paths = dict(pair.parse_fold_path(value) for value in args.fold_runtime)
    manifest = json.loads(args.source_manifest.read_text())
    manifest_self = pair.verify_self_hash(manifest)
    preregister = json.loads(args.preregister.read_text())
    preregister_self = pair.verify_self_hash(preregister)
    if (
        preregister_self != PREREGISTER_SELF_SHA256
        or preregister.get("schema") != "exp699_retrieval_learned_pair_preregister_v1"
        or preregister.get("source_bindings", {}).get("retrieval_source_manifest_self_sha256")
        != manifest_self
        or preregister.get("public_used") is not False
        or preregister.get("sealed_used") is not False
    ):
        raise ValueError("learned-pair preregistration mismatch")
    bindings = preregister["source_bindings"]
    pair_terminal = json.loads(args.pair_verifier_terminal.read_text())
    pair_terminal_self = pair.verify_self_hash(pair_terminal)
    if (
        pair_terminal_self != bindings["pair_verifier_terminal_self_sha256"]
        or pair_terminal.get("schema") != "exp699_retrieval_pair_verifier_terminal_v1"
        or pair_terminal.get("screen_folds") != [0, 3]
        or pair_terminal.get("public_used") is not False
    ):
        raise ValueError("pair-verifier terminal binding mismatch")
    if pair.sha256_file(args.component_outputs) != bindings["component_outputs_sha256"]:
        raise ValueError("component-output binding mismatch")
    if pair.sha256_file(args.topology) != bindings["topology_sha256"]:
        raise ValueError("topology binding mismatch")
    if pair.sha256_file(args.multimodal_embeddings) != bindings["qwen_train_embeddings_sha256"]:
        raise ValueError("embedding binding mismatch")
    if pair.sha256_file(args.topology_module) != manifest["topology"]["module_sha256"]:
        raise ValueError("topology-module binding mismatch")
    if pair.sha256_file(args.component_audit) != manifest["component_outputs"]["audit_file_sha256"]:
        raise ValueError("component-audit binding mismatch")
    if set(runtime_paths) != set(pair.FOLDS):
        raise ValueError("exact folds0..4 are required")
    for fold, path in runtime_paths.items():
        if pair.sha256_file(path) != manifest["fold_runtime_sha256"][str(fold)]:
            raise ValueError(f"fold{fold} runtime binding mismatch")
    topology = json.loads(args.topology.read_text())
    topology_self = pair.verify_self_hash(topology)
    topology_module = pair.load_module(args.topology_module, "exp699_learned_pair_topology")
    runtime_rows = pair.load_runtime_rows(runtime_paths)
    component_rows = pair.load_component_rows(args.component_outputs)
    registry, runtime_rows, component_rows = pair.align_sources(
        topology, runtime_rows, component_rows
    )

    ids = np.asarray([str(row["id"]) for row in registry], dtype=str)
    folds = np.asarray([int(row["fold"]) for row in registry], dtype=np.int8)
    categories = np.asarray([str(row["category"]) for row in registry], dtype=str)
    components = np.asarray([str(row["component_id"]) for row in registry], dtype=str)
    texts = [topology_module.row_text(row, mask_digits=False) for row in runtime_rows]
    normalized_texts = [topology_module.row_text(row, mask_digits=True) for row in runtime_rows]
    cues = np.vstack([pair.cue_vector(text) for text in texts])

    embedding_source = np.load(args.multimodal_embeddings, allow_pickle=False)
    embedding_ids = embedding_source["ids"].astype(str)
    if len(set(embedding_ids)) != len(embedding_ids):
        raise ValueError("Qwen embedding ids are duplicated")
    position = {value: index for index, value in enumerate(embedding_ids)}
    if any(value not in position for value in ids):
        raise ValueError("Qwen embeddings do not cover source rows")
    embeddings = embedding_source["embeddings"][[position[value] for value in ids]].astype(
        np.float32
    )
    if embeddings.shape != (len(ids), 2048) or not np.isfinite(embeddings).all():
        raise ValueError("Qwen embeddings must be finite 2048-dimensional rows")
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)

    fingerprints: list[dict[str, Any]] = []
    for row_id in ids:
        image_path = topology_module.image_cache_path(args.image_cache, str(row_id))
        topology_module.validate_image_cache_entry(image_path, args.image_resolved_root)
        fingerprints.append(topology_module.image_fingerprints(image_path))

    # Labels enter only after all label-free source features are frozen above.
    labels = np.asarray([int(row["label"]) for row in component_rows], dtype=np.int8)
    baseline_scores = np.asarray(
        [
            float(row["ensemble"]["production_fixed_baseline_before_prior"]["score"])
            for row in component_rows
        ],
        dtype=np.float32,
    )
    baseline_before = np.asarray(
        [
            int(row["ensemble"]["production_fixed_baseline_before_prior"]["prediction"])
            for row in component_rows
        ],
        dtype=np.int8,
    )
    baseline_after = np.asarray(
        [int(row["ensemble"]["production_fixed_baseline_after_prior"]) for row in component_rows],
        dtype=np.int8,
    )
    prior_source, prior_value = prior_arrays(component_rows)
    family_counts = Counter(
        (str(category), str(component)) for category, component in zip(categories, components)
    )
    family_sizes = np.asarray(
        [
            family_counts[(str(category), str(component))]
            for category, component in zip(categories, components)
        ],
        dtype=np.int32,
    )
    harmful_five = v1_harmful_five(args.v1_report, pair)

    arrays: dict[str, np.ndarray] = {
        "ids": ids,
        "folds": folds,
        "categories": categories,
        "components": components,
        "labels": labels,
        "embeddings": embeddings.astype(np.float16),
        "baseline_scores": baseline_scores,
        "baseline_before": baseline_before,
        "baseline_after": baseline_after,
        "prior_source": prior_source,
        "prior_value": prior_value,
        "family_sizes": family_sizes,
        "v1_harmful_five": harmful_five,
    }
    fold_reports: dict[str, Any] = {}
    for outer_fold in SCREEN_FOLDS:
        fused, evidence = pair.build_candidates(
            outer_fold=outer_fold,
            texts=texts,
            normalized_texts=normalized_texts,
            folds=folds,
            categories=categories,
            components=components,
            embeddings=embeddings,
            fingerprints=fingerprints,
            topology_module=topology_module,
        )
        flammable = categories == pair.FLAMMABLE
        outer_train = np.flatnonzero((folds != outer_fold) & flammable)
        heldout = np.flatnonzero((folds == outer_fold) & flammable)
        component_stats = pair.component_statistics(components, labels, outer_train)

        train_queries: list[int] = []
        train_donors: list[int] = []
        train_targets: list[int] = []
        train_weights: list[float] = []
        train_scalar: list[np.ndarray] = []
        retained_queries = 0
        for query in outer_train:
            selected = select_balanced_hard_pairs(int(query), fused[int(query)], labels)
            if not selected:
                continue
            retained_queries += 1
            weight = 1.0 / len(selected)
            for donor, target in selected:
                if folds[donor] == folds[query] or components[donor] == components[query]:
                    raise ValueError("connected-safe training-pair exclusion failed")
                train_queries.append(int(query))
                train_donors.append(donor)
                train_targets.append(target)
                train_weights.append(weight)
                train_scalar.append(
                    pair.pair_feature(
                        int(query),
                        donor,
                        evidence=evidence[int(query)][donor],
                        donor_labels=labels,
                        component_stats=component_stats,
                        components=components,
                        baseline_scores=baseline_scores,
                        baseline_predictions=baseline_before,
                        cues=cues,
                        texts=texts,
                        fingerprints=fingerprints,
                    )
                )

        eval_queries: list[int] = []
        eval_donors: list[int] = []
        eval_scalar: list[np.ndarray] = []
        eval_offsets = [0]
        for query in heldout:
            donors = fused[int(query)]
            for donor in donors:
                if folds[donor] == folds[query] or components[donor] == components[query]:
                    raise ValueError("connected-safe evaluation-pair exclusion failed")
                eval_queries.append(int(query))
                eval_donors.append(donor)
                eval_scalar.append(
                    pair.pair_feature(
                        int(query),
                        donor,
                        evidence=evidence[int(query)][donor],
                        donor_labels=labels,
                        component_stats=component_stats,
                        components=components,
                        baseline_scores=baseline_scores,
                        baseline_predictions=baseline_before,
                        cues=cues,
                        texts=texts,
                        fingerprints=fingerprints,
                    )
                )
            eval_offsets.append(len(eval_donors))

        prefix = f"f{outer_fold}_"
        arrays[prefix + "train_queries"] = np.asarray(train_queries, dtype=np.int64)
        arrays[prefix + "train_donors"] = np.asarray(train_donors, dtype=np.int64)
        arrays[prefix + "train_targets"] = np.asarray(train_targets, dtype=np.int8)
        arrays[prefix + "train_weights"] = np.asarray(train_weights, dtype=np.float32)
        arrays[prefix + "train_scalar"] = np.vstack(train_scalar).astype(np.float32)
        arrays[prefix + "heldout_queries"] = heldout.astype(np.int64)
        arrays[prefix + "eval_queries"] = np.asarray(eval_queries, dtype=np.int64)
        arrays[prefix + "eval_donors"] = np.asarray(eval_donors, dtype=np.int64)
        arrays[prefix + "eval_offsets"] = np.asarray(eval_offsets, dtype=np.int64)
        arrays[prefix + "eval_scalar"] = np.vstack(eval_scalar).astype(np.float32)
        query_features = np.vstack(
            [
                pair.query_feature(
                    int(query),
                    baseline_scores,
                    baseline_before,
                    cues,
                    texts,
                    fingerprints,
                )
                for query in outer_train
            ]
        ).astype(np.float32)
        heldout_features = np.vstack(
            [
                pair.query_feature(
                    int(query),
                    baseline_scores,
                    baseline_before,
                    cues,
                    texts,
                    fingerprints,
                )
                for query in heldout
            ]
        ).astype(np.float32)
        arrays[prefix + "query_train_indices"] = outer_train.astype(np.int64)
        arrays[prefix + "query_train_features"] = query_features
        arrays[prefix + "query_train_targets"] = labels[outer_train]
        arrays[prefix + "query_eval_features"] = heldout_features

        pair_counts = Counter(train_queries)
        target_balance = {
            query: Counter(
                target
                for local_query, target in zip(train_queries, train_targets)
                if local_query == query
            )
            for query in pair_counts
        }
        if any(local[0] != local[1] for local in target_balance.values()):
            raise ValueError("per-query targets are not balanced")
        fold_reports[str(outer_fold)] = {
            "outer_train_queries": len(outer_train),
            "retained_balanced_queries": retained_queries,
            "train_pairs": len(train_queries),
            "train_positive_pairs": int(sum(train_targets)),
            "train_negative_pairs": int(len(train_targets) - sum(train_targets)),
            "heldout_queries": len(heldout),
            "evaluation_pairs": len(eval_donors),
            "same_fold_pairs": 0,
            "same_component_pairs": 0,
            "per_query_balanced": True,
        }

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **arrays)
    npz_sha = pair.sha256_file(args.output_npz)
    bytes_in_memory = int(sum(value.nbytes for value in arrays.values()))
    report = {
        "schema": "exp699_learned_pair_preflight_v1",
        "experiment": 699,
        "decision": "ACCEPT_LEARNED_PAIR_PREFLIGHT",
        "public_used": False,
        "sealed_rows": 0,
        "gpu_used": 0,
        "candidate_generation_labels_used": False,
        "labels_entered_pair_selection_after_top30_frozen": True,
        "embedding_dimensions": 2048,
        "embedding_finite": True,
        "screen_folds": [0, 3],
        "hard_pair_selection": "top-ranked min(4,same,opposite) per class; both classes required",
        "folds": fold_reports,
        "packet": {
            "file_sha256": npz_sha,
            "bytes": args.output_npz.stat().st_size,
            "uncompressed_array_bytes": bytes_in_memory,
        },
        "source_bindings": {
            "source_manifest_self_sha256": manifest_self,
            "preregister_self_sha256": preregister_self,
            "topology_self_sha256": topology_self,
            "component_outputs_sha256": pair.sha256_file(args.component_outputs),
            "qwen_embeddings_sha256": pair.sha256_file(args.multimodal_embeddings),
            "preflight_code_sha256": pair.sha256_file(Path(__file__).resolve()),
            "pair_verifier_code_sha256": pair.sha256_file(PAIR_PATH),
            "pair_verifier_terminal_file_sha256": pair.sha256_file(args.pair_verifier_terminal),
            "pair_verifier_terminal_self_sha256": pair_terminal_self,
            "baseline_semantics": manifest["baseline_semantics"],
            "candidate_name": preregister["candidate_name"],
        },
    }
    file_sha, self_sha = pair.write_self_hashed(args.output_report, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "npz_sha256": npz_sha,
                "report_file_sha256": file_sha,
                "report_self_sha256": self_sha,
                "folds": fold_reports,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
