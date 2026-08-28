from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import stat
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import soft_cache_runtime as runtime


EXPECTED_PREREGISTER_SELF_SHA256 = runtime.PREREGISTER_SELF_SHA256
EXPECTED_PREREGISTER_FILE_SHA256 = (
    "85ed3fa0a8916d6c25ef5ee5774b0f0dd807dbc01f47524b5e3fe27bb88e316e"
)
EXPECTED_SOURCE_MANIFEST_SELF_SHA256 = runtime.SOURCE_MANIFEST_SELF_SHA256
EXPECTED_SOLUTION140_RUN_SHA256 = (
    "9318db4cc0f26eabaa73ae9856dbf9daffd06787a162e6f5db71f249f6db7146"
)
EXPECTED_SOLUTION140_MANIFEST_FILE_SHA256 = (
    "f60f100bd8cce488a0538a5e9a3df80cb775f781d8b0f29fd0976aee757485c4"
)
EXPECTED_SOLUTION140_MANIFEST_SELF_SHA256 = (
    "ce31287e20068b010dece380ec66c2090f147fe465978c3e66cf17fe13fa8bbe"
)
EXPECTED_METADATA_SHA256 = (
    "f1c93f797b285b31c28d3fdb981a64fd0e223552fbfc81b55869eea9ab037303"
)
EXPECTED_PRIOR_SHA256 = (
    "0b08c0d2397455994b61c66e32bbdff5d60445a8126ad0d889fce127df56a91e"
)
EXPECTED_ROWS = 11118
EXPECTED_DONORS = 4716
FOLDS = (0, 1, 2, 3, 4)
EXCLUDED_PARTS = {"__pycache__", ".git", ".pytest_cache", ".DS_Store"}
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_self_hashed(
    path: Path, expected_schema: str | None
) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"contract must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(value)
    declared = str(payload.pop("self_sha256", ""))
    if declared != canonical_sha256(payload) or (
        expected_schema is not None and value.get("schema") != expected_schema
    ):
        raise ValueError(f"self-hashed contract mismatch: {path}")
    return value, declared


def regular_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"source/package member is symlinked: {relative}")
        if path.is_file():
            result[relative.as_posix()] = path
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSONL input must be a regular file: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def parse_fold_path(value: str) -> tuple[int, Path]:
    raw_fold, raw_path = value.split("=", 1)
    fold = int(raw_fold)
    if fold not in FOLDS:
        raise ValueError("fold paths must bind exact folds0..4")
    return fold, Path(raw_path)


def validate_contracts(args: argparse.Namespace) -> dict[str, Any]:
    preregister, preregister_self = load_self_hashed(
        args.preregister, "exp699_retrieval_exploratory_preregister_v3"
    )
    if (
        preregister_self != EXPECTED_PREREGISTER_SELF_SHA256
        or sha256_file(args.preregister) != EXPECTED_PREREGISTER_FILE_SHA256
        or preregister.get("stage") != "frozen_before_soft_cache_metrics"
        or preregister.get("public_used_for_recipe") is not False
        or preregister.get("frozen_recipe", {}).get("candidate_channels")
        != {
            "exact_text": 2.0,
            "normalized_text": 1.5,
            "tfidf": 1.0,
            "bm25": 1.0,
            "first_image_exact_bytes": 2.0,
            "first_image_near_hash": 1.0,
        }
    ):
        raise ValueError("frozen preregistration binding mismatch")

    source_manifest, source_manifest_self = load_self_hashed(
        args.source_manifest, "exp699_retrieval_source_manifest_v1"
    )
    if (
        source_manifest_self != EXPECTED_SOURCE_MANIFEST_SELF_SHA256
        or source_manifest.get("decision")
        != "ACCEPT_INPUTS_FOR_DIAGNOSTIC_AND_PACKAGE_PARITY_BUILD"
        or source_manifest.get("public_used") is not False
        or source_manifest.get("preregister")
        != {
            "file_sha256": EXPECTED_PREREGISTER_FILE_SHA256,
            "self_sha256": EXPECTED_PREREGISTER_SELF_SHA256,
        }
    ):
        raise ValueError("retrieval source manifest mismatch")

    solution_manifest, solution_manifest_self = load_self_hashed(
        args.solution140_source_manifest, "exp699_solution140_source_manifest_v1"
    )
    if (
        sha256_file(args.solution140_source_manifest)
        != EXPECTED_SOLUTION140_MANIFEST_FILE_SHA256
        or solution_manifest_self != EXPECTED_SOLUTION140_MANIFEST_SELF_SHA256
        or solution_manifest.get("decision") != "ACCEPT_SOURCE"
    ):
        raise ValueError("solution140 source manifest binding mismatch")
    source_files = regular_files(args.solution140_source)
    declared_files = solution_manifest.get("files")
    if not isinstance(declared_files, dict) or set(source_files) != set(declared_files):
        raise ValueError("solution140 source inventory mismatch")
    actual_files = {
        relative: sha256_file(path) for relative, path in source_files.items()
    }
    if actual_files != declared_files:
        raise ValueError("solution140 source member SHA mismatch")
    expected_source = {
        "run_sha256": EXPECTED_SOLUTION140_RUN_SHA256,
        "metadata_sha256": EXPECTED_METADATA_SHA256,
        "annotator_prior_sha256": EXPECTED_PRIOR_SHA256,
        "manifest_file_sha256": EXPECTED_SOLUTION140_MANIFEST_FILE_SHA256,
        "manifest_self_sha256": EXPECTED_SOLUTION140_MANIFEST_SELF_SHA256,
    }
    if source_manifest.get("solution140_source") != expected_source:
        raise ValueError("source-manifest solution140 binding mismatch")

    component_audit, component_audit_self = load_self_hashed(
        args.component_audit, "exp699_component_output_audit_v1"
    )
    component_rows = sum(1 for _ in args.component_outputs.open(encoding="utf-8"))
    expected_component = {
        "file_sha256": sha256_file(args.component_outputs),
        "audit_file_sha256": sha256_file(args.component_audit),
        "audit_self_sha256": component_audit_self,
        "rows": component_rows,
    }
    if (
        source_manifest.get("component_outputs") != expected_component
        or component_audit.get("public_used") is not False
        or component_audit.get("gpu_used") != 0
        or component_audit.get("local_downloads") != 0
        or component_rows != EXPECTED_ROWS
    ):
        raise ValueError("component-output source binding mismatch")

    topology, topology_self = load_self_hashed(args.topology, None)
    expected_topology = {
        "file_sha256": sha256_file(args.topology),
        "self_sha256": topology_self,
        "module_sha256": sha256_file(args.topology_module),
    }
    if source_manifest.get("topology") != expected_topology:
        raise ValueError("topology source binding mismatch")

    runtime_paths = dict(parse_fold_path(value) for value in args.fold_runtime)
    if set(runtime_paths) != set(FOLDS):
        raise ValueError("exact folds0..4 are required")
    runtime_hashes = {
        str(fold): sha256_file(path) for fold, path in runtime_paths.items()
    }
    if source_manifest.get("fold_runtime_sha256") != runtime_hashes:
        raise ValueError("fold runtime SHA binding mismatch")
    actual_packages = {
        "python": ".".join(sys.version.split()[0].split(".")),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "pillow": importlib.metadata.version("Pillow"),
    }
    if source_manifest.get("runtime_packages") != actual_packages:
        raise ValueError("runtime package version binding mismatch")

    evaluation, evaluation_self = load_self_hashed(
        args.evaluation, "exp699_soft_cache_exploratory_oof_v1"
    )
    mixed = evaluation.get("comparisons", {}).get("mixed_all", {}).get(
        "after_prior", {}
    )
    if (
        evaluation.get("decision") != "EXPLORATORY_READY_FOR_PACKAGE"
        or evaluation.get("public_used") is not False
        or evaluation.get("preregister_self_sha256")
        != EXPECTED_PREREGISTER_SELF_SHA256
        or evaluation.get("source_bindings", {}).get(
            "source_manifest_self_sha256"
        )
        != EXPECTED_SOURCE_MANIFEST_SELF_SHA256
        or type(evaluation.get("final_changed_rows")) is not int
        or evaluation.get("final_changed_rows", 0) <= 0
        or mixed.get("changed") != evaluation.get("final_changed_rows")
    ):
        raise ValueError("terminal exploratory evaluation binding mismatch")
    return {
        "preregister": preregister,
        "source_manifest": source_manifest,
        "solution_manifest": solution_manifest,
        "source_files": source_files,
        "runtime_paths": runtime_paths,
        "bindings": {
            "preregister_file_sha256": EXPECTED_PREREGISTER_FILE_SHA256,
            "preregister_self_sha256": preregister_self,
            "source_manifest_file_sha256": sha256_file(args.source_manifest),
            "source_manifest_self_sha256": source_manifest_self,
            "solution140_source_manifest_file_sha256": (
                EXPECTED_SOLUTION140_MANIFEST_FILE_SHA256
            ),
            "solution140_source_manifest_self_sha256": solution_manifest_self,
            "solution140_run_sha256": EXPECTED_SOLUTION140_RUN_SHA256,
            "component_outputs_sha256": expected_component["file_sha256"],
            "component_audit_sha256": expected_component["audit_file_sha256"],
            "topology_sha256": expected_topology["file_sha256"],
            "topology_module_sha256": expected_topology["module_sha256"],
            "fold_runtime_sha256": runtime_hashes,
            "runtime_packages": actual_packages,
            "evaluation_file_sha256": sha256_file(args.evaluation),
            "evaluation_self_sha256": evaluation_self,
            "evaluation_final_changed_rows": evaluation["final_changed_rows"],
            "evaluation_macro_delta": evaluation["macro_delta"],
            "evaluation_corrections": mixed["corrections"],
            "evaluation_regressions": mixed["regressions"],
        },
    }


def align_donors(
    runtime_paths: dict[int, Path], component_outputs: Path
) -> list[dict[str, Any]]:
    runtime_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        local = read_jsonl(runtime_paths[fold])
        if any(int(row.get("fold", -1)) != fold for row in local):
            raise ValueError(f"fold{fold} runtime binding mismatch")
        if any("label" in row or "target" in row for row in local):
            raise ValueError("labels entered label-free donor candidate generation")
        runtime_rows.extend(local)
    runtime_rows.sort(key=lambda row: int(row["global_index"]))
    if [int(row["global_index"]) for row in runtime_rows] != list(
        range(EXPECTED_ROWS)
    ):
        raise ValueError("runtime global-index coverage mismatch")
    component_rows = read_jsonl(component_outputs)
    component_by_key = {
        (str(row["id"]), int(row["fold"]), str(row["category"])): row
        for row in component_rows
    }
    if len(component_by_key) != EXPECTED_ROWS:
        raise ValueError("component-output key coverage mismatch")
    donors: list[dict[str, Any]] = []
    for row in runtime_rows:
        if str(row["category"]) != runtime.FLAMMABLE:
            continue
        key = (str(row["id"]), int(row["fold"]), str(row["category"]))
        component = component_by_key.get(key)
        if component is None or int(component["label"]) not in (0, 1):
            raise ValueError("donor label binding mismatch")
        donors.append(
            {
                "id": str(row["id"]),
                "global_index": int(row["global_index"]),
                "name": str(row.get("name") or ""),
                "description": str(row.get("description") or ""),
                "label": int(component["label"]),
            }
        )
    if len(donors) != EXPECTED_DONORS:
        raise ValueError(f"expected exactly {EXPECTED_DONORS} flammable donors")
    return donors


def validated_image_path(cache_root: Path, resolved_root: Path, row_id: str) -> Path:
    path = cache_root / f"{hashlib.sha256(row_id.encode()).hexdigest()}.img"
    if not path.is_file():
        raise ValueError(f"missing donor image-cache entry: {path.name}")
    resolved = path.resolve(strict=True)
    approved = resolved_root.resolve(strict=True)
    try:
        resolved.relative_to(approved)
    except ValueError as error:
        raise ValueError(f"donor image escapes approved root: {path.name}") from error
    if not resolved.is_file():
        raise ValueError(f"resolved donor image is not a file: {path.name}")
    return path


def _sorted_groups(groups: dict[Any, list[int]], *, stringify: bool = False) -> dict:
    items = sorted(groups.items(), key=lambda item: item[0])
    return {
        str(key) if stringify else key: sorted(int(value) for value in values)
        for key, values in items
    }


def build_cache(
    donors: list[dict[str, Any]], image_cache: Path, image_resolved_root: Path
) -> dict[str, Any]:
    texts = [
        runtime.row_text(row["name"], row["description"], mask_digits=False)
        for row in donors
    ]
    normalized = [
        runtime.row_text(row["name"], row["description"], mask_digits=True)
        for row in donors
    ]
    exact_groups: dict[str, list[int]] = defaultdict(list)
    normalized_groups: dict[str, list[int]] = defaultdict(list)
    for index, (text, masked) in enumerate(zip(texts, normalized, strict=True)):
        if text:
            exact_groups[runtime.text_key(text)].append(index)
        if masked:
            normalized_groups[runtime.text_key(masked)].append(index)

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=200_000,
        sublinear_tf=True,
        norm="l2",
    )
    tfidf_matrix = tfidf.fit_transform(texts).astype(np.float32).tocsr()
    counter = CountVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=200_000, binary=False
    )
    counts = counter.fit_transform(texts).tocsr()
    bm25_matrix = runtime.bm25_documents(counts)

    image_exact: dict[str, list[int]] = defaultdict(list)
    image_phash: dict[int, list[int]] = defaultdict(list)
    image_dhash = np.zeros(len(donors), dtype=np.uint64)
    image_std = np.zeros(len(donors), dtype=np.float32)
    for index, donor in enumerate(donors):
        path = validated_image_path(image_cache, image_resolved_root, donor["id"])
        fingerprint = runtime.image_fingerprints(path)
        image_dhash[index] = np.uint64(fingerprint["dhash64"])
        image_std[index] = np.float32(fingerprint["grayscale_std"])
        if float(fingerprint["grayscale_std"]) >= runtime.MINIMUM_GRAYSCALE_STD:
            image_exact[str(fingerprint["file_sha256"])].append(index)
            image_phash[int(fingerprint["phash64"])].append(index)

    return {
        "schema": runtime.SCHEMA,
        "preregister_self_sha256": EXPECTED_PREREGISTER_SELF_SHA256,
        "source_manifest_self_sha256": EXPECTED_SOURCE_MANIFEST_SELF_SHA256,
        "category": runtime.FLAMMABLE,
        "donor_count": len(donors),
        "donor_order": np.asarray(
            [row["global_index"] for row in donors], dtype=np.int32
        ),
        "donor_labels": np.asarray([row["label"] for row in donors], dtype=np.int8),
        "exact_text_groups": _sorted_groups(exact_groups),
        "normalized_text_groups": _sorted_groups(normalized_groups),
        "tfidf_vectorizer": tfidf,
        "tfidf_matrix": tfidf_matrix,
        "bm25_vectorizer": counter,
        "bm25_matrix": bm25_matrix,
        "image_exact_groups": _sorted_groups(image_exact),
        "image_phash_groups": _sorted_groups(image_phash, stringify=True),
        "image_dhash": image_dhash,
        "image_grayscale_std": image_std,
        "channel_weights": dict(runtime.CHANNEL_WEIGHTS),
        "candidate_top_k": runtime.CANDIDATE_TOP_K,
        "cache_top_k": runtime.CACHE_TOP_K,
        "rrf_offset": runtime.RRF_OFFSET,
        "minimum_unique_donors": runtime.MINIMUM_DONORS,
        "minimum_donor_label_agreement": runtime.MINIMUM_AGREEMENT,
        "minimum_evidence_channels": runtime.MINIMUM_CHANNELS,
        "exact_text_can_satisfy_evidence_gate_alone": True,
        "cache_alpha": runtime.CACHE_ALPHA,
        "threshold": runtime.FLAMMABLE_THRESHOLD,
        "image_thresholds": {
            "minimum_grayscale_std": runtime.MINIMUM_GRAYSCALE_STD,
            "phash_hamming_max": runtime.PHASH_HAMMING_MAX,
            "dhash_hamming_max": runtime.DHASH_HAMMING_MAX,
        },
        "labels_used_only_as_donor_values": True,
        "public_used": False,
        "sealed_rows_used": 0,
    }


def patch_solution140_run(source: str) -> str:
    import_anchor = "from src.model import compose_text, fingerprint, normalize\n"
    import_replacement = import_anchor + (
        "from soft_cache_runtime import apply_runtime_cache\n"
    )
    prediction_anchor = "    predictions = np.zeros(len(frame), dtype=np.int8)\n"
    prediction_replacement = (
        "    final_scores = np.zeros(len(frame), dtype=np.float64)\n"
        + prediction_anchor
    )
    threshold_anchor = (
        "        predictions[mask] = (combined >= config[\"threshold\"]).astype(np.int8)\n"
    )
    threshold_replacement = (
        "        final_scores[mask] = combined\n" + threshold_anchor
    )
    prior_anchor = "    # The organizer confirmed that test labels come from the same ambiguous\n"
    prior_replacement = '''    predictions_before_soft_cache = predictions.copy()
    _, predictions, soft_cache_audit = apply_runtime_cache(
        frame,
        final_scores,
        predictions,
        ROOT / "soft_cache.joblib",
    )
    bad_mask = frame["category"].astype(str).to_numpy() == "БАД"
    if not np.array_equal(
        predictions[bad_mask], predictions_before_soft_cache[bad_mask]
    ):
        raise RuntimeError("soft cache changed BAD predictions")
    print(
        f"soft_cache_gate_rows={len(soft_cache_audit)} "
        f"soft_cache_changed_rows={sum(row['baseline_prediction'] != row['candidate_prediction'] for row in soft_cache_audit)}",
        flush=True,
    )

''' + prior_anchor
    replacements = (
        (import_anchor, import_replacement),
        (prediction_anchor, prediction_replacement),
        (threshold_anchor, threshold_replacement),
        (prior_anchor, prior_replacement),
    )
    output = source
    for old, new in replacements:
        if output.count(old) != 1:
            raise ValueError("solution140 soft-cache patch anchor mismatch")
        output = output.replace(old, new)
    if output.count("apply_runtime_cache(") != 1:
        raise ValueError("soft-cache source substitution count mismatch")
    return output


def deterministic_zip(source: Path, archive: Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as output:
        for relative, path in regular_files(source).items():
            info = zipfile.ZipInfo(relative, FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.create_system = 3
            output.writestr(info, path.read_bytes(), compresslevel=6)


def build_package_from_verified_inputs(
    *,
    source: Path,
    runtime_module: Path,
    cache_file: Path,
    destination: Path,
    archive: Path,
    bindings: dict[str, Any],
) -> dict[str, Any]:
    if destination.exists() or archive.exists():
        raise FileExistsError("refusing to overwrite exploratory package")
    source_files = regular_files(source)
    if sha256_file(source_files["run.py"]) != EXPECTED_SOLUTION140_RUN_SHA256:
        raise ValueError("solution140 run.py SHA mismatch")
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(*EXCLUDED_PARTS),
    )
    patched = patch_solution140_run(source_files["run.py"].read_text(encoding="utf-8"))
    (destination / "run.py").write_text(patched, encoding="utf-8")
    shutil.copyfile(runtime_module, destination / "soft_cache_runtime.py")
    shutil.copyfile(cache_file, destination / "soft_cache.joblib")

    unchanged: dict[str, str] = {}
    for relative, source_path in source_files.items():
        if relative == "run.py":
            continue
        candidate_path = destination / relative
        source_hash = sha256_file(source_path)
        if sha256_file(candidate_path) != source_hash:
            raise RuntimeError(f"unrelated solution140 member drifted: {relative}")
        unchanged[relative] = source_hash
    manifest: dict[str, Any] = {
        "schema": "exp699_solution140_soft_cache_candidate_v1",
        "experiment": 699,
        "candidate_name": "solution140_flammable_graph_aware_soft_cache_v1",
        "parent_submission": "140",
        "status": "ExploratoryReadyNotSubmitted",
        "route": {
            "БАД": "solution140_byte_identical",
            runtime.FLAMMABLE: "solution140_plus_soft_cache",
        },
        "source_run_sha256": EXPECTED_SOLUTION140_RUN_SHA256,
        "candidate_run_sha256": sha256_file(destination / "run.py"),
        "soft_cache_runtime_sha256": sha256_file(
            destination / "soft_cache_runtime.py"
        ),
        "soft_cache_artifact_sha256": sha256_file(
            destination / "soft_cache.joblib"
        ),
        "soft_cache_artifact_size": (destination / "soft_cache.joblib").stat().st_size,
        "donor_count": EXPECTED_DONORS,
        "bindings": bindings,
        "unchanged_files_sha256": unchanged,
        "unrelated_solution140_members_changed": 0,
        "qwen3vl_adapter_unchanged": True,
        "qwen35_adapter_unchanged": True,
        "base_models_unchanged": True,
        "fusion_weights_unchanged": True,
        "thresholds_unchanged": True,
        "production_priors_unchanged": True,
        "public_used_for_recipe": False,
        "sealed_rows_used": 0,
        "ods_submit_authorized": False,
        "decision": "ACCEPT_BUILD_FOR_RUNTIME_SMOKE_NO_PUBLIC_SUBMIT",
    }
    manifest["self_sha256"] = canonical_sha256(manifest)
    (destination / "exp699_candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    deterministic_zip(destination, archive)
    return {
        "decision": manifest["decision"],
        "status": manifest["status"],
        "manifest_self_sha256": manifest["self_sha256"],
        "archive_sha256": sha256_file(archive),
        "archive_size": archive.stat().st_size,
        "donor_count": EXPECTED_DONORS,
        "ods_submit_authorized": False,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    contracts = validate_contracts(args)
    donors = align_donors(contracts["runtime_paths"], args.component_outputs)
    cache = build_cache(donors, args.image_cache, args.image_resolved_root)
    cache_path = args.cache_output
    if cache_path.exists():
        raise FileExistsError("refusing to overwrite soft-cache artifact")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    runtime.dump_cache(cache, cache_path)
    loaded = runtime.load_cache(cache_path)
    if len(loaded["donor_labels"]) != EXPECTED_DONORS:
        raise ValueError("serialized donor-cache count mismatch")
    return build_package_from_verified_inputs(
        source=args.solution140_source,
        runtime_module=args.runtime_module,
        cache_file=cache_path,
        destination=args.destination,
        archive=args.archive,
        bindings=contracts["bindings"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--solution140-source", type=Path, required=True)
    parser.add_argument("--solution140-source-manifest", type=Path, required=True)
    parser.add_argument("--component-outputs", type=Path, required=True)
    parser.add_argument("--component-audit", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--topology-module", type=Path, required=True)
    parser.add_argument("--fold-runtime", action="append", required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--image-resolved-root", type=Path, required=True)
    parser.add_argument(
        "--runtime-module",
        type=Path,
        default=Path(__file__).with_name("soft_cache_runtime.py"),
    )
    parser.add_argument("--cache-output", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
