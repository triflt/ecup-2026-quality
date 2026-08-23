"""Build the label-free semantic-v3 replay bundle for experiment 635.

The builder may read development labels only to construct outer-fold donor-only
repeat rules.  Labels are never serialized into the replay bundle.  Every
learned component score must already be out-of-fold for the target row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FOLDS = (0, 1, 2, 3, 4)
CATEGORIES = ("БАД", "Легковоспламеняющиеся")
PRIOR_CONFIG = {
    "БАД": (2, 2 / 3, 2, 0.999),
    "Легковоспламеняющиеся": (1, 0.999, 999, 0.999),
}
REQUIRED_ARRAYS = (
    "ids",
    "categories",
    "folds",
    "semantic_components",
    "robust_base_score",
    "qwen3vl_score",
    "qwen35_original_logit",
    "qwen35_seed632_logit",
    "seed632_evidence",
    "seed632_concept",
    "seed632_char_start",
    "seed632_char_end",
    "canonical_text",
    "prior_override",
    "prior_source",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower().replace("ё", "е")).strip()


def repeat_text(name: object, description: object) -> str:
    normalized_name = normalize(name)
    return f"{normalized_name}\n{normalized_name}\n{normalize(description)}"


def fingerprint(value: object) -> str:
    return hashlib.sha1(normalize(value).encode("utf-8")).hexdigest()


def canonical_text(name: object, description: object) -> str:
    return f"Название: {'' if pd.isna(name) else str(name)}\nОписание: {'' if pd.isna(description) else str(description)}"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def _load_runtime_rows(runtime_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        path = runtime_root / f"fold{fold}" / "validation.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                if int(item["development_fold"]) != fold:
                    raise ValueError(f"runtime fold mismatch: {path}")
                rows.append(
                    {
                        "id": str(item["id"]),
                        "name": item.get("name", ""),
                        "description": item.get("description", ""),
                        "runtime_fold": fold,
                    }
                )
    frame = pd.DataFrame(rows)
    if len(frame) != 11118 or frame["id"].duplicated().any():
        raise ValueError("runtime text rows must contain 11118 unique development ids")
    frame["canonical_text"] = [
        canonical_text(name, description)
        for name, description in zip(frame["name"], frame["description"], strict=True)
    ]
    frame["repeat_hash"] = [
        fingerprint(repeat_text(name, description))
        for name, description in zip(frame["name"], frame["description"], strict=True)
    ]
    frame["normalized_name"] = frame["name"].map(normalize)
    return frame.set_index("id", verify_integrity=True)


def _load_fold_predictions(pattern: str, *, required: set[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for fold in FOLDS:
        path = Path(pattern.format(fold=fold))
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, dtype={"id": str, "category": str})
        if not required.issubset(frame.columns):
            raise ValueError(f"prediction schema mismatch: {path}")
        if frame["id"].duplicated().any() or set(frame["fold"].astype(int)) != {fold}:
            raise ValueError(f"prediction fold/id contract mismatch: {path}")
        frames.append(frame)
        hashes[str(fold)] = sha256_file(path)
    merged = pd.concat(frames, ignore_index=True)
    if len(merged) != 11118 or merged["id"].duplicated().any():
        raise ValueError("OOF predictions must cover 11118 unique development ids")
    return merged.set_index("id", verify_integrity=True), hashes


def donor_only_prior(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    overrides = np.full(len(frame), -1, dtype=np.int8)
    sources = np.full(len(frame), "NONE", dtype="<U5")
    provenance: dict[str, Any] = {"folds": {}}
    labels = frame["label"].astype(np.int8)
    for fold in FOLDS:
        target = frame["fold"].eq(fold)
        donors = ~target
        fold_overrides = 0
        exact_overrides = 0
        name_overrides = 0
        for category in CATEGORIES:
            exact_min, exact_conf, name_min, name_conf = PRIOR_CONFIG[category]
            local_donors = frame.loc[donors & frame["category"].eq(category)]
            exact_stats = local_donors.groupby("repeat_hash")["label"].agg(["count", "mean"])
            name_stats = local_donors.groupby("normalized_name")["label"].agg(["count", "mean"])
            for index, row in frame.loc[target & frame["category"].eq(category)].iterrows():
                position = int(row["position"])
                exact = exact_stats.loc[row["repeat_hash"]] if row["repeat_hash"] in exact_stats.index else None
                if exact is not None:
                    confidence = max(float(exact["mean"]), 1.0 - float(exact["mean"]))
                    if int(exact["count"]) >= exact_min and confidence >= exact_conf and float(exact["mean"]) != 0.5:
                        overrides[position] = int(float(exact["mean"]) >= 0.5)
                        sources[position] = "EXACT"
                        exact_overrides += 1
                        fold_overrides += 1
                        continue
                name = name_stats.loc[row["normalized_name"]] if row["normalized_name"] in name_stats.index else None
                if name is not None:
                    confidence = max(float(name["mean"]), 1.0 - float(name["mean"]))
                    if int(name["count"]) >= name_min and confidence >= name_conf and float(name["mean"]) != 0.5:
                        overrides[position] = int(float(name["mean"]) >= 0.5)
                        sources[position] = "NAME"
                        name_overrides += 1
                        fold_overrides += 1
        donor_payload = frame.loc[donors, ["id", "category", "fold", "label", "repeat_hash", "normalized_name"]].to_dict("records")
        provenance["folds"][str(fold)] = {
            "target_fold_excluded": True,
            "artifact_sha256": canonical_sha256(donor_payload),
            "donor_rows": int(donors.sum()),
            "target_rows": int(target.sum()),
            "overrides": fold_overrides,
            "exact_overrides": exact_overrides,
            "name_overrides": name_overrides,
        }
    if not np.array_equal(labels.to_numpy(), frame["label"].to_numpy(np.int8)):
        raise AssertionError("label alignment changed while building prior")
    return overrides, sources, provenance


def _array_schema(bundle: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {"dtype": str(bundle[name].dtype), "shape": list(bundle[name].shape)}
        for name in REQUIRED_ARRAYS
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.contract.exists():
        raise FileExistsError("refusing to overwrite replay output")
    spec = _load_json(HERE / "frozen_spec.json")
    registry = pd.read_csv(args.registry, dtype={"id": str, "category": str, "split": str})
    registry = registry.loc[registry["split"].eq("development")].copy().reset_index(drop=True)
    if len(registry) != 11118 or registry["id"].duplicated().any():
        raise ValueError("semantic-v3 development registry mismatch")
    registry["fold"] = registry["development_fold"].astype(np.int8)
    registry["position"] = np.arange(len(registry), dtype=np.int64)
    if set(registry["fold"]) != set(FOLDS) or set(registry["category"]) != set(CATEGORIES):
        raise ValueError("semantic-v3 fold/category mismatch")

    texts = _load_runtime_rows(args.runtime_root)
    registry = registry.join(texts, on="id", validate="one_to_one")
    if registry[["canonical_text", "repeat_hash", "normalized_name"]].isna().any().any():
        raise ValueError("missing runtime text")
    if not np.array_equal(registry["fold"].to_numpy(), registry["runtime_fold"].to_numpy()):
        raise ValueError("runtime folds differ from registry")

    with np.load(args.visual_bundle, allow_pickle=False) as payload:
        expected = {"ids", "labels", "categories", "folds", "semantic_components", "robust_base_score", "qwen3vl_score"}
        if not expected.issubset(payload.files):
            raise ValueError("visual component bundle schema mismatch")
        visual = {key: np.asarray(payload[key]) for key in expected}
    ids = np.asarray(registry["id"].astype(str).tolist(), dtype=str)
    if not np.array_equal(visual["ids"].astype(str), ids):
        raise ValueError("visual bundle id/order mismatch")
    if not np.array_equal(visual["labels"].astype(np.int8), registry["label"].to_numpy(np.int8)):
        raise ValueError("visual bundle labels differ from frozen registry")
    if not np.array_equal(visual["folds"].astype(np.int8), registry["fold"].to_numpy(np.int8)):
        raise ValueError("visual bundle folds differ from registry")

    original, original_hashes = _load_fold_predictions(
        args.original_pattern,
        required={"id", "category", "label", "fold", "lora_score"},
    )
    candidate, candidate_hashes = _load_fold_predictions(
        args.candidate_pattern,
        required={"id", "category", "fold", "lora_score", "evidence", "concept", "char_start", "char_end"},
    )
    original = original.loc[ids]
    candidate = candidate.loc[ids]
    for name, frame in (("original", original), ("candidate", candidate)):
        if not np.array_equal(frame["category"].astype(str).to_numpy(), registry["category"].to_numpy(str)):
            raise ValueError(f"{name} category alignment mismatch")
        if not np.array_equal(frame["fold"].astype(np.int8).to_numpy(), registry["fold"].to_numpy(np.int8)):
            raise ValueError(f"{name} fold alignment mismatch")
    if "label" in original and not np.array_equal(original["label"].to_numpy(np.int8), registry["label"].to_numpy(np.int8)):
        raise ValueError("original prediction labels differ from registry")

    overrides, sources, prior_provenance = donor_only_prior(registry)
    bundle = {
        "ids": ids,
        "categories": np.asarray(registry["category"].astype(str).tolist(), dtype=str),
        "folds": registry["fold"].to_numpy(np.int8),
        "semantic_components": np.asarray(
            registry["semantic_component"].astype(str).tolist(), dtype=str
        ),
        "robust_base_score": visual["robust_base_score"].astype(np.float32),
        "qwen3vl_score": visual["qwen3vl_score"].astype(np.float32),
        "qwen35_original_logit": original["lora_score"].to_numpy(np.float32),
        "qwen35_seed632_logit": candidate["lora_score"].to_numpy(np.float32),
        "seed632_evidence": np.asarray(
            candidate["evidence"].fillna("NO_EVIDENCE").astype(str).tolist(), dtype=str
        ),
        "seed632_concept": np.asarray(
            candidate["concept"].fillna("NO_EVIDENCE").astype(str).tolist(), dtype=str
        ),
        "seed632_char_start": candidate["char_start"].to_numpy(np.int64),
        "seed632_char_end": candidate["char_end"].to_numpy(np.int64),
        "canonical_text": np.asarray(registry["canonical_text"].astype(str).tolist(), dtype=str),
        "prior_override": overrides,
        "prior_source": sources,
    }
    if set(bundle) != set(REQUIRED_ARRAYS) or any(value.dtype.hasobject for value in bundle.values()):
        raise ValueError("replay bundle contains unexpected or object arrays")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **bundle)

    visual_hash = sha256_file(args.visual_bundle)
    visual_contract_hash = sha256_file(args.visual_contract)
    visual_folds = {
        str(fold): {
            "target_fold_excluded": True,
            "artifact_sha256": visual_hash,
            "contract_sha256": visual_contract_hash,
        }
        for fold in FOLDS
    }
    component_provenance = {
        "robust_base": {"folds": visual_folds},
        "qwen3vl": {"folds": visual_folds},
        "qwen35_original": {
            "folds": {
                str(fold): {"target_fold_excluded": True, "artifact_sha256": original_hashes[str(fold)]}
                for fold in FOLDS
            }
        },
        "qwen35_seed632": {
            "folds": {
                str(fold): {"target_fold_excluded": True, "artifact_sha256": candidate_hashes[str(fold)]}
                for fold in FOLDS
            }
        },
        "prior_override": prior_provenance,
    }
    contract: dict[str, Any] = {
        **spec["required_replay_contract"],
        "experiment_id": "635",
        "bundle_sha256": sha256_file(args.output),
        "registry_sha256": sha256_file(args.registry),
        "accepted_632_report_sha256": sha256_file(args.accepted_632_report),
        "source_sha256": spec["source_sha256"],
        "route": spec["route"],
        "component_provenance": component_provenance,
        "array_schema": _array_schema(bundle),
        "input_sha256": {
            "visual_bundle": visual_hash,
            "visual_contract": visual_contract_hash,
            "runtime_fold_jsonl": {
                str(fold): sha256_file(args.runtime_root / f"fold{fold}" / "validation.jsonl")
                for fold in FOLDS
            },
        },
        "historical_note": "semantic-v3 OOF replay of the frozen 140 recipe; not a historical pre-semantic-v3 OOF artifact",
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    args.contract.parent.mkdir(parents=True, exist_ok=True)
    args.contract.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build experiment-635 semantic-v3 replay.")
    parser.add_argument("--registry", type=Path, default=ROOT / "validation/semantic_family_v3/folds.csv")
    parser.add_argument(
        "--visual-bundle",
        type=Path,
        default=ROOT / "experiments/601_semantic_v3_visual_base_baselines/.local/components_strict_v2/semantic_v3_visual_base_components.npz",
    )
    parser.add_argument(
        "--visual-contract",
        type=Path,
        default=ROOT / "experiments/601_semantic_v3_visual_base_baselines/.local/components_strict_v2/component_contract.json",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=ROOT / "experiments/623_semantic_v3_multitask_span_head/.local/runtime_inputs",
    )
    parser.add_argument(
        "--original-pattern",
        default="experiments/600_semantic_v3_qwen35_baselines/.local/artifacts/original/fold{fold}/artifact/lora_holdout_predictions.csv",
    )
    parser.add_argument(
        "--candidate-pattern",
        default="experiments/632_span_head_seed_repeat/.local/evaluation/fold{fold}/validation_predictions.csv",
    )
    parser.add_argument(
        "--accepted-632-report",
        type=Path,
        default=ROOT / "experiments/632_span_head_seed_repeat/.local/evaluation/full.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    contract = build(parse_args())
    print(json.dumps(contract, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
