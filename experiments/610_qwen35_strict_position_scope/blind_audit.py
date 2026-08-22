from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
from position_protocol import (
    AUDIT_MINIMUM_ROWS,
    AUDIT_REQUIRED_KAPPA,
    AUDIT_REQUIRED_PASS_RATE,
    AUDIT_TARGET_ROWS,
    EXPERIMENT_ID,
    FLAMMABLE,
    INDEPENDENT_AUDIT_FOLDS,
    PARSER_VERSION,
    analyze_scope_sentence,
    canonical_sha256,
    move_whole_sentence_to_front,
)
from position_shared import sha256_file

IMMUTABLE_SAMPLE_COLUMNS = (
    "audit_key",
    "cue_type",
    "length_band",
    "name",
    "original_description",
    "transformed_description",
)
REVIEW_COLUMNS = ("reviewer_a", "reviewer_b", "failure_reason_a", "failure_reason_b")
VALID_REVIEW_VALUES = {"preserved", "rejected"}


def length_band(value: str) -> str:
    if len(value) <= 300:
        return "short"
    if len(value) <= 900:
        return "medium"
    return "long"


def _sample_hash(frame: pd.DataFrame) -> str:
    return canonical_sha256(frame.loc[:, IMMUTABLE_SAMPLE_COLUMNS].fillna("").to_dict("records"))


def _allocate_stratified(pool: pd.DataFrame, count: int) -> pd.DataFrame:
    if count >= len(pool):
        return pool.sort_values(["stratum", "selection_hash"], kind="stable").reset_index(drop=True)
    groups = {key: local.copy() for key, local in pool.groupby("stratum", sort=True)}
    exact = {key: count * len(local) / len(pool) for key, local in groups.items()}
    allocation = {key: min(len(groups[key]), math.floor(value)) for key, value in exact.items()}
    remaining = count - sum(allocation.values())
    order = sorted(
        groups,
        key=lambda key: (-(exact[key] - math.floor(exact[key])), key),
    )
    for key in order:
        if remaining and allocation[key] < len(groups[key]):
            allocation[key] += 1
            remaining -= 1
    if remaining:
        for key in sorted(groups):
            available = len(groups[key]) - allocation[key]
            take = min(available, remaining)
            allocation[key] += take
            remaining -= take
            if not remaining:
                break
    selected = []
    for key in sorted(groups):
        local = groups[key].sort_values("selection_hash", kind="stable")
        selected.append(local.iloc[: allocation[key]])
    output = pd.concat(selected, ignore_index=True)
    if len(output) != count:
        raise AssertionError("stratified sample size mismatch")
    return output.sort_values("selection_hash", kind="stable").reset_index(drop=True)


def build_sample(
    *,
    data: pd.DataFrame,
    folds: pd.DataFrame,
    legacy_ids: set[str],
) -> tuple[pd.DataFrame, dict]:
    required_data = {"id", "category", "name", "description"}
    required_folds = {"id", "split", "development_fold"}
    if missing := required_data.difference(data.columns):
        raise ValueError(f"audit data lacks columns: {sorted(missing)}")
    if missing := required_folds.difference(folds.columns):
        raise ValueError(f"audit folds lack columns: {sorted(missing)}")
    # Intentionally construct a new label-free frame so future code cannot
    # accidentally inspect labels while deciding eligibility or sample order.
    label_free = data.loc[:, ["id", "category", "name", "description"]].copy()
    label_free["id"] = label_free["id"].astype(str)
    aligned = folds.loc[:, ["id", "split", "development_fold"]].copy()
    aligned["id"] = aligned["id"].astype(str)
    if label_free["id"].duplicated().any() or aligned["id"].duplicated().any():
        raise ValueError("blind-audit inputs contain duplicate IDs")
    joined = label_free.merge(aligned, on="id", how="left", validate="one_to_one")
    if joined["split"].isna().any():
        raise ValueError("blind-audit data contains IDs absent from folds")
    allowed = (
        joined["split"].eq("development")
        & joined["development_fold"].astype(int).isin(INDEPENDENT_AUDIT_FOLDS)
        & joined["category"].astype(str).eq(FLAMMABLE)
        & ~joined["id"].isin(legacy_ids)
    )
    candidates = []
    reason_counts: Counter[str] = Counter()
    for row in joined.loc[allowed].itertuples(index=False):
        description = str(row.description or "")
        analysis = analyze_scope_sentence(description)
        reason_counts[analysis.reason] += 1
        if analysis.match is None:
            continue
        transformed = move_whole_sentence_to_front(description, analysis.match)
        audit_key = hashlib.sha256(
            f"{PARSER_VERSION}\0independent-audit\0{row.id}".encode()
        ).hexdigest()[:20]
        band = length_band(description)
        selection_hash = hashlib.sha256(f"{audit_key}\0sample-order".encode()).hexdigest()
        candidates.append(
            {
                "audit_key": audit_key,
                "cue_type": analysis.match.cue_type,
                "length_band": band,
                "name": str(row.name or ""),
                "original_description": description,
                "transformed_description": transformed,
                "stratum": f"{analysis.match.cue_type}:{band}",
                "selection_hash": selection_hash,
            }
        )
    pool = pd.DataFrame(candidates)
    if pool.empty:
        sample = pd.DataFrame(columns=IMMUTABLE_SAMPLE_COLUMNS)
    else:
        if pool["audit_key"].duplicated().any():
            raise ValueError("blind audit key collision")
        sample = _allocate_stratified(pool, min(AUDIT_TARGET_ROWS, len(pool)))
        sample = sample.loc[:, IMMUTABLE_SAMPLE_COLUMNS]
    for column in REVIEW_COLUMNS:
        sample[column] = ""
    pool_size = len(pool)
    status = (
        "NO_GO_INSUFFICIENT_INDEPENDENT_PAIRS"
        if pool_size < AUDIT_MINIMUM_ROWS
        else "READY_FOR_BLIND_REVIEW"
    )
    report = {
        "experiment_id": EXPERIMENT_ID,
        "parser_version": PARSER_VERSION,
        "audit_version": "strict_position_scope_independent_blind_v1",
        "label_columns_read": False,
        "model_outputs_read": False,
        "source_folds": list(INDEPENDENT_AUDIT_FOLDS),
        "screen_folds_used": [],
        "legacy_audit_ids_excluded": len(legacy_ids),
        "independent_pool_rows": pool_size,
        "target_sample_rows": AUDIT_TARGET_ROWS,
        "minimum_independent_rows": AUDIT_MINIMUM_ROWS,
        "sample_rows": len(sample),
        "sampling": "deterministic proportional strata by cue type and description length",
        "pool_strata": dict(sorted(Counter(pool.get("stratum", [])).items())),
        "sample_strata": dict(
            sorted(Counter((sample["cue_type"] + ":" + sample["length_band"]).tolist()).items())
        ) if len(sample) else {},
        "parser_reason_counts": dict(sorted(reason_counts.items())),
        "required_pass_rate": AUDIT_REQUIRED_PASS_RATE,
        "required_cohen_kappa": AUDIT_REQUIRED_KAPPA,
        "sample_immutable_sha256": _sample_hash(sample),
        "decision": status,
        "gpu_launch_allowed": False,
    }
    report["report_sha256"] = canonical_sha256(report)
    return sample, report


def cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    labels = sorted(VALID_REVIEW_VALUES)
    expected = sum(
        (left.count(label) / len(left)) * (right.count(label) / len(right)) for label in labels
    )
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def score_reviews(sample: pd.DataFrame, provenance: dict) -> dict:
    if _sample_hash(sample) != provenance.get("sample_immutable_sha256"):
        raise ValueError("reviewed sample content differs from frozen blind sample")
    if len(sample) < AUDIT_MINIMUM_ROWS:
        raise ValueError("independent blind sample is below the frozen minimum")
    missing = [column for column in REVIEW_COLUMNS if column not in sample.columns]
    if missing:
        raise ValueError(f"review columns are missing: {missing}")
    left = sample["reviewer_a"].astype(str).str.strip().str.lower().tolist()
    right = sample["reviewer_b"].astype(str).str.strip().str.lower().tolist()
    unexpected = sorted((set(left) | set(right)) - VALID_REVIEW_VALUES)
    if unexpected:
        raise ValueError(f"reviews must be complete preserved/rejected values: {unexpected}")
    strict_pass = [a == "preserved" and b == "preserved" for a, b in zip(left, right, strict=True)]
    pass_rate = sum(strict_pass) / len(strict_pass)
    raw_agreement = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    kappa = cohen_kappa(left, right)
    gates = {
        "independent_rows_at_least_100": len(sample) >= AUDIT_MINIMUM_ROWS,
        "strict_pass_rate_at_least_0_98": pass_rate >= AUDIT_REQUIRED_PASS_RATE,
        "cohen_kappa_at_least_0_80": kappa is not None and kappa >= AUDIT_REQUIRED_KAPPA,
        "screen_folds_absent": provenance.get("screen_folds_used") == [],
        "labels_not_read": provenance.get("label_columns_read") is False,
        "model_outputs_not_read": provenance.get("model_outputs_read") is False,
    }
    report = {
        "experiment_id": EXPERIMENT_ID,
        "parser_version": PARSER_VERSION,
        "audit_version": "strict_position_scope_independent_blind_v1",
        "reviewed_pairs": len(sample),
        "strict_preserved_pairs": int(sum(strict_pass)),
        "strict_pass_rate": pass_rate,
        "raw_reviewer_agreement": raw_agreement,
        "cohen_kappa": kappa,
        "sample_immutable_sha256": provenance["sample_immutable_sha256"],
        "gates": gates,
        "decision": "GO" if all(gates.values()) else "NO_GO",
        "gpu_launch_allowed": all(gates.values()),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def _legacy_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["row_id"]) for row in payload["rows"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or score the independent blind audit.")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--runtime-dir", required=True, type=Path)
    build.add_argument("--legacy-audit", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)
    score = commands.add_parser("score")
    score.add_argument("--sample", required=True, type=Path)
    score.add_argument("--provenance", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "score":
        if args.output.exists():
            raise FileExistsError("refusing to overwrite blind-audit result")
        sample = pd.read_csv(args.sample, dtype=str).fillna("")
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
        report = score_reviews(sample, provenance)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"decision": report["decision"], "output": str(args.output)}))
        return 0

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite blind-audit directory")
    runtime = args.runtime_dir.resolve()
    audit_path = runtime / "zero_sealed_runtime_audit.json"
    data_path = runtime / "development_data.csv"
    folds_path = runtime / "development_folds.csv"
    runtime_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if runtime_audit.get("decision") != "GO" or runtime_audit.get("sealed_rows_in_runtime_inputs") != 0:
        raise ValueError("runtime inputs do not prove physical exclusion of sealed rows")
    for path in (data_path, folds_path):
        if runtime_audit.get("output_sha256", {}).get(path.name) != sha256_file(path):
            raise ValueError(f"runtime input checksum mismatch: {path.name}")
    data = pd.read_csv(
        data_path,
        dtype={"id": str},
        usecols=["id", "category", "name", "description"],
    ).fillna("")
    folds = pd.read_csv(
        folds_path,
        dtype={"id": str},
        usecols=["id", "split", "development_fold"],
    )
    sample, report = build_sample(
        data=data,
        folds=folds,
        legacy_ids=_legacy_ids(args.legacy_audit),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "blind_audit_sample.csv"
    provenance_path = args.output_dir / "blind_audit_provenance.json"
    sample.to_csv(sample_path, index=False)
    report["input_sha256"] = {
        "development_data.csv": sha256_file(data_path),
        "development_folds.csv": sha256_file(folds_path),
        "legacy_audit": sha256_file(args.legacy_audit),
    }
    report["report_sha256"] = canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    provenance_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": report["decision"], "sample_rows": len(sample)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
