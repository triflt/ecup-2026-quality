from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from grid_contract import CELL_SPECS, GRID_CONTRACT_SHA256, canonical_sha256, sha256_file

EXPECTED_FAST_PATH_PACKAGES = {
    "transformers": "5.15.1",
    "triton": "3.6.0",
    "einops": "0.8.2",
    "fla_core": "0.5.2",
    "flash_linear_attention": "0.5.2",
    "causal_conv1d": "1.6.2.post1",
    "kernels": "0.16.0",
    "kernels_data": "0.16.0",
    "ziglang": "0.16.0",
}
EXPECTED_BINDING_PREFIXES = {
    "chunk_gated_delta_rule": "fla.",
    "recurrent_gated_delta_rule": "fla.",
    "causal_conv1d_fn": "causal_conv1d",
    "causal_conv1d_update": "causal_conv1d",
}
FORBIDDEN_PREDICTION_FIELDS = {
    "label",
    "target",
    "evidence_target",
    "sealed",
    "public",
    "split",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _verify_adapter_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"adapter ZIP CRC failure: {bad_member}")
        members = [PurePosixPath(name) for name in archive.namelist()]
        if any(member.is_absolute() or ".." in member.parts for member in members):
            raise ValueError("adapter ZIP contains an unsafe path")
        basenames = {member.name for member in members if member.name}
        required = {"adapter_config.json", "adapter_model.safetensors"}
        if not required.issubset(basenames):
            raise ValueError(f"adapter ZIP is incomplete: {sorted(required - basenames)}")
    return {"members": len(members), "sha256": sha256_file(path)}


def verify_artifact(
    artifact_dir: Path,
    *,
    experiment_id: str,
    fold: int,
    technical_smoke: bool,
) -> dict[str, Any]:
    if experiment_id not in CELL_SPECS:
        raise ValueError(f"unknown grid experiment: {experiment_id}")
    spec = CELL_SPECS[experiment_id]
    contract_path = artifact_dir / "output_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError("output_contract.json is missing")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    payload = dict(contract)
    digest = payload.pop("contract_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("output contract self-hash mismatch")
    expected = {
        "experiment_id": experiment_id,
        "outer_fold": fold,
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "objective": spec.objective,
        "grid_contract_sha256": GRID_CONTRACT_SHA256,
        "technical_smoke": technical_smoke,
        "sealed_rows_used": 0,
        "validation_labels_read": 0,
        "threshold": 0.0,
        "threshold_tuned": False,
        "decision": "TECHNICAL_SMOKE_ONLY" if technical_smoke else "GO_EVALUATE",
    }
    mismatch = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise ValueError(f"output contract mismatch: {mismatch}")
    if contract.get("fast_path_packages") != EXPECTED_FAST_PATH_PACKAGES:
        raise ValueError("fast-path package versions differ from the frozen contract")
    bindings = contract.get("fast_path_bindings", {})
    if set(bindings) != set(EXPECTED_BINDING_PREFIXES):
        raise ValueError("fast-path binding set is incomplete")
    for name, prefix in EXPECTED_BINDING_PREFIXES.items():
        if not str(bindings[name]).startswith(prefix):
            raise ValueError(f"fast-path binding is not compiled: {name}")
    if contract.get("optimized_training_kernels") is not True:
        raise ValueError("optimized training kernels were not recorded")

    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("artifact checksum map is missing")
    required_predictions = (
        {"predictions.jsonl"}
        if spec.objective == "class_only"
        else {"predictions_class_first.jsonl", "predictions_evidence_first.jsonl"}
    )
    required_artifacts = required_predictions | {"adapter.zip"}
    if not required_artifacts.issubset(artifacts):
        missing = sorted(required_artifacts - set(artifacts))
        raise ValueError(f"required artifacts are missing: {missing}")
    for name, expected_sha in artifacts.items():
        path = artifact_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"contract artifact is missing: {name}")
        if sha256_file(path) != expected_sha:
            raise ValueError(f"artifact checksum mismatch: {name}")

    zip_report = _verify_adapter_zip(artifact_dir / "adapter.zip")
    prediction_rows = 0
    prediction_ids: set[str] = set()
    for name in sorted(required_predictions):
        rows = read_jsonl(artifact_dir / name)
        if not rows:
            raise ValueError(f"prediction artifact is empty: {name}")
        for row in rows:
            forbidden = FORBIDDEN_PREDICTION_FIELDS.intersection(row)
            if forbidden:
                raise ValueError(f"prediction contains supervision fields: {sorted(forbidden)}")
            if int(row.get("fold", -1)) != fold:
                raise ValueError("prediction fold differs from output contract")
            if str(row.get("model_id")) != spec.model_id:
                raise ValueError("prediction model differs from output contract")
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError("prediction score is not finite")
            if int(row["prediction"]) != int(score >= 0.0):
                raise ValueError("prediction differs from the frozen zero threshold")
            row_id = str(row["id"])
            if row_id in prediction_ids:
                raise ValueError("prediction IDs overlap across artifacts")
            prediction_ids.add(row_id)
        prediction_rows += len(rows)
    if prediction_rows != int(contract.get("validation_rows", -1)):
        raise ValueError("prediction count differs from output contract")

    return {
        "experiment_id": experiment_id,
        "outer_fold": fold,
        "technical_smoke": technical_smoke,
        "prediction_rows": prediction_rows,
        "adapter_zip": zip_report,
        "contract_sha256": digest,
        "decision": "ACCEPT_ARTIFACT",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify one Qwen scale-grid training artifact.")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True, choices=sorted(CELL_SPECS))
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_artifact(
        args.artifact_dir,
        experiment_id=args.experiment_id,
        fold=args.fold,
        technical_smoke=args.technical_smoke,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite verification output")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
