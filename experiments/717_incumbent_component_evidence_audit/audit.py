from __future__ import annotations

import hashlib
import json
from pathlib import Path

MAIN = Path("/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-2026-quality")
QC = Path("/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC")
EXPECTED_DATA_SHA256 = "4bc59e640563160fa04572b570606ceb1dd3d31627c6cf7fd1750ae4ea61f510"
EXPECTED_FOLDS_SHA256 = "03baaa25bd5a3aef6ad94e02067cccda114041f98d7a35a9e06330a425166e4d"
INCUMBENT_MACRO = 0.9118425205786493
EXPECTED_COMPONENTS = {
    "robust_base": (
        MAIN / "research/oof-cache-extracted/oof_scores.npz",
        "5d7467c48fc8a5a73f947f5aa1300071c77ba699b29c12250caf1bcd3176d7ac",
    ),
    "qwen3vl": (
        MAIN / "research/lora-hard-5fold-robust-fusion-report.npz",
        "ba432e13624e6c3b1c7304ced8cacf580f4ffcc0a0cde1af8b6afb098bf6dc01",
    ),
    "qwen35": (
        MAIN / "research/qwen35-hard-5fold-robust-fusion-report.npz",
        "147f2b2b87d8220566526b38ab0e085c0bc44cd82b0442baca6b019516c3f1d8",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def adapter_record(path: Path) -> dict:
    config_path = path.parent / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "config_sha256": sha256(config_path),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "use_rslora": config.get("use_rslora"),
        "target_modules": sorted(config.get("target_modules", [])),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    output = root / "results/incumbent_evidence_audit.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite incumbent evidence audit")
    data = QC / "data/data.csv"
    folds = MAIN / "validation/grouped_text_v1/folds.csv"
    if sha256(data) != EXPECTED_DATA_SHA256 or sha256(folds) != EXPECTED_FOLDS_SHA256:
        raise ValueError("canonical dataset or fold checksum mismatch")

    metrics_path = MAIN / "experiments/140_dual_lora_fusion/results/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    historical = metrics["historical_results"][0]
    actual_incumbent = float(historical["macro_f1"])
    if actual_incumbent != INCUMBENT_MACRO:
        raise ValueError("experiment-140 incumbent metric changed")

    manifest_path = MAIN / "validation/locked_190_nested_v1/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_expected = {
        "robust_base": manifest["checksums"]["robust_base_oof_sha256"],
        "qwen3vl": manifest["checksums"]["qwen3vl_oof_sha256"],
        "qwen35": manifest["checksums"]["qwen35_seed42_oof_sha256"],
    }
    component_status = {}
    for name, (path, expected_sha) in EXPECTED_COMPONENTS.items():
        if manifest_expected[name] != expected_sha:
            raise ValueError(f"locked manifest checksum changed for {name}")
        exists = path.is_file()
        actual_sha = sha256(path) if exists else None
        component_status[name] = {
            "path": str(path),
            "exists": exists,
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "checksum_valid": exists and actual_sha == expected_sha,
        }

    production = {
        "qwen3vl": adapter_record(
            QC / "step1_140ocr/submission_ocr/adapter_qwen3vl/adapter_model.safetensors"
        ),
        "qwen35": adapter_record(
            QC / "step1_140ocr/submission_ocr/adapter_qwen35/adapter_model.safetensors"
        ),
    }
    qc_step2 = {
        "qwen3vl": adapter_record(
            QC / "step2_synth/lora_runs/qwen3vl_real_f0/adapter/adapter_model.safetensors"
        ),
        "qwen35": adapter_record(
            QC / "step2_synth/lora_runs/real_f0/adapter/adapter_model.safetensors"
        ),
    }
    for name in ("qwen3vl", "qwen35"):
        qc_step2[name]["compatible_with_production_adapter"] = (
            qc_step2[name]["bytes"] == production[name]["bytes"]
            and qc_step2[name]["target_modules"] == production[name]["target_modules"]
            and qc_step2[name]["sha256"] == production[name]["sha256"]
        )
        if qc_step2[name]["compatible_with_production_adapter"]:
            raise ValueError(f"unexpectedly compatible QC step2 adapter: {name}")

    locked_report_path = MAIN / "validation/locked_190_nested_v1/adapter_replacement_report.json"
    locked_report = json.loads(locked_report_path.read_text(encoding="utf-8"))
    locked_macro = sum(
        locked_report["baseline"][category]["nested_f1"]
        for category in ("БАД", "Легковоспламеняющиеся")
    ) / 2
    exact_available = all(row["checksum_valid"] for row in component_status.values())
    result = {
        "schema_version": "exp717_incumbent_evidence_audit_v1",
        "experiment_id": "717",
        "data_sha256": sha256(data),
        "folds_sha256": sha256(folds),
        "incumbent_140": {
            "nested_macro_f1": actual_incumbent,
            "bad_f1": float(historical["bad_f1"]),
            "flammable_f1": float(historical["flammable_f1"]),
            "metrics_sha256": sha256(metrics_path),
        },
        "exact_component_oof": component_status,
        "exact_component_oof_available": exact_available,
        "production_adapters": production,
        "rejected_qc_step2_substitutes": qc_step2,
        "locked_190_context": {
            "nested_macro_f1": locked_macro,
            "report_sha256": sha256(locked_report_path),
            "manifest_sha256": sha256(manifest_path),
            "same_as_clean_incumbent_140": False,
            "usable_as_aggregate_context_only": True,
        },
        "evidence_capabilities": {
            "measure_exact_fixed_140_fusion_delta": exact_available,
            "measure_candidate_standalone_grouped_oof": True,
            "run_connected_family_guard_v2": True,
            "build_isolated_qwen35_adapter_replacement": True,
            "retune_fusion_without_exact_component_oof": False,
        },
        "policy": (
            "Exact solution-140 component OOF is unavailable: keep fusion weights and thresholds "
            "frozen; do not claim a measured incumbent-fusion delta. Promotion requires standalone "
            "grouped OOF plus connected-family gates, followed by an isolated adapter replacement."
            if not exact_available
            else "Exact component OOF is checksum-verified; fixed-fusion replacement may be evaluated."
        ),
        "decision": "FREEZE_FUSION_USE_ISOLATED_REPLACEMENT_GATES" if not exact_available else "EXACT_FUSION_EVALUATION_AVAILABLE",
    }
    payload = dict(result)
    result["contract_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
