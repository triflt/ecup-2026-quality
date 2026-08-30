from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

EXPECTED_BASE_SHA256 = "0fa35652e178f316f782c7b425255b7d393390b186617d6d1bb319fae96a1cef"
EXPECTED_ALLOWED_TREE_SHA256 = "3a237b0aaaacbdb43ec8f40aafc55db180e3beb1447125b81c305d42d5d46cfe"
MAX_BYTES = 5 * 1024**3
ADAPTER_PREFIX = "adapter_qwen35/"
EXPECTED_IMAGE_PREPROCESSING = "solution140_first_image_thumbnail_448_lanczos_v1"
REQUIRED_UNCHANGED = {
    "run.py",
    "adapter_qwen3vl/adapter_config.json",
    "adapter_qwen3vl/adapter_model.safetensors",
}
EXPECTED_INCUMBENT_MACRO = 0.9118425205786493


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: dict) -> str:
    return bytes_sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    )


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in name
    )


def verify_full_refit(
    adapter_dir: Path,
    contract_path: Path,
    *,
    allow_fold_contract: bool = False,
    allow_control_mode: bool = False,
) -> tuple[dict, dict[str, bytes]]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    payload = dict(contract)
    digest = payload.pop("contract_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("full-refit contract self-hash mismatch")
    is_fold = contract.get("schema_version") == "exp698_fold_output_v2"
    if is_fold and not allow_fold_contract:
        raise ValueError("fold adapter contract is not authorized for this package")
    expected = (
        {
            "schema_version": "exp698_fold_output_v2",
            "experiment_id": "698",
            "source_experiment_id": "697",
            "technical_smoke": False,
            "submission_eligible_base_model": True,
            "teacher_model_required_at_inference": False,
            "image_preprocessing": EXPECTED_IMAGE_PREPROCESSING,
        }
        if is_fold
        else {
            "schema_version": "exp715_full_refit_v1",
            "experiment_id": "715",
            "source_experiment_id": "698",
            "full_data": True,
            "technical_smoke": False,
            "teacher_required_at_inference": False,
            "submission_base_model": "Qwen/Qwen3.5-4B",
            "image_preprocessing": EXPECTED_IMAGE_PREPROCESSING,
        }
    )
    mismatch = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatch:
        raise ValueError(f"full-refit contract mismatch: {mismatch}")
    mode = contract.get("mode") if is_fold else contract.get("selected_mode")
    allowed_modes = {"hardneg_candidate", "rank_candidate"}
    if allow_control_mode:
        allowed_modes.add("gold_control")
    if mode not in allowed_modes:
        raise ValueError("full-refit mode is not a distillation candidate")
    if not adapter_dir.is_dir():
        raise FileNotFoundError("full-refit adapter directory is missing")
    files = {
        path.name: path.read_bytes()
        for path in sorted(adapter_dir.iterdir())
        if path.is_file()
    }
    required = {"adapter_config.json", "adapter_model.safetensors"}
    if not required.issubset(files):
        raise ValueError("full-refit adapter lacks required PEFT files")
    if bytes_sha256(files["adapter_model.safetensors"]) != contract["adapter_model_sha256"]:
        raise ValueError("full-refit adapter checksum mismatch")
    adapter_bytes = int(contract.get("adapter_bytes", len(files["adapter_model.safetensors"])))
    if len(files["adapter_model.safetensors"]) != adapter_bytes:
        raise ValueError("full-refit adapter size mismatch")
    if is_fold and bytes_sha256(files["adapter_config.json"]) != contract.get(
        "adapter_config_sha256"
    ):
        raise ValueError("fold adapter config checksum mismatch")
    config = json.loads(files["adapter_config.json"])
    config_expected = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "use_rslora": True,
    }
    config_mismatch = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in config_expected.items()
        if config.get(key) != value
    }
    if config_mismatch:
        raise ValueError(f"adapter config mismatch: {config_mismatch}")
    if set(config.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj", "o_proj"}:
        raise ValueError("adapter target modules mismatch")
    normalized = dict(contract)
    normalized["selected_mode"] = mode
    normalized["adapter_bytes"] = adapter_bytes
    normalized["adapter_scope"] = "outer_fold" if is_fold else "full_data"
    return normalized, files


def verify_incumbent_evidence(path: Path) -> dict:
    audit = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(audit)
    digest = payload.pop("contract_sha256", None)
    if digest != canonical_sha256(payload):
        raise ValueError("incumbent evidence audit self-hash mismatch")
    expected = {
        "schema_version": "exp717_incumbent_evidence_audit_v1",
        "experiment_id": "717",
        "exact_component_oof_available": False,
        "decision": "FREEZE_FUSION_USE_ISOLATED_REPLACEMENT_GATES",
    }
    mismatch = {
        key: {"expected": value, "actual": audit.get(key)}
        for key, value in expected.items()
        if audit.get(key) != value
    }
    if mismatch:
        raise ValueError(f"incumbent evidence audit mismatch: {mismatch}")
    if audit.get("incumbent_140", {}).get("nested_macro_f1") != EXPECTED_INCUMBENT_MACRO:
        raise ValueError("incumbent evidence metric mismatch")
    capabilities = audit.get("evidence_capabilities", {})
    capability_expected = {
        "measure_exact_fixed_140_fusion_delta": False,
        "retune_fusion_without_exact_component_oof": False,
        "build_isolated_qwen35_adapter_replacement": True,
    }
    capability_mismatch = {
        key: {"expected": value, "actual": capabilities.get(key)}
        for key, value in capability_expected.items()
        if capabilities.get(key) != value
    }
    if capability_mismatch:
        raise ValueError(f"incumbent evidence capability mismatch: {capability_mismatch}")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", default=EXPECTED_BASE_SHA256)
    parser.add_argument("--base-freeze-report", type=Path)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--full-refit-contract", type=Path, required=True)
    parser.add_argument("--incumbent-evidence-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-fold-contract", action="store_true")
    parser.add_argument("--allow-control-mode", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite package output")
    if sha256(args.base_submission) != args.expected_base_sha256:
        raise ValueError("base submission checksum mismatch")
    base_freeze_binding = None
    if args.base_freeze_report is not None:
        freeze = json.loads(args.base_freeze_report.read_text(encoding="utf-8"))
        expected_freeze = {
            "schema_version": "exp716_allowed_base_freeze_v1",
            "experiment_id": "716",
            "source_read_only": True,
            "source_tree_sha256": EXPECTED_ALLOWED_TREE_SHA256,
            "forbidden_model_references_absent": True,
            "output_sha256": sha256(args.base_submission),
            "zip_integrity": "PASS",
        }
        freeze_mismatch = {
            key: {"expected": value, "actual": freeze.get(key)}
            for key, value in expected_freeze.items()
            if freeze.get(key) != value
        }
        if freeze_mismatch:
            raise ValueError(f"allowed base freeze mismatch: {freeze_mismatch}")
        base_freeze_binding = {
            "report_sha256": sha256(args.base_freeze_report),
            "source_tree_sha256": freeze["source_tree_sha256"],
            "allowed_model_mounts": freeze["allowed_model_mounts"],
        }
    evidence_audit = verify_incumbent_evidence(args.incumbent_evidence_audit)
    full_contract, adapter_files = verify_full_refit(
        args.adapter_dir,
        args.full_refit_contract,
        allow_fold_contract=args.allow_fold_contract,
        allow_control_mode=args.allow_control_mode,
    )

    with zipfile.ZipFile(args.base_submission) as source:
        if source.testzip() is not None:
            raise ValueError("base submission ZIP integrity failure")
        infos = source.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or any(not safe_member(name) for name in names):
            raise ValueError("base submission contains duplicate or unsafe paths")
        if not REQUIRED_UNCHANGED.issubset(names) or "metadata.json" not in names:
            raise ValueError("base submission lacks required runtime members")
        old_adapter_members = {name for name in names if name.startswith(ADAPTER_PREFIX)}
        if not {
            f"{ADAPTER_PREFIX}adapter_config.json",
            f"{ADAPTER_PREFIX}adapter_model.safetensors",
        }.issubset(old_adapter_members):
            raise ValueError("base submission lacks replaceable Qwen3.5 adapter")
        unchanged_hashes = {
            info.filename: bytes_sha256(source.read(info))
            for info in infos
            if not info.is_dir()
            and not info.filename.startswith(ADAPTER_PREFIX)
            and info.filename != "metadata.json"
        }
        metadata = json.loads(source.read("metadata.json"))
        metadata["description"] = (
            str(metadata.get("description", "")).strip()
            + "; Qwen3.5 adapter replaced by "
            + f"exp{full_contract['experiment_id']} {full_contract['selected_mode']}"
        ).lstrip("; ")
        metadata["distillation"] = {
            "teacher_in_submission": False,
            "student_base": "Qwen/Qwen3.5-4B",
            "adapter_experiment": full_contract["experiment_id"],
            "selected_mode": full_contract["selected_mode"],
            "adapter_scope": full_contract["adapter_scope"],
            "full_refit_contract_sha256": sha256(args.full_refit_contract),
            "incumbent_evidence_audit_sha256": sha256(args.incumbent_evidence_audit),
            "fusion_policy": evidence_audit["decision"],
            "image_preprocessing": full_contract["image_preprocessing"],
        }
        metadata_bytes = (
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()

        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with zipfile.ZipFile(temporary, "w", allowZip64=True) as destination:
                for info in infos:
                    if info.filename.startswith(ADAPTER_PREFIX) or info.filename == "metadata.json":
                        continue
                    destination.writestr(info, source.read(info))
                metadata_info = zipfile.ZipInfo("metadata.json")
                metadata_info.compress_type = zipfile.ZIP_DEFLATED
                metadata_info.external_attr = 0o644 << 16
                destination.writestr(metadata_info, metadata_bytes)
                for name, payload in sorted(adapter_files.items()):
                    info = zipfile.ZipInfo(f"{ADAPTER_PREFIX}{name}")
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o644 << 16
                    destination.writestr(info, payload)
            temporary.replace(args.output)
        finally:
            if temporary.exists():
                temporary.unlink()

    if args.output.stat().st_size >= MAX_BYTES:
        raise ValueError("replacement submission exceeds 5 GiB")
    with zipfile.ZipFile(args.output) as result:
        if result.testzip() is not None:
            raise ValueError("replacement submission ZIP integrity failure")
        result_names = [info.filename for info in result.infolist()]
        if len(result_names) != len(set(result_names)):
            raise ValueError("replacement submission contains duplicate paths")
        actual_unchanged = {
            info.filename: bytes_sha256(result.read(info))
            for info in result.infolist()
            if not info.is_dir()
            and not info.filename.startswith(ADAPTER_PREFIX)
            and info.filename != "metadata.json"
        }
        if actual_unchanged != unchanged_hashes:
            raise ValueError("a supposedly fixed base member changed")
        for name, payload in adapter_files.items():
            if result.read(f"{ADAPTER_PREFIX}{name}") != payload:
                raise ValueError("packaged Qwen3.5 adapter differs from full refit")
    report = {
        "schema_version": "exp716_submission_package_v1",
        "experiment_id": "716",
        "architecture": "fixed_base_submission_with_qwen35_adapter_replacement",
        "single_changed_model_component": "adapter_qwen35",
        "base_submission_sha256": sha256(args.base_submission),
        "base_submission_bytes": args.base_submission.stat().st_size,
        "base_freeze_binding": base_freeze_binding,
        "full_refit_contract_sha256": sha256(args.full_refit_contract),
        "incumbent_evidence_audit_sha256": sha256(args.incumbent_evidence_audit),
        "fusion_policy": evidence_audit["decision"],
        "exact_fixed_140_fusion_delta_measured": False,
        "selected_mode": full_contract["selected_mode"],
        "image_preprocessing": full_contract["image_preprocessing"],
        "adapter_model_sha256": full_contract["adapter_model_sha256"],
        "adapter_bytes": full_contract["adapter_bytes"],
        "teacher_in_submission": False,
        "base_model_in_submission": False,
        "submission_base_model": "Qwen/Qwen3.5-4B",
        "unchanged_member_count": len(unchanged_hashes),
        "unchanged_members_manifest_sha256": canonical_sha256(unchanged_hashes),
        "output_sha256": sha256(args.output),
        "output_bytes": args.output.stat().st_size,
        "under_5_gib": args.output.stat().st_size < MAX_BYTES,
        "zip_integrity": "PASS",
        "runtime_smoke": "PENDING",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
