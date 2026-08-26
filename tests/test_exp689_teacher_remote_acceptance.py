from __future__ import annotations

import json
import os
import sys
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

EXPERIMENT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "689_qwen35_4b_grounded_transaction_graph_kd"
)
sys.path.insert(0, str(EXPERIMENT))

import build_target_audit as common
import build_teacher_model_contract as model_builder
import build_teacher_smoke as smoke_builder
import verify_teacher_run as verifier


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def make_model_contract(root: Path) -> tuple[Path, dict[str, Any]]:
    model_root = root / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text('{"model_type":"qwen3_6"}\n')
    (model_root / "model.safetensors").write_bytes(b"base-weights")
    (model_root / "tokenizer.json").write_text('{"version":"1"}\n')
    (model_root / "processor_config.json").write_text('{"image_size":64}\n')
    contract_path = root / "model_contract.json"
    contract = model_builder.build(
        remote_root=root, model_root=model_root, output_path=contract_path
    )
    return contract_path, contract


def make_bundle(root: Path) -> tuple[Path, str, dict[str, str]]:
    bundle_path = root / "teacher_code.zip"
    members = {
        "build_target_audit.py": b"build-contract",
        "frozen_target_audit_spec.json": b"{}\n",
        "run_teacher.py": b"frozen-runner",
        "target_audit_schema_v1.json": b"{}\n",
    }
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    member_shas = {name: common.sha256_bytes(payload) for name, payload in sorted(members.items())}
    return bundle_path, common.sha256_file(bundle_path), member_shas


def valid_target(request: dict[str, Any]) -> dict[str, Any]:
    evidence_id = request["evidence_candidates"][0]["candidate_id"]
    return {
        "schema_version": "exp689_teacher_selection_row_v2",
        "audit_id": request["audit_id"],
        "record_id": request["record_id"],
        "request_row_sha256": request["request_row_sha256"],
        "sold_object": "fuel_consumable",
        "substance": "solid_fuel",
        "relation": "included",
        "object_evidence_candidate_ids": [evidence_id],
        "substance_evidence_candidate_ids": [evidence_id],
        "relation_evidence_candidate_ids": [evidence_id],
        "support_status": "supported",
        "supervise": True,
    }


def expand_inputs(smoke_dir: Path, root: Path, rows: int) -> tuple[Path, Path]:
    base_requests = read_jsonl(smoke_dir / "teacher_request.jsonl")
    base_images = read_jsonl(smoke_dir / "image_manifest.jsonl")
    requests: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    for index in range(rows):
        request = deepcopy(base_requests[index % len(base_requests)])
        request["audit_id"] = f"G689-{index + 1:03d}"
        request["record_id"] = f"remote-record-{index + 1:03d}"
        request["request_row_sha256"] = None
        request["request_row_sha256"] = common.request_row_hash(request)
        requests.append(request)
        image = deepcopy(base_images[index % len(base_images)])
        image["audit_id"] = request["audit_id"]
        image["reference"] = request["first_image"]["reference"]
        images.append(image)
    request_path = root / f"requests_{rows}.jsonl"
    image_manifest_path = root / f"images_{rows}.jsonl"
    common.write_jsonl(request_path, requests)
    common.write_jsonl(image_manifest_path, images)
    return request_path, image_manifest_path


def make_runner_output(
    root: Path,
    *,
    scope: str,
    request_path: Path,
    image_manifest_path: Path,
    model_contract: dict[str, Any],
    runner_sha: str,
    prompt_sha: str,
) -> tuple[Path, dict[str, str], dict[str, int], str]:
    output_dir = root / "runner_output"
    output_dir.mkdir()
    requests = read_jsonl(request_path)
    targets = [valid_target(request) for request in requests]
    targets_path = output_dir / "targets.jsonl"
    common.write_jsonl(targets_path, targets)
    source_sha = common.sha256_file(request_path)
    pixel_set_sha = common.sha256_bytes(
        common.canonical_json_bytes(
            [row["pixel_sha256"] for row in read_jsonl(image_manifest_path)]
        )
    )
    selection_sha = common.sha256_bytes(common.canonical_json_bytes(targets))
    checks = {key: True for key in verifier.TECHNICAL_CHECKS}
    packages = {"torch": "test", "transformers": "test"}
    report = common.with_self_hash(
        {
            "schema_version": "exp689_teacher_run_report_v1",
            "experiment_id": "689",
            "scope": scope,
            "model_id": model_builder.MODEL_ID,
            "model_revision": model_builder.MODEL_REVISION,
            "model_tree_sha256": model_contract["model_tree_sha256"],
            "processor_sha256": model_contract["processor_sha256"],
            "code_sha256": runner_sha,
            "prompt_sha256": prompt_sha,
            "source_sha256": source_sha,
            "image_manifest_sha256": common.sha256_file(image_manifest_path),
            "pixel_set_sha256": pixel_set_sha,
            "input_contract_self_sha256": "1" * 64,
            "accepted_smoke_self_sha256": None if scope == "technical_smoke" else "2" * 64,
            "selection_payload_sha256": selection_sha,
            "targets_sha256": common.sha256_file(targets_path),
            "rows": len(targets),
            "decoding": {
                "do_sample": False,
                "enable_thinking": False,
                "max_new_tokens": 160,
                "repair_attempts": 0,
            },
            "runtime_seconds": 12.5,
            "peak_cuda_bytes": 60 * 1024**3,
            "total_cuda_bytes": 80 * 1024**3,
            "cuda_device_name": "NVIDIA H100 80GB HBM3",
            "model_class": "Qwen3_6ForConditionalGeneration",
            "packages": packages,
            "packages_sha256": common.sha256_bytes(common.canonical_json_bytes(packages)),
            "technical_checks": checks,
            "labels_read": 0,
            "sealed_rows": 0,
            "public_used": False,
            "quality_evaluated": False,
            "student_gpu_authorized": False,
            "self_sha256": None,
        }
    )
    report_path = output_dir / "report.json"
    write_json(report_path, report)
    runner_acceptance = common.with_self_hash(
        {
            "schema_version": "exp689_teacher_run_acceptance_v1",
            "experiment_id": "689",
            "scope": scope,
            "status": "accepted",
            "decision": (
                "TECHNICAL_SMOKE_PASS_NOT_QUALITY"
                if scope == "technical_smoke"
                else "TARGETS_READY_FOR_DUAL_BLIND_REVIEW"
            ),
            "rows": len(targets),
            "model_id": model_builder.MODEL_ID,
            "model_revision": model_builder.MODEL_REVISION,
            "model_tree_sha256": model_contract["model_tree_sha256"],
            "processor_sha256": model_contract["processor_sha256"],
            "code_sha256": runner_sha,
            "prompt_sha256": prompt_sha,
            "source_sha256": source_sha,
            "pixel_set_sha256": pixel_set_sha,
            "selection_payload_sha256": selection_sha,
            "targets_sha256": common.sha256_file(targets_path),
            "report_sha256": common.sha256_file(report_path),
            "runtime_seconds": 12.5,
            "peak_cuda_bytes": 60 * 1024**3,
            "technical_checks": checks,
            "labels_read": 0,
            "sealed_rows": 0,
            "public_used": False,
            "quality_evaluated": False,
            "full_teacher_authorized": False,
            "student_gpu_authorized": False,
            "terminal_job_metadata_bound": False,
            "approved_remote_output_bound": False,
            "self_sha256": None,
        }
    )
    write_json(output_dir / "acceptance.json", runner_acceptance)
    inventory = {name: common.sha256_file(output_dir / name) for name in verifier.OUTPUT_FILES}
    sizes = {name: (output_dir / name).stat().st_size for name in verifier.OUTPUT_FILES}
    return output_dir, inventory, sizes, pixel_set_sha


def make_receipt(
    root: Path,
    *,
    scope: str,
    commit: str,
    bundle_sha: str,
    bundle_members: dict[str, str],
    runner_sha: str,
    request_path: Path,
    image_manifest_path: Path,
    pixel_set_sha: str,
    model_contract_path: Path,
    model_contract: dict[str, Any],
    model_input_identity: str,
    inventory: dict[str, str],
    sizes: dict[str, int],
) -> Path:
    s3_prefix = "s3://approved-runtime/exp689/immutable-run/"
    receipt = common.with_self_hash(
        {
            "schema_version": "exp689_teacher_remote_receipt_v1",
            "experiment_id": "689",
            "scope": scope,
            "job_identity_sha256": common.sha256_text("opaque-job"),
            "terminal_state": "SUCCESS",
            "exit_code": 0,
            "failure_reason": None,
            "submitted_at_utc": "2026-08-26T10:00:00Z",
            "started_at_utc": "2026-08-26T10:01:00Z",
            "finished_at_utc": "2026-08-26T10:20:00Z",
            "attempt": 1,
            "gpu_count": 1,
            "gpu_model": "NVIDIA H100 80GB HBM3",
            "commit_sha": commit,
            "code_bundle_sha256": bundle_sha,
            "code_bundle_members": bundle_members,
            "runner_sha256": runner_sha,
            "teacher_request_sha256": common.sha256_file(request_path),
            "image_manifest_sha256": common.sha256_file(image_manifest_path),
            "pixel_set_sha256": pixel_set_sha,
            "model_registry_input": {
                "input_identity_sha256": model_input_identity,
                "model_id": model_builder.MODEL_ID,
                "model_revision": model_builder.MODEL_REVISION,
                "model_contract_sha256": common.sha256_file(model_contract_path),
                "model_contract_self_sha256": model_contract["self_sha256"],
                "model_tree_sha256": model_contract["model_tree_sha256"],
                "processor_sha256": model_contract["processor_sha256"],
            },
            "output_ref": {
                "schema_version": "exp689_immutable_s3_output_ref_v1",
                "s3_prefix": s3_prefix,
                "immutable": True,
                "objects": [
                    {
                        "name": name,
                        "uri": s3_prefix + name,
                        "version_id": f"version-{index + 1}",
                        "sha256": inventory[name],
                        "size": sizes[name],
                    }
                    for index, name in enumerate(sorted(verifier.OUTPUT_FILES))
                ],
            },
            "runner_runtime_seconds": 12.5,
            "peak_cuda_bytes": 60 * 1024**3,
            "labels_read": 0,
            "sealed_rows": 0,
            "public_used": False,
            "self_sha256": None,
        }
    )
    path = root / "remote_receipt.json"
    write_json(path, receipt)
    return path


def make_packet(tmp_path: Path, scope: str) -> dict[str, Any]:
    smoke_dir = tmp_path / "synthetic"
    smoke_builder.build(remote_root=tmp_path, output_dir=smoke_dir)
    rows = 12 if scope == "technical_smoke" else 300
    request_path, image_manifest_path = expand_inputs(smoke_dir, tmp_path, rows)
    model_contract_path, model_contract = make_model_contract(tmp_path)
    bundle_path, bundle_sha, bundle_members = make_bundle(tmp_path)
    runner_sha = bundle_members["run_teacher.py"]
    prompt_sha = common.sha256_text("frozen-prompt")
    output_dir, inventory, sizes, pixel_set_sha = make_runner_output(
        tmp_path,
        scope=scope,
        request_path=request_path,
        image_manifest_path=image_manifest_path,
        model_contract=model_contract,
        runner_sha=runner_sha,
        prompt_sha=prompt_sha,
    )
    commit = "a" * 40
    model_input_identity = common.sha256_text("opaque-model-registry-input")
    receipt_path = make_receipt(
        tmp_path,
        scope=scope,
        commit=commit,
        bundle_sha=bundle_sha,
        bundle_members=bundle_members,
        runner_sha=runner_sha,
        request_path=request_path,
        image_manifest_path=image_manifest_path,
        pixel_set_sha=pixel_set_sha,
        model_contract_path=model_contract_path,
        model_contract=model_contract,
        model_input_identity=model_input_identity,
        inventory=inventory,
        sizes=sizes,
    )
    return {
        "remote_root": tmp_path,
        "scope": scope,
        "output_dir": output_dir,
        "receipt_path": receipt_path,
        "teacher_request_path": request_path,
        "image_manifest_path": image_manifest_path,
        "model_contract_path": model_contract_path,
        "code_bundle_path": bundle_path,
        "expected_commit": commit,
        "expected_bundle_sha256": bundle_sha,
        "expected_runner_sha256": runner_sha,
        "expected_prompt_sha256": prompt_sha,
        "expected_source_sha256": common.sha256_file(request_path),
        "expected_image_manifest_sha256": common.sha256_file(image_manifest_path),
        "expected_pixel_set_sha256": pixel_set_sha,
        "expected_accepted_smoke_self_sha256": (
            None if scope == "technical_smoke" else "2" * 64
        ),
        "expected_model_contract_sha256": common.sha256_file(model_contract_path),
        "expected_model_input_identity_sha256": model_input_identity,
        "approved_output_prefix": "s3://approved-runtime/exp689/",
        "acceptance_output_path": tmp_path / "remote_acceptance.json",
    }


@pytest.mark.parametrize(
    ("scope", "decision"),
    [
        ("technical_smoke", "OPEN_FULL_TEACHER"),
        ("full", "TARGETS_READY_FOR_DUAL_BLIND_REVIEW"),
    ],
)
def test_remote_verifier_accepts_exact_smoke_and_full(
    tmp_path: Path, scope: str, decision: str
) -> None:
    packet = make_packet(tmp_path, scope)
    result = verifier.verify(**packet)
    assert result["status"] == "accepted"
    assert result["decision"] == decision
    assert result["quality_evaluated"] is False
    assert result["student_gpu_authorized"] is False
    assert result["full_teacher_technical_gate_open"] is (scope == "technical_smoke")
    common.validate_self_hash(result, "remote acceptance")


def rewrite_receipt(packet: dict[str, Any], mutate: Any) -> None:
    path = packet["receipt_path"]
    receipt = json.loads(path.read_text(encoding="utf-8"))
    mutate(receipt)
    receipt = common.with_self_hash({**receipt, "self_sha256": None})
    write_json(path, receipt)


def test_nonterminal_receipt_fails_closed(tmp_path: Path) -> None:
    packet = make_packet(tmp_path, "technical_smoke")
    rewrite_receipt(packet, lambda receipt: receipt.update({"terminal_state": "RUNNING"}))
    with pytest.raises(common.ContractError, match="terminal SUCCESS"):
        verifier.verify(**packet)


def test_unapproved_or_unversioned_s3_output_fails_closed(tmp_path: Path) -> None:
    packet = make_packet(tmp_path, "technical_smoke")

    def mutate(receipt: dict[str, Any]) -> None:
        receipt["output_ref"]["objects"][0]["version_id"] = ""

    rewrite_receipt(packet, mutate)
    with pytest.raises(common.ContractError, match="version ID"):
        verifier.verify(**packet)


def test_output_inventory_rejects_extra_member(tmp_path: Path) -> None:
    packet = make_packet(tmp_path, "technical_smoke")
    (packet["output_dir"] / "raw_generation.txt").write_text("forbidden\n")
    with pytest.raises(common.ContractError, match="inventory mismatch"):
        verifier.verify(**packet)


def test_code_bundle_rejects_unknown_member(tmp_path: Path) -> None:
    packet = make_packet(tmp_path, "technical_smoke")
    bundle = packet["code_bundle_path"]
    with zipfile.ZipFile(bundle, "a") as archive:
        archive.writestr("unknown.py", b"forbidden")
    packet["expected_bundle_sha256"] = common.sha256_file(bundle)
    with pytest.raises(common.ContractError, match="whitelist mismatch"):
        verifier.verify(**packet)


def test_source_sha_pin_and_registry_identity_are_fail_closed(tmp_path: Path) -> None:
    packet = make_packet(tmp_path, "technical_smoke")
    packet["expected_source_sha256"] = "0" * 64
    with pytest.raises(common.ContractError, match="source SHA pin"):
        verifier.verify(**packet)


def test_labels_sealed_or_public_receipt_is_rejected(tmp_path: Path) -> None:
    packet = make_packet(tmp_path, "technical_smoke")
    rewrite_receipt(packet, lambda receipt: receipt.update({"labels_read": 1}))
    with pytest.raises(common.ContractError, match="labels, sealed rows, or Public"):
        verifier.verify(**packet)


def test_model_contract_builder_is_deterministic_and_immutable(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_path, first = make_model_contract(first_root)
    second_path, second = make_model_contract(second_root)
    assert first == second
    assert common.sha256_file(first_path) == common.sha256_file(second_path)
    with pytest.raises(FileExistsError, match="immutable"):
        model_builder.build(
            remote_root=first_root,
            model_root=first_root / "model",
            output_path=first_path,
        )


def test_model_contract_builder_rejects_adapter_and_symlink(tmp_path: Path) -> None:
    adapter_root = tmp_path / "adapter-case"
    adapter_root.mkdir()
    model_root = adapter_root / "model"
    model_root.mkdir()
    (model_root / "tokenizer.json").write_text("{}\n")
    (model_root / "adapter_config.json").write_text("{}\n")
    with pytest.raises(common.ContractError, match="adapter/class-LoRA"):
        model_builder.build(
            remote_root=adapter_root,
            model_root=model_root,
            output_path=adapter_root / "contract.json",
        )

    symlink_root = tmp_path / "symlink-case"
    symlink_root.mkdir()
    symlink_model = symlink_root / "model"
    symlink_model.mkdir()
    target = symlink_model / "tokenizer.json"
    target.write_text("{}\n")
    os.symlink(target, symlink_model / "processor_config.json")
    with pytest.raises(common.ContractError, match="forbidden symlink"):
        model_builder.build(
            remote_root=symlink_root,
            model_root=symlink_model,
            output_path=symlink_root / "contract.json",
        )
