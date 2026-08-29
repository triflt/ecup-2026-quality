from __future__ import annotations

import json
import sys
from argparse import Namespace
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
import build_teacher_smoke as smoke
import run_teacher as teacher


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def make_model(root: Path) -> tuple[Path, Path]:
    model_root = root / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text('{"architectures":["Qwen3_6"]}\n')
    (model_root / "model.safetensors").write_bytes(b"synthetic-model-weights")
    (model_root / "tokenizer.json").write_text('{"version":"1"}\n')
    (model_root / "processor_config.json").write_text('{"image_size":64}\n')
    model_sha, model_files = teacher.tree_sha256(model_root)
    processor_sha, processor_files = teacher.tree_sha256(
        model_root, predicate=teacher.is_processor_file
    )
    contract = common.with_self_hash(
        {
            "schema_version": "exp689_teacher_model_contract_v1",
            "model_id": teacher.MODEL_ID,
            "model_revision": teacher.MODEL_REVISION,
            "model_tree_sha256": model_sha,
            "model_tree_files": model_files,
            "processor_sha256": processor_sha,
            "processor_files": processor_files,
            "base_only": True,
            "class_lora_present": False,
            "self_sha256": None,
        }
    )
    contract_path = root / "model_contract.json"
    write_json(contract_path, contract)
    return model_root, contract_path


def valid_target(request: dict[str, Any]) -> dict[str, Any]:
    evidence_id = request["evidence_candidates"][0]["candidate_id"]
    return {
        "schema_version": "exp689_teacher_selection_row_v2",
        "audit_id": request["audit_id"],
        "record_id": request["record_id"],
        "request_row_sha256": request["request_row_sha256"],
        "sold_object": "device",
        "substance": "none",
        "relation": "absent",
        "object_evidence_candidate_ids": [evidence_id],
        "substance_evidence_candidate_ids": [evidence_id],
        "relation_evidence_candidate_ids": [evidence_id],
        "support_status": "supported",
        "supervise": True,
    }


def fake_backend(
    args: Namespace, requests: list[dict[str, Any]], images: list[Any]
) -> tuple[list[str], dict[str, Any]]:
    assert args.scope == "technical_smoke"
    assert len(images) == len(requests) == 12
    return [json.dumps(valid_target(request)) for request in requests], {
        "cuda_device_count": 1,
        "cuda_device_name": "NVIDIA H100 80GB HBM3",
        "cuda_forward_rows": 12,
        "pixel_tensor_rows": 12,
        "visible_image_regions": 60,
        "cpu_offload": False,
        "disk_offload": False,
        "base_only": True,
        "class_lora_present": False,
        "model_class": "Qwen3_6ForConditionalGeneration",
        "runtime_seconds": 1.25,
        "peak_cuda_bytes": 1024,
        "total_cuda_bytes": 80 * 1024**3,
        "packages": {"torch": "test", "transformers": "test"},
    }


def smoke_args(root: Path, smoke_dir: Path, model_root: Path, model_contract: Path) -> Namespace:
    return Namespace(
        scope="technical_smoke",
        remote_root=root,
        teacher_request=smoke_dir / "teacher_request.jsonl",
        input_contract=smoke_dir / "teacher_smoke_contract.json",
        image_root=smoke_dir,
        image_manifest=smoke_dir / "image_manifest.jsonl",
        model_root=model_root,
        model_contract=model_contract,
        accepted_smoke=None,
        accepted_smoke_sha256=None,
        accepted_smoke_gate=None,
        accepted_smoke_gate_sha256=None,
        output_dir=root / "teacher_output",
    )


def build_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    smoke_dir = tmp_path / "smoke"
    smoke.build(remote_root=tmp_path, output_dir=smoke_dir)
    model_root, model_contract = make_model(tmp_path)
    return smoke_dir, model_root, model_contract


def test_smoke_builder_is_deterministic_and_structurally_complete(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    contract_a = smoke.build(remote_root=tmp_path, output_dir=first)
    contract_b = smoke.build(remote_root=tmp_path, output_dir=second)
    assert common.sha256_file(first / "teacher_request.jsonl") == common.sha256_file(
        second / "teacher_request.jsonl"
    )
    assert common.sha256_file(first / "image_manifest.jsonl") == common.sha256_file(
        second / "image_manifest.jsonl"
    )
    assert contract_a == contract_b
    assert contract_a["teacher_request_rows"] == 12
    assert contract_a["structural_enum_coverage"] == {
        key: common.load_spec()["enums"][key]
        for key in ("sold_object", "substance", "relation")
    }
    assert contract_a["quality_evaluated"] is False
    assert contract_a["full_teacher_authorized"] is False


def test_smoke_runner_emits_contract_bound_non_authorizing_artifacts(tmp_path: Path) -> None:
    smoke_dir, model_root, model_contract = build_inputs(tmp_path)
    args = smoke_args(tmp_path, smoke_dir, model_root, model_contract)
    acceptance = teacher.run(args, backend=fake_backend)
    assert acceptance["status"] == "accepted"
    assert acceptance["decision"] == "TECHNICAL_SMOKE_PASS_NOT_QUALITY"
    assert acceptance["rows"] == 12
    assert acceptance["quality_evaluated"] is False
    assert acceptance["full_teacher_authorized"] is False
    assert acceptance["student_gpu_authorized"] is False
    assert len(read_jsonl(args.output_dir / "targets.jsonl")) == 12
    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["decoding"] == {
        "do_sample": False,
        "enable_thinking": False,
        "max_new_tokens": 160,
        "repair_attempts": 0,
    }
    assert report["source_sha256"] == common.sha256_file(args.teacher_request)
    assert report["targets_sha256"] == common.sha256_file(args.output_dir / "targets.jsonl")


def test_full_runner_refuses_without_exact_accepted_smoke() -> None:
    args = Namespace(
        scope="full",
        accepted_smoke=None,
        accepted_smoke_sha256=None,
        accepted_smoke_gate=None,
        accepted_smoke_gate_sha256=None,
    )
    with pytest.raises(common.ContractError, match="remote smoke acceptance and promotion"):
        teacher.run(args)


def test_accepted_smoke_gate_binds_code_prompt_model_and_processor(tmp_path: Path) -> None:
    smoke_dir, model_root, model_contract_path = build_inputs(tmp_path)
    args = smoke_args(tmp_path, smoke_dir, model_root, model_contract_path)
    teacher.run(args, backend=fake_backend)
    model_contract = json.loads(model_contract_path.read_text(encoding="utf-8"))
    inventory = {
        name: common.sha256_file(args.output_dir / name)
        for name in ("acceptance.json", "report.json", "targets.jsonl")
    }
    acceptance = common.with_self_hash(
        {
            "schema_version": "exp689_teacher_remote_acceptance_v1",
            "experiment_id": "689",
            "scope": "technical_smoke",
            "status": "accepted",
            "decision": "OPEN_FULL_TEACHER",
            "technical_only": True,
            "quality_evaluated": False,
            "student_gpu_authorized": False,
            "terminal_job_metadata_bound": True,
            "approved_remote_output_bound": True,
            "full_teacher_technical_gate_open": True,
            "commit_sha": "a" * 40,
            "code_bundle_sha256": "1" * 64,
            "code_bundle_members": {"run_teacher.py": common.sha256_file(Path(teacher.__file__))},
            "runner_sha256": common.sha256_file(Path(teacher.__file__)),
            "prompt_sha256": teacher.PROMPT_SHA256,
            "source_sha256": "2" * 64,
            "image_manifest_sha256": "3" * 64,
            "pixel_set_sha256": "4" * 64,
            "accepted_smoke_self_sha256": None,
            "accepted_smoke_promotion_gate_self_sha256": None,
            "model_contract_sha256": common.sha256_file(model_contract_path),
            "model_contract_self_sha256": model_contract["self_sha256"],
            "model_registry_input_identity_sha256": "5" * 64,
            "model_tree_sha256": model_contract["model_tree_sha256"],
            "processor_sha256": model_contract["processor_sha256"],
            "runner_output_inventory": inventory,
            "runner_output_inventory_sha256": common.sha256_bytes(
                common.canonical_json_bytes(inventory)
            ),
            "remote_receipt_self_sha256": "6" * 64,
            "remote_receipt_sha256": "7" * 64,
            "remote_output_ref_sha256": "8" * 64,
            "runtime_seconds": 1.25,
            "peak_cuda_bytes": 1024,
            "labels_read": 0,
            "sealed_rows": 0,
            "public_used": False,
            "jobs_launched_by_verifier": 0,
            "uploads_by_verifier": 0,
            "presets_built_by_verifier": 0,
            "bundles_built_by_verifier": 0,
            "self_sha256": None,
        }
    )
    acceptance_path = tmp_path / "remote_smoke_acceptance.json"
    write_json(acceptance_path, acceptance)
    accepted = teacher.validate_smoke_acceptance(
        acceptance_path,
        common.sha256_file(acceptance_path),
        model_binding=model_contract,
    )
    assert accepted["decision"] == "OPEN_FULL_TEACHER"
    assert accepted["terminal_job_metadata_bound"] is True
    with pytest.raises(common.ContractError, match="file SHA mismatch"):
        teacher.validate_smoke_acceptance(
            acceptance_path, "0" * 64, model_binding=model_contract
        )


def test_full_runner_rejects_runner_local_smoke_acceptance(tmp_path: Path) -> None:
    smoke_dir, model_root, model_contract_path = build_inputs(tmp_path)
    args = smoke_args(tmp_path, smoke_dir, model_root, model_contract_path)
    teacher.run(args, backend=fake_backend)
    local_acceptance = args.output_dir / "acceptance.json"
    model_contract = json.loads(model_contract_path.read_text(encoding="utf-8"))
    with pytest.raises(common.ContractError, match="exact fields|required"):
        teacher.validate_smoke_acceptance(
            local_acceptance,
            common.sha256_file(local_acceptance),
            model_binding=model_contract,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update({"confidence": 1.0}), "exact fields required"),
        (
            lambda row: row.update({"relation_evidence_candidate_ids": ["f" * 64]}),
            "nonexistent evidence candidate ID",
        ),
        (lambda row: row.pop("relation"), "exact fields required"),
    ],
)
def test_generated_target_fails_on_extra_unknown_or_missing_fields(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    smoke_dir = tmp_path / "smoke"
    smoke.build(remote_root=tmp_path, output_dir=smoke_dir)
    request = read_jsonl(smoke_dir / "teacher_request.jsonl")[0]
    target = valid_target(request)
    mutation(target)
    with pytest.raises(common.ContractError, match=message):
        teacher.parse_generated_target(json.dumps(target), request)


def test_generated_target_does_not_repair_markdown_or_multiple_json(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "smoke"
    smoke.build(remote_root=tmp_path, output_dir=smoke_dir)
    request = read_jsonl(smoke_dir / "teacher_request.jsonl")[0]
    target = json.dumps(valid_target(request))
    for raw in (f"```json\n{target}\n```", f"{target}\n{target}"):
        with pytest.raises(common.ContractError, match="no repair allowed"):
            teacher.parse_generated_target(raw, request)


def test_image_hash_failure_is_fail_closed(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "smoke"
    smoke.build(remote_root=tmp_path, output_dir=smoke_dir)
    first_image = smoke_dir / "images" / "smoke_001.png"
    first_image.write_bytes(first_image.read_bytes() + b"corruption")
    requests = read_jsonl(smoke_dir / "teacher_request.jsonl")
    with pytest.raises(common.ContractError, match="encoded image SHA mismatch"):
        teacher.load_images(smoke_dir, smoke_dir / "image_manifest.jsonl", requests)


def test_multimodal_content_exposes_exact_full_and_2x2_regions(tmp_path: Path) -> None:
    smoke_dir = tmp_path / "smoke"
    smoke.build(remote_root=tmp_path, output_dir=smoke_dir)
    requests = read_jsonl(smoke_dir / "teacher_request.jsonl")
    request = requests[0]
    images, _ = teacher.load_images(
        smoke_dir, smoke_dir / "image_manifest.jsonl", requests
    )
    image = images[0]
    try:
        content, visible = teacher.multimodal_content(request, image)
        assert len(visible) == 5
        assert [item["type"] for item in content] == [
            "image", "text", "image", "text", "image", "text",
            "image", "text", "image", "text", "text",
        ]
        assert [item["image"].size for item in content if item["type"] == "image"] == [
            (64, 64), (32, 32), (32, 32), (32, 32), (32, 32)
        ]
        for region, label in zip(teacher.IMAGE_REGIONS, content[1:10:2], strict=True):
            assert f"IMAGE_REGION={region}" in label["text"]
    finally:
        for visible_image in visible[1:]:
            visible_image.close()
        for loaded_image in images:
            loaded_image.close()


def test_base_model_contract_rejects_adapter_or_class_lora_artifact(tmp_path: Path) -> None:
    model_root, model_contract_path = make_model(tmp_path)
    (model_root / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    contract = json.loads(model_contract_path.read_text(encoding="utf-8"))
    with pytest.raises(common.ContractError, match="adapter/class-LoRA"):
        teacher.validate_base_model_tree(model_root, contract)


def test_output_directory_is_immutable(tmp_path: Path) -> None:
    smoke_dir, model_root, model_contract = build_inputs(tmp_path)
    args = smoke_args(tmp_path, smoke_dir, model_root, model_contract)
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError, match="immutable teacher output"):
        teacher.run(args, backend=fake_backend)
