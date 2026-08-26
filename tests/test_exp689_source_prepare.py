from __future__ import annotations

import csv
import io
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

EXPERIMENT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "689_qwen35_4b_grounded_transaction_graph_kd"
)
sys.path.insert(0, str(EXPERIMENT))

import build_target_audit as target
import prepare_source_universe as source
import verify_source_prepare as verifier

try:
    import run_teacher as teacher
except ModuleNotFoundError:
    teacher = None


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(source.canonical_json_bytes(row) + b"\n" for row in rows))


def image_bytes(size: tuple[int, int] = (32, 24)) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", size, (73, 31, 149)).save(stream, format="PNG")
    return stream.getvalue()


def make_row(fold: int, kind: str, index: int) -> dict[str, Any]:
    if kind == "device":
        name = f"Газовая горелка {fold}-{index}"
        description = "Устройство совместимо с газовым баллоном. Баллон приобретается отдельно."
    elif kind == "direct":
        name = f"Газ пропан топливный {fold}-{index}"
        description = "Горючий товар в заводской упаковке; предназначен для бытового применения."
    elif kind == "singleton_cue_free":
        name = f"Керамическая ваза {fold}-{index}"
        description = "Декоративный товар. Матовая поверхность."
    else:
        name = f"Керамическая ваза совместимая {fold}-{index}"
        description = "Совместимый аксессуар."
    row_id = f"row-{fold}-{kind}-{index:02d}"
    return {
        "category": "Легковоспламеняющиеся",
        "description": description,
        "fold": fold,
        "global_index": fold * 1000 + len(kind) * 100 + index,
        "id": row_id,
        "image_url": f"memory://{row_id}",
        "name": name,
        "ocr_images": [
            {
                "image_index": 0,
                "detections": [
                    {
                        "text": "FORGED_UNVERIFIED_OCR_LABEL_LIKE_TEXT",
                        "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    }
                ],
            }
        ],
        "semantic_component": f"component-{fold}-{kind}-{index:02d}",
    }


def make_inputs(root: Path) -> dict[str, Any]:
    spec = deepcopy(source.load_spec())
    spec["folds"] = {}
    runtime_dirs: list[Path] = []
    for fold in range(5):
        runtime_dir = root / f"runtime-fold-{fold}"
        runtime_dir.mkdir(parents=True)
        rows = (
            [make_row(fold, "device", index) for index in range(20)]
            + [make_row(fold, "direct", index) for index in range(20)]
            + [make_row(fold, "singleton_cue_free", index) for index in range(10)]
            + [make_row(fold, "singleton_other", index) for index in range(10)]
        )
        validation = runtime_dir / "validation.jsonl"
        write_jsonl(validation, rows)
        audit = {
            "schema_version": 1,
            "experiment_id": "641",
            "outer_fold": fold,
            "input_sha256": {
                "data": spec["input_data_sha256"],
                "folds": spec["input_registry_sha256"],
            },
            "output_sha256": {"validation.jsonl": source.sha256_file(validation)},
            "decision": "GO",
        }
        audit["contract_sha256"] = source.sha256_bytes(source.canonical_json_bytes(audit))
        write_json(runtime_dir / "runtime_audit.json", audit)
        spec["folds"][str(fold)] = {
            "source_runtime_contract_sha256": audit["contract_sha256"],
            "validation_rows": len(rows),
            "validation_sha256": source.sha256_file(validation),
        }
        runtime_dirs.append(runtime_dir)

    exclusion_670 = root / "exp670.csv"
    with exclusion_670.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["semantic_component"])
        writer.writeheader()
        writer.writerows(
            {"semantic_component": f"old-670-{index:03d}"} for index in range(300)
        )
    exclusion_672 = root / "exp672.json"
    write_json(
        exclusion_672,
        {"ordered_components": [f"old-672-{index:03d}" for index in range(40)]},
    )
    spec["self_sha256"] = None
    spec["self_sha256"] = source.sha256_bytes(source.canonical_json_bytes(spec))
    return {
        "spec": spec,
        "runtime_dirs": runtime_dirs,
        "exclusion_670_path": exclusion_670,
        "exclusion_672_path": exclusion_672,
        "exclusion_670_sha256": source.sha256_file(exclusion_670),
        "exclusion_672_sha256": source.sha256_file(exclusion_672),
        "builder_revision": "a" * 40,
    }


def run_prepare(root: Path, inputs: dict[str, Any], name: str = "prepared", **changes: Any) -> Path:
    arguments = dict(inputs)
    arguments.update(changes)
    output = root / name
    source.prepare(
        **arguments,
        output_dir=output,
        image_fetcher=lambda _url: image_bytes(),
    )
    return output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_prepare_builds_exact_300_compatible_rows(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    output = run_prepare(tmp_path, inputs)
    rows = read_jsonl(output / "source_rows.jsonl")
    source_rows_text = source.canonical_json_bytes(rows).decode()
    contract = json.loads((output / "source_contract.json").read_text())
    manifest = read_jsonl(output / "images" / "image_manifest.jsonl")

    assert len(rows) == len(manifest) == 300
    assert [row["record_id"] for row in rows] == [f"G689-{index:03d}" for index in range(1, 301)]
    assert len({row["component_token"] for row in rows}) == 300
    assert {row["stratum"] for row in rows} == {
        "direct_included_fuel",
        "device_accessory_compatible_mention",
        "singleton_new_family_ambiguous",
    }
    assert all(len(row["evidence_candidates"]) >= 6 for row in rows)
    assert all(source_item["source_kind"] != "ocr" for row in rows for source_item in row["sources"])
    assert "FORGED_UNVERIFIED_OCR_LABEL_LIKE_TEXT" not in source_rows_text
    target.validate_source_contract(contract, output / "source_rows.jsonl", 300)
    target_spec = target.load_spec()
    for index, row in enumerate(rows):
        target.validate_source_row(row, target_spec, index)
    assert not any(
        forbidden in source_rows_text.lower()
        for forbidden in ('"label"', '"score"', '"target"')
    )


def test_prepare_is_byte_deterministic(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    first = run_prepare(tmp_path, inputs, "first")
    second = run_prepare(tmp_path, inputs, "second")
    for relative in ("source_rows.jsonl", "source_contract.json", "images/image_manifest.jsonl"):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
    assert sorted(path.name for path in (first / "images").glob("*.jpg")) == sorted(
        path.name for path in (second / "images").glob("*.jpg")
    )


def test_existing_output_is_rejected_before_fetch(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    calls = 0

    def fetch(_url: str) -> bytes:
        nonlocal calls
        calls += 1
        return image_bytes()

    with pytest.raises(FileExistsError, match="existing output"):
        source.prepare(**inputs, output_dir=output, image_fetcher=fetch)
    assert calls == 0


@pytest.mark.parametrize("field", ["label", "target", "teacher_score"])
def test_supervision_fields_are_rejected(tmp_path: Path, field: str) -> None:
    inputs = make_inputs(tmp_path)
    validation = inputs["runtime_dirs"][0] / "validation.jsonl"
    rows = read_jsonl(validation)
    rows[0][field] = 1
    write_jsonl(validation, rows)
    inputs["spec"]["folds"]["0"]["validation_sha256"] = source.sha256_file(validation)
    audit_path = inputs["runtime_dirs"][0] / "runtime_audit.json"
    audit = json.loads(audit_path.read_text())
    audit.pop("contract_sha256")
    audit["output_sha256"]["validation.jsonl"] = source.sha256_file(validation)
    audit["contract_sha256"] = source.sha256_bytes(source.canonical_json_bytes(audit))
    write_json(audit_path, audit)
    inputs["spec"]["folds"]["0"]["source_runtime_contract_sha256"] = audit["contract_sha256"]
    with pytest.raises(source.PrepareError, match="forbidden field"):
        run_prepare(tmp_path, inputs)


def test_duplicate_id_and_cross_fold_component_are_rejected(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    validation = inputs["runtime_dirs"][1] / "validation.jsonl"
    rows = read_jsonl(validation)
    rows[0]["id"] = read_jsonl(inputs["runtime_dirs"][0] / "validation.jsonl")[0]["id"]
    write_jsonl(validation, rows)
    inputs["spec"]["folds"]["1"]["validation_sha256"] = source.sha256_file(validation)
    audit_path = inputs["runtime_dirs"][1] / "runtime_audit.json"
    audit = json.loads(audit_path.read_text())
    audit.pop("contract_sha256")
    audit["output_sha256"]["validation.jsonl"] = source.sha256_file(validation)
    audit["contract_sha256"] = source.sha256_bytes(source.canonical_json_bytes(audit))
    write_json(audit_path, audit)
    inputs["spec"]["folds"]["1"]["source_runtime_contract_sha256"] = audit["contract_sha256"]
    with pytest.raises(source.PrepareError, match="duplicate ID"):
        run_prepare(tmp_path, inputs)


def test_exclusion_sha_and_cardinality_are_frozen(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    with pytest.raises(source.PrepareError, match="frozen input SHA"):
        run_prepare(tmp_path, inputs, exclusion_670_sha256="0" * 64)

    with inputs["exclusion_672_path"].open("w", encoding="utf-8") as stream:
        json.dump({"ordered_components": ["one"]}, stream)
    inputs["exclusion_672_sha256"] = source.sha256_file(inputs["exclusion_672_path"])
    with pytest.raises(source.PrepareError, match="unique component count"):
        run_prepare(tmp_path, inputs, name="bad-count")


@pytest.mark.parametrize("payload", [b"", b"not-an-image"])
def test_image_failure_is_atomic(tmp_path: Path, payload: bytes) -> None:
    inputs = make_inputs(tmp_path)
    output = tmp_path / "prepared"
    with pytest.raises(source.PrepareError, match="first-image"):
        source.prepare(
            **inputs,
            output_dir=output,
            image_fetcher=lambda _url: payload,
        )
    assert not output.exists()


def test_image_is_fetched_once_and_area_capped(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    calls = 0
    large = image_bytes((1024, 1024))

    def fetch(_url: str) -> bytes:
        nonlocal calls
        calls += 1
        return large

    output = tmp_path / "prepared"
    source.prepare(**inputs, output_dir=output, image_fetcher=fetch)
    assert calls == 300
    manifest = read_jsonl(output / "images" / "image_manifest.jsonl")
    assert all(row["width"] * row["height"] <= 262144 for row in manifest)
    assert all(
        set(row["region_sha256"]) == {"full", "q00", "q01", "q10", "q11"}
        for row in manifest
    )


@pytest.mark.skipif(teacher is None, reason="teacher runner lands in a separate commit")
def test_raw_rgb_and_quadrants_load_in_teacher_runner(tmp_path: Path) -> None:
    assert teacher is not None
    inputs = make_inputs(tmp_path)
    output = tmp_path / "prepared"
    source.prepare(
        **inputs,
        output_dir=output,
        image_fetcher=lambda _url: image_bytes((33, 25)),
    )
    verifier_arguments = make_verifier_inputs(tmp_path, inputs, output)
    verifier.verify(**verifier_arguments)
    target_output = tmp_path / "target-prepare"
    target.prepare(
        remote_root=tmp_path,
        source_rows_path=output / "source_rows.jsonl",
        source_contract_path=output / "source_contract.json",
        source_prepare_acceptance_path=verifier_arguments["acceptance_path"],
        source_prepare_acceptance_sha256=target.sha256_file(
            verifier_arguments["acceptance_path"]
        ),
        exclusion_670_path=inputs["exclusion_670_path"],
        exclusion_672_path=inputs["exclusion_672_path"],
        exclusion_670_sha256=inputs["exclusion_670_sha256"],
        exclusion_672_sha256=inputs["exclusion_672_sha256"],
        exclusion_670_adapter="csv_component_column",
        exclusion_672_adapter="json_component_list",
        exclusion_670_component_locator="semantic_component",
        exclusion_672_component_locator="ordered_components",
        exclusion_670_token_mode="sha256_utf8_v1",
        exclusion_672_token_mode="sha256_utf8_v1",
        output_dir=target_output,
    )
    requests = read_jsonl(target_output / "teacher_request.jsonl")
    manifest_path = target_output / "teacher_image_manifest.jsonl"
    manifest = json.loads((target_output / "selection_manifest.json").read_text())
    assert manifest["teacher_image_manifest_sha256"] == target.sha256_file(manifest_path)
    assert manifest["source_prepare_acceptance_sha256"] == target.sha256_file(
        verifier_arguments["acceptance_path"]
    )
    images, pixel_set_sha = teacher.load_images(output, manifest_path, requests)
    assert len(images) == 300
    assert verifier.HEX64.fullmatch(pixel_set_sha)
    for image in images:
        image.close()


def test_quota_failure_happens_before_fetch(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    validation = inputs["runtime_dirs"][4] / "validation.jsonl"
    rows = read_jsonl(validation)[:-1]
    write_jsonl(validation, rows)
    inputs["spec"]["folds"]["4"]["validation_rows"] = len(rows)
    inputs["spec"]["folds"]["4"]["validation_sha256"] = source.sha256_file(validation)
    audit_path = inputs["runtime_dirs"][4] / "runtime_audit.json"
    audit = json.loads(audit_path.read_text())
    audit.pop("contract_sha256")
    audit["output_sha256"]["validation.jsonl"] = source.sha256_file(validation)
    audit["contract_sha256"] = source.sha256_bytes(source.canonical_json_bytes(audit))
    write_json(audit_path, audit)
    inputs["spec"]["folds"]["4"]["source_runtime_contract_sha256"] = audit["contract_sha256"]
    calls = 0

    def fetch(_url: str) -> bytes:
        nonlocal calls
        calls += 1
        return image_bytes()

    with pytest.raises(source.PrepareError, match=r"insufficient 10\+10"):
        source.prepare(**inputs, output_dir=tmp_path / "prepared", image_fetcher=fetch)
    assert calls == 0


def make_verifier_inputs(
    root: Path, inputs: dict[str, Any], prepared: Path
) -> dict[str, Any]:
    bundle_root = root / "bundle"
    relative_root = Path("experiments/689_qwen35_4b_grounded_transaction_graph_kd")
    bundle_experiment = bundle_root / relative_root
    bundle_experiment.mkdir(parents=True)
    for name in ("prepare_source_universe.py", "verify_source_prepare.py"):
        shutil.copyfile(EXPERIMENT / name, bundle_experiment / name)
    spec_path = bundle_experiment / "source_prepare_spec_v1.json"
    write_json(spec_path, inputs["spec"])
    builder_revision = inputs["builder_revision"]
    manifest_files = []
    for relative in sorted(verifier.ALLOWED_BUNDLE_PATHS):
        path = bundle_root / relative
        manifest_files.append(
            {
                "path": relative,
                "sha256": verifier.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    bundle_manifest = {
        "schema_version": "exp689_source_prepare_bundle_manifest_v1",
        "builder_revision": builder_revision,
        "files": manifest_files,
        "self_sha256": None,
    }
    bundle_manifest["self_sha256"] = verifier.sha256_bytes(
        verifier.canonical_json_bytes(bundle_manifest)
    )
    bundle_manifest_path = root / "bundle_manifest.json"
    write_json(bundle_manifest_path, bundle_manifest)

    archives: list[Path] = []
    archive_refs: list[str] = []
    archive_shas: list[str] = []
    for fold in range(5):
        archive = root / f"runtime-{fold}.tar"
        archive.write_bytes(f"exact-runtime-archive-{fold}".encode())
        archives.append(archive)
        archive_refs.append(f"s3://approved-inputs/runtime-fold-{fold}.tar")
        archive_shas.append(verifier.sha256_file(archive))
    output_ref = "s3://approved-output/exp689/source-prepare"
    terminal = {
        "schema_version": "exp689_source_prepare_terminal_v1",
        "job_id": "source-prepare-test-job",
        "status": "SUCCESS",
        "finished_at": "2026-08-26T01:00:00Z",
        "output_ref": output_ref,
        "self_sha256": None,
    }
    terminal["self_sha256"] = verifier.sha256_bytes(
        verifier.canonical_json_bytes(terminal)
    )
    terminal_path = root / "terminal.json"
    write_json(terminal_path, terminal)
    return {
        "prepare_dir": prepared,
        "runtime_dirs": inputs["runtime_dirs"],
        "runtime_archives": archives,
        "runtime_archive_refs": archive_refs,
        "runtime_archive_sha256": archive_shas,
        "bundle_root": bundle_root,
        "bundle_manifest_path": bundle_manifest_path,
        "bundle_manifest_sha256": verifier.sha256_file(bundle_manifest_path),
        "builder_revision": builder_revision,
        "spec_path": spec_path,
        "terminal_metadata_path": terminal_path,
        "terminal_metadata_sha256": verifier.sha256_file(terminal_path),
        "approved_s3_output_ref": output_ref,
        "acceptance_path": root / "source_prepare_acceptance.json",
    }


def test_independent_verifier_emits_self_hashed_acceptance(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    prepared = run_prepare(tmp_path, inputs)
    arguments = make_verifier_inputs(tmp_path, inputs, prepared)
    acceptance = verifier.verify(**arguments)
    assert acceptance["decision"] == "ACCEPT"
    assert acceptance["source_row_count"] == 300
    assert len(acceptance["runtime_archives"]) == 5
    assert len(acceptance["output_inventory"]) == 303
    verifier._self_hash(acceptance, "acceptance")
    assert json.loads(arguments["acceptance_path"].read_text()) == acceptance


def test_verifier_rejects_output_byte_tamper(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    prepared = run_prepare(tmp_path, inputs)
    arguments = make_verifier_inputs(tmp_path, inputs, prepared)
    image = prepared / "images" / "G689-001.jpg"
    image.write_bytes(image.read_bytes() + b"tamper")
    with pytest.raises(verifier.VerificationError, match="JPEG SHA mismatch"):
        verifier.verify(**arguments)
    assert not arguments["acceptance_path"].exists()


def test_verifier_rejects_non_whitelisted_teacher_file(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    prepared = run_prepare(tmp_path, inputs)
    arguments = make_verifier_inputs(tmp_path, inputs, prepared)
    manifest = json.loads(arguments["bundle_manifest_path"].read_text())
    manifest["files"].append(
        {
            "path": "experiments/689_qwen35_4b_grounded_transaction_graph_kd/run_teacher.py",
            "sha256": "0" * 64,
            "size_bytes": 0,
        }
    )
    manifest["self_sha256"] = None
    manifest["self_sha256"] = verifier.sha256_bytes(verifier.canonical_json_bytes(manifest))
    write_json(arguments["bundle_manifest_path"], manifest)
    arguments["bundle_manifest_sha256"] = verifier.sha256_file(arguments["bundle_manifest_path"])
    with pytest.raises(verifier.VerificationError, match="exact source-prep whitelist"):
        verifier.verify(**arguments)


def test_verifier_rejects_nonterminal_metadata(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    prepared = run_prepare(tmp_path, inputs)
    arguments = make_verifier_inputs(tmp_path, inputs, prepared)
    terminal = json.loads(arguments["terminal_metadata_path"].read_text())
    terminal["status"] = "RUNNING"
    terminal["self_sha256"] = None
    terminal["self_sha256"] = verifier.sha256_bytes(verifier.canonical_json_bytes(terminal))
    write_json(arguments["terminal_metadata_path"], terminal)
    arguments["terminal_metadata_sha256"] = verifier.sha256_file(arguments["terminal_metadata_path"])
    with pytest.raises(verifier.VerificationError, match="terminal SUCCESS"):
        verifier.verify(**arguments)
