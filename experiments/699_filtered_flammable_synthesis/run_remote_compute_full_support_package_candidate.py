from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from build_solution140_full_support_submission import canonical_sha256, sha256_file
from run_remote_compute_package_candidate import (
    regular_files,
    run_capture,
    safe_extract_exact,
)


def wait_for(path: Path, timeout_seconds: int) -> None:
    started = time.monotonic()
    while not path.is_file():
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(10)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke_root.exists():
        raise FileExistsError("refusing to overwrite package smoke root")
    wait_for(args.refit_output / "output_contract.json", args.wait_timeout_seconds)
    args.smoke_root.mkdir(parents=True)
    build_log = args.smoke_root / "package_build.json"
    run_capture(
        [
            sys.executable,
            str(args.package_builder),
            "--source",
            str(args.source),
            "--runtime-dir",
            str(args.runtime_dir),
            "--refit-output",
            str(args.refit_output),
            "--destination",
            str(args.candidate_dir),
            "--archive",
            str(args.archive),
            "--route",
            args.route,
            "--preprocessing",
            args.preprocessing,
            "--synth-cap",
            str(args.synth_cap),
            "--expected-source-run-sha256",
            args.expected_source_run_sha256,
            "--expected-runtime-contract",
            args.expected_runtime_contract,
            "--expected-synth-ids-sha256",
            args.expected_synth_ids_sha256,
            "--expected-synth-payload-sha256",
            args.expected_synth_payload_sha256,
        ],
        path=build_log,
    )
    build_report = json.loads(build_log.read_text(encoding="utf-8"))
    if build_report.get("decision") != "GO_PACKAGE_SMOKE":
        raise ValueError("full-support builder did not open package smoke")
    if build_report.get("archive_sha256") != sha256_file(args.archive):
        raise ValueError("package archive binding mismatch")
    extracted = args.smoke_root / "extracted"
    extracted_files = safe_extract_exact(args.archive, extracted)
    candidate_files = regular_files(args.candidate_dir)
    if extracted_files != candidate_files:
        raise ValueError("ZIP extraction differs from candidate directory")
    manifest = json.loads(
        (extracted / "exp699_candidate_manifest.json").read_text(encoding="utf-8")
    )
    manifest_payload = dict(manifest)
    manifest_self = manifest_payload.pop("self_sha256", None)
    if manifest_self != canonical_sha256(manifest_payload):
        raise ValueError("candidate manifest self-hash mismatch")

    output = args.smoke_root / "submission.csv"
    environment = dict(os.environ)
    environment.update(
        {
            "QWEN_EMBED_MODEL_PATH": "/models/qwen3vl_embedding_2b",
            "QWEN_INSTRUCT_MODEL_PATH": "/models/qwen3vl_2b",
            "QWEN35_MODEL_PATH": "/models/qwen35_4b",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }
    )
    runtime_log = args.smoke_root / "runtime.log"
    run_capture(
        [
            sys.executable,
            "-u",
            str(extracted / "run.py"),
            "-i",
            str(args.smoke_input / "data.csv"),
            "-o",
            str(output),
        ],
        path=runtime_log,
        env=environment,
    )
    if args.route == "flammable_only":
        runtime_text = runtime_log.read_text(encoding="utf-8")
        expected_routes = {
            "qwen35_route='БАД' rows=4 adapter=adapter_qwen35",
            (
                "qwen35_route='Легковоспламеняющиеся' rows=4 "
                "adapter=adapter_qwen35_flammable"
            ),
        }
        missing = sorted(route for route in expected_routes if route not in runtime_text)
        if missing:
            raise ValueError(f"category route was not exercised by smoke: {missing}")
    verify_log = args.smoke_root / "smoke_acceptance.json"
    run_capture(
        [
            sys.executable,
            str(args.smoke_verifier),
            "verify",
            "--input-dir",
            str(args.smoke_input),
            "--output",
            str(output),
        ],
        path=verify_log,
    )
    acceptance = json.loads(verify_log.read_text(encoding="utf-8"))
    if acceptance.get("decision") != "ACCEPT_PACKAGE_RUNTIME_SMOKE":
        raise ValueError("package runtime smoke was not accepted")
    acceptance_payload = dict(acceptance)
    acceptance_self = acceptance_payload.pop("self_sha256", None)
    if acceptance_self != canonical_sha256(acceptance_payload):
        raise ValueError("smoke acceptance self-hash mismatch")
    report: dict[str, Any] = {
        "schema": "exp699_full_support_package_terminal_v1",
        "experiment_id": "699",
        "route": args.route,
        "preprocessing_key": args.preprocessing,
        "synth_cap": args.synth_cap,
        "archive_sha256": sha256_file(args.archive),
        "archive_size": args.archive.stat().st_size,
        "candidate_manifest_self_sha256": manifest_self,
        "refit_output_contract_sha256": manifest[
            "refit_output_contract_sha256"
        ],
        "refit_runtime_contract_sha256": args.expected_runtime_contract,
        "refit_adapter_zip_sha256": manifest["refit_binding"][
            "adapter_zip_sha256"
        ],
        "smoke_acceptance_self_sha256": acceptance_self,
        "smoke_output_sha256": acceptance["output_sha256"],
        "rows": acceptance["rows"],
        "category_route_exercised": args.route == "flammable_only",
        "unrelated_solution140_members_changed": manifest[
            "unrelated_solution140_members_changed"
        ],
        "validation_labels_read": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "decision": "ACCEPT_PACKAGE_NO_PUBLIC_SUBMIT",
    }
    report["self_sha256"] = canonical_sha256(report)
    (args.smoke_root / "terminal.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--refit-output", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--smoke-input", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--package-builder", type=Path, required=True)
    parser.add_argument("--smoke-verifier", type=Path, required=True)
    parser.add_argument("--route", choices=("all", "flammable_only"), required=True)
    parser.add_argument(
        "--preprocessing", choices=("thumbnail448", "area_cap262144"), required=True
    )
    parser.add_argument("--synth-cap", type=int, choices=(5, 10), required=True)
    parser.add_argument("--expected-source-run-sha256", required=True)
    parser.add_argument("--expected-runtime-contract", required=True)
    parser.add_argument("--expected-synth-ids-sha256", required=True)
    parser.add_argument("--expected-synth-payload-sha256", required=True)
    parser.add_argument("--wait-timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
