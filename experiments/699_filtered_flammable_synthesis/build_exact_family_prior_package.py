from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path
from typing import Any


FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
PREREGISTER_SELF_SHA256 = (
    "4eeac3aa45049a5529d0f86585ce5140e925a728aca677684cd009619829e02e"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def load_self_hashed(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    payload = dict(value)
    declared = str(payload.pop("self_sha256", ""))
    if value.get("schema") != schema or declared != canonical_sha256(payload):
        raise ValueError(f"self-hashed contract mismatch: {path}")
    return value, declared


def regular_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in {"__pycache__", ".git", ".pytest_cache"} for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"symlinked package member: {relative}")
        if path.is_file():
            result[relative.as_posix()] = path
    return result


def deterministic_zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, path in regular_files(source).items():
            info = zipfile.ZipInfo(relative, FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, path.read_bytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-candidate", type=Path, required=True)
    parser.add_argument("--preregister", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_zip.exists():
        raise FileExistsError("refusing to overwrite exact-family package")
    preregister, preregister_self = load_self_hashed(
        args.preregister, "exp699_exact_family_prior_preregister_v2"
    )
    if (
        preregister_self != PREREGISTER_SELF_SHA256
        or preregister.get("public_used_for_recipe") is not False
        or preregister.get("sealed_rows_used") != 0
    ):
        raise ValueError("exact-family preregistration mismatch")
    parent_manifest, parent_self = load_self_hashed(
        args.parent_candidate / "exp699_candidate_manifest.json",
        "exp699_solution140_soft_cache_candidate_v1",
    )
    if (
        parent_manifest.get("status") != "ExploratoryReadyNotSubmitted"
        or parent_manifest.get("parent_submission") != "140"
        or parent_manifest.get("public_used_for_recipe") is not False
    ):
        raise ValueError("parent package is not accepted exploratory solution140")
    shutil.copytree(args.parent_candidate, args.output_dir)
    destination_runtime = args.output_dir / "exact_family_prior_runtime.py"
    shutil.copyfile(args.runtime, destination_runtime)
    run_path = args.output_dir / "run.py"
    run_text = run_path.read_text(encoding="utf-8")
    old_import = "from soft_cache_runtime import apply_runtime_cache"
    if run_text.count(old_import) != 1 or run_text.count("apply_runtime_cache(") != 1:
        raise ValueError("parent run.py soft-cache hook mismatch")
    run_text = run_text.replace(
        old_import,
        "from exact_family_prior_runtime import apply_runtime_exact_family_prior",
    ).replace("apply_runtime_cache(", "apply_runtime_exact_family_prior(")
    run_path.write_text(run_text, encoding="utf-8")
    parent_manifest_path = args.output_dir / "exp699_candidate_manifest.json"
    parent_manifest_path.unlink()
    members_before_manifest = {
        relative: sha256_file(path)
        for relative, path in regular_files(args.output_dir).items()
    }
    manifest: dict[str, Any] = {
        "schema": "exp699_exact_family_prior_candidate_v1",
        "experiment": 699,
        "status": "ExploratoryReadyNotSubmitted",
        "candidate_name": "solution140_flammable_exact_family_prior_v1",
        "parent_submission": "140",
        "parent_candidate_manifest_file_sha256": sha256_file(
            args.parent_candidate / "exp699_candidate_manifest.json"
        ),
        "parent_candidate_manifest_self_sha256": parent_self,
        "preregister_file_sha256": sha256_file(args.preregister),
        "preregister_self_sha256": preregister_self,
        "exact_family_runtime_sha256": sha256_file(destination_runtime),
        "route": {
            "БАД": "byte_identical_solution140",
            "Легковоспламеняющиеся": "solution140_plus_unanimous_exact_family_prior",
        },
        "changed_factor": "replace broad soft-cache residual with unanimous exact-family hard override",
        "public_used": False,
        "sealed_rows_used": 0,
        "package_authorized": True,
        "ods_submit_authorized": False,
        "members": members_before_manifest,
        "self_hash_algorithm": "sha256_canonical_json_without_self_sha256",
    }
    manifest["self_sha256"] = canonical_sha256(manifest)
    parent_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    deterministic_zip(args.output_dir, args.output_zip)
    print(
        json.dumps(
            {
                "decision": "EXPLORATORY_READY_NOT_SUBMITTED",
                "package_sha256": sha256_file(args.output_zip),
                "package_size": args.output_zip.stat().st_size,
                "manifest_file_sha256": sha256_file(parent_manifest_path),
                "manifest_self_sha256": manifest["self_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
