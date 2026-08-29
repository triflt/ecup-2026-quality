from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "689_qwen35_4b_grounded_transaction_graph_kd"
sys.path.insert(0, str(EXPERIMENT))

import build_source_prepare_retry_bundle as retry


def test_retry_bundle_is_deterministic_and_transport_only(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative in retry.FILES:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture"],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    first = retry.build(repo, revision, tmp_path / "first")
    second = retry.build(repo, revision, tmp_path / "second")
    assert first["archive_sha256"] == second["archive_sha256"]
    assert [item["path"] for item in first["files"]] == list(retry.FILES)
    assert any(
        item["path"].endswith("extract_source_archive_transport.py")
        for item in first["files"]
    )
    assert first["teacher_code_members"] == first["student_code_members"] == 0
