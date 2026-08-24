from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/634_immutable_ocr_dataset"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
SPEC = importlib.util.spec_from_file_location("remote_634", EXP / "build_remote.py")
assert SPEC is not None and SPEC.loader is not None
REMOTE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REMOTE)

PRESET_SPEC = importlib.util.spec_from_file_location(
    "preset_634", EXP / "build_private_preset.py"
)
assert PRESET_SPEC is not None and PRESET_SPEC.loader is not None
PRESET = importlib.util.module_from_spec(PRESET_SPEC)
PRESET_SPEC.loader.exec_module(PRESET)


def write_shards(path: Path, *, unsafe: bool = False) -> None:
    path.mkdir()
    for shard in range(REMOTE.EXPECTED_SHARDS):
        with zipfile.ZipFile(path / f"paddleocr_full_s{shard:02d}.zip", "w") as archive:
            archive.writestr("report.json", json.dumps({"shard_index": shard}))
            archive.writestr("spotting.jsonl", "{}\n")
            if unsafe and shard == 0:
                archive.writestr("../escape", "bad")


def test_extract_shards_requires_exact_safe_member_set(tmp_path: Path) -> None:
    archives = tmp_path / "archives"
    write_shards(archives)
    shards = REMOTE.extract_shards(archives, tmp_path / "output")
    assert len(shards) == REMOTE.EXPECTED_SHARDS
    assert {path.name for path in shards[0].iterdir()} == REMOTE.EXPECTED_MEMBERS


def test_extract_shards_rejects_unsafe_or_extra_member(tmp_path: Path) -> None:
    archives = tmp_path / "archives"
    write_shards(archives, unsafe=True)
    with pytest.raises(ValueError, match="member set mismatch"):
        REMOTE.extract_shards(archives, tmp_path / "output")


def test_source_url_manifest_rejects_incomplete_scope(tmp_path: Path) -> None:
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "shard": 0,
                    "url": "https://example.invalid/shard.zip",
                    "archive_sha256": "0" * 64,
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly 32"):
        REMOTE.download_source_archives(manifest, tmp_path / "output")


def write_source_jobs(path: Path, *, count: int = 32, unsafe: bool = False) -> None:
    records = [
        {"shard": shard, "job_name": f"paddleocr-source-{shard:02d}"}
        for shard in range(count)
    ]
    if unsafe:
        records[0]["job_name"] = "bad/name"
    path.write_text(json.dumps({"records": records}), encoding="utf-8")


def test_artifact_preset_uses_exact_job_outputs_without_http_source_urls(tmp_path: Path) -> None:
    base = tmp_path / "base.yml"
    base.write_text(
        """job:
  flavor: test-flavor
  region: test-region
  image: test-image
  preemption: forbidden
  work_dir: /work
  env:
    TOKENIZERS_PARALLELISM: \"false\"
    PYTORCH_ALLOC_CONF: expandable_segments:True
""",
        encoding="utf-8",
    )
    runtime_url = tmp_path / "runtime_url.txt"
    runtime_url.write_text("https://example.invalid/runtime.tar.gz\n", encoding="utf-8")
    jobs = tmp_path / "jobs.json"
    write_source_jobs(jobs)
    rendered = PRESET.build(
        Namespace(
            base=base,
            runtime_url_file=runtime_url,
            source_jobs_manifest=jobs,
            repair_jobs_manifest=None,
            output=tmp_path / "output.yml",
        )
    )
    assert rendered.count("type: artifact") == 32
    assert "paddleocr-source-00/paddleocr_full_s00" in rendered
    assert "paddleocr-source-31/paddleocr_full_s31" in rendered
    assert "--source-url-manifest" not in rendered
    assert "--verify-only" in rendered


def test_artifact_preset_can_attach_complete_repair_overlay(tmp_path: Path) -> None:
    base = tmp_path / "base.yml"
    base.write_text(
        """job:
  flavor: test-flavor
  region: test-region
  image: test-image
  preemption: forbidden
  work_dir: /work
  env:
    TOKENIZERS_PARALLELISM: \"false\"
    PYTORCH_ALLOC_CONF: expandable_segments:True
""",
        encoding="utf-8",
    )
    runtime_url = tmp_path / "runtime_url.txt"
    runtime_url.write_text("https://example.invalid/runtime.tar.gz\n", encoding="utf-8")
    jobs = tmp_path / "jobs.json"
    write_source_jobs(jobs)
    repairs = tmp_path / "repairs.json"
    repairs.write_text(
        json.dumps(
            {
                "records": [
                    {"repair_shard": shard, "job_name": f"repair-{shard:02d}"}
                    for shard in range(16)
                ]
            }
        ),
        encoding="utf-8",
    )
    rendered = PRESET.build(
        Namespace(
            base=base,
            runtime_url_file=runtime_url,
            source_jobs_manifest=jobs,
            repair_jobs_manifest=repairs,
            output=tmp_path / "output.yml",
        )
    )
    assert rendered.count("type: artifact") == 48
    assert "repair-00/paddleocr_repair_s00" in rendered
    assert "repair-15/paddleocr_repair_s15" in rendered
    assert "--repair-dir ${repair_dir}" in rendered


@pytest.mark.parametrize(("count", "unsafe"), [(31, False), (32, True)])
def test_artifact_preset_rejects_incomplete_or_unsafe_jobs(
    tmp_path: Path, count: int, unsafe: bool
) -> None:
    jobs = tmp_path / "jobs.json"
    write_source_jobs(jobs, count=count, unsafe=unsafe)
    with pytest.raises(ValueError):
        PRESET.artifact_inputs(jobs)
