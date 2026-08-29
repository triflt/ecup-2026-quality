from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/685_qwen35_4b_structured_rank_distillation"
sys.path.insert(0, str(EXP))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXP / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PAIR = load("build_pair_runtime")
VERIFY = load("verify_pair_runtime")
STAGE = load("stage_training_input")
VENDOR = load("bridge_peft_vendor")
VERIFY_CODE = load("verify_code_bundle")
BUILD_CODE = load("build_code_bundle")


def row(index: int, label: int) -> dict[str, object]:
    return {
        "global_index": index,
        "id": str(index),
        "category": PAIR.FLAMMABLE,
        "fold": 1,
        "occurrence_index": 0,
        "semantic_component": f"component-{index}",
        "label": label,
    }


def make_pair_input(root: Path) -> tuple[Path, dict[str, object]]:
    source = root / "source"
    runtime = source / "runtime"
    source_runtime = source / "source_runtime"
    runtime.mkdir(parents=True)
    source_runtime.mkdir()
    positive_count = 64
    negative_count = 192
    rows = [row(index, 1) for index in range(positive_count)] + [
        row(index, 0)
        for index in range(positive_count, positive_count + negative_count)
    ]
    scores = [2.0 - index / 100 for index in range(positive_count)] + [
        0.5 - index / 100 for index in range(negative_count)
    ]
    total_pairs = positive_count * (PAIR.HARD_NEGATIVES + PAIR.UNIFORM_NEGATIVES)
    max_endpoint_count = int(2 * total_pairs * PAIR.MAX_ITEM_WEIGHT_SHARE)
    pairs, summary = PAIR.build_pair_records(
        rows, scores, fold=0, max_endpoint_count=max_endpoint_count
    )
    ranks, normal = PAIR.normal_ranks(scores)
    targets = [
        {
            **{field: value[field] for field in PAIR.KEY_FIELDS},
            "semantic_component": value["semantic_component"],
            "label": value["label"],
            "teacher_score": score,
            "teacher_rank": rank_value,
            "teacher_normal_rank": normal_value,
        }
        for value, score, rank_value, normal_value in zip(
            rows, scores, ranks, normal, strict=True
        )
    ]
    train_payload = PAIR.jsonl_bytes(targets)
    pairs_payload = PAIR.jsonl_bytes(pairs)
    (runtime / "train_targets.jsonl").write_bytes(train_payload)
    (runtime / "pairs.jsonl").write_bytes(pairs_payload)
    audit = {
        "schema_version": 1,
        "experiment_id": PAIR.EXPERIMENT_ID,
        "stage": "685A_R0",
        "outer_fold": 0,
        "ordinary_oof_merge_used": False,
        "outer_validation_teacher_overlap": 0,
        "validation_labels_written": 0,
        "sealed_rows_used": 0,
        "public_used": False,
        "derived_680_output_sha256": {
            "train.jsonl": "a" * 64,
            "validation.jsonl": "b" * 64,
        },
        "pair_summary": summary,
        "output_sha256": {
            "train_targets.jsonl": PAIR.sha256_bytes(train_payload),
            "pairs.jsonl": PAIR.sha256_bytes(pairs_payload),
        },
        "decision": "GO_TECHNICAL_SMOKE",
    }
    audit["contract_sha256"] = PAIR.canonical_sha256(audit)
    (runtime / "runtime_audit.json").write_text(
        json.dumps(audit, sort_keys=True), encoding="utf-8"
    )
    acceptance = VERIFY.verify(runtime)
    (source / "r0_acceptance.json").write_text(
        json.dumps(acceptance, sort_keys=True), encoding="utf-8"
    )
    for name in ("runtime_audit.json", "train.jsonl", "validation.jsonl"):
        (source_runtime / name).write_text("{}\n", encoding="utf-8")
    return source, acceptance


class RemoteHardeningTests(unittest.TestCase):
    def test_code_bundle_is_manifest_bound_to_clean_git_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.name", "Test"],
                check=True,
            )
            (repo / "payload.py").write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "payload.py"], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True
            )
            revision = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
            ).strip()
            bundle = root / "code.tar.gz"
            result = BUILD_CODE.build(repo, bundle, source_paths=("payload.py",))
            self.assertEqual(result["git_revision"], revision)
            extracted = root / "extracted"
            extracted.mkdir()
            with tarfile.open(bundle) as archive:
                archive.extractall(extracted)
            accepted = VERIFY_CODE.verify(extracted, expected_revision=revision)
            self.assertEqual(accepted["decision"], "ACCEPT_CODE_BUNDLE")
            (extracted / "payload.py").write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
                VERIFY_CODE.verify(extracted, expected_revision=revision)

    def test_training_stage_ignores_only_transport_metadata_and_binds_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, acceptance = make_pair_input(root)
            (source / "source_runtime/._runtime_audit.json").write_bytes(b"metadata")
            output = root / "output"
            result = STAGE.stage(
                source,
                output,
                fold=0,
                expected_pair_acceptance_sha256=acceptance["acceptance_sha256"],
                expected_pair_runtime_contract_sha256=acceptance[
                    "runtime_contract_sha256"
                ],
            )
            self.assertEqual(result["decision"], "ACCEPT_TRAINING_INPUT")
            self.assertEqual(
                result["ignored_transport_metadata"],
                ["source_runtime/._runtime_audit.json"],
            )
            self.assertFalse((output / "source_runtime/._runtime_audit.json").exists())

    def test_training_stage_rejects_unexpected_member(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, acceptance = make_pair_input(root)
            (source / "unexpected.bin").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "unexpected remote pair input"):
                STAGE.stage(
                    source,
                    root / "output",
                    fold=0,
                    expected_pair_acceptance_sha256=acceptance["acceptance_sha256"],
                    expected_pair_runtime_contract_sha256=acceptance[
                        "runtime_contract_sha256"
                    ],
                )

    def test_pair_verifier_rejects_self_consistent_sampler_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = make_pair_input(root)
            runtime = source / "runtime"
            pairs = PAIR.read_jsonl(runtime / "pairs.jsonl")
            pairs[0]["negative_key"] = pairs[1]["negative_key"]
            pairs[0]["negative_semantic_component"] = pairs[1][
                "negative_semantic_component"
            ]
            pairs[0]["negative_teacher_score"] = pairs[1]["negative_teacher_score"]
            pairs[0]["negative_teacher_rank"] = pairs[1]["negative_teacher_rank"]
            pairs[0]["negative_normal_rank"] = pairs[1]["negative_normal_rank"]
            pairs[0]["raw_pair_target"] = pairs[1]["raw_pair_target"]
            pairs[0]["pair_target"] = pairs[1]["pair_target"]
            payload = PAIR.jsonl_bytes(pairs)
            (runtime / "pairs.jsonl").write_bytes(payload)
            audit = json.loads((runtime / "runtime_audit.json").read_text())
            audit.pop("contract_sha256")
            audit["output_sha256"]["pairs.jsonl"] = PAIR.sha256_bytes(payload)
            audit["contract_sha256"] = PAIR.canonical_sha256(audit)
            (runtime / "runtime_audit.json").write_text(json.dumps(audit))
            with self.assertRaises(ValueError):
                VERIFY.verify(runtime)

    def test_peft_bridge_extracts_one_pinned_member(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vendor_buffer = io.BytesIO()
            with zipfile.ZipFile(vendor_buffer, "w") as archive:
                archive.writestr(
                    "peft/__init__.py",
                    """
__version__ = "0.20.0"
class LoraConfig: pass
class PeftModel: pass
class TaskType: pass
def get_peft_model(): pass
""".lstrip(),
                )
                for member in (
                    "peft/config.py",
                    "peft/peft_model.py",
                    "peft/tuners/lora/config.py",
                    "peft/tuners/lora/model.py",
                    "peft/utils/peft_types.py",
                ):
                    archive.writestr(member, "")
                archive.writestr(
                    "peft-0.20.0.dist-info/METADATA",
                    "Metadata-Version: 2.1\nVersion: 0.20.0\n",
                )
            source = root / "source.tar.gz"
            payload = vendor_buffer.getvalue()
            with tarfile.open(source, "w:gz") as archive:
                info = tarfile.TarInfo(VENDOR.EXPECTED_MEMBER)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            output = root / "output"
            result = VENDOR.bridge(
                source,
                output,
                expected_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
            self.assertEqual(result["decision"], "ACCEPT_PEFT_VENDOR_BRIDGE")
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"peft-0.20.0.zip", "acceptance.json"},
            )


if __name__ == "__main__":
    unittest.main()
