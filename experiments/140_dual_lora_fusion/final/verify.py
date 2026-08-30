from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
ROOT = EXPERIMENT.parents[1]
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_PUBLIC = 0.8923976821312729
EXPECTED_ARCHIVE_SHA = "6cc2fda9d17d959880050889d964c3b971b92adbfa606505df7f04b46a819cd3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_repository() -> dict:
    required = [
        EXPERIMENT / "experiment.toml",
        EXPERIMENT / "results/metrics.json",
        EXPERIMENT / "submission/metadata.json",
        EXPERIMENT / "submission/run.py",
        EXPERIMENT / "submission/src/model.py",
        EXPERIMENT / "submission/src/output.py",
        ROOT / "datasets/registry.toml",
        ROOT / "validation/grouped_text_v1/folds.csv",
        ROOT / "validation/grouped_text_v1/manifest.json",
        ROOT / "validation/semantic_family_v3/manifest.json",
        ROOT / "validation/locked_190_nested_v1/manifest.json",
        ROOT / "research/qwen3vl_lora_holdout.py",
        ROOT / "research/aggregate_lora_oof.py",
        ROOT / "research/nested_multimodel_fusion.py",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"solution 140 repository contract is incomplete: {missing}")

    metadata = load_json(EXPERIMENT / "submission/metadata.json")
    if metadata != {
        "image": "odsai/ecup26-quality-baseline:1.0",
        "entry_point": "python -u run.py",
    }:
        raise ValueError("solution 140 metadata.json drift")

    runner_path = EXPERIMENT / "submission/run.py"
    runner = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(runner)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    forbidden_network = sorted(imports & {"aiohttp", "httpx", "requests", "socket", "urllib"})
    if forbidden_network:
        raise ValueError(f"solution 140 runtime imports network clients: {forbidden_network}")
    for token in (
        '"-i", "--test_data_path", "--test-data-path"',
        '"-o", "--output_path", "--output-path"',
        'images_root = data_path.parent / "images"',
        '"file://" + str(Path(path).resolve())',
        'pd.DataFrame({"id": frame["id"], "result": results})',
        "OUTPUT_RE.fullmatch(result)",
    ):
        if token not in runner:
            raise ValueError(f"solution 140 runtime contract missing {token!r}")

    metrics = load_json(EXPERIMENT / "results/metrics.json")
    champion = load_json(ROOT / "reports/champion.json")
    artifacts = load_json(HERE / "artifact-contract.json")
    if metrics.get("experiment_id") != "140":
        raise ValueError("solution 140 metrics identity mismatch")
    if float(metrics.get("public_macro_f1")) != EXPECTED_PUBLIC:
        raise ValueError("solution 140 metrics Public score drift")
    if float(champion["leaderboard"]["public_macro_f1"]) != EXPECTED_PUBLIC:
        raise ValueError("solution 140 champion Public score drift")
    if champion.get("canonical_experiment_id") != "140_dual_lora_fusion":
        raise ValueError("solution 140 champion identity mismatch")
    if artifacts["historical_submission"]["sha256"] != EXPECTED_ARCHIVE_SHA:
        raise ValueError("solution 140 archive identity drift")

    checked_weights = 0
    if artifacts.get("weights_published"):
        for item in artifacts["required_runtime_artifacts"]:
            expected = item.get("sha256")
            if not isinstance(expected, str) or not HEX64.fullmatch(expected):
                raise ValueError(f"missing published SHA for {item['path']}")
            path = EXPERIMENT / "submission" / item["path"]
            if not path.exists():
                raise ValueError(f"published artifact is missing: {item['path']}")
            if path.is_file() and sha256_file(path) != expected:
                raise ValueError(f"published artifact SHA mismatch: {item['path']}")
            checked_weights += 1

    return {
        "solution": "140",
        "decision": "REPOSITORY_CONTRACT_PASS",
        "public_macro_f1": EXPECTED_PUBLIC,
        "weights_published": bool(artifacts.get("weights_published")),
        "published_weight_artifacts_checked": checked_weights,
        "runtime_network_imports": forbidden_network,
        "offline_image_inputs": "local_file_uris_only",
        "required_files": len(required),
    }


if __name__ == "__main__":
    print(json.dumps(verify_repository(), ensure_ascii=False, indent=2, sort_keys=True))
