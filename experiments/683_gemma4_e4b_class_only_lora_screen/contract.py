from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "683"
CONTROL_EXPERIMENT_ID = "641"
MODEL_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
GRID_CONTRACT_SHA256 = "aad63f99ee9ddd5ecfa133575ea5cf1a803c11ab3fd26b5328d894f8631b3224"
PROMPT_VERSION = "qwen_scale_grid_prompt_v1"
PREPROCESSING_VERSION = "qwen_scale_grid_first_image_v1+gemma4_native_280"
SCREEN_FOLDS = (0, 3)
EXPECTED_RUNTIME_CONTRACT = {
    0: "38802115365cef7e3a0c1a82abc5efc5ce92a41e0046f1ddad4ab9e02647c568",
    3: "e08a51c2db16163953c45841f3dd1e7b30b293a7265b2bbd8084d1479c20ea36",
}
EXPECTED_TRAIN_SHA256 = {
    0: "3b79e512c60f4f18531da6d1b5c866ff6f1927dc6b26ac9e52b21da841744964",
    3: "404146157667adca1419d272a9bb5200a027d3ba0d15711bcfee9da07df805cf",
}
EXPECTED_VALIDATION_SHA256 = {
    0: "a9abfc5bc8dc8cb62c8add9f061d2ba9e9f1e0d165add23fca518519a848f4d0",
    3: "3381a41f8e7a8819d9831d4e6d3ba8a03580d81e741df4fbe8818fb855c7ce31",
}
EXPECTED_VALIDATION_ROWS = {0: 2224, 3: 2224}
EXPECTED_TRAIN_OCCURRENCES = 4892
SEED = 42
EPOCHS = 1
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 8
EFFECTIVE_BATCH_SIZE = 16
LEARNING_RATE = 2e-4
MAX_LENGTH = 1536
MAX_SOURCE_PIXELS = 262144
GEMMA_SOFT_IMAGE_TOKENS = 280
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
EXPECTED_TARGET_COUNTS = {"q_proj": 42, "o_proj": 42, "k_proj": 24, "v_proj": 24}
EXPECTED_TARGET_TOTAL = 132


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_self_hash(value: dict[str, Any], field: str = "contract_sha256") -> str:
    payload = dict(value)
    digest = payload.pop(field, None)
    if digest != canonical_sha256(payload):
        raise ValueError(f"{field} self-hash mismatch")
    return str(digest)
