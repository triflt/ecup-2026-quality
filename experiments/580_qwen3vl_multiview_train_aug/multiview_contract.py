from __future__ import annotations

import hashlib

EXPERIMENT_ID = "580"
SCREEN_FOLDS = (0, 3)
SEED = 42
EPOCHS = 1
FIRST_IMAGE_MAX_EDGE = 448
FIRST_IMAGE_MAX_PIXELS = 262_144
TRAINING_VIEW_POLICY = "deterministic_single_gallery_view_v1"
INFERENCE_VIEW_POLICY = "first_image_only_v1"


def gallery_offset(*, item_id: str, seed: int, epoch: int, gallery_size: int) -> int:
    if gallery_size < 1:
        raise ValueError("gallery_size must be positive")
    payload = f"{seed}\0{epoch}\0{item_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % gallery_size


def training_image_index(
    *, item_id: str, seed: int, epoch: int, occurrence: int, gallery_size: int
) -> int:
    """Choose one real gallery view and cycle through all views on repetition."""

    if occurrence < 0:
        raise ValueError("occurrence must be non-negative")
    return (
        gallery_offset(
            item_id=item_id,
            seed=seed,
            epoch=epoch,
            gallery_size=gallery_size,
        )
        + occurrence
    ) % gallery_size


def inference_image_index(*, gallery_size: int) -> int:
    if gallery_size < 1:
        raise ValueError("gallery_size must be positive")
    return 0
