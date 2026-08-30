from __future__ import annotations

"""Fail-closed first-image preparation from the frozen selector manifest."""

import hashlib
import io
import os
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import BinaryIO

import pandas as pd
from PIL import Image

DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT_SECONDS = 60
DOWNLOAD_WORKERS = 32


class ImagePreparationError(RuntimeError):
    def __init__(self, report: dict[str, object]) -> None:
        self.report = report
        super().__init__(
            "manifest image preparation failed for "
            f"{report['failed_images']} of {report['expected_images']} expected ids"
        )


def cache_name(item_id: str) -> str:
    return hashlib.sha256(item_id.encode()).hexdigest() + ".jpg"


def id_set_sha256(ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def decode_rgb(payload: bytes) -> Image.Image:
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        return image.convert("RGB")


def validate_cached_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            return image.mode == "RGB" and image.width > 0 and image.height > 0
    except (OSError, ValueError):
        return False


def download_one(
    *,
    item_id: str,
    image_url: str,
    destination: Path,
    attempts: int,
    timeout: int,
    opener: Callable[..., BinaryIO],
) -> tuple[str, bool, int, str]:
    last_error_type = ""
    for attempt in range(1, attempts + 1):
        temporary = destination.with_suffix(f".attempt-{attempt}.tmp")
        try:
            with opener(image_url, timeout=timeout) as response:
                payload = response.read()
            image = decode_rgb(payload)
            image.save(temporary, format="JPEG", quality=94)
            if not validate_cached_image(temporary):
                raise ValueError("encoded cache image failed RGB validation")
            os.replace(temporary, destination)
            return item_id, True, attempt, ""
        except Exception as error:  # noqa: BLE001 - retry all transport/decode failures.
            last_error_type = type(error).__name__
            temporary.unlink(missing_ok=True)
    return item_id, False, attempts, last_error_type


def prepare_manifest_images(
    *,
    manifest: pd.DataFrame,
    expected_ids: set[str],
    cache_dir: Path,
    attempts: int = DOWNLOAD_ATTEMPTS,
    timeout: int = DOWNLOAD_TIMEOUT_SECONDS,
    workers: int = DOWNLOAD_WORKERS,
    opener: Callable[..., BinaryIO] = urllib.request.urlopen,
) -> tuple[ManifestImageStore, dict[str, object]]:
    if attempts != DOWNLOAD_ATTEMPTS:
        raise ValueError(f"screen requires exactly {DOWNLOAD_ATTEMPTS} download attempts")
    if workers < 1 or timeout < 1 or not expected_ids:
        raise ValueError("invalid manifest image preparation settings")
    required = {"id", "image_url"}
    missing_columns = sorted(required - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"frozen selector manifest lacks columns: {missing_columns}")
    frame = manifest[["id", "image_url"]].copy()
    frame["id"] = frame.id.astype(str)
    if frame.id.duplicated().any():
        raise ValueError("frozen selector manifest contains duplicate image ids")
    if not expected_ids <= set(frame.id):
        raise ValueError("frozen selector manifest lacks expected image ids")
    selected = frame.set_index("id").loc[sorted(expected_ids)]
    if selected.image_url.isna().any() or (selected.image_url.astype(str).str.strip() == "").any():
        raise ValueError("frozen selector manifest has an empty image URL")

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {item_id: cache_dir / cache_name(item_id) for item_id in sorted(expected_ids)}
    cached_ids = {item_id for item_id, path in paths.items() if validate_cached_image(path)}
    pending_ids = sorted(expected_ids - cached_ids)
    outcomes: list[tuple[str, bool, int, str]] = []
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(pending_ids)))) as pool:
        futures = [
            pool.submit(
                download_one,
                item_id=item_id,
                image_url=str(selected.loc[item_id, "image_url"]),
                destination=paths[item_id],
                attempts=attempts,
                timeout=timeout,
                opener=opener,
            )
            for item_id in pending_ids
        ]
        outcomes.extend(future.result() for future in as_completed(futures))

    failed = [outcome for outcome in outcomes if not outcome[1]]
    prepared_ids = {item_id for item_id, ok, _, _ in outcomes if ok} | cached_ids
    decoded_ids = {item_id for item_id in prepared_ids if validate_cached_image(paths[item_id])}
    one_path_per_id = len(set(paths.values())) == len(expected_ids)
    error_types: dict[str, int] = {}
    for _, ok, _, error_type in outcomes:
        if not ok:
            error_types[error_type] = error_types.get(error_type, 0) + 1
    report: dict[str, object] = {
        "image_source": "manifest",
        "expected_images": len(expected_ids),
        "cache_hits": len(cached_ids),
        "download_requests": len(pending_ids),
        "downloaded_images": sum(ok for _, ok, _, _ in outcomes),
        "download_attempts": sum(used for _, _, used, _ in outcomes),
        "configured_attempts_per_image": attempts,
        "decoded_rgb_images": len(decoded_ids),
        "expected_ids_sha256": id_set_sha256(expected_ids),
        "prepared_ids_sha256": id_set_sha256(decoded_ids),
        "failed_images": len(failed),
        "failure_error_types": error_types,
        "white_fallbacks": 0,
        "exactly_one_image_per_expected_id": one_path_per_id,
        "ready": not failed and decoded_ids == expected_ids and one_path_per_id,
    }
    if not report["ready"]:
        raise ImagePreparationError(report)
    return ManifestImageStore(paths), report


class ManifestImageStore:
    def __init__(self, paths: dict[str, Path]) -> None:
        self.paths = dict(paths)

    def rgb(self, item_id: str) -> Image.Image:
        try:
            path = self.paths[item_id]
        except KeyError as error:
            raise ValueError("requested image id was not prepared") from error
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB":
                raise ValueError("prepared image is not RGB")
            return image.copy()

    def close(self) -> None:
        return None
