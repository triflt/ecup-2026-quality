from __future__ import annotations

"""Label-blind rendering and effective-resolution audit for first images."""

import argparse
import hashlib
import io
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps
from resolution_contract import (
    BASELINE_MAX_EDGE,
    CANDIDATE_MAX_EDGE,
    CANDIDATE_MAX_PIXELS,
    MIN_ADDITIONAL_PIXEL_FRACTION,
    MIN_PIXEL_GAIN,
    RENDER_AUDIT_ROWS,
)

FIRST_IMAGE_MEMBER = re.compile(
    r"^images/(?P<id>[^/]+)/0\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE
)
AUDIT_VERSION = "qwen3vl_first_image_rendering_audit_v1"


@dataclass(frozen=True)
class RenderRecord:
    item_id: str
    member: str
    source_sha256: str
    source_width: int
    source_height: int
    baseline_width: int
    baseline_height: int
    candidate_width: int
    candidate_height: int
    same_cover_image: bool
    baseline_aspect_preserved: bool
    candidate_aspect_preserved: bool
    additional_pixel_fraction: float
    gains_more_than_25_percent: bool
    decode_ok: bool
    error_type: str


def stable_sample_key(item_id: str) -> str:
    return hashlib.sha256(f"{AUDIT_VERSION}\0{item_id}".encode()).hexdigest()


def select_first_images(
    archive: zipfile.ZipFile, *, sample_size: int
) -> list[tuple[str, zipfile.ZipInfo]]:
    by_id: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        match = FIRST_IMAGE_MEMBER.match(info.filename)
        if not match:
            continue
        item_id = match.group("id")
        if item_id in by_id:
            raise ValueError(f"duplicate first-image member for id={item_id}")
        by_id[item_id] = info
    if len(by_id) < sample_size:
        raise ValueError(f"archive has only {len(by_id)} first images; need {sample_size}")
    selected_ids = sorted(by_id, key=lambda item_id: (stable_sample_key(item_id), item_id))[
        :sample_size
    ]
    return [(item_id, by_id[item_id]) for item_id in selected_ids]


def decoded_pixel_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(f"{image.mode}:{image.width}:{image.height}\0".encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def decode_rgb(payload: bytes) -> Image.Image:
    with Image.open(io.BytesIO(payload)) as source:
        source.load()
        return ImageOps.exif_transpose(source).convert("RGB")


def render(image: Image.Image, max_edge: int) -> Image.Image:
    output = image.copy()
    output.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return output


def aspect_preserved(source: Image.Image, rendered: Image.Image) -> bool:
    # Pillow rounds each integer dimension independently. One source pixel of
    # cross-product error is therefore the strict useful tolerance.
    cross_error = abs(rendered.width * source.height - rendered.height * source.width)
    return cross_error <= max(source.width, source.height)


def audit_payload(item_id: str, member: str, payload: bytes) -> RenderRecord:
    source_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        baseline_source = decode_rgb(payload)
        candidate_source = decode_rgb(payload)
        same_cover = decoded_pixel_sha256(baseline_source) == decoded_pixel_sha256(
            candidate_source
        )
        baseline = render(baseline_source, BASELINE_MAX_EDGE)
        candidate = render(candidate_source, CANDIDATE_MAX_EDGE)
        baseline_pixels = baseline.width * baseline.height
        candidate_pixels = candidate.width * candidate.height
        additional = candidate_pixels / max(baseline_pixels, 1) - 1.0
        return RenderRecord(
            item_id=item_id,
            member=member,
            source_sha256=source_sha256,
            source_width=baseline_source.width,
            source_height=baseline_source.height,
            baseline_width=baseline.width,
            baseline_height=baseline.height,
            candidate_width=candidate.width,
            candidate_height=candidate.height,
            same_cover_image=same_cover,
            baseline_aspect_preserved=aspect_preserved(baseline_source, baseline),
            candidate_aspect_preserved=aspect_preserved(candidate_source, candidate),
            additional_pixel_fraction=float(additional),
            gains_more_than_25_percent=additional > MIN_PIXEL_GAIN,
            decode_ok=True,
            error_type="",
        )
    except (OSError, ValueError) as error:
        return RenderRecord(
            item_id=item_id,
            member=member,
            source_sha256=source_sha256,
            source_width=0,
            source_height=0,
            baseline_width=0,
            baseline_height=0,
            candidate_width=0,
            candidate_height=0,
            same_cover_image=False,
            baseline_aspect_preserved=False,
            candidate_aspect_preserved=False,
            additional_pixel_fraction=0.0,
            gains_more_than_25_percent=False,
            decode_ok=False,
            error_type=type(error).__name__,
        )


def audit_archive(images_zip: Path, *, sample_size: int = RENDER_AUDIT_ROWS) -> dict:
    if sample_size < 1:
        raise ValueError("sample size must be positive")
    records: list[RenderRecord] = []
    with zipfile.ZipFile(images_zip) as archive:
        selected = select_first_images(archive, sample_size=sample_size)
        for item_id, info in selected:
            records.append(audit_payload(item_id, info.filename, archive.read(info)))
    decode_failures = sum(not record.decode_ok for record in records)
    same_cover_failures = sum(not record.same_cover_image for record in records)
    aspect_failures = sum(
        not (record.baseline_aspect_preserved and record.candidate_aspect_preserved)
        for record in records
    )
    additional_count = sum(record.gains_more_than_25_percent for record in records)
    additional_fraction = additional_count / sample_size
    gates = {
        "exact_sample_size": len(records) == sample_size,
        "decode_failures_zero": decode_failures == 0,
        "same_cover_failures_zero": same_cover_failures == 0,
        "aspect_failures_zero": aspect_failures == 0,
        "additional_pixel_fraction_at_least_20_percent": (
            additional_fraction >= MIN_ADDITIONAL_PIXEL_FRACTION
        ),
    }
    return {
        "audit_version": AUDIT_VERSION,
        "status": "go_for_two_fold_screen" if all(gates.values()) else "no_go",
        "label_blind": True,
        "label_or_category_columns_read": False,
        "first_image_only": True,
        "sample_selection": "lowest deterministic SHA-256 ranks of item ids",
        "sample_size": sample_size,
        "baseline_max_edge": BASELINE_MAX_EDGE,
        "candidate_max_edge": CANDIDATE_MAX_EDGE,
        "candidate_max_pixels": CANDIDATE_MAX_PIXELS,
        "decode_failures": decode_failures,
        "same_cover_failures": same_cover_failures,
        "aspect_failures": aspect_failures,
        "more_than_25_percent_additional_pixels": additional_count,
        "additional_pixel_fraction": additional_fraction,
        "gates": gates,
        "records": [asdict(record) for record in records],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = audit_archive(args.images_zip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "sample_size": report["sample_size"],
                "decode_failures": report["decode_failures"],
                "additional_pixel_fraction": report["additional_pixel_fraction"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
