"""Build a deterministic 12-row image-aware technical-smoke packet for exp689."""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path
from typing import Any

import run_teacher
from build_target_audit import (
    canonical_json_bytes,
    load_spec,
    make_teacher_request,
    require_remote_path,
    sha256_bytes,
    sha256_file,
    sha256_text,
    with_self_hash,
    write_json,
    write_jsonl,
)

SMOKE_ROWS = 12
WIDTH = 64
HEIGHT = 64
REGIONS = ("full", "q00", "q01", "q10", "q11")
SOURCE_KINDS = ("name", "description", "ocr")


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(
        ">I", zlib.crc32(kind + payload) & 0xFFFFFFFF
    )


def deterministic_rgb(index: int) -> bytes:
    values = bytearray()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            values.extend(
                (
                    (17 * index + 3 * x) % 256,
                    (29 * index + 5 * y) % 256,
                    (11 * index + x + y) % 256,
                )
            )
    return bytes(values)


def encode_png(rgb: bytes) -> bytes:
    stride = WIDTH * 3
    scanlines = b"".join(
        b"\x00" + rgb[offset : offset + stride]
        for offset in range(0, len(rgb), stride)
    )
    header = struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + png_chunk(b"IEND", b"")
    )


def quadrant_bytes(rgb: bytes, region: str) -> bytes:
    if region == "full":
        return rgb
    row_half = HEIGHT // 2
    col_half = WIDTH // 2
    row_start = 0 if region[1] == "0" else row_half
    col_start = 0 if region[2] == "0" else col_half
    pieces = []
    for y in range(row_start, row_start + row_half):
        start = (y * WIDTH + col_start) * 3
        pieces.append(rgb[start : start + col_half * 3])
    return b"".join(pieces)


def make_candidate(**fields: Any) -> dict[str, Any]:
    candidate = {"candidate_id": None, **fields}
    candidate["candidate_id"] = sha256_bytes(canonical_json_bytes(candidate))
    return candidate


def make_row(index: int, relative_image: str, rgb: bytes, png: bytes) -> dict[str, Any]:
    spec = load_spec()
    sold = spec["enums"]["sold_object"][index % len(spec["enums"]["sold_object"])]
    substance = spec["enums"]["substance"][index % len(spec["enums"]["substance"])]
    relation = spec["enums"]["relation"][index % len(spec["enums"]["relation"])]
    source_kind = SOURCE_KINDS[index % len(SOURCE_KINDS)]
    text = (
        f"Synthetic structural exercise {index:02d}: sold_object={sold}; "
        f"substance={substance}; relation={relation}."
    )
    sources = [
        {
            "source_index": 0,
            "source_kind": source_kind,
            "text": text,
            "text_sha256": sha256_text(text),
        }
    ]
    first_image = {
        "reference": relative_image,
        "content_sha256": sha256_bytes(png),
        "decoded_rgb_sha256": sha256_bytes(rgb),
        "media_type": "image/png",
        "width": WIDTH,
        "height": HEIGHT,
    }
    candidates = [
        make_candidate(
            candidate_index=0,
            evidence_kind="text_span",
            source_index=0,
            char_start=0,
            char_end=len(text),
            image_region=None,
            evidence_sha256=sha256_text(text),
        )
    ]
    for candidate_index, region in enumerate(REGIONS, 1):
        candidates.append(
            make_candidate(
                candidate_index=candidate_index,
                evidence_kind="image_region",
                source_index=None,
                char_start=None,
                char_end=None,
                image_region=region,
                evidence_sha256=sha256_bytes(quadrant_bytes(rgb, region)),
            )
        )
    source_card = {"sources": sources, "first_image": first_image}
    source_row = {
        "record_id": f"synthetic-smoke-{index + 1:03d}",
        "source_card_sha256": sha256_bytes(canonical_json_bytes(source_card)),
        "sources": sources,
        "first_image": first_image,
        "evidence_candidates": candidates,
    }
    return make_teacher_request(source_row, f"G689-{index + 1:03d}")


def build(*, remote_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = require_remote_path(
        remote_root, output_dir, context="teacher smoke output", must_exist=False
    )
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite immutable teacher-smoke output")
    output_dir.mkdir(parents=True)
    image_dir = output_dir / "images"
    image_dir.mkdir()

    requests: list[dict[str, Any]] = []
    image_manifest: list[dict[str, Any]] = []
    for index in range(SMOKE_ROWS):
        rgb = deterministic_rgb(index)
        png = encode_png(rgb)
        relative_image = f"images/smoke_{index + 1:03d}.png"
        image_path = output_dir / relative_image
        image_path.write_bytes(png)
        request = make_row(index, relative_image, rgb, png)
        requests.append(request)
        image_manifest.append(
            {
                "schema_version": "exp689_teacher_image_manifest_row_v1",
                "audit_id": request["audit_id"],
                "reference": relative_image,
                "relative_path": relative_image,
                "content_sha256": sha256_bytes(png),
                "decoded_rgb_sha256": sha256_bytes(rgb),
                "pixel_sha256": sha256_bytes(rgb),
                "media_type": "image/png",
                "width": WIDTH,
                "height": HEIGHT,
            }
        )

    request_path = output_dir / "teacher_request.jsonl"
    image_manifest_path = output_dir / "image_manifest.jsonl"
    write_jsonl(request_path, requests)
    write_jsonl(image_manifest_path, image_manifest)
    spec = load_spec()
    contract = with_self_hash(
        {
            "schema_version": "exp689_teacher_smoke_input_contract_v1",
            "experiment_id": "689",
            "scope": "synthetic_technical_smoke",
            "teacher_request_sha256": sha256_file(request_path),
            "teacher_request_rows": SMOKE_ROWS,
            "image_manifest_sha256": sha256_file(image_manifest_path),
            "pixel_set_sha256": sha256_bytes(
                canonical_json_bytes(
                    [row["pixel_sha256"] for row in image_manifest]
                )
            ),
            "runner_code_sha256": sha256_file(Path(run_teacher.__file__)),
            "prompt_sha256": run_teacher.PROMPT_SHA256,
            "model_id": run_teacher.MODEL_ID,
            "model_revision": run_teacher.MODEL_REVISION,
            "structural_enum_coverage": {
                key: spec["enums"][key] for key in ("sold_object", "substance", "relation")
            },
            "evidence_kind_coverage": ["image_region", "text_span"],
            "image_region_coverage": list(REGIONS),
            "source_kind_coverage": list(SOURCE_KINDS),
            "labels_present": 0,
            "sealed_rows": 0,
            "public_used": False,
            "quality_evaluated": False,
            "full_teacher_authorized": False,
            "student_gpu_authorized": False,
            "jobs_launched": 0,
            "uploads": 0,
            "presets_built": 0,
            "bundles_built": 0,
            "self_sha256": None,
        }
    )
    write_json(output_dir / "teacher_smoke_contract.json", contract)
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = build(remote_root=args.remote_root, output_dir=args.output_dir)
    print(contract["self_sha256"])


if __name__ == "__main__":
    main()
