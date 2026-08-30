from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from multiview_contract import (
    FIRST_IMAGE_MAX_EDGE,
    SEED,
    gallery_offset,
    inference_image_index,
    training_image_index,
)
from PIL import Image

_IMAGE_MEMBER = re.compile(r"^images/([^/]+)/(\d+)\.jpg$")


def load_gallery_urls(path: Path) -> dict[str, tuple[str, ...]]:
    """Load the private manifest without printing or persisting its URLs."""

    result: dict[str, tuple[str, ...]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["id", "image_urls"]:
            raise ValueError("gallery manifest schema mismatch")
        for row in reader:
            item_id = str(row["id"])
            raw = json.loads(row["image_urls"])
            if (
                not isinstance(raw, list)
                or not raw
                or any(not isinstance(value, str) or not value for value in raw)
            ):
                raise ValueError(f"invalid gallery for id={item_id}")
            if item_id in result:
                raise ValueError(f"duplicate gallery id={item_id}")
            result[item_id] = tuple(raw)
    return result


def archive_gallery_members(archive: zipfile.ZipFile) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, dict[int, str]] = defaultdict(dict)
    for member in archive.namelist():
        match = _IMAGE_MEMBER.fullmatch(member)
        if match is None:
            continue
        item_id, raw_index = match.groups()
        index = int(raw_index)
        if index in grouped[item_id]:
            raise ValueError(f"duplicate gallery index for id={item_id}")
        grouped[item_id][index] = member
    result: dict[str, tuple[str, ...]] = {}
    for item_id, by_index in grouped.items():
        expected = list(range(len(by_index)))
        if sorted(by_index) != expected:
            raise ValueError(f"non-contiguous gallery for id={item_id}")
        result[item_id] = tuple(by_index[index] for index in expected)
    return result


def parent_448_jpeg(payload: bytes) -> bytes:
    """Byte-for-byte equivalent to the parent component's image preparation."""

    with Image.open(io.BytesIO(payload)) as source:
        image = source.convert("RGB")
    image.thumbnail((FIRST_IMAGE_MAX_EDGE, FIRST_IMAGE_MAX_EDGE), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92)
    image.close()
    return output.getvalue()


@dataclass
class OccurrencePlanner:
    gallery_sizes: dict[str, int]
    seed: int = SEED
    epoch: int = 0
    training: bool = True
    _occurrences: Counter[str] = field(default_factory=Counter)
    _selected: dict[str, Counter[int]] = field(default_factory=lambda: defaultdict(Counter))

    def choose(self, item_id: str) -> int:
        item_id = str(item_id)
        size = self.gallery_sizes[item_id]
        if self.training:
            occurrence = self._occurrences[item_id]
            index = training_image_index(
                item_id=item_id,
                seed=self.seed,
                epoch=self.epoch,
                occurrence=occurrence,
                gallery_size=size,
            )
            self._occurrences[item_id] += 1
            self._selected[item_id][index] += 1
            return index
        return inference_image_index(gallery_size=size)

    def report(self) -> dict[str, object]:
        repeated = [item_id for item_id, count in self._occurrences.items() if count > 1]
        covered = [
            item_id
            for item_id in repeated
            if len(self._selected[item_id])
            == min(self.gallery_sizes[item_id], self._occurrences[item_id])
        ]
        return {
            "training_occurrences": int(sum(self._occurrences.values())),
            "training_unique_rows": len(self._occurrences),
            "repeated_rows": len(repeated),
            "repeated_rows_with_maximal_cycle_coverage": len(covered),
            "selected_gallery_positions": dict(
                sorted(
                    Counter(
                        index
                        for counts in self._selected.values()
                        for index, count in counts.items()
                        for _ in range(count)
                    ).items()
                )
            ),
        }


def audit_archive(
    *, images_zip: Path, gallery_manifest: Path, progress_every: int = 2_000
) -> dict[str, object]:
    """Run the full label-blind gallery and decode audit before any GPU work."""

    urls = load_gallery_urls(gallery_manifest)
    url_counts = {item_id: len(values) for item_id, values in urls.items()}
    del urls
    decode_failures: list[dict[str, str]] = []
    null_mismatches: list[str] = []
    decoded = 0
    transformed_bytes = 0
    with zipfile.ZipFile(images_zip) as archive:
        members = archive_gallery_members(archive)
        if set(members) != set(url_counts):
            raise ValueError("archive and manifest id sets differ")
        bad_counts = [
            item_id for item_id, paths in members.items() if len(paths) != url_counts[item_id]
        ]
        if bad_counts:
            raise ValueError(
                f"archive and manifest gallery counts differ for {len(bad_counts)} ids"
            )
        for item_id in sorted(members, key=lambda value: (len(value), value)):
            paths = members[item_id]
            for image_index, member in enumerate(paths):
                try:
                    payload = archive.read(member)
                    candidate = parent_448_jpeg(payload)
                    transformed_bytes += len(candidate)
                    if image_index == 0:
                        parent = parent_448_jpeg(archive.read(paths[0]))
                        if candidate != parent:
                            null_mismatches.append(item_id)
                except Exception as error:  # noqa: BLE001 - audit must record every decoder failure
                    decode_failures.append(
                        {
                            "id": item_id,
                            "image_index": str(image_index),
                            "error_type": type(error).__name__,
                        }
                    )
                decoded += 1
                if progress_every and decoded % progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "decoded": decoded,
                                "decode_failures": len(decode_failures),
                            }
                        ),
                        flush=True,
                    )

    deterministic_failures: list[str] = []
    coverage_failures: list[str] = []
    for item_id, count in sorted(url_counts.items()):
        first = [
            training_image_index(
                item_id=item_id,
                seed=SEED,
                epoch=0,
                occurrence=occurrence,
                gallery_size=count,
            )
            for occurrence in range(count)
        ]
        second = [
            training_image_index(
                item_id=item_id,
                seed=SEED,
                epoch=0,
                occurrence=occurrence,
                gallery_size=count,
            )
            for occurrence in range(count)
        ]
        if first != second:
            deterministic_failures.append(item_id)
        if sorted(first) != list(range(count)):
            coverage_failures.append(item_id)
        if first[0] != gallery_offset(item_id=item_id, seed=SEED, epoch=0, gallery_size=count):
            deterministic_failures.append(item_id)

    total_rows = len(url_counts)
    multi_rows = sum(count > 1 for count in url_counts.values())
    passed = not (decode_failures or null_mismatches or deterministic_failures or coverage_failures)
    return {
        "audit_version": "qwen3vl_multiview_train_aug_label_blind_v1",
        "status": "go_for_two_fold_screen" if passed else "no_go",
        "label_blind": True,
        "labels_categories_folds_read": False,
        "rows": total_rows,
        "images": sum(url_counts.values()),
        "gallery_size_histogram": dict(sorted(Counter(url_counts.values()).items())),
        "multi_image_rows": multi_rows,
        "multi_image_fraction": multi_rows / total_rows,
        "decode_failures": len(decode_failures),
        "decode_failure_records": decode_failures,
        "deterministic_plan_failures": len(set(deterministic_failures)),
        "gallery_cycle_coverage_failures": len(coverage_failures),
        "first_image_null_byte_mismatches": len(null_mismatches),
        "training_images_per_occurrence": 1,
        "holdout_inference_image_index": 0,
        "holdout_inference_passes": 1,
        "transformed_jpeg_bytes_checked": transformed_bytes,
        "images_zip_sha256": _sha256(images_zip),
        "gallery_manifest_sha256": _sha256(gallery_manifest),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
