from __future__ import annotations

"""Build a label-blind semantic-family graph audit, never a sealed validation.

The builder reads a local CSV and a local image ZIP without extracting it. It
persists only row IDs, content fingerprints, edge provenance and aggregate
audits. Raw text, image bytes and remote URLs are never written or printed.

This module deliberately emits ``draft_requires_manual_audit``. Its partitions
are diagnostics for balance and component integrity; they are not valid for
candidate scoring until a separately reviewed immutable protocol is created.
"""

import argparse
import hashlib
import io
import json
import math
import re
import subprocess
import time
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecup_quality.data.text import normalize_text
from ecup_quality.validation.semantic_families import (
    JOINT_STRATA,
    assign_balanced_component_folds,
    description_similarity,
    masked_quantity_name,
)

STATUS = "draft_requires_manual_audit"
GRAPH_VERSION = "semantic_family_graph_audit_draft_v3"
STAGES = (
    "exact_full_text",
    "exact_name_corroborated",
    "masked_name_corroborated",
    "exact_first_image",
    "exact_auxiliary_images",
    "perceptual_corroborated",
)
IMAGE_MEMBER = re.compile(r"^images/([^/]+)/(\d+)\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)
IDENTITY_TOKEN = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
GENERIC_IDENTITY_TOKENS = frozenset(
    {
        "бад",
        "биодобавка",
        "добавка",
        "капсула",
        "капсулы",
        "комплекс",
        "набор",
        "натуральный",
        "порошок",
        "продукт",
        "средство",
        "таблетка",
        "таблетки",
        "для",
        "здоровье",
        "иммунитет",
        "мощный",
        "похудение",
        "рост",
        "снижение",
        "эффективный",
        "вес",
        "мышца",
        "и",
        "из",
        "к",
        "на",
        "от",
        "по",
        "с",
        "у",
        "в",
        "г",
        "кг",
        "мг",
        "мл",
        "л",
        "шт",
        "zippo",
        "зиппо",
    }
)
GENERIC_IDENTITY_STEMS = (
    "активн",
    "антиоксидант",
    "бад",
    "безопасн",
    "бензин",
    "биодобав",
    "беспламен",
    "вес",
    "горюч",
    "детокс",
    "добав",
    "жидк",
    "жиросжиг",
    "здоров",
    "иммун",
    "капсул",
    "комплекс",
    "мощн",
    "набор",
    "нагреват",
    "натурал",
    "питан",
    "порош",
    "похуд",
    "продукт",
    "разогрев",
    "реторт",
    "рост",
    "сниж",
    "средств",
    "сухпай",
    "таблет",
    "топлив",
    "турист",
    "услов",
    "хлорофилл",
    "шлак",
    "эффектив",
)


@dataclass(frozen=True)
class AuditConfig:
    description_similarity_min: float = 0.88
    exact_name_max_degree: int = 64
    masked_name_max_degree: int = 32
    first_exact_max_degree: int = 32
    auxiliary_exact_max_degree: int = 64
    perceptual_max_degree: int = 16
    contrast_min: float = 12.0
    entropy_min: float = 3.5
    maximum_component_fraction: float = 0.02
    manual_samples_per_kind: int = 32


@dataclass(frozen=True)
class ImageFingerprint:
    item_id: str
    position: int
    exact_sha1: str
    phash: str
    dhash: str
    aspect_bucket: int
    contrast: float
    entropy: float
    width: int
    height: int

    def informative(self, config: AuditConfig) -> bool:
        return self.contrast >= config.contrast_min and self.entropy >= config.entropy_min

    @property
    def perceptual_key(self) -> str:
        return f"{self.phash}:{self.dhash}:{self.aspect_bucket}"


@dataclass(frozen=True)
class AuditEdge:
    stage: str
    left_id: str
    right_id: str
    key_hash: str
    key_degree: int
    left_positions: str
    right_positions: str
    corroboration: str
    description_similarity: float
    informative_guard: bool
    degree_guard: bool
    label_blind: bool = True


@dataclass(frozen=True)
class GraphAudit:
    component_ids: np.ndarray
    component_sizes: np.ndarray
    edges: list[dict[str, Any]]
    key_degrees: list[dict[str, Any]]
    manual_samples: list[dict[str, Any]]


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.weight = np.ones(size, dtype=np.int32)

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != value:
            parent = int(self.parent[value])
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> bool:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return False
        if self.weight[left_root] < self.weight[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.weight[left_root] += self.weight[right_root]
        return True


def file_sha256(path: Path) -> str:
    # On macOS a freshly closed compressed output can briefly fail its first
    # reopen while provenance metadata is being attached.  Retry only that
    # transient local-read failure; any persistent error still fails closed.
    for attempt in range(4):
        try:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        except PermissionError:
            if attempt == 3:
                break
            time.sleep(0.25 * (attempt + 1))
    # Some macOS files carrying provenance metadata remain readable to the
    # system checksum utility while Python's first direct opens are denied.
    # This fallback is reachable only after repeated PermissionError failures;
    # every other read error still propagates unchanged.
    completed = subprocess.run(
        ["shasum", "-a", "256", "--", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout
    if not output.endswith("\n") or output.count("\n") != 1:
        raise RuntimeError("invalid shasum output: expected exactly one line")
    digest, separator, reported_path = output[:-1].partition("  ")
    if (
        separator != "  "
        or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
        or reported_path != str(path)
    ):
        raise RuntimeError("invalid shasum output: digest/path mismatch")
    return digest.lower()


def opaque_key(stage: str, value: object) -> str:
    payload = f"{GRAPH_VERSION}\0{stage}\0{value}".encode()
    return hashlib.sha256(payload).hexdigest()


def normalized_full_text(name: object, description: object) -> str:
    name_key = normalize_text(name)
    description_key = normalize_text(description)
    if not name_key and not description_key:
        return ""
    return f"{name_key}\n{description_key}"


def product_identity_signature(name: object) -> str:
    """Return a conservative text-only product identity or an empty value.

    Quantity variants share the masked title when that representation is
    available.  Broad product forms, use-cases and marketing words cannot form
    an identity by themselves.  Requiring either two remaining lexical tokens
    or a model-like alphanumeric token deliberately trades recall for precision.
    """

    normalized = normalize_text(name)
    masked = masked_quantity_name(name)
    source = masked or normalized
    tokens = []
    for token in IDENTITY_TOKEN.findall(source):
        if token == "#" or token in GENERIC_IDENTITY_TOKENS:
            continue
        if any(token.startswith(stem) for stem in GENERIC_IDENTITY_STEMS):
            continue
        tokens.append(token)
    lexical = [token for token in tokens if token.isalpha()]
    model_like = [
        token
        for token in tokens
        if any(character.isalpha() for character in token)
        and any(character.isdigit() for character in token)
    ]
    if not model_like and (len(set(lexical)) < 2 or sum(len(token) for token in lexical) < 10):
        return ""
    return " ".join(tokens)


def component_hash(ids: list[str]) -> str:
    payload = GRAPH_VERSION + "\0" + "\0".join(sorted(ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_bits(bits: np.ndarray) -> str:
    value = 0
    for index, bit in enumerate(bits.reshape(-1)):
        value |= int(bool(bit)) << index
    width = math.ceil(bits.size / 4)
    return f"{value:0{width}x}"


def fingerprint_image_bytes(item_id: str, position: int, payload: bytes) -> ImageFingerprint:
    try:
        from PIL import Image, ImageOps, ImageStat
        from scipy.fft import dctn
    except ImportError as error:  # pragma: no cover - exercised in runtime environments
        raise RuntimeError(
            "local image fingerprinting requires the optional Pillow dependency"
        ) from error

    import io

    with Image.open(io.BytesIO(payload)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        gray = image.convert("L")
        dhash_pixels = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
        dhash = _hash_bits(dhash_pixels[:, 1:] > dhash_pixels[:, :-1])
        phash_pixels = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
        coefficients = dctn(phash_pixels, type=2, norm="ortho")[:8, :8]
        median = float(np.median(coefficients.reshape(-1)[1:]))
        phash = _hash_bits(coefficients > median)
        contrast = float(ImageStat.Stat(gray.resize((64, 64))).stddev[0])
        entropy = float(gray.entropy())
    ratio = max(width, 1) / max(height, 1)
    aspect_bucket = round(math.log(max(ratio, 1e-9)) / math.log(1.02))
    return ImageFingerprint(
        item_id=item_id,
        position=position,
        exact_sha1=hashlib.sha1(payload, usedforsecurity=False).hexdigest(),
        phash=phash,
        dhash=dhash,
        aspect_bucket=aspect_bucket,
        contrast=contrast,
        entropy=entropy,
        width=width,
        height=height,
    )


def read_zip_fingerprints(images_zip: Path, expected_ids: set[str]) -> list[ImageFingerprint]:
    """Stream fingerprints from a local ZIP without extraction or URL handling."""

    members: list[tuple[str, int, zipfile.ZipInfo]] = []
    seen_names: set[str] = set()
    with zipfile.ZipFile(images_zip) as archive:
        for info in archive.infolist():
            match = IMAGE_MEMBER.match(info.filename)
            if not match:
                continue
            if info.filename in seen_names:
                raise ValueError(f"duplicate ZIP member: {info.filename}")
            seen_names.add(info.filename)
            members.append((match.group(1), int(match.group(2)), info))
        members.sort(key=lambda item: (item[0], item[1], item[2].filename))
        grouped_positions: dict[str, list[int]] = defaultdict(list)
        for item_id, position, _ in members:
            grouped_positions[item_id].append(position)
        missing = sorted(expected_ids - set(grouped_positions))
        extra = sorted(set(grouped_positions) - expected_ids)
        if missing or extra:
            raise ValueError(f"ZIP/data id mismatch: missing={missing[:10]} extra={extra[:10]}")
        invalid = {
            item_id: positions
            for item_id, positions in grouped_positions.items()
            if positions != list(range(len(positions))) or not 1 <= len(positions) <= 5
        }
        if invalid:
            first = next(iter(sorted(invalid.items())))
            raise ValueError(f"non-contiguous ZIP image positions: {first}")

        fingerprints: list[ImageFingerprint] = []
        for number, (item_id, position, info) in enumerate(members, 1):
            fingerprints.append(fingerprint_image_bytes(item_id, position, archive.read(info)))
            if number % 2000 == 0 or number == len(members):
                print(f"fingerprinted_images={number}/{len(members)}", flush=True)
    return fingerprints


def _read_fingerprint_frame(path: Path) -> pd.DataFrame:
    read_options = {"dtype": str, "keep_default_na": False}
    try:
        return pd.read_csv(path, **read_options)
    except PermissionError:
        if not path.name.lower().endswith(".csv.gz"):
            raise
        completed = subprocess.run(
            ["gzip", "-dc", "--", str(path)],
            check=True,
            capture_output=True,
        )
        return pd.read_csv(io.BytesIO(completed.stdout), **read_options)


def _parse_integer_column(
    frame: pd.DataFrame,
    column: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    allow_negative: bool = False,
) -> pd.Series:
    values = frame[column].astype(str)
    pattern = r"-?(?:0|[1-9]\d*)" if allow_negative else r"0|[1-9]\d*"
    if not values.str.fullmatch(pattern).all():
        raise ValueError(f"invalid fingerprint integer field: {column}")
    parsed = values.astype(np.int64)
    if minimum is not None and (parsed < minimum).any():
        raise ValueError(f"fingerprint field below minimum: {column}")
    if maximum is not None and (parsed > maximum).any():
        raise ValueError(f"fingerprint field above maximum: {column}")
    return parsed


def _parse_float_column(
    frame: pd.DataFrame,
    column: str,
    *,
    minimum: float,
    maximum: float,
) -> pd.Series:
    values = frame[column].astype(str)
    if not values.str.fullmatch(r"(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?").all():
        raise ValueError(f"invalid fingerprint float field: {column}")
    parsed = values.astype(np.float64)
    if (
        not np.isfinite(parsed.to_numpy()).all()
        or (parsed < minimum).any()
        or (parsed > maximum).any()
    ):
        raise ValueError(f"fingerprint field outside valid range: {column}")
    return parsed


def read_reused_fingerprints(
    fingerprint_csv: Path, expected_ids: set[str]
) -> list[ImageFingerprint]:
    """Read a complete prior fingerprint CSV under strict fail-closed checks."""

    frame = _read_fingerprint_frame(fingerprint_csv)
    expected_columns = [field.name for field in fields(ImageFingerprint)]
    if frame.columns.tolist() != expected_columns:
        raise ValueError(
            "fingerprint schema mismatch: "
            f"expected={expected_columns} actual={frame.columns.tolist()}"
        )
    if frame.empty:
        raise ValueError("fingerprint CSV must be nonempty")
    if (frame.astype(str).apply(lambda column: column.str.len() == 0)).any().any():
        raise ValueError("fingerprint CSV contains empty fields")

    frame["item_id"] = frame.item_id.astype(str)
    actual_ids = set(frame.item_id)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ValueError(f"fingerprint/data id mismatch: missing={missing[:10]} extra={extra[:10]}")

    frame["position"] = _parse_integer_column(frame, "position", minimum=0, maximum=4)
    if frame.duplicated(["item_id", "position"]).any():
        raise ValueError("duplicate fingerprint (item_id, position)")
    grouped_positions = frame.groupby("item_id", sort=False).position.apply(list)
    invalid_positions = {
        item_id: positions
        for item_id, positions in grouped_positions.items()
        if positions != list(range(len(positions))) or not 1 <= len(positions) <= 5
    }
    if invalid_positions:
        first = next(iter(sorted(invalid_positions.items())))
        raise ValueError(f"non-contiguous fingerprint positions: {first}")

    expected_order = sorted(
        zip(frame.item_id.tolist(), frame.position.tolist()),
        key=lambda value: (value[0], value[1]),
    )
    actual_order = list(zip(frame.item_id, frame.position))
    if actual_order != expected_order:
        raise ValueError("fingerprint rows must be sorted by (item_id, position)")

    for column, width in (("exact_sha1", 40), ("phash", 16), ("dhash", 16)):
        if not frame[column].str.fullmatch(rf"[0-9a-f]{{{width}}}").all():
            raise ValueError(f"invalid lowercase hexadecimal fingerprint: {column}")
    frame["aspect_bucket"] = _parse_integer_column(
        frame,
        "aspect_bucket",
        minimum=-100_000,
        maximum=100_000,
        allow_negative=True,
    )
    frame["contrast"] = _parse_float_column(frame, "contrast", minimum=0.0, maximum=127.5)
    frame["entropy"] = _parse_float_column(frame, "entropy", minimum=0.0, maximum=8.0)
    frame["width"] = _parse_integer_column(frame, "width", minimum=1, maximum=100_000)
    frame["height"] = _parse_integer_column(frame, "height", minimum=1, maximum=100_000)
    expected_aspect = np.rint(
        np.log(frame.width.to_numpy() / frame.height.to_numpy()) / np.log(1.02)
    ).astype(np.int64)
    if not np.array_equal(frame.aspect_bucket.to_numpy(), expected_aspect):
        raise ValueError("aspect_bucket does not match image dimensions")

    return [
        ImageFingerprint(
            item_id=str(row.item_id),
            position=int(row.position),
            exact_sha1=str(row.exact_sha1),
            phash=str(row.phash),
            dhash=str(row.dhash),
            aspect_bucket=int(row.aspect_bucket),
            contrast=float(row.contrast),
            entropy=float(row.entropy),
            width=int(row.width),
            height=int(row.height),
        )
        for row in frame.itertuples(index=False, name="FingerprintRow")
    ]


def _group_indices(values: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        if value:
            groups[value].append(index)
    return groups


def _position_json(values: set[int] | list[int] | tuple[int, ...]) -> str:
    return json.dumps(sorted(set(map(int, values))), separators=(",", ":"))


def _sample_ids(indices: list[int] | set[int], ids: np.ndarray, limit: int = 8) -> str:
    values = sorted(str(ids[index]) for index in indices)[:limit]
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def _append_manual_sample(
    samples: list[dict[str, Any]],
    counts: dict[str, int],
    *,
    config: AuditConfig,
    sample_type: str,
    stage: str,
    key_hash: str,
    key_degree: int,
    ids: str,
    left_id: str = "",
    right_id: str = "",
    reason: str,
) -> None:
    if counts[sample_type] >= config.manual_samples_per_kind:
        return
    samples.append(
        {
            "sample_type": sample_type,
            "stage": stage,
            "key_hash": key_hash,
            "key_degree": key_degree,
            "left_id": left_id,
            "right_id": right_id,
            "sample_ids": ids,
            "reason": reason,
        }
    )
    counts[sample_type] += 1


def _pair(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def _ordered_pair_ids(left: int, right: int, ids: np.ndarray) -> tuple[str, str]:
    left_id, right_id = str(ids[left]), str(ids[right])
    return (left_id, right_id) if left_id < right_id else (right_id, left_id)


def _image_occurrences(
    fingerprints: list[ImageFingerprint],
    id_to_index: dict[str, int],
    config: AuditConfig,
) -> tuple[
    dict[str, dict[int, set[int]]],
    dict[str, dict[int, set[int]]],
    dict[str, dict[int, set[int]]],
    dict[str, dict[int, set[int]]],
]:
    all_exact: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    all_perceptual: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    informative_exact: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    informative_perceptual: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for fingerprint in fingerprints:
        if fingerprint.item_id not in id_to_index:
            raise ValueError(f"image fingerprint has unknown id: {fingerprint.item_id}")
        index = id_to_index[fingerprint.item_id]
        all_exact[fingerprint.exact_sha1][index].add(fingerprint.position)
        all_perceptual[fingerprint.perceptual_key][index].add(fingerprint.position)
        if not fingerprint.informative(config):
            continue
        informative_exact[fingerprint.exact_sha1][index].add(fingerprint.position)
        informative_perceptual[fingerprint.perceptual_key][index].add(fingerprint.position)
    return (
        dict(all_exact),
        dict(all_perceptual),
        dict(informative_exact),
        dict(informative_perceptual),
    )


def _eligible_pair_support(
    occurrences: dict[str, dict[int, set[int]]],
    *,
    degree_occurrences: dict[str, dict[int, set[int]]],
    maximum_degree: int,
) -> tuple[
    dict[tuple[int, int], set[str]],
    dict[str, int],
]:
    pair_support: dict[tuple[int, int], set[str]] = defaultdict(set)
    degrees = {key: len(rows) for key, rows in degree_occurrences.items()}
    for key, rows in occurrences.items():
        if len(rows) < 2 or degrees[key] > maximum_degree:
            continue
        for left, right in combinations(sorted(rows), 2):
            pair_support[(left, right)].add(key)
    return dict(pair_support), degrees


def _cross_identity_reused_keys(
    occurrences: dict[str, dict[int, set[int]]],
    identities: list[str],
) -> set[str]:
    """Find image keys shared by more than one known product identity."""

    reused: set[str] = set()
    for key, rows in occurrences.items():
        signatures = {identities[index] for index in rows if identities[index]}
        if len(signatures) > 1:
            reused.add(key)
    return reused


def _without_keys(
    occurrences: dict[str, dict[int, set[int]]], blocked: set[str]
) -> dict[str, dict[int, set[int]]]:
    return {key: rows for key, rows in occurrences.items() if key not in blocked}


def _first_image_keys(
    occurrences: dict[str, dict[int, set[int]]],
) -> dict[int, set[str]]:
    output: dict[int, set[str]] = defaultdict(set)
    for key, rows in occurrences.items():
        for index, positions in rows.items():
            if 0 in positions:
                output[index].add(key)
    return dict(output)


def _informative_first_images_conflict(
    left: int,
    right: int,
    *,
    first_exact: dict[int, set[str]],
    first_perceptual: dict[int, set[str]],
) -> bool:
    """Return true only for observable disagreement between two first images."""

    if left not in first_perceptual or right not in first_perceptual:
        return False
    return not (
        first_exact.get(left, set()) & first_exact.get(right, set())
        or first_perceptual[left] & first_perceptual[right]
    )


def _pair_has_strong_anchor(
    left: int,
    right: int,
    *,
    identities: list[str],
    first_exact: dict[int, set[str]],
    first_perceptual: dict[int, set[str]],
) -> bool:
    """Require a pair-specific identity or informative first-image anchor."""

    if identities[left] and identities[left] == identities[right]:
        return True
    return bool(
        first_exact.get(left, set()) & first_exact.get(right, set())
        or first_perceptual.get(left, set()) & first_perceptual.get(right, set())
    )


def _filter_component_safe_edges(
    ids: np.ndarray,
    edges: list[dict[str, Any]],
    *,
    identities: list[str],
    first_exact: dict[int, set[str]],
    first_perceptual: dict[int, set[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply a deterministic pairwise strong-anchor veto before component closure.

    An individually plausible edge can still create an impure component through
    transitivity.  A proposed union is therefore accepted only when every pair
    across the two existing components has either the same non-generic identity
    signature or a shared informative first-image fingerprint.
    """

    id_to_index = {str(item_id): index for index, item_id in enumerate(ids)}
    dsu = DisjointSet(len(ids))
    members: dict[int, set[int]] = {index: {index} for index in range(len(ids))}
    stage_order = {stage: index for index, stage in enumerate(STAGES)}
    ordered_edges = sorted(
        edges,
        key=lambda edge: (
            stage_order[edge["stage"]],
            edge["left_id"],
            edge["right_id"],
            edge["key_hash"],
        ),
    )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for edge in ordered_edges:
        left = id_to_index[edge["left_id"]]
        right = id_to_index[edge["right_id"]]
        left_root, right_root = dsu.find(left), dsu.find(right)
        if left_root == right_root:
            accepted.append(edge)
            continue
        left_members = members[left_root]
        right_members = members[right_root]
        incompatible = next(
            (
                (left_member, right_member)
                for left_member in sorted(left_members)
                for right_member in sorted(right_members)
                if not _pair_has_strong_anchor(
                    left_member,
                    right_member,
                    identities=identities,
                    first_exact=first_exact,
                    first_perceptual=first_perceptual,
                )
            ),
            None,
        )
        if incompatible is not None:
            rejected.append(
                {
                    **edge,
                    "component_veto": "missing_pairwise_strong_anchor",
                    "witness_left_id": str(ids[incompatible[0]]),
                    "witness_right_id": str(ids[incompatible[1]]),
                }
            )
            continue
        merged_members = left_members | right_members
        dsu.union(left_root, right_root)
        merged_root = dsu.find(left_root)
        members.pop(left_root, None)
        members.pop(right_root, None)
        members[merged_root] = merged_members
        accepted.append(edge)
    return accepted, rejected


def _text_corroboration(
    left: int,
    right: int,
    *,
    full_text: list[str],
    names: list[str],
    masked_names: list[str],
    descriptions: list[str],
    identities: list[str],
    threshold: float,
) -> tuple[str, float]:
    del names, masked_names
    if not identities[left] or identities[left] != identities[right]:
        return "", 0.0
    similarity = description_similarity(descriptions[left], descriptions[right])
    if full_text[left] and full_text[left] == full_text[right]:
        return "exact_product_identity_and_full_text", 1.0
    if similarity >= threshold:
        return "exact_product_identity_and_description", similarity
    return "exact_product_identity", similarity


def _edge(
    *,
    stage: str,
    left: int,
    right: int,
    ids: np.ndarray,
    key_hash: str,
    key_degree: int,
    left_positions: set[int] | tuple[int, ...] = (),
    right_positions: set[int] | tuple[int, ...] = (),
    corroboration: str,
    similarity: float = 0.0,
    informative_guard: bool = True,
    degree_guard: bool = True,
) -> dict[str, Any]:
    if str(ids[right]) < str(ids[left]):
        left, right = right, left
        left_positions, right_positions = right_positions, left_positions
    value = AuditEdge(
        stage=stage,
        left_id=str(ids[left]),
        right_id=str(ids[right]),
        key_hash=key_hash,
        key_degree=key_degree,
        left_positions=_position_json(left_positions),
        right_positions=_position_json(right_positions),
        corroboration=corroboration,
        description_similarity=round(float(similarity), 8),
        informative_guard=informative_guard,
        degree_guard=degree_guard,
    )
    return asdict(value)


def _component_arrays(
    ids: np.ndarray, edges: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    id_to_index = {str(item_id): index for index, item_id in enumerate(ids)}
    dsu = DisjointSet(len(ids))
    for edge in edges:
        dsu.union(id_to_index[edge["left_id"]], id_to_index[edge["right_id"]])
    roots = np.asarray([dsu.find(index) for index in range(len(ids))], dtype=np.int32)
    component_ids = np.empty(len(ids), dtype=object)
    component_sizes = np.empty(len(ids), dtype=np.int32)
    for root in np.unique(roots):
        positions = np.flatnonzero(roots == root)
        value = component_hash(ids[positions].astype(str).tolist())
        component_ids[positions] = value
        component_sizes[positions] = len(positions)
    return component_ids.astype(str), component_sizes


def build_label_blind_graph(
    frame: pd.DataFrame,
    fingerprints: list[ImageFingerprint],
    *,
    config: AuditConfig,
) -> GraphAudit:
    """Return graph topology using no category, label, fold or model score."""

    required = {"id", "name", "description"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing graph columns: {sorted(missing)}")
    if frame.id.astype(str).duplicated().any():
        raise ValueError("data ids must be unique")
    ids = frame.id.astype(str).to_numpy()
    id_to_index = {item_id: index for index, item_id in enumerate(ids)}
    names_raw = frame.name.fillna("").astype(str).tolist()
    descriptions = frame.description.fillna("").astype(str).tolist()
    names = [normalize_text(value) for value in names_raw]
    masked_names = [masked_quantity_name(value) for value in names_raw]
    identities = [product_identity_signature(value) for value in names_raw]
    full_text = [
        normalized_full_text(name, description)
        for name, description in zip(names_raw, descriptions)
    ]
    (
        all_exact_occurrences,
        all_perceptual_occurrences,
        exact_occurrences,
        perceptual_occurrences,
    ) = _image_occurrences(fingerprints, id_to_index, config)
    first_exact_image_keys = _first_image_keys(exact_occurrences)
    first_perceptual_image_keys = _first_image_keys(perceptual_occurrences)
    reused_exact_keys = _cross_identity_reused_keys(all_exact_occurrences, identities)
    reused_perceptual_keys = _cross_identity_reused_keys(all_perceptual_occurrences, identities)
    eligible_exact_occurrences = _without_keys(exact_occurrences, reused_exact_keys)
    eligible_perceptual_occurrences = _without_keys(perceptual_occurrences, reused_perceptual_keys)
    exact_pair_support, exact_degrees = _eligible_pair_support(
        eligible_exact_occurrences,
        degree_occurrences=all_exact_occurrences,
        maximum_degree=config.auxiliary_exact_max_degree,
    )
    perceptual_pair_support, perceptual_degrees = _eligible_pair_support(
        eligible_perceptual_occurrences,
        degree_occurrences=all_perceptual_occurrences,
        maximum_degree=config.perceptual_max_degree,
    )

    first_exact_pairs: dict[tuple[int, int], str] = {}
    for key, rows in eligible_exact_occurrences.items():
        if len(rows) < 2 or exact_degrees[key] > config.first_exact_max_degree:
            continue
        first_rows = sorted(index for index, positions in rows.items() if 0 in positions)
        for left, right in combinations(first_rows, 2):
            first_exact_pairs[(left, right)] = key
    strong_image_pairs = set(first_exact_pairs) | set(exact_pair_support)

    edges: list[dict[str, Any]] = []
    key_degrees: list[dict[str, Any]] = []
    manual_samples: list[dict[str, Any]] = []
    sample_counts: dict[str, int] = defaultdict(int)

    def key_audit(
        *,
        stage: str,
        key: str,
        indices: list[int] | set[int],
        status: str,
        reason: str,
        accepted_edges: int,
    ) -> None:
        key_degrees.append(
            {
                "stage": stage,
                "key_hash": opaque_key(stage, key),
                "key_degree": len(indices),
                "status": status,
                "reason": reason,
                "accepted_edges": accepted_edges,
                "manual_required": status.startswith("quarantined")
                or stage == "perceptual_corroborated",
                "sample_ids": _sample_ids(set(indices), ids),
            }
        )

    quarantined_generic_text_pairs: set[tuple[int, int]] = set()
    for key, indices in sorted(_group_indices(full_text).items()):
        if len(indices) < 2:
            continue
        ordered = sorted(indices, key=lambda index: str(ids[index]))
        key_hash = opaque_key("exact_full_text", key)
        low_information = all(not identities[index] for index in ordered)
        conflicting_pairs = [
            (left, right)
            for left, right in combinations(ordered, 2)
            if _informative_first_images_conflict(
                left,
                right,
                first_exact=first_exact_image_keys,
                first_perceptual=first_perceptual_image_keys,
            )
        ]
        if low_information or conflicting_pairs:
            quarantined_generic_text_pairs.update(
                _pair(left, right) for left, right in combinations(ordered, 2)
            )
            if conflicting_pairs:
                status = "quarantined_first_image_conflict"
                reason = "exact text conflicts across informative first product images"
                sample_reason = "non-independent exact text and conflicting first images"
            else:
                status = "quarantined_nonindependent_text"
                reason = "generic exact text has no independent product-identity anchor"
                sample_reason = "generic copied text is non-independent identity evidence"
            key_audit(
                stage="exact_full_text",
                key=key,
                indices=indices,
                status=status,
                reason=reason,
                accepted_edges=0,
            )
            sample_left, sample_right = (
                conflicting_pairs[0] if conflicting_pairs else (ordered[0], ordered[1])
            )
            left_id, right_id = _ordered_pair_ids(sample_left, sample_right, ids)
            _append_manual_sample(
                manual_samples,
                sample_counts,
                config=config,
                sample_type="quarantined_generic_text_group",
                stage="exact_full_text",
                key_hash=key_hash,
                key_degree=len(indices),
                ids=_sample_ids(set(indices), ids),
                left_id=left_id,
                right_id=right_id,
                reason=sample_reason,
            )
            continue
        for other in ordered[1:]:
            edges.append(
                _edge(
                    stage="exact_full_text",
                    left=ordered[0],
                    right=other,
                    ids=ids,
                    key_hash=key_hash,
                    key_degree=len(indices),
                    corroboration="direct_exact_full_text",
                    similarity=1.0,
                )
            )
        key_audit(
            stage="exact_full_text",
            key=key,
            indices=indices,
            status="accepted_direct",
            reason="exact full text is a direct label-blind edge",
            accepted_edges=len(indices) - 1,
        )

    def corroborated_name_stage(*, stage: str, values: list[str], maximum_degree: int) -> None:
        for key, indices in sorted(_group_indices(values).items()):
            if len(indices) < 2:
                continue
            key_hash = opaque_key(stage, key)
            if len(indices) > maximum_degree:
                key_audit(
                    stage=stage,
                    key=key,
                    indices=indices,
                    status="quarantined_degree",
                    reason=f"degree {len(indices)} exceeds fail-closed cap {maximum_degree}",
                    accepted_edges=0,
                )
                _append_manual_sample(
                    manual_samples,
                    sample_counts,
                    config=config,
                    sample_type="quarantined_text_key",
                    stage=stage,
                    key_hash=key_hash,
                    key_degree=len(indices),
                    ids=_sample_ids(set(indices), ids),
                    reason="high-degree text key requires manual audit",
                )
                continue
            accepted = 0
            best_rejected: tuple[float, str, str, int, int] | None = None
            for left, right in combinations(sorted(indices), 2):
                if _pair(left, right) in quarantined_generic_text_pairs:
                    continue
                similarity = description_similarity(descriptions[left], descriptions[right])
                identity_matches = bool(
                    identities[left] and identities[left] == identities[right]
                )
                first_image_conflict = _informative_first_images_conflict(
                    left,
                    right,
                    first_exact=first_exact_image_keys,
                    first_perceptual=first_perceptual_image_keys,
                )
                if identity_matches and not first_image_conflict and (
                    similarity >= config.description_similarity_min
                ):
                    corroboration = "description_similarity"
                elif (
                    identity_matches
                    and not first_image_conflict
                    and _pair(left, right) in strong_image_pairs
                ):
                    corroboration = "strong_image_pair"
                else:
                    left_id, right_id = _ordered_pair_ids(left, right, ids)
                    candidate = (similarity, left_id, right_id, left, right)
                    if best_rejected is None or (
                        candidate[0] > best_rejected[0]
                        or (
                            candidate[0] == best_rejected[0] and candidate[1:3] < best_rejected[1:3]
                        )
                    ):
                        best_rejected = candidate
                    continue
                edges.append(
                    _edge(
                        stage=stage,
                        left=left,
                        right=right,
                        ids=ids,
                        key_hash=key_hash,
                        key_degree=len(indices),
                        corroboration=corroboration,
                        similarity=similarity,
                    )
                )
                accepted += 1
            status = "accepted_corroborated" if accepted else "rejected_no_corroboration"
            key_audit(
                stage=stage,
                key=key,
                indices=indices,
                status=status,
                reason=(
                    "pair-specific description or strong-image corroboration"
                    if accepted
                    else "no pair passed description/image corroboration"
                ),
                accepted_edges=accepted,
            )
            if best_rejected is not None:
                similarity, left_id, right_id, left, right = best_rejected
                _append_manual_sample(
                    manual_samples,
                    sample_counts,
                    config=config,
                    sample_type="rejected_text_pair",
                    stage=stage,
                    key_hash=key_hash,
                    key_degree=len(indices),
                    ids=_sample_ids({left, right}, ids),
                    left_id=left_id,
                    right_id=right_id,
                    reason=f"closest rejected description similarity={similarity:.6f}",
                )

    corroborated_name_stage(
        stage="exact_name_corroborated",
        values=names,
        maximum_degree=config.exact_name_max_degree,
    )
    corroborated_name_stage(
        stage="masked_name_corroborated",
        values=masked_names,
        maximum_degree=config.masked_name_max_degree,
    )

    first_edges_by_key: dict[str, int] = defaultdict(int)
    for (left, right), key in sorted(first_exact_pairs.items()):
        rows = eligible_exact_occurrences[key]
        edges.append(
            _edge(
                stage="exact_first_image",
                left=left,
                right=right,
                ids=ids,
                key_hash=opaque_key("exact_first_image", key),
                key_degree=exact_degrees[key],
                left_positions=rows[left],
                right_positions=rows[right],
                corroboration="both_informative_first_images",
            )
        )
        first_edges_by_key[key] += 1
    for key, all_rows in sorted(all_exact_occurrences.items()):
        rows = exact_occurrences.get(key, {})
        if len(all_rows) < 2:
            continue
        if key in reused_exact_keys:
            status = "quarantined_cross_identity_reuse"
            reason = "image key occurs across distinct product identities"
        elif len(all_rows) > config.first_exact_max_degree:
            status, reason = "quarantined_degree", "exact first-image key exceeds degree cap"
        elif first_edges_by_key[key]:
            status, reason = "accepted_direct", "informative exact first images"
        elif len(rows) < 2:
            status, reason = "rejected_uninformative", "fewer than two informative occurrences"
        else:
            status, reason = "rejected_position", "fewer than two occurrences at position zero"
        key_audit(
            stage="exact_first_image",
            key=key,
            indices=set(all_rows),
            status=status,
            reason=reason,
            accepted_edges=first_edges_by_key[key],
        )
        if status.startswith("quarantined"):
            _append_manual_sample(
                manual_samples,
                sample_counts,
                config=config,
                sample_type=(
                    "quarantined_reused_image_key"
                    if status == "quarantined_cross_identity_reuse"
                    else "quarantined_image_key"
                ),
                stage="exact_first_image",
                key_hash=opaque_key("exact_first_image", key),
                key_degree=len(all_rows),
                ids=_sample_ids(set(all_rows), ids),
                reason=reason,
            )

    exact_aux_accepted: dict[str, int] = defaultdict(int)
    for pair, shared_keys in sorted(exact_pair_support.items()):
        left, right = pair
        if pair in first_exact_pairs:
            continue
        text_reason, similarity = _text_corroboration(
            left,
            right,
            full_text=full_text,
            names=names,
            masked_names=masked_names,
            descriptions=descriptions,
            identities=identities,
            threshold=config.description_similarity_min,
        )
        first_image_conflict = _informative_first_images_conflict(
            left,
            right,
            first_exact=first_exact_image_keys,
            first_perceptual=first_perceptual_image_keys,
        )
        if not text_reason or first_image_conflict:
            key = next(iter(shared_keys))
            left_id, right_id = _ordered_pair_ids(left, right, ids)
            _append_manual_sample(
                manual_samples,
                sample_counts,
                config=config,
                sample_type="rejected_auxiliary_pair",
                stage="exact_auxiliary_images",
                key_hash=opaque_key("exact_auxiliary_images", key),
                key_degree=exact_degrees[key],
                ids=_sample_ids({left, right}, ids),
                left_id=left_id,
                right_id=right_id,
                reason=(
                    "auxiliary exact matches require independent non-generic product "
                    "identity and no informative first-image conflict"
                ),
            )
            continue
        corroboration = f"exact_auxiliary_plus_{text_reason}"
        combined_key = "|".join(sorted(shared_keys))
        left_positions = set().union(
            *(eligible_exact_occurrences[key][left] for key in shared_keys)
        )
        right_positions = set().union(
            *(eligible_exact_occurrences[key][right] for key in shared_keys)
        )
        edges.append(
            _edge(
                stage="exact_auxiliary_images",
                left=left,
                right=right,
                ids=ids,
                key_hash=opaque_key("exact_auxiliary_images", combined_key),
                key_degree=max(exact_degrees[key] for key in shared_keys),
                left_positions=left_positions,
                right_positions=right_positions,
                corroboration=corroboration,
                similarity=similarity,
            )
        )
        left_id, right_id = _ordered_pair_ids(left, right, ids)
        _append_manual_sample(
            manual_samples,
            sample_counts,
            config=config,
            sample_type="accepted_auxiliary_pair",
            stage="exact_auxiliary_images",
            key_hash=opaque_key("exact_auxiliary_images", combined_key),
            key_degree=max(exact_degrees[key] for key in shared_keys),
            ids=_sample_ids({left, right}, ids),
            left_id=left_id,
            right_id=right_id,
            reason=corroboration,
        )
        for key in shared_keys:
            exact_aux_accepted[key] += 1
    for key, all_rows in sorted(all_exact_occurrences.items()):
        rows = exact_occurrences.get(key, {})
        if len(all_rows) < 2:
            continue
        if key in reused_exact_keys:
            status = "quarantined_cross_identity_reuse"
            reason = "image key occurs across distinct product identities"
        elif len(all_rows) > config.auxiliary_exact_max_degree:
            status, reason = "quarantined_degree", "auxiliary exact-image key exceeds cap"
        elif exact_aux_accepted[key]:
            status = "accepted_corroborated"
            reason = "independent product-identity text corroboration"
        elif len(rows) < 2:
            status, reason = "rejected_uninformative", "fewer than two informative occurrences"
        else:
            status, reason = "rejected_no_corroboration", "single auxiliary match is insufficient"
        key_audit(
            stage="exact_auxiliary_images",
            key=key,
            indices=set(all_rows),
            status=status,
            reason=reason,
            accepted_edges=exact_aux_accepted[key],
        )
        if status == "quarantined_cross_identity_reuse":
            _append_manual_sample(
                manual_samples,
                sample_counts,
                config=config,
                sample_type="quarantined_reused_image_key",
                stage="exact_auxiliary_images",
                key_hash=opaque_key("exact_auxiliary_images", key),
                key_degree=len(all_rows),
                ids=_sample_ids(set(all_rows), ids),
                reason=reason,
            )

    perceptual_accepted: dict[str, int] = defaultdict(int)
    for pair, shared_keys in sorted(perceptual_pair_support.items()):
        left, right = pair
        if pair in exact_pair_support:
            continue
        text_reason, similarity = _text_corroboration(
            left,
            right,
            full_text=full_text,
            names=names,
            masked_names=masked_names,
            descriptions=descriptions,
            identities=identities,
            threshold=config.description_similarity_min,
        )
        first_image_conflict = _informative_first_images_conflict(
            left,
            right,
            first_exact=first_exact_image_keys,
            first_perceptual=first_perceptual_image_keys,
        )
        both_first = any(
            0 in eligible_perceptual_occurrences[key][left]
            and 0 in eligible_perceptual_occurrences[key][right]
            for key in shared_keys
        )
        if both_first:
            corroboration = "two_hashes_per_match_and_both_first_positions"
        elif text_reason and not first_image_conflict:
            corroboration = f"two_hashes_per_match_plus_{text_reason}"
        else:
            key = next(iter(shared_keys))
            left_id, right_id = _ordered_pair_ids(left, right, ids)
            _append_manual_sample(
                manual_samples,
                sample_counts,
                config=config,
                sample_type="rejected_perceptual_pair",
                stage="perceptual_corroborated",
                key_hash=opaque_key("perceptual_corroborated", key),
                key_degree=perceptual_degrees[key],
                ids=_sample_ids({left, right}, ids),
                left_id=left_id,
                right_id=right_id,
                reason=(
                    "perceptual auxiliary matches require an informative first-image "
                    "match or independent product identity"
                ),
            )
            continue
        combined_key = "|".join(sorted(shared_keys))
        left_positions = set().union(
            *(eligible_perceptual_occurrences[key][left] for key in shared_keys)
        )
        right_positions = set().union(
            *(eligible_perceptual_occurrences[key][right] for key in shared_keys)
        )
        edges.append(
            _edge(
                stage="perceptual_corroborated",
                left=left,
                right=right,
                ids=ids,
                key_hash=opaque_key("perceptual_corroborated", combined_key),
                key_degree=max(perceptual_degrees[key] for key in shared_keys),
                left_positions=left_positions,
                right_positions=right_positions,
                corroboration=corroboration,
                similarity=similarity,
            )
        )
        left_id, right_id = _ordered_pair_ids(left, right, ids)
        _append_manual_sample(
            manual_samples,
            sample_counts,
            config=config,
            sample_type="accepted_perceptual_pair",
            stage="perceptual_corroborated",
            key_hash=opaque_key("perceptual_corroborated", combined_key),
            key_degree=max(perceptual_degrees[key] for key in shared_keys),
            ids=_sample_ids({left, right}, ids),
            left_id=left_id,
            right_id=right_id,
            reason=corroboration,
        )
        for key in shared_keys:
            perceptual_accepted[key] += 1
    for key, all_rows in sorted(all_perceptual_occurrences.items()):
        rows = perceptual_occurrences.get(key, {})
        if len(all_rows) < 2:
            continue
        if key in reused_perceptual_keys:
            status = "quarantined_cross_identity_reuse"
            reason = "image key occurs across distinct product identities"
        elif len(all_rows) > config.perceptual_max_degree:
            status, reason = "quarantined_degree", "perceptual key exceeds degree cap"
        elif perceptual_accepted[key]:
            status, reason = "accepted_requires_manual_audit", "position/image/text corroboration"
        elif len(rows) < 2:
            status, reason = "rejected_uninformative", "fewer than two informative occurrences"
        else:
            status, reason = "rejected_no_corroboration", "perceptual key alone is insufficient"
        key_audit(
            stage="perceptual_corroborated",
            key=key,
            indices=set(all_rows),
            status=status,
            reason=reason,
            accepted_edges=perceptual_accepted[key],
        )
        if status == "quarantined_cross_identity_reuse":
            _append_manual_sample(
                manual_samples,
                sample_counts,
                config=config,
                sample_type="quarantined_reused_image_key",
                stage="perceptual_corroborated",
                key_hash=opaque_key("perceptual_corroborated", key),
                key_degree=len(all_rows),
                ids=_sample_ids(set(all_rows), ids),
                reason=reason,
            )

    edges, component_vetoes = _filter_component_safe_edges(
        ids,
        edges,
        identities=identities,
        first_exact=first_exact_image_keys,
        first_perceptual=first_perceptual_image_keys,
    )
    for veto in component_vetoes:
        _append_manual_sample(
            manual_samples,
            sample_counts,
            config=config,
            sample_type="quarantined_component_merge",
            stage=veto["stage"],
            key_hash=veto["key_hash"],
            key_degree=int(veto["key_degree"]),
            ids=json.dumps(
                sorted([veto["witness_left_id"], veto["witness_right_id"]]),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            left_id=veto["witness_left_id"],
            right_id=veto["witness_right_id"],
            reason="component merge lacks pairwise strong identity or first-image anchors",
        )
    stage_order = {stage: index for index, stage in enumerate(STAGES)}
    edges.sort(
        key=lambda edge: (
            stage_order[edge["stage"]],
            edge["left_id"],
            edge["right_id"],
            edge["key_hash"],
        )
    )
    key_degrees.sort(key=lambda row: (stage_order[row["stage"]], row["key_hash"]))
    component_ids, component_sizes = _component_arrays(ids, edges)
    largest_limit = max(1, math.floor(len(frame) * config.maximum_component_fraction))
    component_groups = pd.Series(component_ids).groupby(component_ids).groups
    for value, positions in sorted(
        component_groups.items(), key=lambda item: (-len(item[1]), str(item[0]))
    ):
        if len(positions) <= largest_limit:
            break
        _append_manual_sample(
            manual_samples,
            sample_counts,
            config=config,
            sample_type="oversized_component",
            stage="final_component",
            key_hash=str(value),
            key_degree=len(positions),
            ids=_sample_ids(set(map(int, positions)), ids),
            reason=f"component exceeds {config.maximum_component_fraction:.4f} of rows",
        )
    manual_samples.sort(
        key=lambda row: (row["sample_type"], row["stage"], row["key_hash"], row["left_id"])
    )
    return GraphAudit(
        component_ids=component_ids,
        component_sizes=component_sizes,
        edges=edges,
        key_degrees=key_degrees,
        manual_samples=manual_samples,
    )


def incremental_component_audit(
    frame: pd.DataFrame, edges: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ids = frame.id.astype(str).to_numpy()
    id_to_index = {item_id: index for index, item_id in enumerate(ids)}
    dsu = DisjointSet(len(frame))
    output: list[dict[str, Any]] = []
    for stage in STAGES:
        new_unions = 0
        stage_edges = [edge for edge in edges if edge["stage"] == stage]
        for edge in stage_edges:
            new_unions += int(
                dsu.union(id_to_index[edge["left_id"]], id_to_index[edge["right_id"]])
            )
        roots = np.asarray([dsu.find(index) for index in range(len(frame))])
        audit = (
            pd.DataFrame(
                {
                    "root": roots,
                    "category": frame.category.astype(str).to_numpy(),
                    "label": frame.label.astype(int).to_numpy(),
                }
            )
            .groupby("root")
            .agg(
                rows=("root", "size"),
                categories=("category", "nunique"),
                labels=("label", "nunique"),
            )
        )
        output.append(
            {
                "stage": stage,
                "accepted_edges": len(stage_edges),
                "new_unions": new_unions,
                "components": len(audit),
                "non_singleton_components": int((audit.rows > 1).sum()),
                "rows_in_non_singletons": int(audit.loc[audit.rows > 1, "rows"].sum()),
                "mixed_category_components": int((audit.categories > 1).sum()),
                "mixed_label_components": int((audit.labels > 1).sum()),
                "largest_component": int(audit.rows.max()),
                "component_size_p95": float(audit.rows.quantile(0.95)),
                "component_size_p99": float(audit.rows.quantile(0.99)),
            }
        )
    return output


def _assignment_manifest(assignment: Any) -> dict[str, Any]:
    return {
        "fold_counts_columns": ["rows", *JOINT_STRATA],
        "fold_counts": assignment.fold_counts.astype(int).tolist(),
        "objective": [float(value) for value in assignment.objective],
        "trials": int(assignment.trials),
    }


def target_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "rows": output_dir / "rows.csv",
        "edges": output_dir / "edges.csv",
        "key_degrees": output_dir / "key_degrees.csv",
        "incremental": output_dir / "incremental_components.json",
        "manual_samples": output_dir / "manual_audit_samples.csv",
        "fingerprints": output_dir / "image_fingerprints.csv.gz",
        "manifest": output_dir / "manifest.json",
    }


def ensure_targets_absent(paths: dict[str, Path]) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing graph-audit targets: " + ", ".join(map(str, existing))
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    fingerprint_source = parser.add_mutually_exclusive_group(required=True)
    fingerprint_source.add_argument("--images-zip", type=Path)
    fingerprint_source.add_argument("--reuse-fingerprints", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assignment-trials", type=int, default=128)
    parser.add_argument("--draft-only", action="store_true")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    if not args.draft_only:
        raise SystemExit(
            "fail-closed: this builder emits only a draft graph audit; "
            "pass --draft-only and never use its partitions for candidate scoring"
        )
    paths = target_paths(args.output_dir)
    ensure_targets_absent(paths)
    if args.assignment_trials < 1:
        raise ValueError("assignment trials must be positive")

    frame = pd.read_csv(args.data, dtype={"id": str})
    required = {"id", "name", "description", "category", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing data columns: {sorted(missing)}")
    if frame.id.duplicated().any():
        raise ValueError("data ids must be unique")
    strata = frame.category.astype(str) + "|" + frame.label.astype(str)
    unexpected = sorted(set(strata) - set(JOINT_STRATA))
    if unexpected:
        raise ValueError(f"unexpected joint strata: {unexpected}")

    config = AuditConfig()
    expected_ids = set(frame.id.astype(str))
    reuse_fingerprints = args.reuse_fingerprints is not None
    source_path = args.reuse_fingerprints or args.images_zip
    assert source_path is not None
    if reuse_fingerprints:
        fingerprints = read_reused_fingerprints(source_path, expected_ids)
    else:
        fingerprints = read_zip_fingerprints(source_path, expected_ids)
    fingerprint_source_sha256 = file_sha256(source_path)
    graph = build_label_blind_graph(frame, fingerprints, config=config)
    incremental = incremental_component_audit(frame, graph.edges)

    rows = frame[["id", "category", "label"]].copy()
    rows["semantic_component"] = graph.component_ids
    rows["component_size"] = graph.component_sizes
    outer = assign_balanced_component_folds(
        rows,
        component_column="semantic_component",
        n_splits=7,
        seed=20260822,
        trials=args.assignment_trials,
    )
    rows["draft_partition_7"] = outer.row_folds
    draft_holdout_candidate = outer.row_folds == 0
    rows["draft_holdout_candidate_not_sealed"] = draft_holdout_candidate
    rows["draft_dev_fold_5"] = -1
    development = rows.loc[~draft_holdout_candidate].copy()
    dev_assignment = assign_balanced_component_folds(
        development,
        component_column="semantic_component",
        n_splits=5,
        seed=314159,
        trials=args.assignment_trials,
    )
    rows.loc[~draft_holdout_candidate, "draft_dev_fold_5"] = dev_assignment.row_folds

    component_split_audit = rows.groupby("semantic_component").agg(
        partition_values=("draft_partition_7", "nunique"),
        draft_roles=("draft_holdout_candidate_not_sealed", "nunique"),
        dev_values=("draft_dev_fold_5", lambda values: values[values >= 0].nunique()),
    )
    invariants = {
        "all_ids_unique": bool(rows.id.is_unique),
        "all_rows_accounted_for": bool(len(rows) == len(frame)),
        "component_crosses_partition_7": int((component_split_audit.partition_values > 1).sum()),
        "component_crosses_draft_role": int((component_split_audit.draft_roles > 1).sum()),
        "component_crosses_dev_fold": int((component_split_audit.dev_values > 1).sum()),
        "draft_candidate_has_dev_fold_minus_one": bool(
            (rows.loc[draft_holdout_candidate, "draft_dev_fold_5"] == -1).all()
        ),
        "development_has_assigned_fold": bool(
            (rows.loc[~draft_holdout_candidate, "draft_dev_fold_5"] >= 0).all()
        ),
        "topology_uses_labels": False,
        "candidate_scores_used": False,
        "sealed": False,
        "valid_for_candidate_scoring": False,
    }
    if not all(
        [
            invariants["all_ids_unique"],
            invariants["all_rows_accounted_for"],
            invariants["component_crosses_partition_7"] == 0,
            invariants["component_crosses_draft_role"] == 0,
            invariants["component_crosses_dev_fold"] == 0,
            invariants["draft_candidate_has_dev_fold_minus_one"],
            invariants["development_has_assigned_fold"],
        ]
    ):
        raise RuntimeError(f"graph-audit invariant failed: {invariants}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(paths["rows"], index=False)
    pd.DataFrame(graph.edges).to_csv(paths["edges"], index=False)
    pd.DataFrame(graph.key_degrees).to_csv(paths["key_degrees"], index=False)
    pd.DataFrame(graph.manual_samples).to_csv(paths["manual_samples"], index=False)
    pd.DataFrame([asdict(value) for value in fingerprints]).to_csv(
        paths["fingerprints"], index=False, compression="gzip"
    )
    paths["incremental"].write_text(
        json.dumps(incremental, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "graph_version": GRAPH_VERSION,
        "status": STATUS,
        "draft": True,
        "requires_manual_audit": True,
        "sealed": False,
        "valid_for_candidate_scoring": False,
        "topology": {
            "label_blind": True,
            "candidate_scores_used": False,
            "edge_stages": list(STAGES),
            "configuration": asdict(config),
            "perceptual_rule": (
                "exact agreement of DCT-pHash, dHash and aspect bucket; never alone; "
                "auxiliary matches require an informative first-image match or an "
                "independent product-identity text signature"
            ),
            "auxiliary_rule": (
                "exact auxiliary matches require an independent non-generic product-identity "
                "text signature and no informative first-image conflict; image "
                "multiplicity and marketing creatives are insufficient"
            ),
            "generic_asset_rule": (
                "image keys observed across distinct product-identity signatures are "
                "non-informative and quarantined"
            ),
            "generic_text_rule": (
                "generic exact text is non-independent evidence; every exact-text or "
                "name edge is vetoed on informative first-image conflict"
            ),
            "identity_normalization_rule": (
                "conservative Russian stems remove inflected generic, marketing and "
                "commodity-class terms before identity corroboration"
            ),
            "component_closure_rule": (
                "every cross-pair in a proposed component merge requires an equal "
                "non-generic identity or a shared informative first-image fingerprint"
            ),
        },
        "rows": len(rows),
        "components": int(rows.semantic_component.nunique()),
        "largest_component": int(rows.component_size.max()),
        "edges": len(graph.edges),
        "key_degree_records": len(graph.key_degrees),
        "manual_samples": len(graph.manual_samples),
        "incremental_components": incremental,
        "draft_assignment": {
            "warning": "diagnostic only; fold zero is not a sealed holdout",
            "partition_7": _assignment_manifest(outer),
            "holdout_candidate_fold": 0,
            "development_5": _assignment_manifest(dev_assignment),
        },
        "invariants": invariants,
        "input_sha256": {
            "data": file_sha256(args.data),
            (
                "reused_fingerprints" if reuse_fingerprints else "images_zip"
            ): fingerprint_source_sha256,
        },
        "fingerprint_source": {
            "reuse": reuse_fingerprints,
            "sha256": fingerprint_source_sha256,
        },
        "output_sha256": {
            name: file_sha256(path) for name, path in paths.items() if name != "manifest"
        },
        "forbidden_next_actions": [
            "do not score a candidate on draft partitions",
            "do not rename this audit as sealed",
            "do not promote before manual edge/component review and a new immutable version",
        ],
    }
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": STATUS,
                "rows": len(rows),
                "components": manifest["components"],
                "manual_samples": manifest["manual_samples"],
                "sealed": False,
                "valid_for_candidate_scoring": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
