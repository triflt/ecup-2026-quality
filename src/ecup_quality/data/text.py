from __future__ import annotations

import hashlib
import html
import re
import unicodedata


def normalize_text(value: object) -> str:
    """Normalize product text for exact grouping without removing digits."""
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    value = unicodedata.normalize("NFKC", value).lower().replace("ё", "е")
    value = re.sub(r"[^0-9a-zа-я]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def canonicalize(value: object, *, mask_digits: bool = False) -> str:
    """Canonicalize wording variants used by conservative family priors."""
    value = html.unescape(str(value or ""))
    value = re.sub(r"<[^>]+>", " ", value)
    value = unicodedata.normalize("NFKC", value).lower().replace("ё", "е")
    if mask_digits:
        value = re.sub(r"\d+(?:[.,]\d+)?", " # ", value)
    value = re.sub(r"[^0-9a-zа-я#]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def compose_text(name: object, description: object) -> str:
    """Match the train/inference text layout used by the strongest text head."""
    normalized_name = normalize_text(name)
    return f"{normalized_name}\n{normalized_name}\n{normalize_text(description)}"


def text_fingerprint(name: object, description: object) -> str:
    payload = compose_text(name, description).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()
