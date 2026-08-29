from __future__ import annotations

import hashlib
import html
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

DATA_VERSION = "competition_train_v1"
EVALUATION_VERSION = "semantic_family_v3"
PROMPT_VERSION = "qwen_scale_binary_v1"
PREPROCESSING_VERSION = "qwen_scale_first_image_v1"
GRID_CONTRACT_VERSION = "qwen_scale_2x3_v1"
SCREEN_FOLDS = (0, 3)
FULL_FOLDS = (0, 1, 2, 3, 4)
CATEGORIES = ("БАД", "Легковоспламеняющиеся")
NO_EVIDENCE = "NO_EVIDENCE"
CONCEPTS = (
    "OBJECT_OF_SALE",
    "COMPOSITION",
    "COMPLETENESS",
    "FUEL_OR_IGNITION",
    "NEGATION",
)
MODEL_REVISIONS = {
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.8-27B": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
}
RULES = {
    "БАД": (
        "Метка 1 только если в описании или на упаковке есть прямое указание БАД "
        "или dietary supplement. Спортивное питание без такой маркировки, явное "
        "отрицание или отсутствие маркировки — метка 0."
    ),
    "Легковоспламеняющиеся": (
        "Метка 1 для самостоятельного источника огня, горючего вещества или газа, "
        "либо если такой товар входит в комплект. Пустое оборудование, встроенный "
        "источник, горючий материал только как компонент или предмет не в комплекте — 0."
    ),
}


@dataclass(frozen=True)
class CellSpec:
    experiment_id: str
    model_id: str
    objective: str
    evidence_auxiliary_weight: float = 0.0

    @property
    def model_revision(self) -> str:
        return MODEL_REVISIONS[self.model_id]

    @property
    def model_size(self) -> str:
        return "4b" if self.model_id.endswith("4B") else "27b"

    @property
    def micro_batch_size(self) -> int:
        return 4 if self.model_size == "4b" else 1

    @property
    def gradient_accumulation(self) -> int:
        return 4 if self.model_size == "4b" else 16


CELL_SPECS = {
    "641": CellSpec("641", "Qwen/Qwen3.5-4B", "class_only"),
    "642": CellSpec("642", "Qwen/Qwen3.8-27B", "class_only"),
    "643": CellSpec("643", "Qwen/Qwen3.5-4B", "grounded_evidence", 0.05),
    "644": CellSpec("644", "Qwen/Qwen3.8-27B", "grounded_evidence", 0.05),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def clean_text(value: object, limit: int) -> str:
    missing = value is None or (isinstance(value, float) and math.isnan(value))
    text = "" if missing else html.unescape(str(value))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    return text[:head].rstrip() + " … " + text[-(limit - head) :].lstrip()


def base_prompt(row: SimpleNamespace | Any) -> str:
    category = str(row.category)
    if category not in RULES:
        raise ValueError(f"unsupported category: {category}")
    return (
        f"Категория: {category}\n"
        f"Название: {clean_text(row.name, 320)}\n"
        f"Описание: {clean_text(row.description, 1800)}\n"
        f"Правило: {RULES[category]}\n"
        "Определи правильность категории. Ответь только одной цифрой: 1 или 0."
    )


def evidence_prompt(row: SimpleNamespace | Any, order: str) -> str:
    if order not in {"class_first", "evidence_first"}:
        raise ValueError("order must be class_first or evidence_first")
    order_instruction = (
        "Сначала verdict, затем quote и concept."
        if order == "class_first"
        else "Сначала quote и concept, затем verdict."
    )
    return (
        base_prompt(row)
        + "\nВерни один JSON-объект без дополнительного текста. "
        + order_instruction
        + " quote должен быть точной цитатой из названия, описания или видимого "
        "текста изображения; иначе quote=NO_EVIDENCE и concept=NO_EVIDENCE."
    )


def conditioned_verdict_prompt(row: SimpleNamespace | Any, evidence: dict[str, Any]) -> str:
    quote = str(evidence.get("quote", NO_EVIDENCE))
    concept = str(evidence.get("concept", NO_EVIDENCE))
    return (
        base_prompt(row)
        + "\nПроверенное доказательство: "
        + json.dumps({"quote": quote, "concept": concept}, ensure_ascii=False, sort_keys=True)
        + "\nС учётом только этого проверенного доказательства ответь одной цифрой: 1 или 0."
    )


def structured_target(*, verdict: int, quote: str, concept: str, order: str) -> str:
    if verdict not in (0, 1):
        raise ValueError("verdict must be binary")
    if quote == NO_EVIDENCE:
        concept = NO_EVIDENCE
    elif concept not in CONCEPTS:
        raise ValueError("grounded evidence concept is outside the closed ontology")
    if order == "class_first":
        payload = {"verdict": verdict, "quote": quote, "concept": concept}
    elif order == "evidence_first":
        payload = {"quote": quote, "concept": concept, "verdict": verdict}
    else:
        raise ValueError("unknown target order")
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_structured_generation(raw: str, *, expected_order: str) -> dict[str, Any]:
    result = {
        "format_valid": False,
        "generated_verdict": None,
        "quote": NO_EVIDENCE,
        "concept": NO_EVIDENCE,
    }
    try:
        payload = json.loads(raw.strip())
    except (json.JSONDecodeError, TypeError):
        return result
    expected_keys = (
        ["verdict", "quote", "concept"]
        if expected_order == "class_first"
        else ["quote", "concept", "verdict"]
    )
    if not isinstance(payload, dict) or list(payload) != expected_keys:
        return result
    verdict = payload.get("verdict")
    quote = payload.get("quote")
    concept = payload.get("concept")
    if verdict not in (0, 1) or not isinstance(quote, str) or not isinstance(concept, str):
        return result
    if quote == NO_EVIDENCE and concept != NO_EVIDENCE:
        return result
    if quote != NO_EVIDENCE and concept not in CONCEPTS:
        return result
    return {
        "format_valid": True,
        "generated_verdict": int(verdict),
        "quote": quote,
        "concept": concept,
    }


def canonical_text(row: SimpleNamespace | dict[str, Any] | Any) -> str:
    getter = (
        row.get if isinstance(row, dict) else lambda key, default="": getattr(row, key, default)
    )
    return f"Название: {getter('name', '')}\nОписание: {getter('description', '')}"


def _nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def resolve_grounding(row: dict[str, Any], quote: str) -> dict[str, Any]:
    """Resolve a generated quote without fuzzy matching or generated coordinates."""

    if quote == NO_EVIDENCE:
        return {
            "grounded": True,
            "source": "none",
            "image_index": -1,
            "region_index": -1,
            "polygon": None,
        }
    text = canonical_text(row)
    hits = [match.start() for match in re.finditer(re.escape(quote), text)]
    if len(hits) == 1:
        return {
            "grounded": True,
            "source": "text",
            "char_start": hits[0],
            "char_end": hits[0] + len(quote),
            "image_index": -1,
            "region_index": -1,
            "polygon": None,
        }
    ocr_hits: list[tuple[int, int, Any]] = []
    for image in row.get("ocr_images", []):
        image_index = int(image["image_index"])
        for region_index, region in enumerate(image.get("detections", [])):
            if _nfc(str(region.get("text", ""))) == _nfc(quote):
                ocr_hits.append((image_index, region_index, region.get("polygon")))
    if len(ocr_hits) == 1:
        image_index, region_index, polygon = ocr_hits[0]
        return {
            "grounded": True,
            "source": "ocr",
            "image_index": image_index,
            "region_index": region_index,
            "polygon": polygon,
        }
    return {
        "grounded": False,
        "source": "unresolved",
        "image_index": -1,
        "region_index": -1,
        "polygon": None,
    }


def grid_contract_payload() -> dict[str, Any]:
    return {
        "version": GRID_CONTRACT_VERSION,
        "data_version": DATA_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "screen_folds": list(SCREEN_FOLDS),
        "full_folds": list(FULL_FOLDS),
        "prompt_version": PROMPT_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "models": MODEL_REVISIONS,
        "threshold": 0.0,
        "threshold_tuned": False,
        "thinking": False,
        "temperature": 0.0,
        "image_policy": "first_image_only_max_262144_pixels",
        "description_limit": 1800,
        "name_limit": 320,
        "training": {
            "seed": 42,
            "epochs": 1,
            "effective_batch_size": 16,
            "learning_rate": 0.0002,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "grounded_auxiliary_weight": 0.05,
            "grounded_order_weights": {
                "class_first": 0.5,
                "evidence_first": 0.5,
            },
        },
        "evidence": {
            "free_coordinates": False,
            "resolution": "exact unique listing substring, else exact unique OCR-region text",
            "no_evidence": NO_EVIDENCE,
            "primary_inference_order": "evidence_first",
            "diagnostic_inference_order": "class_first",
        },
    }


GRID_CONTRACT_SHA256 = canonical_sha256(grid_contract_payload())
