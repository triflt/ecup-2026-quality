"""Label-blind causal attributes for the rare flammable class.

The extractor is deliberately conservative.  It emits a supervised auxiliary
target only when the sold object, regulated substance and their relation are
supported by an exact surface span.  Ambiguous rows fail closed and remain
available for the ordinary verdict loss.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Final

SOLD_OBJECTS: Final = ("consumable", "device", "accessory", "kit", "unknown")
REGULATED_SUBSTANCES: Final = (
    "bad_marker",
    "gas",
    "flammable_liquid",
    "solid_fuel",
    "ignition_aid",
    "none",
)
RELATIONS: Final = (
    "sold_object",
    "included",
    "compatible_external",
    "mentioned_only",
    "negated",
    "unknown",
)


@dataclass(frozen=True)
class Match:
    source: str
    start: int
    end: int
    span: str
    concept: str


@dataclass(frozen=True)
class CausalTarget:
    sold_object: str
    regulated_substance: str
    relation: str
    evidence_source: str | None
    evidence_start: int | None
    evidence_end: int | None
    evidence_span: str | None
    supported: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _compile(items: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("(?:" + "|".join(items) + ")", re.IGNORECASE | re.UNICODE)


SUBSTANCE_PATTERNS: Final = {
    "flammable_liquid": _compile(
        (
            r"жидкост(?:ь|и)\s+для\s+розжига",
            r"топливн(?:ая|ой)\s+жидкост(?:ь|и)",
            r"бензин(?:а|ом)?",
            r"керосин(?:а|ом)?",
            r"биоэтанол(?:а|ом)?",
            r"спиртов(?:ое|ого)\s+топлив",
        )
    ),
    "solid_fuel": _compile(
        (
            r"древесн(?:ый|ого)\s+угол[ьяь]",
            r"каменн(?:ый|ого)\s+угол[ьяь]",
            r"угол[ьяь]",
            r"дров(?:а|ами|яной)",
            r"пеллет(?:ы|ами)?",
            r"топливн(?:ые|ых)\s+брикет",
        )
    ),
    "ignition_aid": _compile(
        (
            r"сух(?:ое|ого)\s+горюч",
            r"гель\s+для\s+розжига",
            r"растопк(?:а|и|ой)",
            r"средств(?:о|а)\s+для\s+розжига",
            r"спич(?:ки|ек)",
            r"зажигалк(?:а|и|ой)",
            r"огнив(?:о|а)",
        )
    ),
    "gas": _compile(
        (
            r"газов(?:ый|ая|ое|ые|ого|ому|ым|ом)\s+баллон",
            r"баллон(?:а|ом|ы)?\s+(?:с\s+)?газ",
            r"газов(?:ая|ой|ую)\s+смес",
            r"пропан(?:а|ом)?",
            r"бутан(?:а|ом)?",
            r"изобутан(?:а|ом)?",
            r"газ(?:а|ом)?",
        )
    ),
    "bad_marker": _compile(
        (
            r"\bБАД\b",
            r"биологически\s+активн(?:ая|ой|ую)\s+добав",
            r"биодобавк(?:а|и|ой|у)",
            r"пищев(?:ая|ой|ую)\s+добавк",
        )
    ),
}

DEVICE_PATTERN: Final = _compile(
    (
        r"горелк(?:а|и|ой|у)",
        r"плит(?:а|ка|ки|ой|у)",
        r"мангал(?:а|ом)?",
        r"грил(?:ь|я|ем)",
        r"печ(?:ь|и|ью|ка)",
        r"камин(?:а|ом)?",
        r"обогревател",
        r"ламп(?:а|ы|ой)",
        r"резак(?:а|ом)?",
        r"заправочн(?:ая|ое)\s+устройств",
    )
)
ACCESSORY_PATTERN: Final = _compile(
    (
        r"чех(?:ол|ла)",
        r"переходник",
        r"адаптер",
        r"шланг",
        r"редуктор",
        r"держател",
        r"подставк",
        r"насадк",
        r"запчаст",
    )
)
KIT_PATTERN: Final = _compile((r"\bнабор\b", r"\bкомплект\b", r"\bkit\b"))
CONSUMABLE_PATTERN: Final = _compile(
    (
        r"баллон",
        r"жидкост",
        r"топлив",
        r"угол[ьяь]",
        r"дров",
        r"пеллет",
        r"брикет",
        r"растопк",
        r"сух(?:ое|ого)\s+горюч",
        r"гель\s+для\s+розжига",
        r"спич",
        r"зажигал",
    )
)

NEGATED_PATTERN: Final = _compile(
    (
        r"не\s+входит",
        r"не\s+включ[её]н",
        r"без\s+(?:газ|баллон|топлив|угл|дров|жидкост|спич|зажигал)",
        r"поставля(?:ется|ются)\s+без",
    )
)
EXTERNAL_PATTERN: Final = _compile(
    (
        r"приобрета(?:ется|ются)\s+отдельно",
        r"прода(?:ется|ются)\s+отдельно",
        r"совместим",
        r"подход(?:ит|ят)\s+(?:для|к)",
        r"работа(?:ет|ют)\s+от",
        r"под\s+(?:газов(?:ый|ые)|цангов(?:ый|ые))\s+баллон",
        r"для\s+(?:газов(?:ого|ых)|цангов(?:ого|ых))\s+баллон",
    )
)
INCLUDED_PATTERN: Final = _compile(
    (
        r"в\s+комплект(?:е|\s+входит)",
        r"комплекту(?:ется|ются)",
        r"включ(?:ает|ают|ен|ены)",
        r"набор\s+содержит",
        r"в\s+состав\s+входит",
        r"поставля(?:ется|ются)\s+с",
    )
)


def _first_match(source: str, text: str, pattern: re.Pattern[str], concept: str) -> Match | None:
    found = pattern.search(text)
    if found is None:
        return None
    return Match(source, found.start(), found.end(), text[found.start() : found.end()], concept)


def _substance_match(name: str, description: str, category: str) -> Match | None:
    concepts = ("bad_marker",) if category == "БАД" else (
        "flammable_liquid",
        "solid_fuel",
        "ignition_aid",
        "gas",
    )
    for source, text in (("name", name), ("description", description)):
        for concept in concepts:
            match = _first_match(source, text, SUBSTANCE_PATTERNS[concept], concept)
            if match is not None:
                return match
    return None


def _object_match(name: str, description: str) -> tuple[str, Match | None]:
    for sold_object, pattern in (
        ("device", DEVICE_PATTERN),
        ("accessory", ACCESSORY_PATTERN),
        ("kit", KIT_PATTERN),
        ("consumable", CONSUMABLE_PATTERN),
    ):
        match = _first_match("name", name, pattern, sold_object)
        if match is not None:
            return sold_object, match
    for sold_object, pattern in (("device", DEVICE_PATTERN), ("accessory", ACCESSORY_PATTERN)):
        match = _first_match("description", description, pattern, sold_object)
        if match is not None:
            return sold_object, match
    return "unknown", None


def _context(text: str, start: int, end: int, radius: int = 180) -> str:
    return text[max(0, start - radius) : min(len(text), end + radius)]


def extract_causal_target(*, category: str, name: str, description: str) -> CausalTarget:
    """Extract a conservative target without accepting labels, folds or predictions."""

    name = str(name or "")
    description = str(description or "")
    substance = _substance_match(name, description, category)
    sold_object, object_match = _object_match(name, description)
    reasons: list[str] = []

    if category == "БАД" and substance is not None:
        sold_object = "consumable"
    elif sold_object == "kit" and substance is not None and substance.source == "name":
        # A named set remains a kit even when it contains a regulated consumable.
        reasons.append("named_kit")
    elif substance is not None and substance.source == "name" and sold_object == "unknown":
        sold_object = "consumable"

    evidence = substance or object_match
    regulated = substance.concept if substance is not None else "none"
    relation = "unknown"

    if evidence is not None:
        surface = name if evidence.source == "name" else description
        context = _context(surface, evidence.start, evidence.end)
        if regulated != "none" and NEGATED_PATTERN.search(context):
            relation = "negated"
            reasons.append("explicit_negation")
        elif regulated != "none" and EXTERNAL_PATTERN.search(context):
            relation = "compatible_external"
            reasons.append("explicit_external_relation")
        elif regulated != "none" and INCLUDED_PATTERN.search(context):
            relation = "included"
            reasons.append("explicit_inclusion")
        elif regulated == "bad_marker":
            relation = "sold_object"
            reasons.append("explicit_bad_marker")
        elif sold_object == "consumable" and substance is not None and substance.source == "name":
            relation = "sold_object"
            reasons.append("regulated_consumable_named")
        elif sold_object in {"device", "accessory"} and regulated != "none":
            # A fuel word attached to equipment is not evidence that fuel is sold.
            relation = (
                "compatible_external"
                if substance is not None and substance.source == "name" and regulated == "gas"
                else "mentioned_only"
            )
            reasons.append("equipment_scope")
        elif sold_object in {"device", "accessory", "kit"} and regulated == "none":
            relation = "sold_object"
            reasons.append("non_substance_object_named")
        elif substance is not None:
            relation = "mentioned_only"
            reasons.append("substance_not_tied_to_sale")

    supported = bool(
        evidence is not None
        and sold_object != "unknown"
        and relation != "unknown"
        and evidence.span.strip()
    )
    if not supported:
        reasons.append("fail_closed")

    target = CausalTarget(
        sold_object=sold_object,
        regulated_substance=regulated,
        relation=relation,
        evidence_source=None if evidence is None else evidence.source,
        evidence_start=None if evidence is None else evidence.start,
        evidence_end=None if evidence is None else evidence.end,
        evidence_span=None if evidence is None else evidence.span,
        supported=supported,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )
    if target.sold_object not in SOLD_OBJECTS:
        raise AssertionError("unknown sold_object")
    if target.regulated_substance not in REGULATED_SUBSTANCES:
        raise AssertionError("unknown regulated_substance")
    if target.relation not in RELATIONS:
        raise AssertionError("unknown relation")
    if target.evidence_source is not None:
        source = name if target.evidence_source == "name" else description
        if source[target.evidence_start : target.evidence_end] != target.evidence_span:
            raise AssertionError("evidence offsets are not exact")
    return target
