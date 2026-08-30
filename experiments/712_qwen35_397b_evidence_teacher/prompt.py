from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = "explanation_synth_v5"
MODEL_ID = "Qwen/Qwen3.5-397B-A17B-FP8"

RULES = {
    "БАД": (
        "Метка 1 только если в карточке или на упаковке есть прямое указание "
        "«БАД», «биологически активная добавка» или «dietary supplement». "
        "Спортивное питание, витамины и полезный продукт без такой маркировки, "
        "а также явное отрицание маркировки — метка 0."
    ),
    "Легковоспламеняющиеся": (
        "Метка 1 для самостоятельного источника огня, горючего вещества или газа, "
        "а также когда такой предмет явно входит в продаваемый комплект. Пустое "
        "оборудование, компонент без продаваемого топлива, декоративное изображение, "
        "игрушка или упоминание не входящего в комплект предмета — метка 0."
    ),
}

REASONS = {
    "bad_marking_present",
    "bad_marking_missing",
    "bad_marking_negated",
    "standalone_flammable_item",
    "included_flammable_item",
    "empty_equipment",
    "component_or_mention_only",
    "not_enough_evidence",
}
ALLOWED_REASONS = {
    ("БАД", 1): {"bad_marking_present", "not_enough_evidence"},
    ("БАД", 0): {"bad_marking_missing", "bad_marking_negated", "not_enough_evidence"},
    ("Легковоспламеняющиеся", 1): {
        "standalone_flammable_item", "included_flammable_item", "not_enough_evidence",
    },
    ("Легковоспламеняющиеся", 0): {
        "empty_equipment", "component_or_mention_only", "not_enough_evidence",
    },
}
BAD_MARKER_PATTERNS = (
    re.compile(r"(?<!\w)бад(?!\w)", re.IGNORECASE),
    re.compile(r"биологически\s+активн\w*\s+добавк\w*", re.IGNORECASE),
    re.compile(r"dietary\s+supplement", re.IGNORECASE),
)

SYSTEM_PROMPT = """Ты создаёшь короткое проверяемое объяснение для уже заданной gold-метки товарной карточки.

Правила:
1. Значение метки: label=1 — заявленная категория подтверждена, итог «не бан»; label=0 — категория не подтверждена, итог «бан». Всегда верни входную цифру в поле label.
2. Используй только название, описание и приложенные изображения текущей карточки. Не используй веб и сведения о товаре, которых нет во входе.
3. Сначала проверь, не противоречит ли карточка заданной метке. Gold нельзя оспаривать или менять, но нельзя и оправдывать выдумкой. При явном конфликте используй not_enough_evidence. Запрещено объяснять, почему буквальный допустимый маркер якобы «не является маркировкой»: ненегированный маркер всегда считается маркировкой.
4. Найди одно самое сильное основание. Не перечисляй внутреннее рассуждение и не пиши chain-of-thought.
5. evidence — объект {"source": ..., "value": ...}. Для title/description СКОПИРУЙ value посимвольно как одну непрерывную подстроку: не меняй регистр, окончания, пробелы, «е/ё» и пунктуацию. Предпочитай короткую цитату 5–160 символов. Для изображения source имеет вид image:N, а value описывает только чётко видимый объект или дословно читаемый текст.
6. Для БАД label=1 принимаются только буквальные «БАД», «биологически активная добавка» или «dietary supplement». Фразы вроде «vitamin supplement», витамины, капсулы и полезные свойства сами по себе недостаточны. Для label=0 нельзя заявлять отсутствие маркировки, если любой допустимый маркер присутствует без явного отрицания. В частности, предупреждение «Биологически активная добавка не может использоваться как замена питания» содержит прямую маркировку, а не отрицание категории. При конфликте с gold верни not_enough_evidence.
7. Для отсутствующей маркировки БАД используй source="absence" и value="Обязательная маркировка БАД не найдена в названии и описании.". Не используй absence, пока не проверены полностью название и описание.
8. explanation — конкретный русский комментарий именно об этом товаре длиной 50–300 символов. Назови продаваемый объект и решающий признак/отношение. Комментарий должен читаться как самостоятельная проверка карточки, а не отчёт классификатора. Никогда не пиши «соответствует метке», «подтверждает метку», «метка 0/1», «заявленная категория», «gold», «класс», «reason» или «правило категории». Заверши объяснение фактом о товаре, а не ссылкой на решение модели.
9. Если заданную метку невозможно честно обосновать, используй reason="not_enough_evidence", evidence=null и explanation=null. Это правильнее, чем правдоподобная выдумка.
10. Верни только один JSON-объект ровно с четырьмя полями: label, reason, evidence, explanation. Markdown и дополнительный текст запрещены.
"""


def user_prompt(
    *, category: str, label: int, name: str, description: str, image_count: int
) -> str:
    if (category, label) not in ALLOWED_REASONS:
        raise ValueError(f"unsupported category/label: {(category, label)!r}")
    reasons = ", ".join(sorted(ALLOWED_REASONS[(category, label)]))
    return f"""Категория: {category}
Gold-метка: {label}
Название:
<<<{name}>>>
Описание:
<<<{description}>>>
Приложено изображений: {image_count}; индексы 0..{image_count - 1}.
Правило категории: {RULES[category]}
Допустимые reason для этой категории и метки: {reasons}.

Верни только JSON:
{{
  "label": {label},
  "reason": "один допустимый reason",
  "evidence": {{"source": "title|description|image:N|absence", "value": "одно доказательство"}},
  "explanation": "конкретный комментарий 50–300 символов"
}}
Для not_enough_evidence последние два поля должны быть null."""


@dataclass(frozen=True)
class ValidationResult:
    parsed: dict[str, Any]
    errors: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return not self.errors


def _bad_markers_absent(name: str, description: str) -> bool:
    text = f"{name}\n{description}"
    return not any(pattern.search(text) for pattern in BAD_MARKER_PATTERNS)


def _contains_bad_marker(value: str) -> bool:
    return any(pattern.search(value) for pattern in BAD_MARKER_PATTERNS)


def _has_explicit_bad_negation(value: str) -> bool:
    marker = r"(?:бад|биологически\s+активн\w*\s+добавк\w*|dietary\s+supplement)"
    patterns = (
        rf"(?<!\w)не(?!\w)\s+{marker}",
        rf"(?<!\w)не(?!\w)\s+(?:является|относится|считается).{{0,100}}?{marker}",
        rf"{marker}.{{0,50}}?(?<!\w)не(?!\w)\s+(?:является|относится|считается)",
        rf"(?<!\w)без(?!\w)\s+(?:маркировк\w*|указан\w*|обозначен\w*).{{0,80}}?{marker}",
    )
    return any(re.search(pattern, value, re.IGNORECASE | re.DOTALL) for pattern in patterns)


def validate_output(
    payload: Any,
    *,
    category: str,
    expected_label: int,
    name: str,
    description: str,
    image_count: int,
) -> ValidationResult:
    if not isinstance(payload, dict):
        return ValidationResult({}, ("top-level output must be an object",))
    errors: list[str] = []
    expected_fields = {"label", "reason", "evidence", "explanation"}
    if set(payload) != expected_fields:
        errors.append(f"fields must be exactly {sorted(expected_fields)}")
    label = payload.get("label")
    if label != expected_label or isinstance(label, bool):
        errors.append("label does not echo the supplied gold label")
    reason = payload.get("reason")
    if reason not in ALLOWED_REASONS.get((category, expected_label), set()):
        errors.append("reason is incompatible with category/label")
    evidence = payload.get("evidence")
    explanation = payload.get("explanation")
    if reason == "not_enough_evidence":
        if evidence is not None or explanation is not None:
            errors.append("not_enough_evidence requires null evidence and explanation")
        return ValidationResult(dict(payload), tuple(errors))
    if not isinstance(explanation, str) or not 50 <= len(explanation.strip()) <= 300:
        errors.append("explanation must contain 50-300 characters")
    elif re.search(r"\b(?:метк[аеуи]|reason|класс[ауе]?|правил[оу]\s+категории)\b", explanation, re.IGNORECASE):
        errors.append("explanation contains forbidden classifier-meta language")
    if not isinstance(evidence, dict) or set(evidence) != {"source", "value"}:
        errors.append("evidence must contain exactly source and value")
        return ValidationResult(dict(payload), tuple(errors))
    source = evidence.get("source")
    value = evidence.get("value")
    if not isinstance(value, str) or not value.strip() or len(value) > 500:
        errors.append("evidence.value must be a non-empty string up to 500 characters")
    elif source == "title":
        if value not in name:
            errors.append("title evidence is not an exact substring")
    elif source == "description":
        if value not in description:
            errors.append("description evidence is not an exact substring")
    elif source == "absence":
        if category != "БАД" or reason != "bad_marking_missing":
            errors.append("absence evidence is allowed only for missing BAD marking")
        if not _bad_markers_absent(name, description):
            errors.append("BAD marker is present despite absence evidence")
        if value != "Обязательная маркировка БАД не найдена в названии и описании.":
            errors.append("absence evidence must use the frozen canonical value")
    elif isinstance(source, str) and re.fullmatch(r"image:\d+", source):
        image_index = int(source.split(":", 1)[1])
        if not 0 <= image_index < image_count:
            errors.append("image evidence index is out of range")
    else:
        errors.append("unsupported evidence source")
    if category == "БАД" and reason == "bad_marking_missing" and not _bad_markers_absent(name, description):
        errors.append("bad_marking_missing conflicts with a marker in title/description")
    if category == "БАД" and reason == "bad_marking_present":
        if isinstance(value, str) and not _contains_bad_marker(value):
            errors.append("bad_marking_present evidence lacks an allowed literal marker")
    if category == "БАД" and reason == "bad_marking_negated":
        if isinstance(value, str) and not _contains_bad_marker(value):
            errors.append("bad_marking_negated evidence lacks an allowed literal marker")
        if isinstance(value, str) and not _has_explicit_bad_negation(value):
            errors.append("bad_marking_negated evidence does not explicitly negate BAD status")
    return ValidationResult(dict(payload), tuple(errors))


def parse_json_response(raw: str) -> dict[str, Any]:
    value = str(raw).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        value = fenced.group(1)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError("teacher response must contain one JSON object")
    return parsed
