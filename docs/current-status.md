# Текущий статус проекта

Дата среза: 2026-08-21.

## Подтверждено leaderboard

- Лучший Public Macro F1: **0.806579** — category-specific text/Qwen3-VL late fusion.
- Advanced mixed ensemble: 0.785500, что подтвердило риск duplicate/photo overfitting.
- Prompt-only Qwen branches существенно слабее supervised models.

## Сильнейшая validated architecture

Robust base + Qwen3-VL rsLoRA + Qwen3.5 rsLoRA:

- nested BAD F1: 0.951764;
- nested flammable F1: 0.871921;
- nested Macro F1: **0.911843**.

Rare-shingle BAD prior поверх production branch даёт donor-only +0.000853 BAD F1 и выигрывает 20/20 recurrence simulations. Это основной готовый submission candidate.

## Риски

- Flammable содержит только 198 positives.
- Product recurrence в hidden неизвестна.
- Старые leave-one-out recurrence metrics имеют transductive leakage и не используются как честный grouped score.
- Public result для LoRA candidates ещё отсутствует.

## Активная точка

Второй seed Qwen3.5 обучен на пяти folds и full data. Следующая операция — скачать OOF artifacts, агрегировать и принять/reject ensemble по nested Macro F1.
