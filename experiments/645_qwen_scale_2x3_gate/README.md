# 645: честная сетка Qwen 2×3 и допуск к дистилляции

Статус: **подготовлен, ожидает артефакты 640–644**.

Пакет владеет единым кодовым контрактом для шести обученных/необученных ячеек:

| Размер | Prompting | Class-only LoRA | Class + evidence LoRA |
|---|---|---|---|
| 4B | 640/4B | 641 | 643 |
| 27B | 640/27B | 642 | 644 |

Evidence-модель у каждого размера одна. Она обучается на обоих порядках с
равными auxiliary-весами и оценивается двумя путями. Первичный путь заранее
зафиксирован как `evidence_first`; `class_first` — причинный контроль.

## Защита честности

- `semantic_family_v3`, сначала folds `0/3`, затем `0..4`;
- одинаковые поля входа и SHA модельного input view во всех ячейках;
- одинаковый первый снимок, prompt base, очистка и frozen zero threshold;
- validation runtimes физически не содержат labels/targets;
- sealed и Public не читаются;
- quote проверяется повторно как точная подстрока или точный OCR region;
- OCR polygon только восстанавливается из sidecar, а не генерируется;
- evidence→class и class→evidence считаются отдельно;
- prompting `640` сначала нормализуется и проверяется `adapt_prompt640.py`.

## Gates

Screen требует положительную разницу `644−643` на обоих folds, aggregate delta
не меньше `+0.003`, category drop не хуже `−0.002`, отношение
исправлений/ухудшений не ниже `1.5` и выигрыш на label-blind hard cohort.

Дистилляция разрешена только после полной оценки, если 27B evidence-first
выигрывает минимум `4/5` folds, проходит те же classification gates, выдаёт
не менее 50% доказательств, не менее 95% из выданных доказательств точно
привязываются и проходит заранее замороженный ручной аудит 200 строк:
релевантность ≥90%, неподтверждённые утверждения ≤1%, scope correctness ≥95%.

```bash
python experiments/645_qwen_scale_2x3_gate/adapt_prompt640.py --help
python experiments/645_qwen_scale_2x3_gate/freeze_audit.py --help
python experiments/645_qwen_scale_2x3_gate/evaluate.py --help
```

Пакет не содержит platform presets, приватные адреса, данные, веса или ZIP.
