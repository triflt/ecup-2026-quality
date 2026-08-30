# 714: Qwen3.5-4B только для объяснения

Статус: **финальный пакет собран; Public pending**.

Exp714 не меняет решение 140. TF-IDF, мультимодальный embedding-classifier,
Qwen3-VL LoRA и Qwen3.5 LoRA сначала независимо считают признаки; затем
category-specific fusion, thresholds и recurrence exact-id/name prior формируют
окончательный verdict. Reasoner не видит component scores и не участвует ни в
scores, ни в fusion, ни в priors.

## Обучение

Отдельный Qwen3.5-4B explanation-only adapter обучен как rsLoRA `r=16`,
`alpha=32`. На SFT он получает штатные поля карточки, image0 после того же
preprocessing с лимитом 448 px и train label как conditioning verdict. Target —
только русский комментарий, без label, JSON и submission-тегов. На inference
train label заменяется уже готовым frozen verdict solution140.

## Runtime

1. Solution140 фиксирует verdict: `0 → бан`, `1 → не бан`.
2. Reasoner получает карточку, image0 и только этот verdict, затем генерирует
   комментарий.
3. Runtime требует 50–300 символов, русскоязычный текст, законченное предложение
   и отсутствие явного противоречия verdict.
4. При malformed, incomplete или verdict mismatch используется статический
   fail-closed комментарий для нужной категории. Verdict не пересчитывается.
5. CSV содержит ровно `id,result`; формат без закрывающих тегов:
   `<комментарий>{текст}<вердикт>{бан|не бан}`.

Контракты реализованы в
[`runtime_contract.py`](runtime_contract.py),
[`submission_runtime.py`](submission_runtime.py) и
[`research/explanation_submission_contract.py`](../../research/explanation_submission_contract.py).

## Проверенный пример

Карточка БАД: «Полимедэл повязка на рану пленка электретная лечебная».
Solution140 verdict — `0`.

> Товар — электретная лечебная плёнка для наложения на рану. В описании указано,
> что она усиливает действие БАД, но сама не является биологически активной
> добавкой и не имеет соответствующей маркировки. Отрицательный электрический
> заряд не делает её БАД.

```text
<комментарий>Товар — электретная лечебная плёнка для наложения на рану. В описании указано, что она усиливает действие БАД, но сама не является биологически активной добавкой и не имеет соответствующей маркировки. Отрицательный электрический заряд не делает её БАД.<вердикт>бан
```

## Frozen 600-row E2E smoke

| Проверка | Результат |
|---|---:|
| Schema | 600/600 valid |
| Изменения verdict | 0/600 |
| Полное время | 341 с |
| Reasoner | 89.76 с |
| Прогноз Public / Private | 15.16 / 35.99 мин |

Пакет: `exp714-solution140-reasoner-4b-b192-final.zip`, 55 770 475 байт,
SHA-256 `a0695a55a85ca835d18f23e3700ee3eccc9ba03b9d490653719f474c38861bce`.
Пакет загружает пользователь. Public пока не измерен и не используется для
настройки решения.
