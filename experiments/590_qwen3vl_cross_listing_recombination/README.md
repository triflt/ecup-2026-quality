# 590: audited cross-listing recombination for Qwen3-VL

Статус: **NO_GO: blind audit завершён, обучение и model inference не запускались**.

## Единственный фактор

Родитель — exact Qwen3-VL experiment 110 в route400: те же 5 390 records,
hard selector, текст, prompt, labels, LoRA, optimizer, steps и ensemble weights.
Только у ровно 25% повторных eligible occurrences first-image заменяется первым
изображением другой listing того же audited semantic component. Новых строк нет.
Validation и production inference всегда используют исходное первое изображение
448 px, один проход.

## Frozen provenance и запреты

Component assignment читается только из sealed
`validation/semantic_family_v3`; ignored draft v3 используется только как
immutable edge provenance с обязательной SHA-проверкой. Допустимы direct edges:

- `exact_full_text`;
- `exact_first_image`;
- `exact_name_corroborated`.

Оба конца обязаны иметь одинаковые category и training label и находиться в
outer-train. Запрещены sealed holdout, mixed-label components, degree выше 32,
generic/failed guards, masked-name, auxiliary-image и perceptual edges.

CPU preflight получил:

| Fold | Eligible pairs | Components | Repeated occurrences | Recombined |
|---:|---:|---:|---:|---:|
| 0 | 4 773 | 986 | 140 | 35 |
| 3 | 4 538 | 980 | 160 | 40 |

Record multiset, order и steps не меняются. Pair/component coverage gate прошёл,
forbidden/outer/sealed/mixed-label counters равны нулю.

## Почему запуск заблокирован

В `analysis/preflight` заморожена детерминированная label-blind выборка из 300
пар. Два независимых строгих прохода завершены без просмотра gold labels и
downstream scores: оба reviewer приняли 292/300 пар. Cohen's kappa = `1.0`,
raw agreement = `1.0`, но strict same-product precision = `0.9733`, что ниже
требуемых `>=98%`; поэтому решение **NO_GO**. Восемь отклонений — пять случаев
с отличающейся комплектацией/количеством и три случая с другим или неоднозначным
товаром. Подробный протокол и ID пар сохранены в
`analysis/preflight/blind_audit_result.json`, а заполненный шаблон — в
`analysis/preflight/blind_reviews_completed.csv`.

Оба fold audits остаются `NO_GO` с причиной `blind_audit_not_approved`, и train
entrypoint по-прежнему останавливается до импорта модели. Это честный отказ
гипотезы на CPU gate: нельзя безопасно переносить изображения между всеми
выбранными связями без дополнительного, более строгого фильтра.

Повторный CPU audit после независимой разметки:

```bash
uv run python experiments/590_qwen3vl_cross_listing_recombination/local_run.py audit \
  --data /path/to/data.csv \
  --oof /path/to/frozen_oof.npz \
  --sealed-dir validation/semantic_family_v3 \
  --draft-dir validation/.local/semantic_family_graph_audit_draft_v3 \
  --reviews /path/to/completed_blind_reviews.csv \
  --output-dir /path/to/new-frozen-preflight
```

## Экран и null-control

Evaluator заменяет только Qwen3-VL rank при существующих route400 weights.
Exact null-control уже прошёл: ноль changed predictions.

Candidate должен выиграть на обоих folds, иметь mean Macro delta `>=+0.001`, не
увеличивать flammable/safety false negatives, исправлять больше ошибок, чем
добавляет, и не снижать BAD F1 более чем на `0.003`. Screen reject-only и не
разрешает full cycle автоматически.

Private однокарточные presets имеют нейтральные имена и лежат в
`.local/runtime`; они не запускались.
