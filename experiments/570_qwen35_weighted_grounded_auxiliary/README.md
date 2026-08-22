# 570: Qwen3.5 с ослабленным grounded auxiliary loss

Статус: **отклонён на двухfoldовом экране**. Следующие folds, подбор веса и
full train заблокированы.

Это причинный follow-up эксперимента `520`, который был отклонён на двухfoldовом
экране: среднее изменение Macro F1 составило `−0,034344`. Наиболее прямое
объяснение результата — длинный структурированный suffix получил значительно
больше суммарного градиента, чем первая цифра вердикта, потому что в `520` каждый
его токен имел вес `1.0`.

## Единственное изменение

Эксперимент наследует у `520` без изменений:

- Qwen3.5-4B и тот же adapter parent;
- точный selector, порядок и multiset из 5390 training occurrences;
- изображения и multimodal prompt;
- folds `0` и `3`, seed `42`, один epoch, batch `4`, accumulation `4`, optimizer,
  learning rate и число шагов;
- тот же шестистрочный assistant target, включая первый atomic verdict;
- тот же first-token inference и frozen route-`400` evaluator.

Меняется только causal CE. Первый supervised token `0/1` имеет вес `1.0`, каждый
последующий supervised token пяти evidence-полей — фиксированный вес `0.05`.
Loss нормируется на сумму активных весов. Значение `0.05` объявлено до обучения,
не подбирается и не является частью grid. Учитель и псевдоразметка отсутствуют.

## Fail-closed runtime audit

Каждый training batch проверяется до переноса на GPU:

- первый supervised token совпадает с atomic gold verdict;
- на него назначен вес `1.0`;
- все последующие supervised tokens имеют вес `0.05`;
- prompt, padding и прочие unsupervised tokens имеют вес `0`.

До первого backward выполняется exact null-control: та же реализация с единичными
весами обязана побитово совпасть с обычной mean causal CE на тех же logits и
labels. Перед validation создаётся `weighted_loss_audit.runtime.json`; evaluator
отклоняет отсутствующий или несовместимый audit. Отдельный grounded runtime audit
обязан сохранить hashes target plan и record multiset эксперимента `520`.

## Экран и решение

Проверяются только folds `0` и `3`. Кандидат заменяет только flammable Qwen3.5
rank в замороженном route-`400`; BAD route должен оставаться неизменным.

GO возможен одновременно при следующих условиях:

- Macro F1 delta строго положительна на обоих folds;
- средняя Macro F1 delta не ниже `+0.001`;
- число flammable false negatives не растёт ни на одном fold;
- число false negatives в safety-union не растёт ни на одном fold;
- runtime contracts, first-token format и exact evidence checks проходят.

Любое нарушение означает `REJECT`. Повтор seed, подбор веса, другие folds и full
train не разрешены этим пакетом.

## Подготовленные команды

Один fold можно запустить только с замороженным coverage audit соответствующего
fold из `520`:

```bash
uv run --extra vlm python experiments/570_qwen35_weighted_grounded_auxiliary/local_run.py train \
  --data /path/to/data.csv \
  --oof /path/to/frozen_oof.npz \
  --audit-json /path/to/coverage_fold_0.json \
  --image-manifest /path/to/image_manifest.tsv.gz \
  --images /path/to/images \
  --model-root /path/to/model \
  --vendor /path/to/vendor \
  --output-dir /path/to/fold0 \
  --fold 0
```

После двух folds:

```bash
uv run python experiments/570_qwen35_weighted_grounded_auxiliary/local_run.py screen \
  --fold-0 /path/to/fold0/lora_holdout_predictions.csv \
  --fold-3 /path/to/fold3/lora_holdout_predictions.csv \
  --output-dir /path/to/screen
```

Пакет прошёл тесты, lint двух пресетов, implementation null-control и runtime
contracts на обоих folds.

## Результат

| Fold | Δ Macro F1 | Δ flammable F1 | Исправлено | Ухудшено | Δ flammable FN |
|---:|---:|---:|---:|---:|---:|
| 0 | −0,019290 | −0,038580 | 5 | 8 | +2 |
| 3 | +0,022170 | +0,044340 | 3 | 0 | −3 |

Средний прирост равен `+0,001440`, однако обязательная устойчивость не пройдена:
на fold 0 метрика падает и растёт число false negatives целевого класса. Общий
баланс — 8 исправлений и 8 ухудшений. Это поддерживает гипотезу, что слабый
evidence-сигнал иногда полезен, но эффект зависит от состава семейств и пока не
годится для принятия. Рецепт отклонён без подбора веса по увиденным folds.
