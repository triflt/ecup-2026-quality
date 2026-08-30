# 520: grounded auxiliary SFT для Qwen3.5

Статус: **двухfoldовый pilot завершён и отклонён**.

Обе задачи и архивы прошли техническую приёмку, но замороженный route-`400`
экран дал Macro delta `−0,051385` на fold 0 и `−0,017304` на fold 3; среднее
`−0,034344`. Исправлено 5 решений, ухудшено 15. На fold 0 flammable false
negatives выросли на 7, а safety-union false negatives — на 3. Полный цикл,
повтор seed и включение structured suffix в решение запрещены.

## Единственный фактор

Эксперимент наследует фиксированный Qwen3.5 adapter parent решения `400`: тот же
hard selector, порядок и multiplicity records, Qwen3.5-4B, rsLoRA, optimizer,
learning rate, один epoch, batch/accumulation, изображения, folds и first-token
validation. Учитель и отдельная модель объяснений не используются.

Меняется только контракт assistant suffix. Вместо одной цифры target состоит из
шести строк:

```text
1
CONCEPT=BAD_EXPLICIT_MARKING
SOURCE=description
START=24
END=34
EVIDENCE="биодобавки"
```

Первая строка всегда gold training verdict `0/1`. Остальные строки строит
детерминированный extractor опыта `490`: закрытый policy concept, источник,
offsets в нормализованном surface text и точная surface-подстрока. Если
extractor не может безопасно обосновать gold verdict, фиксированный target имеет
`CONCEPT=NO_SAFE_EVIDENCE`, `SOURCE=none`, offsets `-1/-1` и пустую evidence.

Loss — прежний unit-weight causal CE по всему assistant suffix. Нет отдельной
головы, коэффициента, curriculum, grid или дополнительных steps. Runtime
проверяет, что первым supervised token каждого target остаётся тот же atomic
token цифры, который использует parent validation.

## Inference и объяснение

Prompt просит сначала одну цифру, затем пять закрытых полей. Verdict извлекается
из первого atomic `0/1` token одного generated continuation. Второго прохода,
teacher call, retrieval или verdict override нет. `evaluate_outputs.py` отдельно
проверяет формат, закрытость concept, category и точное восстановление evidence
по offsets; только после этих проверок он рендерит краткое объяснение.

Существующий parent holdout evaluator по-прежнему считает score как разность
logits atomic `1` и `0` в первой позиции. Поэтому screen Macro F1 сопоставим с
parent, а качество полного structured continuation оценивается отдельным hook.

## Обязательный coverage audit

Audit пересчитывает targets из исходного CSV и gold verdict только для records
фиксированного outer-train selector. Он сообщает unique и multiplicity-weighted
SAFE coverage для каждой пары category/label. Локальный JSONL опыта `490` можно
передать только как cross-check выбранных outer-train строк; gate не читает
holdout explanations.

```bash
uv run python experiments/520_qwen35_grounded_auxiliary_sft/local_run.py audit \
  --data /path/to/data.csv \
  --oof /path/to/frozen_oof.npz \
  --cross-check-jsonl /path/to/evidence_gold.jsonl \
  --output /path/to/coverage_fold0.json \
  --fold 0
```

GO требует SAFE majority среди training occurrences, не менее 25% и не менее 25
SAFE occurrences в каждом category/label cohort, точных offsets, полного
совпадения optional cross-check и нуля outer-validation occurrences. Если gate
не пройден, launcher отвергает train.

Реальный audit фиксированного selector прошёл на обоих folds:

| Fold | Cohort | SAFE / occurrences | SAFE rate |
|---:|---|---:|---:|
| 0 | БАД, label 0 | 777 / 1500 | 51.80% |
| 0 | БАД, label 1 | 1240 / 1500 | 82.67% |
| 0 | Легковоспламеняющиеся, label 0 | 511 / 1600 | 31.94% |
| 0 | Легковоспламеняющиеся, label 1 | 570 / 790 | 72.15% |
| 3 | БАД, label 0 | 780 / 1500 | 52.00% |
| 3 | БАД, label 1 | 1257 / 1500 | 83.80% |
| 3 | Легковоспламеняющиеся, label 0 | 473 / 1600 | 29.56% |
| 3 | Легковоспламеняющиеся, label 1 | 520 / 790 | 65.82% |

Итого: fold 0 — `3098/5390 = 57.48%` SAFE occurrences; fold 3 —
`3030/5390 = 56.22%`. Fallback не является большинством ни в одном полном
training plan. На каждом fold проверено 4758 selected unique rows против
готового `490` sidecar: ноль расхождений, ноль format/offset failures и ноль
outer-validation occurrences. Решение до обучения: **GO для ограниченного
двухfoldового pilot**, без разрешения full train.

Независимый полный sidecar `490` даёт ожидаемый sanity-check `7697/12971 =
59.34%` SAFE: БАД label 0 — `982/1905`, БАД label 1 — `4699/5564`, flammable
label 0 — `1878/5304`, flammable label 1 — `138/198`. Эти full-data числа не
участвуют в launch gate; решения `GO` выше вычислены только на outer-train.

## Два выполненных pilot folds

Интерфейс существует только для folds `0` и `3`; обучение было выполнено только
для них:

```bash
uv run --extra vlm python experiments/520_qwen35_grounded_auxiliary_sft/local_run.py train \
  --data /path/to/data.csv \
  --oof /path/to/frozen_oof.npz \
  --audit-json /path/to/coverage_fold0.json \
  --image-manifest /path/to/image_manifest.tsv.gz \
  --images /path/to/images \
  --model-root /path/to/model \
  --vendor /path/to/vendor \
  --output-dir /path/to/fold0 \
  --fold 0
```

Fold 3 использует тот же интерфейс и собственный audit. Полное обучение, другие
folds и подбор auxiliary weight запрещены.

Готовые generated continuations проверяются без model call:

```bash
uv run python experiments/520_qwen35_grounded_auxiliary_sft/local_run.py evaluate \
  --input /path/to/generated.csv \
  --output /path/to/structured_report.json \
  --rendered-output /path/to/rendered_comments.csv
```

Pilot отклоняется при неположительном Macro delta на любом fold, среднем delta
ниже `+0.001`, росте flammable/safety false negatives, first-token format ниже
100% или любой неexact evidence у well-formed outputs.
