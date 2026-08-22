# 560: Qwen3.5 flammable hard-pair auxiliary margin

Статус: **CPU-аудиты и null-control готовы; GPU/model inference не запускались**.

## Единственный фактор

Это exact flammable Qwen3.5 parent из route `400` (`260`): те же 5 390
training occurrences, prompt, first image, pointwise assistant CE, 1 epoch,
batch/accumulation/optimizer steps и rsLoRA. Меняется только loss на заранее
собранных batch-local donor pairs:

```text
loss = pointwise_CE + 0.10 * max(0, 1.0 - score(positive) + score(negative))
score = first-token logit("1") - logit("0")
```

Новых строк и forward passes нет. Inference остаётся single-pass first-token
scoring.

## Замороженная topology и leakage contract

Переиспользуется immutable label-blind 909-row selector и topology exp530.
Labels читаются только после заморозки topology, чтобы ориентировать
outer-train donor pair как positive/negative. Из неё берутся лишь edges, оба
конца которых реально присутствуют в exact parent train multiset данного fold.

Pair-aware order расходует существующие occurrences без изменения
multiplicity. Максимум четыре pairs на positive, negative reuse не выше восьми
(фактически один из-за multiplicity родителя). `safe_for_selection=false` и
outer-validation rows запрещены. Runtime до загрузки модели требует точного
равенства полного parent selector и dependency-light selector, затем строго
сверяет multiset, manifest, targets и donor labels.

Ordered-record SHA — только cross-runtime диагностика и никогда не membership
gate. Это не ослабляет строгую проверку runtime multiset или точное
parent-vs-light равенство внутри одного runtime.

Реальный CPU audit:

| Fold | Parent edges | Realized pairs/batches | Same cue | Global fallback | Unsafe / outer |
|---:|---:|---:|---:|---:|---:|
| 0 | 264 | 88 / 88 | 82 | 6 | 0 / 0 |
| 3 | 260 | 88 / 88 | 80 | 8 | 0 / 0 |

На обоих folds ровно 5 390 occurrences, record multiset и steps неизменны,
positive cap равен 4, фактический negative reuse равен 1, решение audit — GO.

## Локальный интерфейс

Dependency-light audit:

```bash
uv run python experiments/560_qwen35_flammable_hard_pairwise/local_run.py audit \
  --data /path/to/data.csv \
  --oof /path/to/frozen_oof.npz \
  --output-dir /path/to/fold0-audit \
  --fold 0
```

Ограниченный launcher допускает только seed 42 и folds 0/3. Эта команда не
запускалась:

```bash
uv run --extra vlm python experiments/560_qwen35_flammable_hard_pairwise/local_run.py train \
  --data /path/to/data.csv \
  --oof /path/to/frozen_oof.npz \
  --audit-json /path/to/fold0-audit/pair_audit.json \
  --image-manifest /path/to/image_manifest.tsv.gz \
  --images /path/to/images \
  --model-root /path/to/model \
  --vendor /path/to/vendor \
  --output-dir /path/to/fold0-output \
  --fold 0
```

## Evaluator и null-control

`evaluate_screen.py` восстанавливает locked route400 и заменяет только
flammable Qwen3.5 rank при заранее фиксированном весе `0.75`. BAD rank и его
route остаются неизменны; weight grid отсутствует.

```bash
uv run python experiments/560_qwen35_flammable_hard_pairwise/evaluate_screen.py \
  --null-control \
  --output-dir /path/to/null-control
```

Реальный CPU null-control прошёл: route400 восстановлен byte-identically и на
folds 0/3 изменено ровно ноль predictions. Candidate screen требует оба folds,
положительный Macro-F1 и flammable delta на каждом, средний Macro-F1 delta не
ниже `+0.001`, отсутствие BAD/safety/connected-safe регрессии и corrected >
regressed. Успешный screen сам по себе не разрешает полный five-fold цикл.
