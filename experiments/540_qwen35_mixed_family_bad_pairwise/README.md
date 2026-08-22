# 540: Qwen3.5 BAD mixed-family pairwise margin

Статус: **двухfoldовый экран завершён и отклонён**.

Артефакты прошли ZIP-проверку и контракт frozen evaluator. Macro delta составил
`−0,001307` на fold 0 и `−0,000176` на fold 3; среднее `−0,000742`.
Исправлено 9 решений, ухудшено 16, flammable-маршрут остался неизменным.
Pairwise margin не улучшил BAD ни на одном fold, поэтому полный цикл и повтор
seed запрещены.

## Гипотеза и единственный фактор

Главный локальный остаток route `400` сосредоточен в BAD mixed-family: около
40.7% ошибок против примерно 2.9% у label-consistent families. Это описательное
наблюдение не используется для выбора отдельных строк.

Эксперимент сохраняет exact parent `400/260`: Qwen3.5-4B, те же изображения и
prompt, hard records и их multiplicity, один epoch, число batches/optimizer
steps, rsLoRA и first-token inference. Единственное изменение — auxiliary hinge:

```text
0.10 * max(0, 1.0 - score(BAD-positive donor) + score(BAD-negative donor))
score = logit("1") - logit("0") в первой assistant-позиции
```

Pair batches лишь меняют порядок неизменного record multiset; новых строк и
forward passes нет. Pointwise assistant-suffix CE остаётся прежним.

## Честный family selector

Topology строится до чтения labels только для BAD outer-train:

1. immutable `connected_family_guard_v2` задаёт exact-text/image components;
2. все `safe_for_selection=false` components исключаются;
3. для safe component фиксируются до восьми ближайших safe components по
   normalized-name token Jaccard (`>=0.45`, минимум два общих токена);
4. только после заморозки topology donor labels ориентируют positive/negative;
5. сначала используются пары внутри component, затем nearest-family;
6. caps: максимум четыре negatives на positive и reuse negative не выше четырёх.

Semantic-family draft намеренно запрещён: его собственный контракт помечает
partitions невалидными для candidate scoring, а ручной аудит имеет `NO_GO`.

## Frozen manifests и leakage gate

```bash
uv run python experiments/540_qwen35_mixed_family_bad_pairwise/local_run.py audit \
  --data /path/to/data.csv \
  --oof /path/to/frozen_oof.npz \
  --guard /path/to/connected_guard_rows.csv \
  --output-dir /path/to/fold0-manifest \
  --fold 0
```

Реальный CPU audit прошёл:

| Fold | Pair batches | Realized pairs | Same family | Nearest family |
|---:|---:|---:|---:|---:|
| 0 | 73 | 146 | 18 | 128 |
| 3 | 80 | 160 | 23 | 137 |

На обоих folds record multiset и steps неизменны, outer-validation и unsafe
donors равны нулю, caps соблюдены. Runtime заново строит manifest и сравнивает
record-order и manifest hashes до загрузки модели. Guard принимается только при
точном SHA-256 immutable `connected_family_guard_v2`; подмена похожим CSV
прекращает audit или train.

## Ограниченный launcher

Разрешены только seed 42 и folds `0/3`; команда ниже не запускалась:

```bash
uv run --extra vlm python experiments/540_qwen35_mixed_family_bad_pairwise/local_run.py train \
  --data /path/to/data.csv \
  --oof /path/to/frozen_oof.npz \
  --guard /path/to/connected_guard_rows.csv \
  --audit-json /path/to/fold0-manifest/pair_audit.json \
  --image-manifest /path/to/image_manifest.tsv.gz \
  --images /path/to/images \
  --model-root /path/to/model \
  --vendor /path/to/vendor \
  --output-dir /path/to/fold0 \
  --fold 0
```

## Null evaluator и screen

`evaluate_screen.py` восстанавливает byte-identical route `400`. Новый rank
заменяет только BAD Qwen3.5 rank с заранее фиксированным прежним весом `0.25`;
robust base/Qwen3-VL остаются `0.50/0.25`, а flammable route неизменен. Weight
grid отсутствует.

```bash
uv run python experiments/540_qwen35_mixed_family_bad_pairwise/evaluate_screen.py \
  --null-control --output-dir /path/to/null-control
```

Реальный CPU null-control пройден: на folds 0 и 3 получено ровно ноль changed
predictions и нулевой Macro delta; веса BAD восстановлены как
`0.50 / 0.25 / 0.25`.

Candidate screen требует оба folds и отклоняется при неположительном Macro/BAD
delta на любом fold, среднем Macro delta ниже `+0.001`, connected-safe регрессии,
числе corrected не больше regressed или любом изменении flammable. Даже успешный
screen не разрешает full train: для этого branch он явно запрещён.
