# Независимое сравнение следующих гипотез после `300` и во время `310`

Дата: 2026-08-22.

Проверяемые ветки:

- **A:** семейно-контрастный прототипный score поверх уже вычисляемых
  `Qwen3-VL-Embedding-2B` признаков;
- **B:** отдельный регуляторный BAD-head для спортивного питания, обычной еды,
  лекарственной и ветеринарной лексики.

Никакое обучение и никакая private compute platform job в рамках этого обзора не запускались.
Общие журналы и карточки экспериментов не изменялись.

## Использованные доказательства

Локальные источники:

- `experiments/290_minicpm_v46_visual_screen/analysis/agent_hard_errors/report.json`;
- `hard_error_cases.csv`, `cohort_metrics.csv`, `component_summary.csv` из того же
  каталога;
- результаты `050`, `080`, `210`, `260`, `280`, `300`;
- карточка выполняющегося `310_qwen35_crossfit_soft_targets`;
- `reports/experiment-log.csv`, `reports/hypothesis-board.csv`,
  `reports/research-lessons.jsonl`, `components/registry.json`;
- сохранённые OOF-признаки и отчёт ранней проверки embedding-head.

Первичные внешние источники:

- [Supervised Contrastive Learning for Product Matching](https://arxiv.org/abs/2202.02098)
  применяет supervised contrastive обучение к товарным предложениям и подчёркивает
  необходимость явного supervision;
- [Block-SCL](https://arxiv.org/abs/2207.02008) показывает, что близкие по исходным
  признакам товары с противоположным match-label являются полезными hard negatives;
  в абляции hard-negative batching дал `+2.8` F1 относительно случайных negatives,
  а авторы сообщают среднее и стандартное отклонение трёх seed;
- [Supervised Contrastive Learning, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)
  даёт общий механизм притяжения примеров одного класса и отталкивания разных;
- [победившее решение Amazon KDD Cup 2022](https://amazonkddcup.github.io/papers/3782.pdf)
  использует дедупликацию, контекстные товарные признаки и cross-validated soft
  labels; это поддерживает и выполняющийся `310`, и идею отдельного контекстного
  специалиста;
- [первое место Rakuten SIGIR eCom 2020](https://sigir-ecom.github.io/ecom20DCPapers/SIGIR_eCom20_DC_paper_4.pdf)
  показывает преимущество decision-level late fusion независимых товарных
  классификаторов над feature-level fusion, но его random split нельзя переносить
  как наш validation protocol;
- [второе место Amazon KDD Cup 2022 Task 2/3](https://amazonkddcup.github.io/posters/8572.pdf)
  использует отдельные cross-encoders, вероятности и небольшой второй уровень, но
  также показывает опасность leaderboard-driven отбора: групповые признаки дали
  заметное падение после публичного роста.

## Состояние сильнейшего решения и остаточные ошибки

После donor-only воспроизведения production priors у `190`:

- BAD: `TP=5315`, `FP=227`, `FN=249`, F1 `0.957140`;
- flammable: `TP=182`, `FP=28`, `FN=16`, F1 `0.892157`;
- общих с заменой `260` ошибок BAD осталось 452, flammable — 27.

### Регуляторные BAD-когорты

| эксклюзивный тип | rows | positives | F1 `190` | общих ошибок | FP | FN | все 9 компонентов ошиблись |
|---|---:|---:|---:|---:|---:|---:|---:|
| sports nutrition | 2246 | 1411 | 0.908565 | 243 | 91 | 152 | 68 |
| medicine language | 666 | 451 | 0.977974 | 19 | 12 | 7 | 8 |
| food/beverage | 137 | 82 | 0.963855 | 5 | 3 | 2 | 3 |
| pet/veterinary | 25 | 9 | 0.875000 | 2 | 0 | 2 | 0 |

`sports_nutrition` содержит **53.8%** всех 452 общих BAD-ошибок. Ошибки не
сосредоточены в одном fold: их количества равны `62/47/46/46/42`. Из 243 ошибок
145 находятся в singleton-семействах, поэтому это не только задача запоминания
повторов. Хотя бы один из девяти компонентов правильный на 175 из 243 строк;
all-images embedding спасает 77, first-image embedding — 79, Gemma — 85. Это
диагностический oracle, а не готовая маршрутизация.

Маркировка неоднозначна, но взаимодействие наблюдаемо:

- sport без явного `БАД/dietary supplement`: 524 строки, 198 positive (`37.8%`);
- sport с явным `БАД/dietary supplement`: 1731 строка, 1221 positive (`70.5%`).

Следовательно, жёсткое правило «sport = negative» неверно для фактической разметки.
Нужен мягкий специалист по взаимодействию свидетельств, а не regex override.

### Сигнал и ограничения embedding-ветки

Сигнал существует: на 452 общих BAD-ошибках all-images embedding даёт правильный
ответ в 137 случаях, first-image embedding — в 145. Однако ранее:

- обычный class-centroid score дал BAD F1 `0.890552` и flammable F1 `0.259319`;
- нелинейный ExtraTrees/четырёхкомпонентный вариант улучшал обычную grouped CV, но
  терял вес на image-duplicate stress, а эксперимент `080` был отклонён;
- donor-only character-TFIDF и rare-token neighbour priors в `210` не превысили
  `190`;
- в consistent repeated families сильный дисбаланс: для BAD `757` positive против
  `116` negative families; для flammable `23` против `851`. Без family caps и hard
  negatives contrastive objective в основном выучит частоту класса и дубликаты.

Это не опровергает обучаемую family-contrastive проекцию, но делает наивный nearest
prototype уже закрытой веткой.

## Обнаруженная проблема frozen folds

Проверка invariant `group_hash -> ровно один fold` обнаружила две exact full-text
семьи, пересекающие folds в `validation/grouped_text_v1/folds.csv`:

- BAD hash `f89080d1...`: 7 строк в folds `0/2/3/4`;
- flammable hash `3e6f8658...`: 2 строки в folds `2/3`.

Всего затронуто 9 из 12 971 строк, поэтому это не объясняет крупные результаты
предыдущих опытов. Но для prototype/nearest-family метода даже такая ошибка создаёт
прямой donor→outer путь. Исторический протокол нельзя переписывать. Перед любой
оценкой A или B нужна новая immutable topology, объединяющая connected components
полного текста и image duplicates, либо эти две компоненты целиком исключаются из
selection score. Нужен автоматический нулевой invariant check.

## Гипотеза A: family-contrastive prototype score

### Причинный механизм

Текущий LinearSVC ищет одну глобальную границу в embedding space. A должна учить
малую проекцию, где близкие карточки одной надёжной семьи сближаются, а визуально и
лексически близкие семьи с противоположной меткой раздвигаются. На применении score
равен разности близости к donor-only positive и negative family prototypes. Это
может переносить статус на новое семейство и извлекать правильное меньшинство,
которое сейчас подавляется общим fusion.

Новый механизм есть только при трёх условиях:

1. prototypes принадлежат разным товарным семьям, а не являются почти одинаковыми
   строками из validation;
2. negative пары действительно сложные: supplement↔sports food,
   fuel↔equipment, gas↔accessory;
3. улучшение сохраняется на singleton families и image-connected split.

Иначе A повторяет `080` или neighbor priors `210`.

### Честный cross-fit protocol

Для каждого outer fold:

1. Все connected text/image families внешнего fold полностью исключить из donors.
2. Из donor rows построить family nodes. Семьи с конфликтующими метками исключить
   из positive pairs и prototypes; каждому семейству дать суммарный вес `1`.
3. Положительные пары брать только внутри consistent donor family. Количество пар
   на семью ограничить заранее.
4. Hard negatives искать только внутри donors: ближайшие в исходном frozen embedding
   семьи противоположной метки, предпочтительно внутри заранее заданной товарной
   когорты. Outer embedding нельзя использовать даже для mining statistics.
5. Обучить только линейную проекцию `2048→64` и нормализацию. Dimension, loss,
   temperature и число negatives зафиксировать до outer scoring; допустим не более
   двух заранее заданных вариантов, выбираемых inner grouped folds.
6. Создать один нормированный prototype на donor family. Score внешней строки —
   робастная разность top-k cosine similarities к положительным и отрицательным
   prototypes; `k`, fusion weight и threshold выбираются только inner OOF donors.
7. В `190` добавить только этот score; все остальные компоненты, priors и правила
   оставить фиксированными. Затем воспроизвести donor-only priors.
8. Показать Macro/category F1, все пять fold deltas, grouped bootstrap,
   corrected/regressed, singleton-only delta, duplicate-connected delta и метрики
   целевых hard-negative cohorts.
9. Победителя повторить вторым seed на лучшем и худшем fold. Full refit использует
   конфигурацию, выбранную большинством outer folds, а не лучшую global OOF.

### Runtime и размер

Foundation-model pass не добавляется: embedding уже вычисляет `190`. Для `64D`
проекции и примерно 4–6 тысяч family prototypes ожидаются:

- дополнительное применение: менее `30` секунд на 3800 строк при пакетном BLAS;
- дополнительная память/архив: ориентировочно `1–5 MB`;
- обучение пяти outer heads на cached embeddings: CPU-десятки минут либо меньше
  `0.5 H100-hour`.

Это оценка, не измерение; перед packaging нужен официальный 600-row runtime smoke.
Полный `2048D` nearest-family индекс не нужен: он увеличивает dot-product cost и
риск запоминания без нового доказательства.

### Условие отказа

A отклоняется, если выполняется хотя бы одно:

- Macro delta меньше `+0.003`, побед меньше `4/5` или bootstrap
  `P(delta>0) < 0.90`;
- любая категория падает более чем на `0.005`;
- singleton-family delta неположительна;
- improvement исчезает или становится отрицательным на image-connected topology;
- больше четверти прироста приходится на повторные/аномально пересекающиеся семьи;
- второй seed на лучшем и худшем folds не даёт положительной средней разницы;
- после production priors полезные изменения не сохраняются;
- наивный centroid/kNN выдаётся за новый эксперимент без обучаемой hard-negative
  проекции.

### Ожидаемый результат и уверенность

Разумный диапазон — `+0.0005…+0.0025` Macro; шанс пройти строгий `+0.003` gate
невысок. Источники подтверждают механизм metric learning, но локальные centroid,
neighbour и duplicate-stress результаты против этой ветки. Уверенность ниже средней.

## Гипотеза B: regulatory BAD-head

### Причинный механизм

Основная ошибка BAD — не распознавание «полезного продукта», а взаимодействие
регуляторных свидетельств: явная маркировка BAD, sports semantics, отрицание,
лекарственная форма, обычная пища и назначение животным. Общий classifier обучает
одну границу на всех товарах и теряет это взаимодействие; отдельный head может
переоценивать только заранее определённые спорные карточки.

Гипотезу следует сузить: **sports nutrition является обучаемой целевой когортой**;
food/medicine/veterinary — контекстные состояния и guardrails. Их совокупный F1 уже
высок, а ошибок лишь `26`; четыре равноправных heads принесут больше variance, чем
сигнала.

### Что именно моделировать

Selector фиксируется без меток до обучения:

- sports/аминокислоты/BCAA/протеин/креатин/карнитин;
- плюс небольшой контекстный union food, medicine, veterinary;
- вне selector решение `190` должно оставаться побитово неизменным.

Head получает:

- frozen all-images Qwen3-VL embedding, который уже вычисляется в `190`;
- положения и количества явных свидетельств отдельно в name и description:
  `БАД/dietary supplement`, sports, `не является лекарством`, food, medicine,
  veterinary, dosage form;
- только заранее заданные взаимодействия, прежде всего
  `sports × explicit_BAD` и `sports × not_a_drug`;
- locked score `190` как prior, а не как новая настраиваемая модель.

Обучается малый family-weighted linear residual head. Он мягко сдвигает BAD score;
hard override запрещён. Нельзя добавлять произвольные metadata interactions после
просмотра outer ошибок.

### Честный cross-fit protocol

Для каждого outer fold:

1. Использовать исправленную connected-family topology. Все preprocessing,
   scaling и head weights обучать только на donors.
2. Selector и список взаимодействий заморозить один раз до результатов.
3. Обучать на regulatory donor rows с весом `1 / family_size`; конфликтующие exact
   families не использовать как чистые доказательства статуса.
4. Внутри donors получить inner OOF specialist score. Выбирать только из короткой
   заранее фиксированной сетки регуляризации и residual weight; threshold `190`
   перенастраивать нельзя, разрешён только ограниченный residual shift.
5. Один раз применить к untouched outer selector. За его пределами прогнозы должны
   совпасть с `190`.
6. После объединения воспроизвести exact/name/numeric/shingle priors donor-only.
7. Считать отдельно sports FP/FN, четыре состояния
   `sports × explicit_BAD`, singleton sports, aggregate non-sports regulatory,
   changed/corrected/regressed и fold dispersion.
8. Проверить grouped bootstrap и новую image/text-connected topology. Для линейного
   детерминированного head повтор seed заменяется на альтернативную заранее
   фиксированную family topology; если используется stochastic projection, нужен
   seed `31415` на лучшем и худшем folds.
9. Full refit строится только после transfer gate. Компонент хранится отдельно и
   подключается к `190` одним residual score.

### Runtime и размер

Нового VLM-прохода нет. При линейном head поверх уже рассчитанного embedding и
нескольких sparse признаков ожидаются:

- применение: менее `5` секунд на Private 3800 rows;
- размер: заметно меньше `1 MB`;
- nested обучение на cached embeddings: CPU-минуты, GPU не требуется.

Это также должно быть подтверждено официальным smoke, но runtime-риск значительно
меньше, чем у отдельного генеративного специалиста.

### Условие отказа

Поскольку flammable остаётся неизменным, общий gate `+0.003 Macro` означает примерно
`+0.006 BAD F1`. Для такого роста нужно около 60–70 чистых исправлений без заметных
регрессий; небольшой cohort gain недостаточен.

B отклоняется, если:

- BAD delta меньше `+0.006` или Macro delta меньше `+0.003`;
- побед меньше `4/5`, bootstrap `P(delta>0)<0.90`;
- sports F1 не растёт минимум на `0.01` либо corrected/regressed меньше `2:1`;
- singleton-sports delta неположительна;
- aggregate food/medicine/veterinary F1 падает больше `0.005`;
- меняется хотя бы одно flammable-решение или решение вне frozen selector;
- эффект исчезает после priors или на connected-family topology;
- основная доля gain приходит из conflicting repeated families;
- реализация превращается в hard regex override или общий meta-stacker.

### Ожидаемый результат и уверенность

Разумный диапазон — `+0.0015…+0.0040` Macro. 243 устойчивых sports-ошибки и
существенная доля правильных независимых embedding-ответов дают реальный запас.
Одновременно label ambiguity ограничивает достижимый эффект. Уверенность средняя и
выше, чем у A.

## Риск повторения закрытых веток

### A повторит прошлое, если

- использовать обычный class centroid: он уже резко проиграл;
- использовать raw nearest neighbours без learned metric: это близко к `210`;
- оценивать по text-grouped folds без image connection: это повтор риска `080`;
- позволить частым семьям генерировать квадратично больше пар: это повторяет
  проблему sampling, проверенную `260/280`.

### B повторит прошлое, если

- сделать `sport => 0` или другой hard override: такие правила в `000` снизили
  Macro до `0.8020`;
- обучить общий stacker на всех scores и metadata: nested meta-fusion `900` дал
  `0.90454`, ниже простого fusion;
- считать новый family sampling достаточным основанием: `260` выиграл локально, но
  не улучшил Public;
- смешать branch с новым threshold, новой fusion и soft-target Qwen: причинность
  снова потеряется как в `230`.

B также пересекается по причине с `310`: обе ветки пытаются справиться с шумными и
неуверенными BAD-целями. Различие состоит в том, что `310` глобально меняет цели
Qwen3.5, а B меняет только decision head в заранее определённом регуляторном режиме.
До результата `310` обучать B не следует.

## Приоритет и условная развилка после `310`

### Приоритет по текущим данным: B выше A

Причины:

1. B отвечает за заранее измеренную, большую и устойчивую к folds когорту — 243
   sports-ошибки.
2. Большая часть этих ошибок — singleton families, поэтому механизм не опирается
   только на recurrence.
3. Нужный независимый signal уже есть в production embedding; нового backbone и
   нового inference pass не требуется.
4. У A есть сильные отрицательные precedents: centroid, neighbour priors и
   image-duplicate overfit.

Но окончательное действие зависит от `310`:

- если `310` проходит transfer gate и уменьшает sports errors минимум на `20%`
  (`49` из текущих `243`) без guardrail breach, B частично исчерпана; тогда A получает
  приоритет как более ортогональная ветка, начиная только с дешёвого cached-embedding
  screen;
- если `310` отклонён либо оставляет не менее `80%` sports errors, B остаётся первым
  полным следующим опытом;
- если `310` улучшает sports, но проваливает другие когорты, сначала проверяется
  возможность использовать его только как диагностический teacher signal; нельзя
  автоматически строить post-hoc gate по правильности outer rows.

## Как применить урок OpenRsi к выбору A/B

OpenRsi полезен циклом `propose → critique → evaluate → verify → keep`, но его
повторная проверка усредняет выбранный удачный запуск с повтором на тех же задачах.
Это уменьшает stochastic variance, но не исправляет multiple selection и adaptation
к одному private set.

Для A/B следует:

1. Сохранить обе карточки как варианты, но выбрать **одну primary branch до sealed
   evaluation**. По текущим данным это B.
2. Второй вариант допускается только к дешёвому механистическому screen, а не к
   параллельному полному подбору по тем же outer labels.
3. Если оба когда-либо проходят полный gate, не выбирать максимум двух noisy deltas:
   нужен заранее заданный приоритет либо paired bootstrap A-vs-B с поправкой на
   selection.
4. Повторять не ту же цифру на тех же folds, а перенос: connected-family topology,
   second seed там, где он есть, singleton cohorts, priors и runtime.
5. Любой проигравший вариант сохранить с точным `reopen_if`; не запускать малые
   вариации до появления нового механизма.

## Рекомендация

Сейчас ничего не запускать и дождаться завершения `310`. Параллельно допустима только
детерминированная подготовка новой topology и invariant tests без чтения её итоговых
метрик автором гипотезы.

Если `310` не устраняет существенную долю sports-ошибок, следующим опытом сделать B,
но как **sports regulatory residual head**, а food/medicine/veterinary оставить
контекстом и guardrails. A оставить следующей ортогональной веткой с обязательным
image-connected screen; простой prototype/kNN не запускать.
