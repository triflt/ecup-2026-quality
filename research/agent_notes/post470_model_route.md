# Модельный маршрут после 440/460: InternVL3.5-2B pairwise flammable ranker

Дата: 2026-08-22. Это read-only исследовательская рекомендация; jobs не
запускались, общие журналы и каталоги экспериментов не менялись.

## Решение

Следующий причинный screen стоит отдать `OpenGVLab/InternVL3_5-2B`, но не
повторять zero-shot attributes из `300`. Проверяемый механизм — supervised
pairwise ranker, который на фактических train-метках учится ставить
подтверждённый flammable-товар выше максимально похожего неподтверждённого.
В production-кандидате он заменяет только ранг Qwen3-VL с весом `0.10` на
заранее замороженном label-blind hard selector; Qwen3.5 `0.75`, robust base
`0.15`, BAD-ветка, priors и пороги `400` остаются неизменными.

Рабочее имя следующего опыта: `480_internvl_pairwise_flammable_ranker`.

Это один компонентный фактор:

> на замороженных трудных flammable-строках заменить Qwen3-VL rank
> pairwise-trained InternVL rank; всё остальное оставить byte-identical `400`.

Здесь нет teacher: используются только outer-donor train labels. В частности,
не требуется 400B-модель, псевдометки hidden или Public-feedback.

## Что именно уже опровергнуто локально

| Ветка | Результат | Что закрыто |
| --- | ---: | --- |
| `440` Qwen3-VL R-Drop | screen folds 0/3: Macro `-0.035676/-0.004907`, среднее `-0.020291`; corrected/regressed `25/36`, flammable FN `+4` | Не подбирать другой `alpha`, seed или остальные folds того же R-Drop |
| `450` transaction-scope text view | Macro `-0.006744`, flammable `-0.013489`, `0/5`, corrected/regressed `1/6`, FN `+4`, connected `P=0.0158` | Не тюнить keyword/sentence extractor |
| `460` Qwen3-VL-Embedding-2B | Macro `-0.016656`, flammable `-0.033311`, `0/5`, corrected/regressed `2/15`, FN `+7`, connected `P=0.0037` | Не тюнить linear/metric head или его вес; embeddings допустимы лишь как диагностика |
| `290` MiniCPM-V-4.6 direct first-image LoRA | folds 0/3: Macro `-0.041895`, flammable `-0.079607`, `0/2` | Не делать ещё одну полную замену сильной системы generic VLM verdict |
| `300` InternVL zero-shot attributes | Macro `-0.006572`, flammable `-0.013145`, `2/5`, corrected/regressed `4/9`; 336/909 ответов не распарсились | Не повторять generic six-bit QA и linear gate; совместимость самой модели не опровергнута |
| `150` Gemma-4-E4B | standalone Macro `0.900292`; nested base+Gemma `0.902686`, ниже Qwen3-VL; flammable fold F1 `0.810--0.974` | Не повторять Gemma direct head меньшим E2B без нового основания |

Остаток flammable семантически контрастный: среди 27 общих ошибок сильных
решений пять когорт (lighter/matches, liquid fuel, candle, charcoal и
equipment/empty container) покрывают 24. Есть неинтуитивные пары: часть refill
fuel и dry fuel имеет label 0, а charcoal — label 1. Поэтому generic вопрос
«горит ли товар» систематически подменяет соревновательную онтологию бытовой.
Нужна обучаемая относительная граница на фактических парах, а не ещё один
zero-shot verdict, OCR-текст или embedding classifier.

## Почему именно InternVL3.5-2B

Официальная [model card](https://huggingface.co/OpenGVLab/InternVL3_5-2B)
указывает `2.3B` total (`0.3B` vision + `2.0B` language), архитектуру
ViT–MLP–LLM, Dynamic High Resolution и `256` visual tokens после pixel shuffle.
То есть модель входит в разрешённый класс и даёт самостоятельный visual path.
Важно для предложенного механизма: официальный training pipeline самой семьи
использует Mixed Preference Optimization с preference/quality/generation losses.
Pairwise ordering поэтому соответствует проверенному режиму семейства, хотя наш
малый loss не объявляется воспроизведением MPO.

Локально совместимость уже измерена, а не предполагается:

- native backward `redacted-job`: `2.3548B` параметров, rank-16 rsLoRA
  `6.4225M` trainable, finite loss/gradients, peak `4.844 GiB` на одной H100;
- inference `redacted-job`: 909 first-image строк, batch 8, `1.6335 min`;
- private compute platform preset успешно использовал
  `huggingface-proxy/OpenGVLab/InternVL3_5-2B/latest`;
- организаторский список фиксирует модель в `SHARED_MODELS_PATH`, поэтому веса
  не входят в submission archive.

Официальная карточка также документирует `AutoModel(...,
trust_remote_code=True)` и bf16 loading. Следует сохранять уже проверенный
GitHub-format native path из `300`, а не снова пробовать несовместимый generic
`AutoProcessor` path.

Почему не ближайшие альтернативы:

- [moondream2](https://huggingface.co/vikhyatk/moondream2) действительно мал и
  имеет native `query/detect/point`, но в проекте нет ни backward, ни
  organizer-image runtime smoke. Его generic visual query всё равно не учит
  неинтуитивную train ontology; сейчас это более высокий инфраструктурный риск.
- [Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) — text-only
  agentic model с custom code (`4B` total на карточке), то есть дублирует уже
  сильную Qwen3.5 text-ветку и не видит содержимое комплекта на изображении.
- [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) —
  официальный `0.9B` document parser/text spotting model; локальная OCR-коррекция
  уже дважды не переносилась, а batch-1 измерение около `4.88 s/image` не годится
  для широкой hidden-ветки.
- [MiniCPM-V-4.6](https://huggingface.co/openbmb/MiniCPM-V-4.6) и Gemma-4-E4B
  уже получили честные отрицательные supervised OOF результаты; E2B не даёт
  нового причинного механизма.

## Полностью замороженный двухфолдовый screen

### 1. Строки и пары

- Outer folds: только `0` и `3`; seed `42`.
- Исходный selector: byte-identical
  `experiments/300_internvl35_attribute_screen/results/attribute_probe_selector_report.json`
  и его 909-row manifest. Он label-blind, выбирает `16.52%` flammable и покрывает
  `37/44` ошибок старого baseline диагностически. Никакой новый keyword search
  или подбор uncertainty fraction не допускается.
- Для outer fold `f` training pool содержит только selector-строки остальных
  четырёх grouped folds, разрешённые connected-family guard. Outer labels,
  unsafe connected rows и строки из fold `f` не участвуют ни в парах, ни в
  выборе масштаба.
- Каждая donor-positive получает ровно четыре donor-negative. Сначала берутся
  ближайшие по donor-only char-TFIDF `name + description` внутри того же уже
  замороженного cue-bitmask selector; при нехватке — ближайшие во всём selector.
  Tie-break: строковый `id`. Один negative допускается максимум в восьми парах;
  дальше берётся следующий сосед. Pair manifest и SHA-256 сохраняются до
  обучения.

TF-IDF здесь не предсказывает label и не входит в submission: он лишь фиксирует
максимально похожие контрастные training pairs. Это не повтор embedding-head из
`460` и не sentence view из `450`.

### 2. Модель и loss

- Exact checkpoint: `OpenGVLab/InternVL3_5-2B`, тот же revision/model-registry
  object, native loader и Transformers `4.57.3`, что в успешном `300` smoke.
- Input: первое доступное изображение, один patch `448x448`; `name` до 320
  символов и `description` до 1400; один фиксированный русский prompt с ответом
  `0/1`.
- Frozen vision tower. Rank-16 rsLoRA только на
  `language_model.*.{q,k,v,o}_proj`: `r=16`, `alpha=32`, dropout `0.05`.
- Score `s(x) = logit(token "1") - logit(token "0")` на первом assistant
  token. До запуска smoke обязан проверить, что обе цифры atomic. Генерация и
  parsing полностью отсутствуют.
- Единственный training loss на паре `(x+, x-)`:
  `softplus(-(s(x+) - s(x-)))`.
- AdamW `lr=2e-4`, weight decay `0.01`, cosine decay, warmup `5%`, bf16,
  gradient checkpointing. Pair batch `4`, gradient accumulation `4`, ровно
  `96` optimizer updates. Детерминированный циклический shuffle pair manifest;
  никаких alpha/margin/lr/epoch trials.

### 3. Единственное изменение inference

Для selected flammable outer rows score InternVL отображается в `[0,1]` через
эмпирическую CDF на donor-selected scores без использования labels. Затем:

`score_candidate = score_400 + 0.10 * (rank_internvl - rank_qwen3vl)`.

Для остальных строк `score_candidate = score_400`. BAD predictions должны быть
byte-identical. Используются уже замороженные outer thresholds `400`:
`0.9564604759216309` для fold 0 и `0.9570291638374329` для fold 3; threshold,
вес `0.10`, selector и CDF formula не тюнятся. Null control
`rank_internvl := rank_qwen3vl` обязан дать ноль изменённых predictions и точные
baseline Macro `0.9312591699999442` (fold 0) и `0.9166759475813936` (fold 3).

### 4. Reject/promote gate

Немедленно reject и не обучать folds 1/2/4, если не выполнено любое условие:

1. `delta Macro > 0` отдельно на fold 0 и fold 3;
2. среднее двух `delta Macro >= +0.003` (эквивалентно среднему flammable
   приросту не менее `+0.006`, поскольку BAD неизменён);
3. суммарно `corrected / regressed >= 1.5` и `regressed > 0` трактуется обычным
   образом; при `regressed=0` gate пройден;
4. flammable FN и safety-union FN не увеличены ни на одном fold;
5. на connected-safe строках оба fold имеют положительный delta, grouped
   bootstrap `10,000`, seed `48042` даёт `P(delta>0) >= 0.80`;
6. BAD changed predictions `=0`, selector membership и pair manifests совпадают
   с замороженными hashes.

Это reject-only screen: успех не разрешает submission, а только полный OOF.

## Полные validation gates после успеха screen

1. Обучить те же immutable manifests/recipe для folds 1/2/4; никаких tuning
   runs между screen и full OOF.
2. На пяти frozen grouped folds относительно `400`: Macro `>= +0.003`,
   flammable F1 `>= +0.006`, минимум `4/5` строгих fold wins, BAD byte-identical,
   corrected/regressed `>=1.5`, flammable FN и safety-union FN не растут.
3. Connected-family guard: positive delta на всех safe rows, минимум `4/5`
   wins, grouped bootstrap `10,000`, seed `48042`, `P(delta>0) >=0.90`.
4. Downstream replay всех exact/name/numeric/shingle priors `400`; прирост должен
   остаться положительным, и не менее `80%` исправлений до priors должны выжить.
5. Независимо повторить только folds 0 и 3 с seed `31415`; оба delta должны
   оставаться положительными, а среднее — не менее `+0.001`.
6. Feature-parity test train/submission, atomic-token assertion, missing-image
   fallback, null control, output-schema smoke и package integrity.
7. До packaging — официальный 600-row stratified H100 runtime smoke с обеими
   категориями и selector hits. Проекция должна быть `<=16 min` Public и
   `<=32 min` Private, иначе кандидат отклоняется даже при хорошем OOF.

## Стоимость и реалистичность

- Backward-совместимость уже доказана на H100. Pairwise batch содержит два
  forward, но 96 updates и 909-row selector дают ожидаемо `30--50 min/fold`,
  то есть `1.0--1.7 H100-hour` на весь screen. Это оценка до измерения, не SLA.
- Screen inference только на selected outer rows 0/3 (исторически 104 и 111),
  ожидаемо меньше минуты суммарно по измерению `1.6335 min / 909 rows`.
- Submission запускает InternVL лишь на label-blind selector. Если консервативно
  добавить полную измеренную стоимость без вычитания сэкономленного Qwen пути к
  известным `400/190` `10.17/24.15 min`, получается примерно `10.4 min` Public и
  `24.7 min` Private при train-like долях категорий/selector. Даже более грубый
  верхний предел, где InternVL запускается на всех flammable, около
  `11.4/27.1 min`; оба ниже целевых `16/32`, но официальный smoke обязателен.
- Memory реалистична: measured LoRA backward peak `4.844 GiB`; последовательная
  загрузка/выгрузка specialist не приближается к H100 80 GB. private compute platform route уже
  рабочий (`h100-1x`, HF proxy model registry). В submission exact модель заранее
  смонтирована организатором.

Главный риск качества — очень мало positives (`198` всего, `116` в selector),
поэтому нельзя расширять поиск гиперпараметров после folds 0/3. Если этот exact
pairwise screen не проходит, закрыть supervised InternVL specialist, а не
перебирать число соседей, margin, selector или fusion weight.

## Проверенные источники

Локальные первичные артефакты:

- `experiments/440_qwen3vl_rdrop/results/metrics.json`;
- `experiments/450_flammable_transaction_scope/results/acceptance_audit.json`;
- `experiments/460_qwen3vl_embedding/results/metrics.json` и
  `results/acceptance_audit.json`;
- `experiments/290_minicpm_v46_visual_screen/results/metrics.json` и
  `analysis/agent_hard_errors/REPORT.md`;
- `experiments/300_internvl35_attribute_screen/results/metrics.json`,
  `results/attribute_gate_report.json` и private compute platform presets;
- `experiments/400_qwen35_category_routed_adapters/results/acceptance_audit.json`;
- `docs/hackathon/data-and-models.md`, `docs/hackathon/task-and-rules.md` и
  `docs/research/archive/public-attempt-2026-08-21.md`.

Официальные карточки/primary docs:

- [InternVL3.5-2B model card](https://huggingface.co/OpenGVLab/InternVL3_5-2B);
- [InternVL3.5 technical report](https://arxiv.org/abs/2508.18265);
- [moondream2 model card](https://huggingface.co/vikhyatk/moondream2);
- [Nanbeige4.2-3B model card](https://huggingface.co/Nanbeige/Nanbeige4.2-3B);
- [PaddleOCR-VL-1.5 model card](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5);
- [MiniCPM-V-4.6 model card](https://huggingface.co/openbmb/MiniCPM-V-4.6);
- [Gemma-4-E4B-it model card](https://huggingface.co/google/gemma-4-E4B-it).
