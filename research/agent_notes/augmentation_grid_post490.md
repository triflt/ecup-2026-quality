# Data/augmentation hypotheses после exp400

Дата: 2026-08-22. Статус: исследовательский дизайн, без запусков. Ограничение: никакого 400B teacher, внешней разметки или настройки по Public.

## Решение в одном абзаце

Приоритет №1 — **position-only transaction-scope augmentation**: в части обучающих flammable-предъявлений переносить уже существующее полное предложение о составе/комплекте в начало описания, не добавляя, не удаляя и не перефразируя ни одного токена. Это единственный кандидат, который напрямую отвечает геометрии остатка `400` (19 ошибок о составе/комплекте, одновременно 19 FN и 17 FP во всей flammable-категории), причинно отличается от keyword-head `450` и ничего не добавляет к inference.

Приоритет №2 — **high-resolution first-image fine-tuning** для одного Qwen3-VL-компонента. Приоритет №3 — **cross-listing same-product image/text recombination**, но он заблокирован до ручного аудита и фиксации semantic-family graph. GPU сегодня не запускать: сначала нужен новый component-disjoint protocol и точный baseline `400` на нём. После этого первым разрешать только двухfoldовый reject-screen гипотезы H1.

## Наблюдаемая мишень и закрытые ветки

У OOF-решения `400` остаётся 538 ошибок:

- БАД: 502 ошибки, 279 FN / 223 FP; 467 connected-safe, 122 лежат в mixed-label components;
- flammable: 36 ошибок, 19 FN / 17 FP; все 36 connected-safe;
- среди flammable-ошибок пересекающиеся когорты: 31 ignition/pyrotechnics, 19 solid fuel, 19 included/kit scope, 15 gas/liquid fuel;
- переход `190 → 400` уменьшил flammable FP `28 → 17`, но увеличил FN `16 → 19`: новый flammable adapter стал точнее, но слишком консервативен на части наборов и явного топлива.

Повторять нельзя:

- family/class resampling и дополнительное class pressure: `241` дал средний сигнал, но лишь 3/5 fold wins; `260` перенёсся локально, но не улучшил Public как отдельная замена; `410` дал Macro `−0.022497` и `+8` flammable FN; balanced continuation `420` ухудшил оба screen-fold;
- ещё одну clean/noisy-label фильтрацию: ранний clean-filter уже проиграл hard-mined обучению, а mixed-label семьи нельзя автоматически объявлять ошибочной разметкой;
- generic regularization: R-Drop `440` дал среднее `−0.020291` на двух screen-fold и `+5` FN на одном из них;
- extra gallery images: first/second/last-image LoRA уже проиграл first-image LoRA на одинаковом fold; просто добавлять изображения или второй inference pass нельзя;
- keyword transaction classifier `450`, generic embedding head `460`, soft targets `310`, neighbour/centroid/diagonal metric и новый fusion-weight sweep;
- adaptive hard-negative mining до нового nested protocol: старый difficulty cache не полностью вложен относительно outer folds;
- teacher-generated examples или pseudo-labels. Внешние данные также не нужны.

## Первичные основания и границы переноса

1. Avigdor et al., **Consistent Text Categorization using Data Augmentation in e-Commerce**, ACL Industry 2023, используют разные версии одного товара как label-preserving augmentation. Generative augmentation дала `+0.28%` F1 и `+4.46%` consistency; масштабное self-training повысило consistency на 7–10%, но снизило F1 на 0.6–1.65% из-за distribution shift и noisy pseudo-labels. Для нас важнее отрицательная часть результата: не увеличивать набор и не добавлять псевдометки, а заменять фиксированную долю уже существующих предъявлений in-distribution трансформациями. [ACL Anthology](https://aclanthology.org/2023.acl-industry.30/), [paper](https://aclanthology.org/2023.acl-industry.30.pdf)
2. Kaushik et al., **Learning the Difference that Makes a Difference with Counterfactually-Augmented Data**, ICLR 2020, показывают ценность минимальных правок, меняющих только причинный признак, а не поверхностный контекст. Наш первый pilot ещё консервативнее: label и текст не меняются вообще, меняется только позиция целого evidence sentence. Настоящие label-flipping синтетические пары без человеческой разметки здесь не разрешены. [OpenReview paper](https://openreview.net/pdf?id=Sklgs0NFvr)
3. Xie et al., **Unsupervised Data Augmentation for Consistency Training**, NeurIPS 2020, показывают, что качество label-preserving transformation важнее простого шума. Это поддерживает ручной semantic-preservation gate для H1/H3 и одновременно объясняет, почему ещё один dropout/R-Drop не является достаточной гипотезой. [Официальная страница NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/44feb0096faa8326192570788b38c1d1-Abstract.html)
4. Xian et al., **Solution for Large-scale Long-tailed Recognition with Noisy Labels**, технический отчёт AliProducts Challenge 2021: high-resolution fine-tuning дал около `+1%` top-1, а enlarge+ten-crop TTA — ещё 1–2%. Мы переносим только resolution mechanism; TTA запрещена условием single-pass. Их 2.5M fine-grained product images значительно отличаются от наших 12,971 карточек, поэтому H2 требует дешёвого reject-screen. [arXiv](https://arxiv.org/abs/2106.10683)
5. Jin et al., **ECLIP**, CVPR 2023, используют изображения одного product instance из разных источников как positive views. Это прямое основание для H3, но работа имеет реальные product identities, около 100M images и 12M products; наш graph не может заменить product ID без ручной проверки точности рёбер. [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2023/html/Jin_Learning_Instance-Level_Representation_for_Large-Scale_Multi-Modal_Pretraining_in_E-Commerce_CVPR_2023_paper.html), [supplement](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Jin_Learning_Instance-Level_Representation_CVPR_2023_supplemental.pdf)
6. MOON2.0, CVPR 2026, подтверждает полезность image-text co-augmentation и dynamic filtering именно для e-commerce, но использует MLLM-generated enrichment и существенно более сложную архитектуру. Мы заимствуем только принцип согласованной пары и fail-closed filtering, не генератор и не MoE. [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Nie_MOON2.0_Dynamic_Modality-balanced_Multimodal_Representation_Learning_for_E-commerce_Product_Understanding_CVPR_2026_paper.html)
7. Официальный first-place report SIGIR 2020 Rakuten строит отдельные text/image classifiers и decision-level fusion. Он не доказывает пользу конкретной augmentation, но поддерживает сохранение модульной архитектуры `400`: в каждом опыте меняется один существующий компонент, не вся система. [Официальный репозиторий решения](https://github.com/Wang-Shuo/SIGIR2020Challenge)

## H1 — position-only transaction-scope augmentation

### Механизм

Для каждой outer-train flammable-карточки label-blind parser разбивает description на предложения. Если ровно одно предложение содержит один из заранее закрытых типов отношения:

- `входит/включён/комплектуется/поставляется вместе`;
- `не входит/без ... в комплекте/приобретается отдельно`;
- `совместим/подходит/используется с`;

создаётся augmented view, где **полное предложение без изменений** переносится сразу после title, а остальные предложения сохраняют взаимный порядок. Дублировать предложение, удалять контекст, менять отрицание, генерировать синонимы или смотреть label/ошибки `400` запрещено. Карточки с несколькими конфликтующими cue sentences, анафорой без явного объекта или неразрешимым HTML fail closed и не меняются.

Трансформация заменяет часть уже существующих training occurrences, а не добавляет шаги. В исходном и изменённом manifest должны точно совпасть row IDs с кратностями, labels, category counts, family counts и shuffle RNG; различается только serialized description выбранного occurrence.

### Почему это не `450`

`450` удалял большую часть текста и строил отдельный линейный rank по keyword-selected предложениям. H1 сохраняет полный исходный текст и обучает прежний cross-attentive Qwen3.5 adapter; единственный intervention — позиция уже существующего evidence sentence. При inference используется обычная карточка и обычный один проход `400`, без нового score или regex rule.

### Training grid

| параметр | null | кандидат |
|---|---:|---:|
| backbone/component | exact Qwen3.5-4B flammable route `400` | тот же |
| initialization | с нуля по точному parent recipe, не continuation | то же |
| transformed eligible occurrences | 0% | **50%**, deterministic SHA parity |
| records / class counts / steps | parent exact | parent exact |
| optimizer, LR, LoRA rank, seed, prompt, image 448 px | parent exact | parent exact |
| inference | `400` | byte-identical `400` preprocessing |

Это намеренно one-point grid. После просмотра fold нельзя пробовать 25/75%, новые cue words, другую позицию или label-flipping edit. Seed repeat разрешён только после full development acceptance.

До обучения — blind audit 200 преобразований, стратифицированный по cue type и длине:

- 100% токенов описания сохранены с той же кратностью;
- отрицание и его объект остались в одном предложении;
- смысл карточки и label-preserving relation не изменились минимум в 98% пар;
- reviewer agreement `κ ≥ 0.80`;
- ни одна строка outer-validation/sealed не использована для словаря или решения об eligibility.

### Compute

Оценка одного Qwen3.5 fold: около 50–70 минут на одной H100; новый preprocessing практически бесплатен. Reject-screen двух заранее зафиксированных outer folds: 2–2.5 GPU-hours. Полные пять folds + full refit: около 6–7 GPU-hours; conditional second seed на двух folds: ещё 2–2.5 GPU-hours. Inference/runtime/archive не меняются.

### Ожидаемый failure mode

- модель выучит position shortcut и ухудшит карточки без cue;
- regex выберет совместимость вместо состава либо разорвёт анафору;
- реальная разметка не следует transaction semantics, как уже намекнул `450`;
- улучшения сконцентрируются в одном повторяющемся family component;
- полнота газа/жидкого топлива упадёт из-за нового отрицательного pressure.

## H2 — high-resolution first-image fine-tuning

### Механизм

Заменить только image cap первого изображения у существующего `Qwen3-VL-2B` adapter: aspect-preserving resize с 448 до **672 px**, processor `max_pixels` согласован с `672²`. Никаких crops, sharpen/OCR, gallery images, TTA или второго прохода. Train и inference используют одинаковую новую resolution; Qwen3.5, base embedding, sampling, prompts, weights, thresholds и priors неизменны.

Это отличается от rejected multi-image LoRA: модель по-прежнему видит один cover image, но мелкая маркировка и предмет занимают больше visual tokens. Мишень — упаковочная надпись/мелкая визуальная деталь для solid fuel, ignition и БАД dosage-form, а не больше контекстных фотографий.

### Training grid

| параметр | null | кандидат |
|---|---:|---:|
| backbone/component | Qwen3-VL-2B component `400` | тот же |
| first-image cap | 448 px | **672 px** |
| max pixels | parent exact | `451,584` |
| crop/augmentation | none | none |
| rows, ordering, optimizer, LoRA, seed, prompt | parent exact | parent exact |
| inference passes | one | one |

Перед GPU — 200-image label-blind rendering audit: no aspect distortion, same cover image, no decompression failure; отдельно измерить долю изображений, которые действительно получают >25% дополнительных pixels. Если таких меньше 20%, H2 отменяется как несущественный intervention.

### Compute

Visual-token count в худшем случае растёт примерно в `2.25×`; ожидаемый fold budget 1.5–2.0 H100-hours. Двухfoldовый reject-screen: 3–4 GPU-hours; пять folds + full refit: 9–12 GPU-hours. До full cycle обязателен 600-row runtime smoke: projected Public ≤16 минут, Private ≤32 минут. Превышение — reject независимо от F1.

### Ожидаемый failure mode

- residual определяется transaction semantics и label noise, а не разрешением;
- мелкая надпись порождает новый OCR shortcut или конфликтует с title;
- больше visual tokens обрезают tail description в 1,536-token контексте;
- training/inference дорожают без достаточного изменения predictions;
- компонент имеет небольшой fusion weight и полезный signal не переживает `400`.

## H3 — audited cross-listing same-product recombination

### Механизм

Только после freeze semantic-family graph брать outer-train пары из strong same-product components: exact full text; либо specific name + corroborating description/product-cover image; auxiliary-image-only и generic/high-degree components запрещены. Для двух outer-train строк одного audited component и одинакового training label создать `(text_a, first_image_b)` и симметричную `(text_b, first_image_a)` view. Полная карточка не смешивается между категориями; mixed-label components не аугментируются, но остаются в обычном train/eval.

Recombined views заменяют 25% повторных occurrences соответствующих исходных строк. Число предъявлений, labels, category/family sampling и шаги не меняются. Inference остаётся исходным single first image 448 px. Это не family balancing (`241/260` меняли представленность families) и не multi-image LoRA (несколько gallery images подавались одновременно): H3 меняет только согласованное сочетание двух реальных views одного product identity в одном прежнем image slot.

### Training grid

| параметр | null | кандидат |
|---|---:|---:|
| backbone/component | Qwen3-VL-2B component `400` | тот же |
| eligible repeated occurrences recombined | 0% | **25%** |
| pair source | — | audited strong outer-train components only |
| rows / labels / family counts / steps | parent exact | parent exact |
| resolution / optimizer / LoRA / seed / prompt | parent exact | parent exact |
| inference | original first image | original first image |

До обучения проверить 300 cross-listing pairs двумя blinded reviewers:

- same product/close packaging variant precision ≥98%;
- no generic component or auxiliary-only edge;
- agreement `κ ≥ 0.80`;
- каждый pair целиком outer-train, sealed coverage zero;
- не менее 200 eligible pairs и не менее 50 independent components на каждом screen fold. Иначе NO-GO: слишком низкое эффективное покрытие.

### Compute

Стоимость почти равна обычному Qwen3-VL fold: 50–70 минут. Reject-screen: 2–2.5 GPU-hours; полный цикл + full refit: 6–7 GPU-hours; repeat двух folds: ещё 2–2.5 GPU-hours. Inference и архив практически без прироста.

### Ожидаемый failure mode

- graph edge означает лишь похожую карточку, а не тот же товар;
- упаковки разных variant/quantity противоречат тексту и превращают augmentation в шум;
- реальных strong components слишком мало, а улучшение держится на recurrence;
- модель теряет точный visual-text alignment, как rejected multi-image branch;
- исключение mixed-label components создаёт лёгкий train subset, но не улучшает полный evaluation.

## Общий nested semantic-family protocol

1. Завершить label-blind graph audit, исправить generic-name/auxiliary-image edges, зафиксировать component assignment и один sealed holdout около 1/7. Никакой кандидат не может влиять на topology.
2. На development complement один раз полностью переобучить exact `400` baseline. Старые adapters невалидны, если они видели sealed или outer-validation components.
3. Для каждого из пяти outer component folds candidate manifest строится только из outer-train:
   - selector/difficulty scores — donor-only inner OOF;
   - H1 cue policy статична до folds, eligibility считается только на outer-train;
   - H3 pair index содержит только outer-train components;
   - thresholds, rank calibration и donor priors fit без outer-validation;
   - все outer-validation rows, включая mixed families, оцениваются, ничего не исключается из основной метрики.
4. Одноточечные grids выше не выбираются по outer result. Первые два заранее назначенных outer folds имеют только право отклонить. Если screen пройден, recipe и hashes не меняются и считаются остальные три folds.
5. После development acceptance сделать conditional seed repeat на тех же двух screen folds. Только затем pre-register full refit и один раз открыть sealed. После просмотра sealed любой новый recipe требует нового holdout.
6. Public не служит ни validation, ни threshold sweep. Downstream exact/name/numeric/shingle priors replay без изменений; отдельно сообщаются before/after-prior решения.

## Frozen gates

### Двухfoldовый reject-screen

Остановить hypothesis, если выполняется хотя бы одно:

- Macro delta против exact retrained `400` ≤0 на любом screen fold;
- средний Macro delta <`+0.001`;
- flammable FN или safety-union FN растут на любом fold;
- corrected ≤ regressed;
- BAD падает >0.003 на H2/H3 либо меняется хотя бы один BAD verdict у H1;
- intervention coverage <10 eligible changed inputs/fold либо candidate меняет <5 решений суммарно;
- выигрыш объясняется одним component размером >10 строк;
- H2 не проходит runtime gate или H3 pair audit.

Screen не может принять модель.

### Full development acceptance

- mean Macro delta `≥ +0.003`, минимум `4/5` outer-fold wins;
- H1: flammable delta `≥ +0.006`, BAD byte-identical;
- H2/H3: ни одна category delta `< −0.003`, flammable delta положителен;
- corrected:regressed `≥1.5`;
- flammable FN и safety-union FN не растут суммарно и ни на одном fold более чем на 1; для H1 требуется строгий нулевой рост;
- component bootstrap `P(delta>0) ≥0.90`, нижний край 95% interval не хуже `−0.0015`;
- singleton-only delta положителен; gain присутствует минимум в трёх заранее объявленных flammable cohorts и не держится на unsafe/repeated families;
- после production-prior replay delta положителен и сохраняется ≥80% candidate changes;
- conditional seed repeat: средний delta положителен, ни один повтор не хуже `−0.003`;
- inference single-pass; H1/H3 не медленнее `400` более чем на 3%, H2 проходит ≤16/32 минут.

### Sealed one-shot

Sealed содержит ориентировочно 28 flammable positives, поэтому это directional confirmation: Macro delta `≥+0.002`, no category drop `<−0.003`, flammable/safety FN не растут, corrected > regressed и component-bootstrap не показывает materially negative tail. Никакого retune после unlock.

## Очерёдность

1. **H1 — запускать первым после готовности validation.** Лучшее соответствие error geometry, нулевая inference-цена, минимальный риск генерации фактов. Сегодня допустимы только immutable transform manifest и blind audit; затем 2-fold screen.
2. **H3 — второй, но только после graph pair gate.** Самый прямой e-commerce precedent, почти бесплатный inference; главный риск — false family identity.
3. **H2 — третий.** Сильный competition precedent, но слабее связь с нашими measured errors и существеннее runtime/GPU риск.

Не объединять hypotheses. Даже если две проходят независимо, их сочетание — отдельный новый single-factor confirmation относительно лучшего принятого parent.
