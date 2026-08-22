# Решение по собственному contrastive embedder для похожих товаров

Дата проверки: 2026-08-22. Статус: **NO-GO на запуск против exp400 сейчас; условный GO только на дешёвый projection-head pilot после аудита и фиксации semantic-family graph.**

## Короткий вывод

Идея не закрыта exp340/360–390/460, но её действительно новая версия очень узка: обучать не moderation classifier и не label-aware метрику, а **label-blind product-identity/view projection** поверх уже вычисляемого `Qwen/Qwen3-VL-Embedding-2B`, а затем использовать его только как высокоточную semantic-family retrieval ветку после детерминированных priors.

Запуск до завершения semantic-family validation не оправдан. В отличие от опубликованных product-matching работ, у набора нет надёжного product ID и размеченных non-match пар. Неаудированные graph edges дадут шумные positives, а «не соединены ребром» — особенно опасные false negatives. Кроме того, exp400 остаётся Public champion, но его локальный выигрыш нестабилен; новый метод надо сравнивать с полностью переобученным exp400 в nested family split, а не с текущим OOF.

## Что уже проверено

1. **exp340, family contrastive prototype — отменён до метрики.** Fixed random Gaussian `2048→256`, labeled family prototypes и rank fusion фактически повторяли rejected centroid/neighbor ideas; в старых folds нашлись cross-fold exact-text families. Это не было обучением embedder, но закрывает naive prototype score на текущих embeddings.
2. **exp360–390 — label-aware metric learning на frozen Qwen3-VL embeddings.** Диагональная метрика обучалась на nearest same/opposite target-label families и применялась через flammable disagreement/gating; разные connected topologies, all-images parity и bagging дали нестабильные fold gains и не прошли acceptance gates. Это закрывает ещё одну target-aware coordinate metric или её retune.
3. **exp460 — LinearSVC на frozen Qwen3-VL-Embedding-2B.** 12,971 нормализованных 2048D embeddings, `title+description+first image`; extraction 6.05 минуты на H100, 5.86 GiB. Добавление head score к exp400 ухудшило Macro F1 на 0.016656, flammable F1 на 0.033311, увеличило FN на 7 и safety FN на 2. Это закрывает ещё один classifier/head/fusion на том же признаке, но не label-independent retrieval geometry.
4. **Старые embedding/neighbor baselines тоже отрицательны.** Qwen embedding SVM был слаб сам по себе; class centroid был особенно плох; char-TFIDF neighbor и rare-token graph priors не обогнали baseline.
5. **Family priors проверены только как label-bearing donor mechanisms.** exp170 exact/name, exp180 numeric/masked и exp190 rare-shingle BAD prior сильны прежде всего в recurrence simulation. Они используют donor labels и не доказывают generalization semantic matching. Текущий порядок — exact full text → exact name → numeric-masked → rare BAD shingle.
6. **Больше label-diverse negatives уже навредили recall.** exp410 ухудшил Macro на 0.022497 и добавил 8 FN. Это ещё один аргумент не считать визуально/семантически похожий, но не соединённый товар «отрицательным» без аудита.
7. **Semantic-family split ещё не запечатан.** Новая реализация лучше сохраняет B12/D3/CoQ10/K-206 и балансирует connected components, но graph policy ещё требует аудита generic exact/masked-name edges, auxiliary-image edges, high-degree components и ручной boundary review. Значит, сейчас нет честной поверхности для выбора contrastive pairs или оценки переноса.

## Что ещё не проверено

- Обучаемая нелинейная projection head с supervision только от **того же товара/варианта и разных его views**, без moderation label.
- Product-identity retrieval на outer-held-out semantic components, где projection, miner, index и donor priors fit только на outer-train.
- False-negative masking для потенциальных semantic-family связей и reciprocal-neighbor evidence вместо безусловного push-away всех unlinked rows.
- Высокоточная fallback route, которая не меняет exp400, если не найден reciprocal semantic-family match; прежние опыты в основном тестировали score fusion или target-aware disagreement.
- End-to-end contrastive LoRA embedding backbone. Его не надо тестировать первым: он либо меняет raw vector, которым пользуется exp400, либо требует второй foundation pass, то есть ломает дешёвую причинную проверку.

## Что говорят первичные источники

- Khosla et al., **Supervised Contrastive Learning**, NeurIPS 2020: multi-positive loss сближает образцы одного класса и отталкивает остальные, устойчивее обычной cross-entropy. Для нас moderation label не должен играть роль их class ID — иначе получится ещё один classifier, а не label-blind product embedder. [Официальная страница NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)
- Jin et al., **ECLIP: Learning Instance-Level Representation for Large-Scale Multi-Modal Pretraining in E-Commerce**, CVPR 2023: разные изображения одного product instance служат positives, другие products — negatives; это прямой прецедент для multi-view карточки. Но масштаб работы — 100M images / 12M products и известные product identities, поэтому их negative assumption нельзя переносить на наш небольшой повторяющийся набор буквально. [CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2023/html/Jin_Learning_Instance-Level_Representation_for_Large-Scale_Multi-Modal_Pretraining_in_E-Commerce_CVPR_2023_paper.html), [supplement](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Jin_Learning_Instance-Level_Representation_CVPR_2023_supplemental.pdf)
- Peeters & Bizer, **Supervised Contrastive Learning for Product Matching**, 2022: positives строятся из product IDs / matching offers; при отсутствии IDs используется source-aware sampling. Авторы также показывают ухудшение от self-supervised contrastive pretraining из-за inherent label noise. Это наиболее прямой аргумент в пользу pair-quality gate до обучения. [arXiv](https://arxiv.org/abs/2202.02098)
- Zhang et al., **Block-SCL: Blocking Matters for Supervised Contrastive Learning in Product Matching**, 2022: candidate blocking даёт более информативные hard negatives и улучшает product matching по сравнению со случайными negatives, но negative pairs у них имеют match supervision. У нас отсутствие graph edge не является таким supervision. [arXiv](https://arxiv.org/abs/2207.02008)
- Li et al., **Retrieval-Enhanced Dual Encoder Training for Product Matching**, EMNLP Industry 2023: retriever первого этапа выбирает информативные training pairs для второго dual encoder и улучшает matching на публичных и production данных. Это хороший второй этап после надёжного identity retriever, но не первый опыт на незапечатанном графе. [ACL Anthology](https://aclanthology.org/2023.emnlp-industry.22/)
- Xiong et al., **ANCE**, ICLR 2021: ANN mining даёт глобальные hard negatives вместо слишком лёгких in-batch negatives. Метод предполагает известные positives; в нашем случае его можно применять только после candidate-edge masking, иначе ближайшие «negative» товары будут как раз вероятными false negatives. [OpenReview paper](https://openreview.net/pdf?id=zeFrfgyZln)
- Zerveas et al., **Mitigating False Negatives in Dense Retrieval with Contrastive Confidence Regularization**, EMNLP 2023: sparse relevance labels превращают часть unlabeled relevant items в false negatives; reciprocal-neighbor evidence используется для сглаживания их вреда. Это прямой прецедент для mask/soft-weight, а не жёсткого отталкивания всех unlinked rows. [ACL Anthology](https://aclanthology.org/2023.emnlp-main.665/)
- Aggarwal et al., **End-to-end Multi-modal Product Matching in Fashion E-commerce**, 2024: contrastively trained projections pretrained image/text encoders дают практичный quality/cost trade-off; релевантная архитектурная поддержка projection-first, хотя fashion matching имеет более сильное identity supervision. [arXiv](https://arxiv.org/abs/2403.11593)
- Huang et al., **AFMRL**, Findings of ACL 2026: attribute-guided contrastive learning выбирает fine-grained hard samples и фильтрует noisy false negatives в identical-product retrieval. Это поддерживает semantic/attribute masking, но MLLM-generated attributes слишком дороги и усложняют первый причинный pilot. [ACL Anthology](https://aclanthology.org/2026.findings-acl.704/)
- Официальные решения Amazon KDD Cup 2022 ESCI показывают важное ограничение переноса: победители relevance ranking использовали relation-aware cross-encoders, self-distillation и ensembles, а не доказали достаточность generic item-item geometry. [Официальная страница соревнования](https://amazonkddcup.github.io/), [Task 1 winner](https://amazonkddcup.github.io/papers/9517.pdf), [Task 2/3 winner](https://amazonkddcup.github.io/papers/3782.pdf)

## Единственная новая гипотеза, которую имеет смысл оставить

**Hypothesis CE-1:** label-blind multi-view / high-precision-family contrastive projection поверх frozen Qwen3-VL embeddings улучшит поиск повторного semantic product identity на unseen components; reciprocal high-confidence matches смогут безопасно применить donor-family verdict после существующих deterministic priors, не ухудшая safety recall.

Это отличается от закрытых веток так:

- supervision — product identity / view relation, **никогда не target label**;
- обучается nonlinear projection, а не LinearSVC, centroid или diagonal target metric;
- output используется для retrieval gate после priors, а не для глобального rank fusion;
- отсутствие безопасного match оставляет prediction exp400 неизменным;
- проверяется component-held-out generalization, а не recurrence через случайно пересекающиеся families.

## Label-blind построение пар

Запрещённые сигналы при создании пар, mining, weighting и thresholding: `label`, logits/scores exp400, OOF errors, Public outcomes, fold deltas и любой derived target statistic.

### Positives

1. **P0, наиболее чистые:** две views одной строки/карточки: `(title+description+all images)` и `(text-only | title+first image | alternate gallery image)`. Это не даёт cross-row match supervision, но учит инвариантности к пропавшей modality/view.
2. **P1, cross-row только из audited strong edges:**
   - exact normalized full text; либо
   - specific exact/masked-quantity name **плюс** независимое corroboration: description similarity или exact first/product image; либо
   - exact first/product image плюс text corroboration;
   - auxiliary/perceptual image match — только при двух независимых image matches или text corroboration.
3. Mixed-label family нельзя исключать по target label: это само по себе внесёт verdict supervision в geometry. Generic/high-degree components надо quarantine по структурным признакам. Weight каждой family обратно пропорционален её размеру, чтобы один 193-row component не доминировал.

### Negatives

1. Базовый вариант — in-batch different-component negatives, но **не предполагать**, что different component означает настоящий non-match.
2. Маскировать пары, у которых есть хотя бы один candidate-relation signal: exact/masked name, rare shingle, exact/perceptual image candidate или reciprocal nearest-neighbor связь в frozen base space.
3. После первого epoch можно добавить semi-hard same-category negatives из полностью frozen, заранее построенного `TF-IDF + raw Qwen embedding` rank, например ranks 20–100 после masking. Самые близкие top ranks не брать: там максимален риск false negative.
4. ANN hard-negative refresh не использовать в первом pilot. Если intrinsic retrieval уже улучшится, его можно добавить отдельной ablation с теми же masks; иначе adaptive mining смешает две гипотезы.

### Обязательный pre-training pair audit

До любого обучения вручную проверить 300–400 пар, стратифицированных по positive-edge и negative-source, двумя blinded reviewers. Gate:

- positive identity / close-variant precision ≥98%;
- hard-negative non-match precision ≥95%;
- agreement κ ≥0.80;
- ни один источник не формирует generic/high-degree hub;
- sealed rows отсутствуют.

Если gate не пройден, решение остаётся NO-GO. Нельзя «исправлять» пары по moderation labels.

## Модель и обучение

- Backbone: разрешённый `Qwen/Qwen3-VL-Embedding-2B`, **frozen**.
- Head: `LayerNorm(2048) → Linear(2048,512) → GELU → Linear(512,256) → L2 normalize` (около 1.2M параметров, существенно меньше 10 MB).
- Loss: symmetric multi-positive SupCon/InfoNCE, фиксированный `temperature=0.07`, inverse-family weights; batch, например, 32 families × 2 views, плюс 1–2 masked semi-hard negative families на anchor.
- В первом опыте допустимы максимум два заранее заданных epoch budgets; epoch выбирается только inner folds. Никакого sweep по fusion weights.
- Raw 2048D embedding сохраняется для неизменного exp400 LinearSVC. Projected 256D используется только в новой retrieval route.

Первый projection-only pilot на cached embeddings должен занимать минуты, полный nested цикл — ориентировочно менее 1 H100-hour. Multi-view extraction потребует отдельных offline view embeddings; один полный pass уже измерен как 6.05 минуты. End-to-end Qwen3-VL LoRA сейчас NO-GO: полные fold fits порядка 50–70 минут каждый и новый inference pass/изменение raw representation затруднят честную ablation.

## Интеграция в exp400

1. Сначала отрабатывают текущие exact/name/masked/shingle priors без изменений.
2. Только если ни один prior не сработал, projected embedding ищет top-k **уникальных donor components**, не rows.
3. Match допустим лишь при reciprocal nearest-neighbor, превышении заранее выбранного label-blind similarity/gap gate и согласии top-3 donor components. Mixed/low-consensus donor verdict не применяется.
4. При допустимом match используется donor-family verdict; иначе prediction ровно exp400.

Это намеренно не continuous score fusion: такой опыт снова смешает CE-1 с закрытой логикой exp340/460. Similarity threshold калибруется только на identity/non-identity pair audit внутри inner train, а не на moderation F1 outer fold.

## Nested evaluation без leakage

1. Сначала завершить graph audit, зафиксировать component policy и один раз выделить sealed semantic-family holdout около 1/7. Sealed не участвует в pair audit, mining, vocabulary/index, calibration или checkpoint selection.
2. На development complement создать 5 outer semantic-family folds. Для каждого outer fold с нуля:
   - строить positives/negative masks только из outer-train components;
   - fit projection и любой miner только на outer-train;
   - строить retrieval index и donor-label aggregates только из outer-train;
   - выбирать epoch/similarity gate только inner component folds;
   - оценивать все outer-validation rows, включая mixed families, без unsafe exclusions.
3. Отдельно считать intrinsic retrieval на полностью held-out components/gallery: Recall@1/5, MRR/mAP, false-match rate, slices по edge source и modality. Intrinsic метрика диагностическая и не заменяет downstream F1.
4. Downstream baseline — **полностью переобученный exact exp400** на тех же outer-train components, включая 190/400 adapters и production priors. Старые adapters невалидны, если они видели sealed rows.
5. Только после прохождения development gates pre-register configs/checkpoint hashes и один раз открыть sealed. После этого holdout retired; Public leaderboard не используется для threshold tuning.

## Acceptance gates

### Development complement

- Pair-audit gates выше выполнены.
- Projection против raw Qwen embedding: Recall@1 не менее +5 pp и mAP не менее +3 pp; улучшение ≥4/5 folds; ни один заранее заданный modality/edge slice не падает >2 pp; false-match rate не хуже.
- Downstream против exact retrained exp400: mean Macro F1 ≥+0.003; wins ≥4/5; ни одна category F1 не падает >0.003; flammable FN и safety-union FN не увеличиваются; corrected:regressed ≥1.5; component bootstrap `P(delta>0) ≥ 0.90`.
- Coverage не должна быть фиктивно нулевой: отчёт обязан показывать eligible matches и изменения по каждому fold. Минимум 10 eligible validation cases/fold для интерпретируемости, но нельзя ослаблять similarity gate ради покрытия.

### Sealed one-shot

Sealed мал и содержит ориентировочно лишь ~28 flammable positives, поэтому один FN меняет recall примерно на 0.036; он годится для directional check, не для точной threshold optimization. Принять только если Macro delta ≥+0.002, ни одна category не падает >0.003, flammable/safety FN не растут и component bootstrap не показывает materially negative tail. Любое изменение config после просмотра sealed означает новый holdout, а не retune.

### Production/runtime

- Один foundation pass: дополнительный Qwen inference запрещён.
- Public runtime ≤11 минут, Private ≤26 минут; прирост архива <10 MB.
- Official smoke/schema checks проходят; raw exp400 branch побитово/численно воспроизводится для rows без retrieval match.

## Go / no-go

- **Против exp400 сейчас: NO-GO.** Не потому что механизм уже проверен, а потому что его единственный честный supervision source — semantic-family graph — пока не прошёл boundary/pair audit, а current exp400 сам требует retrain в новом protocol.
- **После graph freeze: conditional GO на один projection-only CE-1 pilot.** Он дешёвый, использует уже разрешённый/вычисляемый backbone, не требует 400B teacher, не добавляет inference pass и причинно отличается от 340, 360–390 и 460.
- **End-to-end contrastive LoRA, adaptive ANN mining и score fusion: NO-GO до прохождения projection pilot.**
- Если intrinsic retrieval проходит, а downstream gates нет, embedder остаётся только diagnostic artifact и не идёт в submission.

