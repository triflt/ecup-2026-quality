# Второй review: evidence-grounded explanations без изменения verdict 400

Дата: 2026-08-22. Это read-only исследовательский раунд: обучение и inference не запускались, кандидат `400` и общие журналы не изменялись.

## Короткий вывод

Базовое направление в `docs/research/explanation-quality-and-reasoning.md` верное: сначала найти проверяемое свидетельство, затем сопоставить его закрытому policy-концепту и только после этого сформировать короткий комментарий, не позволяя слою объяснений менять verdict `400`.

Однако такой post-hoc слой следует называть **evidence-grounded justification, согласованным с замороженным вердиктом**, а не причинно faithful объяснением классификатора. Он может доказать, что комментарий опирается на карточку и логически поддерживает выданный verdict, но не то, что именно названный признак вызвал решение `400`. Работы о CoT показывают, что правдоподобная рационализация может скрыть реальный смещающий признак; более строгие tests of faithfulness требуют интервенций и наблюдения за распределением предсказаний ([Turpin et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract.html), [Siegel et al., ACL 2024](https://aclanthology.org/2024.acl-short.49/)).

Лучший дешёвый baseline перед любым rationale-LoRA: **verdict-locked deterministic text evidence extractor**. Он оставляет все предсказания `400` побитово неизменными, не требует второго model pass и формирует комментарий только из точной цитаты карточки плюс закрытого policy-концепта. Если безопасного свидетельства нет, baseline должен записать `NO_SAFE_EVIDENCE`, а не придумывать визуальную причину. Это честно измерит text-only coverage и выделит небольшой image-only хвост для последующей отдельной работы.

## Что в текущем документе требует усиления

1. **Groundedness, consistency и causal faithfulness смешаны.** Факт из карточки может поддерживать verdict, но быть не тем, что использовал классификатор. Для текущего решения обещать следует только: (a) factual support карточкой; (b) relevance к правилу; (c) consistency с замороженным verdict. Термин `faithful` допустим только после intervention/counterfactual test либо если verdict вычисляется исключительно из выбранного evidence.

2. **Точная подстрока необходима, но недостаточна.** Цитата может находиться внутри отрицания (`не является БАД`), условия (`подходит для баллона`), отзыва, перечисления совместимости или описания отсутствующего компонента. Нужна проверка локального контекста, полярности и transaction scope, а не только substring match.

3. **«Независимый OCR/атрибутный проход» не гарантирует визуальную опору.** FAITHSCORE декомпозирует ответ на атомарные факты и отдельно проверяет каждый факт по изображению; даже лучший из исследованных авторами open visual entailment verifiers имел около 85% accuracy на их annotated set. Авторы поэтому строят human meta-evaluation с тремя annotators и majority vote ([Jing et al., Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.290/)). Для нас visual claim должен содержать номер изображения и подтверждаться человеком; повторное утверждение другого VLM не считается независимым ground truth.

4. **Утверждения об отсутствии слишком сильны.** По одной цитате нельзя доказать «на карточке нет маркировки» или «в комплекте нет топлива». Допустима только ограниченная формулировка: «в доступных названии и описании прямое указание X не найдено». Изображения упоминаются лишь после их ручной проверки. Нужно отличать `not found in inspected sources` от абсолютного `does not exist`.

5. **Рубрика объединяет разные ошибки.** `specificity`, factual support, policy relevance, verdict agreement, coverage и readability следует размечать раздельно. Иначе красивый, но нерелевантный комментарий может компенсировать выдуманный факт средним баллом.

6. **Нет честной схемы выбора 200 строк и annotator protocol.** Выбор после просмотра объяснений создаст cherry-picking. Нужны frozen seed, family guard, заранее определённые strata, слепая рандомизация вариантов, два независимых русскоязычных reviewer и adjudication. ReproHum показал, что даже почти одинаковый protocol объяснений может дать другой ranking из-за различного понимания критерия annotators ([Gao et al., HumEval 2024](https://aclanthology.org/2024.humeval-1.25/)).

7. **Цель `<1% unsupported` статистически не подтверждается 200 строками.** Даже при 0 ошибках из 200 двухсторонняя 95% Wilson upper bound около 1.9%. Поэтому честная формулировка для этого screen: `0/200 observed critical unsupported claims`; это не доказательство population rate ниже 1%. Для lower 95% bound не ниже 90% по общему row-pass требуется не 180/200, а примерно 189/200.

8. **Rationale distillation не является доказательством explanation quality.** Близкая e-commerce работа подавала teacher rationale как auxiliary decoder objective при обучении BERT cross-encoder и удаляла decoder при inference; это изменяло encoder и классификацию. Она показала прирост ROC-AUC, но не провела human evaluation сгенерированных rationales ([Agrawal et al., COLING Industry 2025](https://aclanthology.org/2025.coling-industry.12/)). Поэтому это precedent для будущего representation supervision, а не основание считать комментарий проверяемым и не безопасный следующий шаг для замороженного `400`.

## Что реально переносится из первичных работ 2023–2025

- **E-commerce auxiliary rationale loss:** label-conditioned краткое rationale можно использовать как дополнительную обучающую цель; decoder не нужен на production inference. Для нашей задачи перенос возможен только в более позднем эксперименте и только после evidence filtering. Сейчас он не подходит: label в teacher prompt облегчает post-hoc rationalization, auxiliary loss меняет verdict path, а качество explanation в статье не проверялось человеком ([Agrawal et al., 2025](https://aclanthology.org/2025.coling-industry.12/)).

- **Не путать правдоподобие с причинностью:** CoT может рационализировать ответ и не назвать реально повлиявший bias. CCT дополнительно показывает, что human-like explanation и association с модельным решением — разные свойства. Практический перенос: в отчёте не писать `faithful to model`; отдельно размечать evidence support и prediction-explanation consistency ([Turpin et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract.html), [Siegel et al., 2024](https://aclanthology.org/2024.acl-short.49/)).

- **Atomic multimodal verification:** разбить комментарий на атомарные проверяемые claims и проверять каждый по исходной модальности. Обычные sentence-overlap metrics плохо коррелируют с human hallucination judgment; FAITHSCORE получил более высокую, но всё ещё умеренную корреляцию, поэтому auto-score остаётся triage, не gate ([Jing et al., 2024](https://aclanthology.org/2024.findings-emnlp.290/)).

- **Human evaluation должна разделять factuality, relevance, justification и usefulness.** В работе Zhan et al. эти четыре критерия оценивали отдельно, по два trained crowd workers на rationale, с отчётом Krippendorff's alpha. Это хороший минимальный каркас, но для товарной модерации `factuality` надо сделать атомарной и добавить policy scope/negation ([Zhan et al., Findings of EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.962/)).

- **Attribution к контексту помогает людям, но доверие не равно faithfulness.** Два user studies Malaviya et al. показывают, что explanations с явной атрибуцией к контексту улучшают reported understanding/trust. Для нас это поддерживает короткую цитату и указание источника, но не заменяет factual audit ([Malaviya et al., NAACL 2024](https://aclanthology.org/2024.naacl-long.168/)).

## Дешёвый baseline: verdict-locked extractive justification

Baseline выполняется после получения `prediction_400` и не имеет права возвращать новый класс.

1. Канонизировать `name` и `description`, но хранить исходную строку и offsets.
2. Извлечь предложения и короткие spans по небольшому закрытому словарю policy concepts, составленному из определения задачи, а не из validation outcomes. Приоритет: явная маркировка/состав/комплектность в title, затем description; ниже — identity товара и отрицательные или scope-фразы.
3. Для каждого кандидата проверить negation, conditional/compatibility language, субъект утверждения и transaction scope (`в комплекте`, `без баллона`, `для заправки`, `совместим с`). Тот же keyword в противоположной области контекста не является support.
4. Выбрать только span, поддерживающий уже замороженный verdict. Сформировать 50–300 символов по concept-specific template. Цитата должна быть точной, а вывод не должен добавлять вещество, состав, объём, комплектность или визуальный объект, которых нет в source span.
5. Если такого span нет или все кандидаты противоречат verdict, сохранить `NO_SAFE_EVIDENCE` и осторожный комментарий без выдуманной причины. Для development sidecar хранить `row_id`, `prediction_400`, `source`, offsets, exact span, concept, polarity, scope и fallback reason. Sidecar не является частью submission format.

Примеры допустимого стиля:

- `В описании прямо указано «БАД к пище» — это явная маркировка биологически активной добавки.`
- `Карточка описывает «горелку без баллона»; продажа топлива или заправленного баллона в комплекте не заявлена.`
- Для отрицательного evidence: `В доступных названии и описании прямое указание «БАД» не найдено; товар назван «сывороточный протеин».` Не писать, что маркировки нет на изображениях, если изображения не проверены.

Почему это полезнее немедленного rationale-LoRA: zero extra model inference, нулевой риск F1 regression, полностью воспроизводимые quotes, явный `NO_SAFE_EVIDENCE` вместо скрытой hallucination и точная оценка того, сколько final cases нельзя закрыть без visual evidence. Keyword extractor здесь **не классифицирует** и потому не повторяет неудачу `450`: он только ищет support после frozen verdict. Если найденный текст противоречит verdict, строка проваливает audit, но verdict не меняется.

## Честный 200-row validation protocol

### Выбор строк до генерации объяснений

- **160 core rows:** по 40 из четырёх клеток `category × frozen verdict_400` (`БАД/Легковоспламеняющиеся × бан/не бан`), равномерный random sample с заранее сохранённым seed. Не более одной карточки из connected product family; при нехватке cell это явно документируется.
- **40 stress rows:** по 5 на категорию в каждом из четырёх заранее заданных risk buckets: (a) image-dependent/no direct text cue; (b) negation/absence; (c) kit/component/compatibility/transaction scope; (d) ошибка `400` или ближайшая к boundary строка. Buckets disjoint по указанному priority order, без просмотра нового explanation.
- Не показывать annotators gold label, confidence, route модели или имя метода. Gold нужен только для post-hoc slice `verdict correct/incorrect`; иначе reviewer будет оценивать правильность класса вместо качества объяснения данного frozen verdict.
- На каждой строке показывать всю карточку, категорию, frozen verdict и два анонимизированных комментария: текущий generic `400` и baseline, в случайном A/B порядке. Абсолютная рубрика заполняется для каждого варианта; pairwise preference — вторичный показатель.

### Рубрика на строку

Сначала разделить комментарий на atomic claims. Любой unsupported atomic claim создаёт hard flag.

| Поле | Оценка | Инструкция reviewer |
|---|---:|---|
| `E: evidence support` | 0/1/2 | 2: каждый claim прямо подтверждён показанным текстом/изображением; 1: traceable, но неоднозначная paraphrase; 0: отсутствует или противоречит карточке. |
| `R: policy relevance` | 0/1/2 | 2: evidence действительно различает нужную категорию и учитывает negation/scope; 1: связано с темой, но недостаточно для вывода; 0: нерелевантно или правило применено наоборот. |
| `J: verdict justification` | 0/1 | Объясняет именно показанный frozen verdict без логического скачка. |
| `S: specificity/locatability` | 0/1 | Reviewer за 30 секунд находит конкретную фразу или объект; для image claim указан номер изображения. |
| `U: usefulness/clarity` | 0/1 | 50–300 символов, понятный русский, evidence и policy link различимы, нет скрытого CoT. |
| `C: coverage status` | direct text / visual / bounded absence / `NO_SAFE_EVIDENCE` | Не качество, а отдельный route для измерения coverage. |

Hard flags: выдуманный факт; изменённая «цитата»; необоснованное абсолютное отсутствие; policy inversion; несогласованный verdict; visual claim без проверяемой локализации. Общий балл не может компенсировать hard flag. Строгий row-pass: `E=2, R=2, J=1, S=1, U=1` и нет hard flags.

### Annotators и отчёт

- Два независимых русскоязычных reviewer после общей calibration на 15 **внеаудитных** примерах: один clear pass, один unsupported visual claim, negation, compatibility, kit scope и bounded absence для каждой категории.
- Все расхождения по `E`, `R` или hard flag рассматривает третий adjudicator. Сохраняются как исходные две оценки, так и adjudicated label.
- Отчёт: raw agreement; weighted Cohen's kappa для `E/R`; Cohen's kappa для hard flags и `J/S/U`; результаты отдельно по четырём core cells, четырём stress buckets, source modality, `NO_SAFE_EVIDENCE` и correctness `400`.
- Pairwise preference сообщать отдельно от factual rubric. Красивый стиль не должен перебить unsupported claim.

### Gates и статистическая оговорка

1. Mechanical: `200/200` verdict точно равен `400`; `200/200` проходят теги и длину.
2. Safety screen: `0/200` observed unsupported/policy-inversion/quote-mismatch hard flags после adjudication.
3. Quality: не менее `189/200` strict row-pass; это даёт примерно 90% lower bound Wilson 95%, тогда как `180/200` даёт лишь около 85%.
4. Coverage обязательно публикуется: доля `NO_SAFE_EVIDENCE` overall и по strata. Нельзя исключать эти строки из denominator или заменять их generic success.
5. Ни один core cell не должен иметь менее `36/40` strict pass; каждый stress bucket показывается отдельно без заявления о population prevalence.

Даже успешный screen означает только «на этой frozen выборке не наблюдались критические hallucinations». Он не доказывает `<1%` в hidden population. Если baseline не проходит, это не повод менять verdict `400`: результат должен определить, нужен ли ограниченный visual-evidence path для image-only rows или ручные concept labels на train.

## Решение по следующему шагу

Сначала сравнить текущие четыре generic comments с deterministic extractive baseline на описанном frozen-200 audit. Не обучать rationale generator до получения двух чисел: strict supported coverage и `NO_SAFE_EVIDENCE` rate. Если text-only coverage уже близка к gate, это самый дешёвый final-quality fix. Если провал сосредоточен в image-only strata, следующая работа должна быть узкой — structured visual evidence с номером изображения и human-verified train targets — а не свободный chain-of-thought и не auxiliary training, меняющее verdict `400`.
