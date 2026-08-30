# Стоит ли сейчас запускать GRPO для объяснений

Дата: 2026-08-22. Режим: независимый read-only аудит. Модели и private compute platform jobs не
запускались; общие журналы и карточки экспериментов не изменялись.

## Решение

**Сейчас GRPO не запускать.** Сначала нужны:

1. детерминированный извлекатель точного evidence и frozen-200 аудит;
2. короткий grounded SFT / rationale-distillation baseline;
3. независимое доказательство, что автоматический reward отличает содержательно
   хорошее объяснение от формально правильного не хуже людей.

После этих этапов допустим один маленький **text-only GRPO screen на
`Qwen3.5-4B`**, причём не для свободной цепочки рассуждений, а только для выбора
точной цитаты и закрытого policy-concept. GRPO на `Qwen3-VL-2B` сейчас — **no-go**:
визуальные утверждения нельзя надёжно проверить автоматически, а второй VLM или
OCR не является ground truth.

Это не общий вывод «GRPO не работает». У GRPO есть сильное место: много разных
траекторий и точный дешёвый проверяющий, как в математике или коде. Наша
содержательная цель другая: конкретное доказательство товара, правильная область
действия отрицания/комплектности и ручная релевантность. Пока reward покрывает
только часть этой цели, RL с высокой вероятностью оптимизирует именно щель между
reward и тем, что увидит жюри.

## 1. Что именно показали исходные работы

### 1.1. Почему GRPO сработал у DeepSeekMath

[DeepSeekMath](https://arxiv.org/abs/2402.03300) ввёл Group Relative Policy
Optimization как более экономную по памяти альтернативу PPO для математического
reasoning. Для одного вопроса политика создаёт группу ответов, reward каждого
ответа сравнивается со средним reward группы, а отдельный critic не обучается.
Это особенно удобно, когда правильность результата можно однозначно проверить.

Наше отличие принципиально: `label` проверяем, но он уже заморожен и explanation
не имеет права его менять. Точная подстрока и формат также проверяемы. А вот
вывод «эта фраза действительно обосновывает правило товара» не сводится к exact
match. Поэтому переносить успех математического GRPO только по названию метода
нельзя.

### 1.2. R1-Zero не является рецептом «RL без подготовки всегда лучше»

Официальная работа
[DeepSeek-R1](https://www.nature.com/articles/s41586-025-09422-z) использовала
для reasoning rule-based accuracy и format rewards и сознательно отказалась от
нейронных reward-моделей на reasoning-задачах из-за наблюдаемой уязвимости к
reward hacking. R1-Zero без SFT показал poor readability и смешение языков.
Полный R1 уже был многоступенчатым: тысячи cold-start примеров, RL, rejection
sampling, новый SFT и ещё один RL-этап. В той же работе direct distillation в
малые Qwen-модели оказался сильнее применения RL к ним.

Следствие для нас: четыре общих комментария `400` — не cold start для grounded
reasoning. Zero-style RL начал бы исследование из поведения, в котором модель не
умеет указывать evidence schema, offsets, scope или безопасно отказываться.

### 1.3. Известные отказы R1-Zero/GRPO совпадают с нашими рисками

[Understanding R1-Zero-Like Training](https://arxiv.org/abs/2503.20783)
показывает bias стандартного GRPO к увеличению длины, особенно у неправильных
ответов, и предлагает length-unbiased Dr. GRPO. Для финального комментария
допустимо только 50–300 символов; длинное «рассуждение» не является ценностью.

[Does Reinforcement Learning Really Incentivize Reasoning Capacity?](https://arxiv.org/abs/2504.13837)
показывает на математических, программных и visual reasoning задачах, что RLVR
часто повышает pass@1, перераспределяя вероятность уже существующих траекторий,
но не расширяет множество решаемых задач при большом pass@k; distillation, в
отличие от RL, может вносить новые знания. Для нас это означает: если
`Qwen3.5-4B` ещё не выдаёт grounded evidence, нет основания ждать, что неполный
reward сам создаст это умение.

Наконец, обычный GRPO получает нулевой или почти нулевой полезный градиент, когда
все ответы группы имеют одинаковый reward. У нас сильный бинарный классификатор,
known frozen verdict и несколько легко соблюдаемых format checks. Группы быстро
станут «все прошли» или «все провалились», то есть дорогие rollouts не дадут
сравнения.

### 1.4. Что известно о не полностью проверяемых задачах

[NOVER](https://aclanthology.org/2025.emnlp-main.378/) прямо начинает с
ограничения RLVR: для свободных text-to-text задач надёжного внешнего verifier
обычно нет. Работа строит proxy reward из perplexity ground-truth ответа при
сгенерированном reasoning и показывает, что RL возможен на стандартных SFT
данных. Но ей всё равно нужен эталонный ответ, а proxy оценивает его
предсказуемость, не фактическую опору объяснения на товар.

Более свежий систематический анализ
[Likelihood-Based Reward Designs for General LLM Reasoning](https://arxiv.org/abs/2602.03979)
сравнивает такие rewards на проверяемых и непроверяемых задачах. На
непроверяемых задачах log-probability reward лишь сравнялся с SFT, probability-
варианты часто не учились, а chain-of-thought сокращался примерно до десяти
токенов. Warm start сохранял длину, но при разумном бюджете всё равно не
превосходил SFT. Это прямое предупреждение: если reference comment уже есть,
cross-entropy SFT — более простой и по меньшей мере не худший первый опыт.

[VerIF](https://aclanthology.org/2025.emnlp-main.1542/) добился RL для instruction
following, но построил примерно 22 тысячи примеров с verification signals и
соединил rule checks с большим reasoning-verifier вроде QwQ-32B. Наши 200
ручных строк — честный audit, но не обучающий reward dataset; использовать их и
для reward, и для финального gate означало бы утечку.

## 2. Почему SFT/rationale distillation ближе к нашей задаче

[Distilling Step-by-Step](https://aclanthology.org/2023.findings-acl.507/)
использует rationale как дополнительную supervised-задачу. Это даёт модели
положительный пример того, **что именно** она должна извлечь, вместо косвенного
наказания после целой генерации.

Ещё ближе
[Rationale-Guided Distillation for E-Commerce Relevance Classification](https://aclanthology.org/2025.coling-industry.12/):
LLM-rationales служили auxiliary objective для небольшого товарного
cross-encoder и улучшали ROC-AUC на многоязычных e-commerce данных. Это не
доказательство качества комментариев и не наша метрика, но это прямой precedent
для схемы «товарная метка + короткая rationale supervision».

Для моделей порядка 2–4B длинные chain-of-thought особенно сомнительны:
[Small Models Struggle to Learn from Strong Reasoners](https://aclanthology.org/2025.findings-acl.1301/)
показывает, что 3B-модели не всегда извлекают пользу из длинных CoT и лучше
учатся на коротких, более простых цепочках. Наш естественный target и так
короткий: `evidence span → closed concept → verdict-locked comment`.

Таким образом, SFT здесь не «менее продвинутый GRPO», а более прямой estimator
нужного поведения. RL становится полезным только после SFT, если остаётся
измеримая задача выбора между несколькими допустимыми evidence и reward умеет
эти варианты честно ранжировать.

## 3. Сопоставление с текущим состоянием проекта

### Доступный supervision

- 12 971 бинарная gold-метка, но не 12 971 объяснение;
- только 198 положительных train-строк `Легковоспламеняющиеся`, поэтому
  category/label reward крайне несбалансирован;
- ровно четыре generic comments в `400`, использующие только category и verdict;
- ручной аудит примерно 200 строк, который должен остаться final evaluation;
- требование организатора: конкретный элемент карточки, а не общее правило;
- explanation не имеет права менять verdict;
- большого учителя до 400B в предлагаемом опыте нет.

### Что из этого следует для reward

Если награждать совпадение с gold-label, модель учится классификации, а не
объяснению, и создаёт риск изменить сильный verdict path. Если передать frozen
verdict в prompt и награждать его повторение, reward тривиален. Если награждать
только длину/JSON/наличие цитаты, модель может копировать нерелевантный span. Если
добавить LLM-as-a-judge, 200 ручных строк недостаточно, чтобы доказать его
надёжность и устойчивость к оптимизации.

Безопасно проверяемая часть задачи уже почти полностью задаёт supervised target:
точный span, offsets, concept enum и детерминированный renderer. Если reward
требует точного совпадения с этим target, GRPO решает ту же задачу дороже, чем
teacher-forced SFT. Если reward разрешает более свободный текст, появляется
непроверяемое пространство для reward hacking.

## 4. Один честный staged experiment

Рабочее название: `grounded_reasoning_ladder_v1`. Это один заранее
зафиксированный протокол с тремя gates, а не три независимо отбираемых опыта.
Следующий этап открывается только после успеха предыдущего.

### Stage A — extractive baseline без GPU

Использовать спецификацию
`research/agent_notes/explanation_baseline_implementation_design.md`:

- полный OOF-аналог verdict `400` после donor-only prior replay;
- frozen-200: 160 core + 40 stress, одна connected-family на строку;
- закрытый concept vocabulary;
- точный surface span, negation/scope checks и `NO_SAFE_EVIDENCE`;
- renderer 50–300 символов без изменения verdict.

Сначала получить два числа: strict supported coverage и долю
`NO_SAFE_EVIDENCE`. Если deterministic baseline уже проходит 189/200 с нулём
critical unsupported claims, обучать генератор не нужно.

### Stage B — короткий grounded SFT

#### Данные

Выбрать до 2 000 connected-family-diverse строк **вне frozen-200 components**:

- все доступные безопасные positive flammable families, остальные клетки
  выровнять по category × verdict × concept;
- один уникальный row-target на family; повторная экспозиция редкого flammable
  target допустима не более трёх раз только в sampler, а не как новая семья;
- `SAFE` targets создаёт frozen deterministic extractor;
- `NO_SAFE_EVIDENCE` является отдельным abstention target и не заменяется
  выдуманной причиной;
- никаких внешних данных и большого учителя;
- 15 calibration примеров и все frozen-200 components исключены из обучения.

Target — не свободный CoT:

```json
{
  "source": "name|description|none",
  "start": 0,
  "end": 0,
  "span": "exact surface substring",
  "concept": "closed enum|null",
  "status": "SAFE|NO_SAFE_EVIDENCE",
  "comment": "50-300 chars"
}
```

Verdict передаётся как неизменяемый input и дописывается только
детерминированным renderer. Gold-label не является generative target.

#### Модель

- `Qwen3.5-4B`, отдельный rank-16 explanation LoRA;
- text-only name+description на первом шаге;
- одна эпоха, teacher-forced cross-entropy, max output 192 tokens;
- classifier/adapter `400` не обновляется и не загружается в optimizer;
- exact same prompt/schema используется затем в optional GRPO.

SFT проходит к следующей стадии только если на отдельном family-held-out
development set:

- 100% verdict lock и contract validity;
- 100% exact-span validity среди `SAFE`;
- ноль critical unsupported claims в ручной проверке 100 строк;
- strict pass не менее 90%;
- `NO_SAFE_EVIDENCE` публикуется, а не удаляется из denominator.

Если SFT хуже deterministic baseline, лестница заканчивается. Если SFT уже
проходит final frozen-200 gate, GRPO также не нужен.

### Stage C — reward qualification

До RL взять 200 **не frozen-200** development prompts и сэмплировать по четыре
ответа SFT-модели (`temperature=0.8`, `top_p=0.95`, max 192 tokens). Сохранить
все 800 ответов. Это preflight, а не обучение.

Reward допускается к RL, только если:

1. среди 200 случайно выбранных reward-positive ответов два человека находят
   `0` critical unsupported/scope errors;
2. не менее 95% reward-positive ответов получают human strict pass;
3. не менее 15% и не более 80% prompt-groups имеют разные rewards; вне этого
   диапазона GRPO либо почти не получает advantage, либо стартовая политика
   слишком плоха;
4. automatic reward и human strict-pass согласуются не менее чем в 95% случаев;
5. словарь, parser, renderer и reward заморожены с SHA-256 до RL;
6. reward-qualified rows не пересекаются с final frozen-200 components.

Если reward — просто exact match одного reference target, Stage C автоматически
заканчивается `NO-GO`: такой сигнал эффективнее оптимизировать SFT.

### Stage D — условный micro-GRPO, только text evidence

Единственная предварительно заданная конфигурация:

```text
base checkpoint: Stage-B Qwen3.5-4B explanation LoRA
train prompts: <= 2,000, no frozen-200 components
group size: 4
temperature/top_p: 0.8 / 0.95
max generated tokens: 192
epochs: 1
optimizer updates: capped at 250
LoRA: separate rank 16 explanation adapter
learning rate: 5e-6
clip range: 0.2
KL beta: 0.02 against frozen Stage-B SFT reference
loss: length-unbiased Dr-GRPO equivalent; no response-length normalization
seed: one predeclared seed for screen; confirmatory repeat only after a pass
```

Не выводить и не обучать `<think>`/свободный CoT. Policy выбирает только
`span + concept + status`; comment строит frozen renderer.

## 5. Reward design

### 5.1. Lexicographic hard gates вместо удобной суммы

Сначала применяются hard invalidators. Любой из них даёт `-1` и обнуляет все
положительные части:

- output не разбирается или содержит лишние поля/текст;
- output пытается изменить frozen verdict;
- source/offset/span не являются точным surface slice;
- concept отсутствует в словаре или несовместим с category/verdict;
- negation, inclusion/exclusion, compatibility или component scope применены
  наоборот;
- comment не равен frozen template для выбранного span/concept;
- модель утверждает visual evidence в text-only режиме;
- `SAFE` поставлен при конфликтующем сильном evidence;
- длина comment вне 50–300.

Только для прошедшего ответа reward равен:

```text
1.00  direct explicit evidence with complete subject+scope
0.90  valid explicit exclusion/negation
0.80  valid bounded-absence evidence
0.60  honest NO_SAFE_EVIDENCE when no safe candidate exists
0.00  NO_SAFE_EVIDENCE despite an unambiguous compatible direct candidate
-1.00 any hard invalidator
```

Дополнительных rewards за длину, число шагов, уверенный стиль, редкие слова или
совпадение с gold-label нет. Несколько разных точных spans могут получить
одинаковый reward; это единственная потенциальная причина использовать policy
optimization вместо exact-target SFT.

### 5.2. Почему нельзя добавить LLM-judge reward сейчас

Ручная цель жюри включает factuality, policy relevance и конкретность. Judge,
который видел ту же rubric, может любить правдоподобный стиль, но пропускать
transaction-scope ошибку. При многократной RL-оптимизации даже редкая false
positive область reward становится основной стратегией policy. Официальный R1
по этой же причине не использовал neural reward model для проверяемого
reasoning. Пока нет отдельного большого judge-dataset и adversarial validation,
LLM-judge допустим только как диагностика после обучения, не как reward.

## 6. Anti-reward-hacking gates

### До обучения

- reward code не импортирует `label`, model score, fold outcome или Public data;
- все reward-positive template claims атомарны и восстанавливаются из source;
- frozen trap set содержит вставку/удаление `не`, `без`, `в комплекте`,
  `приобретается отдельно`, `чехол для X`, перестановку title/description и
  конфликтующие утверждения;
- на каждой мутации reward должен либо изменить scope/concept, либо отклонить
  ответ; verdict остаётся неизменным;
- exact-span verifier тестируется против Unicode/HTML/offset и tag injection;
- отдельно измеряется доля однородных групп и zero-advantage updates.

### Во время обучения

Каждые 25 updates сохранять, но не использовать для optimizer:

- train reward и fixed development human-proxy reward;
- contract/exact-span/scope pass rates по отдельности;
- mean, p95 и maximum output length;
- `NO_SAFE_EVIDENCE` rate;
- долю четырёх одинаковых rewards внутри группы;
- KL к SFT reference;
- частоту повторяющихся шаблонов, смешения русского/английского и копирования
  нерелевантных длинных фрагментов.

Немедленный stop, если proxy reward растёт, а human-qualified strict proxy падает
более чем на 3 п.п.; если critical false-positive появляется на trap set; если
KL/длина уходят за заранее заданный коридор; либо если более 80% групп два
контрольных окна подряд имеют одинаковый reward.

### После обучения

- classifier predictions `400` должны совпасть побитово на всех 12 971 строках;
- SFT и GRPO сравниваются на одних frozen-200 IDs, в слепом A/B порядке;
- два reviewer не видят gold, method, reward, confidence и sample bucket;
- отдельно показываются explanation-only и
  `strict explanation AND verdict==gold` end-to-end результаты;
- все `NO_SAFE_EVIDENCE` остаются в denominator;
- отчёт содержит результаты по category×verdict, stress bucket, correct/error
  verdict и connected-family size.

GRPO считается полезным относительно SFT только если одновременно:

- `0/200` critical unsupported/policy-inversion/quote errors;
- net gain strict-pass не менее 8 строк из 200;
- paired exact test или component bootstrap даёт `p <= 0.05`;
- `NO_SAFE_EVIDENCE` уменьшается хотя бы на 5 п.п. без роста unsupported claims;
- ни одна core-клетка не падает более чем на 5 п.п.;
- verdict/F1 `400` не меняются ни на одной строке;
- официальный runtime smoke проходит с 20% запасом.

Неуспех Stage D отвергает только GRPO; Stage A/B могут оставаться полезными.

## 7. Оценка вычислений и времени

Это planning range, а не измеренный GRPO benchmark.

Локальные опорные измерения:

- full-data Qwen3.5-4B SFT: 6 790 records, 425 updates, одна H100 — **37.8 мин**;
- Qwen3-VL-2B R-Drop fold: 5 390 records, 337 updates, одна H100 — **19.5 мин**;
- текущий двухпроходный `230` inference: 600 товаров — **320 с**; projected
  Private 3 800 — **33.8 мин** из 40, уже без полноценной генерации explanation.

Отсюда разумный бюджет:

| Этап | Объём | Плановый compute |
|---|---|---:|
| deterministic extractor + sampler | 12 971 text rows | CPU, минуты |
| Stage-B Qwen3.5 SFT | до 2 000 коротких targets, 1 epoch | 0.3–0.8 H100-hour |
| Stage-C preflight | 800 text rollouts × ≤192 tokens | 0.2–0.6 H100-hour |
| Stage-D micro-GRPO | 2 000 prompts ×4 rollouts, ≤250 updates | **4–10 H100-hours**, ориентир 2 H100 × 2–5 wall-hours |
| тот же screen с изображениями/Qwen3-VL | 8 000 multimodal rollouts | **12–24 H100-hours** до ручной проверки |
| наивный full 12 971-row GRPO | ≥51 884 rollouts за эпоху | примерно 25–60 H100-hours text-only; не оправдан |

GRPO дороже SFT не только из-за четырёх генераций: нужны autoregressive rollout,
policy log-probabilities, reference/KL pass и backward. Перед любой Stage-D job
обязателен 50-prompt measured benchmark; если экстраполяция выходит за верхнюю
границу, job не запускать.

Production runtime — отдельный blocker. Полная генерация до 192 tokens на каждой
hidden-карточке почти наверняка не помещается в текущий запас `400/230` без
измерений. Даже успешный обучающий опыт можно использовать только:

- offline для улучшения evidence targets; или
- как specialist на небольшой, label-blind доле `NO_SAFE_EVIDENCE` строк;
- после официального 600-row routed smoke и projected Private с 20% запасом.

Deterministic renderer остаётся предпочтительным production path.

## 8. Отдельное решение для двух моделей

### `Qwen3.5-4B`

- **Сейчас:** grounded SFT после extractive baseline.
- **Позже:** условный text-only micro-GRPO, только если reward qualification
  пройдена и SFT оставляет неоднозначные, но автоматически проверяемые варианты.
- **Не делать:** GRPO по gold label, свободный CoT, общий comment judge, изменение
  classifier adapter `400`.

### `Qwen3-VL-2B`

- **Сейчас:** не GRPO; максимум structured visual SFT на human-verified boxes/OCR
  после отдельного пилота.
- **Причина:** exact text verifier не проверяет видимый объект, OCR-текст или
  «товар в комплекте» на изображении. Согласие второго VLM не делает claim
  истинным.
- **Условие будущего GRPO:** сотни независимых human-verified image regions,
  закрытые attributes, deterministic bbox/OCR checks и отдельный visual trap
  set. До этого visual reward hacking вероятнее реального улучшения.

## Итоговый go/no-go

| Решение | Сейчас | Что должно измениться |
|---|---|---|
| Extractive evidence + frozen-200 | **GO** | ничего; это следующий шаг |
| Короткий grounded SFT / rationale distillation | **GO после Stage A** | нужны проверенные targets вне audit families |
| Qwen3.5-4B text micro-GRPO | **CONDITIONAL NO-GO** | SFT pass, честный reward, mixed group rewards, reward-human agreement ≥95% |
| Qwen3-VL-2B visual GRPO | **NO-GO** | нужны human-verified visual evidence и verifier dataset |
| R1-Zero-style свободный reasoning GRPO | **NO-GO** | не соответствует 50–300, verdict-lock и ручной evidence rubric |

Самая честная перспективная гипотеза сейчас — не «научить модель дольше думать»,
а научить её стабильно выбирать короткое проверяемое доказательство. Если
детерминированный extractor и SFT не могут это сделать, текущий reward тем более
не содержит информации, из которой GRPO мог бы создать правильное reasoning.
