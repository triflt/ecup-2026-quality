# Статьи и похожие соревнования

Обновлено: 2026-08-22. В таблице зафиксирован не пересказ abstract, а проверяемое следствие для текущего решения.

| Источник | Наблюдение | Решение для E-CUP |
|---|---|---|
| [Rakuten SIGIR 2020 winning solution](https://arxiv.org/abs/2008.06179) | Победила decision-level fusion отдельных text/image моделей; применялись noise reduction и несколько fusion models | Сохранять независимые modality heads и собирать low-capacity rank fusion |
| [Winning Amazon KDD Cup 2024](https://arxiv.org/abs/2408.04658) | Победители всех пяти треков объединили несколько LoRA в одну модель, ограничили выход допустимыми токенами и отдельно оптимизировали время применения | Сохранить один проход после объединения адаптеров, считать только токены `0/1` и принимать решение вместе с runtime-проверкой |
| [Amazon multi-label product categorization](https://arxiv.org/abs/1907.00420) | Title, description и image дают разные сигналы; late fusion улучшает итог | Не заменять сильный TF-IDF универсальным VLM |
| [Multimodal item categorization with Transformers](https://aclanthology.org/2021.ecnlp-1.13/) | Cross-modal attention полезен, когда изображение действительно уточняет текст | Использовать interaction head как отдельную гипотезу, но проверять distribution shift |
| [Digital leaflet product coding](https://aclanthology.org/2020.ecomnlp-1.2/) | Region detection → OCR → classifier эффективнее неструктурированного OCR | Для БАД сначала извлекать packaging text, затем классифицировать |
| [Large-scale multimodal attribute extraction](https://aclanthology.org/2023.acl-industry.29/) | Product attributes удобно формулировать как QA | VLM должен извлекать признаки, а не обязательно выдавать финальный verdict |
| [MOON2.0](https://openaccess.thecvf.com/content/CVPR2026/html/Nie_MOON2.0_Dynamic_Modality-balanced_Multimodal_Representation_Learning_for_E-commerce_Product_Understanding_CVPR_2026_paper.html) | Dynamic modality balance и sample filtering помогают noisy e-commerce data | Настраивать веса по категориям; сравнивать hard mining с clean filtering |
| [MM-LTP](https://openaccess.thecvf.com/content/CVPR2024W/MULA/html/Hu_De-noised_Vision-language_Fusion_Guided_by_Visual_Cues_for_E-commerce_Product_CVPRW_2024_paper.html) | Невизуальный и шумный product text мешает fusion | Ограничивать redundant description и сохранять decision-bearing части |
| [Learning Instance-Level Representation for E-commerce](https://openaccess.thecvf.com/content/CVPR2023/papers/Jin_Learning_Instance-Level_Representation_for_Large-Scale_Multi-Modal_Pretraining_in_E-Commerce_CVPR_2023_paper.pdf) | Instance-level alignment важен для вариаций одного товара | Отдельно моделировать product families и near-duplicates |
| [Confident Learning](https://arxiv.org/abs/1911.00068) | Out-of-sample probabilities позволяют находить вероятные label errors | Использовать для аудита, но не удалять строки без controlled ablation |
| [LoRA-Ensemble](https://arxiv.org/abs/2405.14438) | Независимые adapters могут образовывать дешёвый ensemble | Проверить второй Qwen3.5 seed и принимать только по nested CV |
| [SoREL](https://openaccess.thecvf.com/content/CVPR2026F/html/Hsieh_SoREL_Soft-Label_Refurbishment_with_Ensemble_Learning_for_Noisy_Long-Tailed_Classification_CVPRF_2026_paper.html) | Ensemble disagreement полезен при noisy long-tail labels | Сохранять soft scores и оценивать rare flammable отдельно |
| [Early-Learning Regularization](https://proceedings.neurips.cc/paper_files/paper/2020/hash/ea89621bee7c88b2c5be6681c8ef4906-Abstract.html) | Модель сначала учит общую закономерность, затем запоминает ошибочные метки | Сохранять историю обучения карточек и уменьшать влияние устойчивых противоречий |
| [DivideMix](https://openreview.net/pdf?id=HJgExaVtwr) | Распределение потерь помогает разделять вероятно чистые и шумные примеры | Сравнить две согласующиеся LoRA-модели и мягкие метки, не удаляя данные вслепую |
| [R-Drop](https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html) | Согласование двух проходов с разными dropout-масками улучшает дообучение | Недорогая проверка устойчивости Qwen-адаптеров |
| [Model Soups](https://proceedings.mlr.press/v162/wortsman22a.html) | Усреднение весов нескольких дообученных моделей может улучшить качество без роста времени применения | Проверить усреднение двух независимо обученных Qwen3.5-адаптеров |
| [AdapterSoup](https://aclanthology.org/2023.findings-eacl.153/) | Усреднение совместимых adapters может улучшать перенос без дополнительного обучения и стоимости применения | После положительного глобального alpha проверять перенос в пространстве adapters, но не считать равный вес оптимальным для редкого класса |
| [Fisher-weighted model merging](https://proceedings.neurips.cc/paper_files/paper/2022/hash/70c26937fbf3d4600b69a129031b66ec-Abstract-Conference.html) | Диагональная Fisher-важность позволяет сохранять чувствительные координаты одной модели и переносить полезные координаты другой | Если независимый seed подтверждает путь 190→260, проверять blockwise importance только на donor-строках и сравнивать с обязательным baseline alpha=0.50 |
| [LT-Soups](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6dddcff5b115b40c998a08fbd1cea4d7-Abstract-Conference.html) | Объединение general и tail-specialists улучшает компромисс между частыми и редкими классами в одном inference-efficient checkpoint | Для flammable отделять сохранение broad recall 190 от переноса редкоклассового направления 260; не возвращаться к отклонённому обучению новой головы |
| [Soup Adapters](https://arxiv.org/abs/2507.05807) | Независимые адаптеры можно объединять в один адаптер, сохраняя разнообразие ансамбля | Проверить точное объединение рангов двух LoRA одним проходом и сравнить с усреднением вероятностей |
| [LoRA Soups](https://arxiv.org/abs/2410.13025) | Простое усреднение не единственный вариант: объединение низкоранговых направлений сохраняет больше информации | Использовать конкатенацию рангов как честный однопроходный вариант, а не усреднять матрицы поэлементно |
| [Compress Then Merge](https://arxiv.org/abs/2606.03723) | Сначала уменьшить избыточность каждого адаптера, затем объединять их устойчивее, чем сжимать готовую смесь | Если ранг 32 полезен, следующим опытом сравнить его с заранее заданным сжатым рангом при том же времени применения |
| [CT-Merging](https://arxiv.org/abs/2607.20561) | Общие направления нескольких дообученных моделей можно отделить от конфликтующих | Анализировать согласованные и противоположные направления двух seed до выбора способа сжатия |
| [Class-Balanced Loss](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html) | Повторяющиеся примеры дают меньше новой информации, чем независимые | Считать баланс по товарным семействам, особенно для редких положительных flammable |
| [Logit Adjustment](https://openreview.net/pdf?id=37nvvqkCo5) | Поправка прогнозов с учётом частоты классов улучшает обучение на несбалансированных данных | Сравнить с текущими весами классов внутри вложенной проверки |
| [Noisy Student](https://openaccess.thecvf.com/content_CVPR_2020/html/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.html) | Учитель выдаёт чистые псевдометки, а ученик обучается с умеренными возмущениями | Использовать открытую большую модель для спорных карточек и обучить воспроизводимого малого ученика |
| [Efficient Learning for Product Attributes](https://openaccess.thecvf.com/content/ICCV2025W/CDEL/html/Kulkarni_Efficient_Learning_for_Product_Attributes_with_Compact_Multimodal_Models_ICCVW_2025_paper.html) | Компактная мультимодальная модель может извлекать отдельные свойства товара без большого универсального учителя | Для flammable предсказывать наличие топлива, газа и источника огня отдельными малыми головами |
| [Early-stopped neural networks are consistent](https://proceedings.mlr.press/v263/lea24a.html) | Ранняя остановка ограничивает запоминание шума при достаточном сигнале | Сравнивать семейный баланс с ранней остановкой по заранее заданному числу обновлений, не выбирая эпоху по outer fold |
| [MiniCPM-V-4.6 official model card](https://huggingface.co/openbmb/MiniCPM-V-4.6) | Компактная модель сочетает SigLIP2-400M и Qwen3.5-0.8B, поддерживает детальный режим `4x` и ориентирована на OCR и эффективный мультимодальный вывод | Проверить как независимый first-image LoRA на контрастных folds; не заменять сильные text/family компоненты до полного nested аудита |
| [InternVL3.5-2B official model card](https://huggingface.co/OpenGVLab/InternVL3_5-2B) | Модель имеет около 2,35 млрд параметров, обучалась на OCR/document data и поддерживает динамическое разбиение изображения на блоки 448×448 | Если MiniCPM не проходит, проверять не ещё один общий verdict, а структурированное извлечение признаков `газ / горючая жидкость / источник огня / пустое оборудование / содержимое комплекта`; ограничить число image tiles ради времени применения |
| [SIGIR eCom 2020 winning solution](https://sigir-ecom.github.io/ecom20DCPapers/SIGIR_eCom20_DC_paper_4.pdf) | Победившее товарное решение выиграло от разнообразия конфигураций и поздних checkpoints; decision-level fusion превзошло feature fusion | В 430 проверяется one-pass перенос комплементарности в пространство LoRA-весов; это наша проверяемая гипотеза, а не утверждение авторов о LoRA |
| [Amazon KDD Cup 2022 Task 2 winner](https://amazonkddcup.github.io/papers/3782.pdf) | Победитель объединял вероятности комплементарных моделей в товарной классификации | Комплементарность полезна, но интерполяция весов обязана отдельно пройти nested-проверку, потому что она не равна среднему вероятностей |
| [Rationale-Guided Distillation for E-Commerce Relevance Classification](https://aclanthology.org/2025.coling-industry.12/) | LLM-обоснования улучшили обучение быстрого 110M cross-encoder на мультиязычных товарных и ESCI-наборах | Получать проверяемые товарные обоснования открытой моделью и обучать отдельный быстрый evidence-layer, не отдавая ему право менять вердикт 400 |
| [Distilling Step-by-Step](https://aclanthology.org/2023.findings-acl.507/) | Метка и обоснование как две обучающие задачи позволяют малой модели использовать более богатый надзор | Проверить multi-task `label + structured rationale` после фильтрации silver-объяснений, а не обучать на свободном CoT |
| [STaR](https://proceedings.neurips.cc/paper_files/paper/2022/hash/639a9a172c044fbb64175b5fad42e9a5-Abstract-Conference.html) | Итеративное обучение сохраняет рационализации правильных ответов и строит новые с известной меткой для ошибок | Известную train-метку использовать для генерации кандидатов, но принимать только объяснения с доказательством во входе, иначе получится правдоподобное оправдание задним числом |
| [ERASER](https://aclanthology.org/2020.acl-main.408/) и [faithful rationalization by construction](https://aclanthology.org/2020.acl-main.409/) | Достоверное объяснение должно выделять достаточный фрагмент, по которому строится решение, а не только правдоподобный текст после ответа | Хранить `evidence span → concept → verdict`; проверять точность цитаты, достаточность и согласованность отдельно от Macro F1 |
| [Language Models Don't Always Say What They Think](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract.html) | Свободный CoT способен скрывать реально повлиявшие признаки и убедительно оправдывать ошибочный ответ | Не выдавать длинное рассуждение; генерировать короткое конкретное наблюдение с закрытым словарём концептов |
| [CLARITY](https://aclanthology.org/2025.starsem-1.33/) | Связка выделенного фрагмента и понятного концепта делает объяснение проверяемым и управляемым | Для БАД/flammable ввести закрытые концепты правил и разрешать комментарий только при найденном текстовом или визуальном основании |

## Что подтвердилось экспериментально

1. **Dual-LoRA с товарными семействами переносится на hidden.** Новая система получила 0.891924 Public против прежнего лидера 0.806579. Одна отправка подтверждает всю архитектуру, но не позволяет отдельно измерить вклад каждого компонента.
2. **Late fusion переносится лучше сложного stacker.** Public 0.806579 у двухголовой fusion против 0.785500 у advanced mixed ensemble.
3. **Целевое дообучение лучше прямых запросов к модели.** Qwen3.5 direct VLM prompt дал 0.468403 Public; supervised Qwen3.5 LoRA улучшил nested three-head fusion.
4. **Выбор сложных примеров полезнее механической очистки.** Отфильтрованный Qwen3-VL на одном и том же разбиении существенно проиграл варианту, обученному на сложных примерах.
5. **Первое изображение — сильнейший проверенный зрительный источник.** Multi-image LoRA не улучшил first-image model.
6. **Разнообразие ошибок важнее общей метрики на всех прогнозах.** Gemma имела высокий общий результат, но проиграла во вложенной проверке из-за нестабильности редкой категории.
7. **Второй Qwen3.5 seed полезен именно для редкой категории.** Среднее вероятностей дало nested Macro F1 0.919237 против 0.911843; у БАД число ошибок почти не изменилось, а у flammable уменьшилось с 52 до 44.
8. **Точное объединение изменений весов не эквивалентно среднему ответов.** Rank-32 LoRA без численной ошибки воспроизводит среднее изменений параметров, но из-за нелинейности сети получил только 0.909984 Macro F1 и был отклонён. Для текущих seed нужен двухпроходный вывод либо обучение отдельного ученика.
9. **Семейно-разнообразное обучение БАД изменило главным образом ошибки flammable.** В фиксированной архитектуре 190 компонент 260 дал Macro `+0.007204`: БАД `-0.002064`, flammable `+0.016473`. Разбор решений связывает выигрыш с газом, топливом и оборудованием, поэтому следующая проверка должна добавлять независимый зрительный/OCR-сигнал, а не новый seed той же модели.
10. **Локальный выигрыш одной замены всё ещё может не перенестись.** В `280`
    компонент `260` был единственным изменением относительно Public-лидера, но
    получил те же `0.8919244237` до отображаемой точности. Для нового компонента
    теперь обязательно измерять не только локальный F1, но и долю его решений,
    переживающих финальный порог и семейные правила.
11. **Дополнительное отрицательное давление ухудшает flammable recall.** В `410`
    разнообразные отрицательные семейства дали Macro `−0,022497` и FN `+8`.
12. **Короткое сбалансированное продолжение не исправляет насыщенный адаптер.**
    В `420` оба заранее выбранных fold ухудшились (`−0,026410` и `−0,007801`),
    исправлений не было, flammable FN выросли на 4 суммарно. Следующая проверка
    поэтому меняет способ объединения весов без новых градиентов.
13. **Глобальная интерполяция 190→260 даёт слабый положительный recall-сигнал.**
    В `430` alpha `0,50` выбиралась на всех пяти outer folds и дала Macro
    `+0,003185`, flammable `+0,006371`, FN `−4`. Однако было лишь `2/5`
    строгих побед, `7/5` исправлений/ухудшений и connected bootstrap `0,7334`;
    поэтому это основание для независимого seed-повтора, а не готовая модель.
14. **Простое текстовое описание смысла сделки не переносится.** В `450`
    cross-fitted представление по title и предложениям о комплекте и топливе
    снизило Macro на `0,006744`, flammable на `0,013489`, выиграло `0/5` folds
    и увеличило FN на 4. Слова и веса этой ветки повторно не подбираются.
15. **Универсальный multimodal embedding не заменяет relation representation.**
    В `460` быстрый `Qwen3-VL-Embedding-2B` прошёл 12 971 строку за 6,05 минуты,
    но линейный flammable-head дал Macro `−0,016656`, `0/5` побед и FN `+7`.
    Векторы разрешены для диагностики и retrieval, но не для повторного подбора
    классификатора на тех же folds.
16. **Текущие объяснения технически корректны, но не конкретны.** В `190/400`
    функция получает только категорию и вердикт, поэтому имеет четыре шаблона и
    не может назвать элемент конкретной карточки. После уточнения организаторов
    это отдельный финальный риск, не отражённый Public/Private Macro F1.

## Открытые исследовательские вопросы

- Улучшает ли равный вес положительных товарных семейств редкий класс без потери БАД?
- Улучшает ли отбор отрицательных flammable-примеров по независимым семействам перенос на новые группы товаров?
- Какова доля повторяющихся товарных семейств в hidden и насколько надёжны их метки?
- Можно ли получить orthogonal gain от OCR/attribute extraction без превышения runtime?
- Воспроизводится ли фиксированная alpha `0,50` пути 190→260 на независимом seed?
- Может ли donor-only blockwise Fisher сохранить broad recall 190 и перенести
  редкоклассовые блоки 260 лучше глобальной alpha `0,50`?
- Улучшает ли поиск редких общих фраз систему отдельно от Dual-LoRA?
- Можно ли воспроизводимо исправить спорные метки без большой модели-учителя?
- Можно ли получить не менее 90% релевантных evidence-grounded комментариев на
  ручном аудите 200 строк, сохранив вердикты `400` и runtime?

Подробная очередность опытов и критерии принятия находятся в [`next-research-program.md`](next-research-program.md).
