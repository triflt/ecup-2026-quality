# Статьи и похожие соревнования

Обновлено: 2026-08-21. В таблице зафиксирован не пересказ abstract, а проверяемое следствие для текущего решения.

| Источник | Наблюдение | Решение для E-CUP |
|---|---|---|
| [Rakuten SIGIR 2020 winning solution](https://arxiv.org/abs/2008.06179) | Победила decision-level fusion отдельных text/image моделей; применялись noise reduction и несколько fusion models | Сохранять независимые modality heads и собирать low-capacity rank fusion |
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
| [Class-Balanced Loss](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html) | Повторяющиеся примеры дают меньше новой информации, чем независимые | Считать баланс по товарным семействам, особенно для редких положительных flammable |
| [Logit Adjustment](https://openreview.net/pdf?id=37nvvqkCo5) | Поправка прогнозов с учётом частоты классов улучшает обучение на несбалансированных данных | Сравнить с текущими весами классов внутри вложенной проверки |
| [Noisy Student](https://openaccess.thecvf.com/content_CVPR_2020/html/Xie_Self-Training_With_Noisy_Student_Improves_ImageNet_Classification_CVPR_2020_paper.html) | Учитель выдаёт чистые псевдометки, а ученик обучается с умеренными возмущениями | Использовать открытую большую модель для спорных карточек и обучить воспроизводимого малого ученика |

## Что подтвердилось экспериментально

1. **Dual-LoRA с товарными семействами переносится на hidden.** Новая система получила 0.891924 Public против прежнего лидера 0.806579. Одна отправка подтверждает всю архитектуру, но не позволяет отдельно измерить вклад каждого компонента.
2. **Late fusion переносится лучше сложного stacker.** Public 0.806579 у двухголовой fusion против 0.785500 у advanced mixed ensemble.
3. **Целевое дообучение лучше прямых запросов к модели.** Qwen3.5 direct VLM prompt дал 0.468403 Public; supervised Qwen3.5 LoRA улучшил nested three-head fusion.
4. **Выбор сложных примеров полезнее механической очистки.** Отфильтрованный Qwen3-VL на одном и том же разбиении существенно проиграл варианту, обученному на сложных примерах.
5. **Первое изображение — сильнейший проверенный зрительный источник.** Multi-image LoRA не улучшил first-image model.
6. **Разнообразие ошибок важнее общей метрики на всех прогнозах.** Gemma имела высокий общий результат, но проиграла во вложенной проверке из-за нестабильности редкой категории.

## Открытые исследовательские вопросы

- Насколько ошибки второго Qwen3.5 seed декоррелированы с первым?
- Какова доля повторяющихся товарных семейств в hidden и насколько надёжны их метки?
- Можно ли получить orthogonal gain от OCR/attribute extraction без превышения runtime?
- Улучшает ли поиск редких общих фраз систему отдельно от Dual-LoRA?
- Можно ли воспроизводимо исправить спорные метки с помощью большой открытой модели?

Подробная очередность опытов и критерии принятия находятся в [`next-research-program.md`](next-research-program.md).
