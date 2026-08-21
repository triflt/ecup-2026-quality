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

## Что подтвердилось экспериментально

1. **Late fusion переносится лучше сложного stacker.** Public 0.806579 у двухголовой fusion против 0.785500 у advanced mixed ensemble.
2. **Targeted adaptation лучше prompting.** Qwen3.5 direct VLM prompt дал 0.468403 Public; supervised Qwen3.5 LoRA улучшил nested three-head fusion.
3. **Hard mining полезнее механической очистки.** Clean-filtered Qwen3-VL на одном и том же fold существенно проиграл hard-mined варианту.
4. **Первое изображение — сильнейший visual view.** Multi-image LoRA не улучшил first-image model.
5. **Diversity важнее глобального OOF.** Gemma имела высокий global fusion score, но проиграла в nested calibration из-за нестабильного flammable fold.

## Открытые исследовательские вопросы

- Насколько ошибки второго Qwen3.5 seed декоррелированы с первым?
- Повторяются ли product families в hidden в доле, близкой к random 70/30 simulation?
- Можно ли получить orthogonal gain от OCR/attribute extraction без превышения runtime?
- Стабильно ли улучшение shingle prior после Public/Private проверки?
