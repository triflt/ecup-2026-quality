# План исследования

## Завершено

- полный аудит train/image export;
- grouped и duplicate-stress validation;
- text, embedding и late-fusion baselines;
- first-image, multi-view, cross-modal и tree heads;
- PaddleOCR hard-case branch;
- Qwen3-VL, Qwen3.5 и Gemma LoRA ablations;
- nested fusion calibration;
- exact/name, numeric-family, shingle, image-hash и neighbour priors;
- official-container schema/runtime smokes;
- Public calibration сложного и устойчивого ensemble.
- Public 0.8919244237 у Dual-LoRA с поиском похожих карточек.

## В работе

- агрегация второго Qwen3.5 seed;
- обновление primary candidate после независимой seed validation;
- проверка второй независимо обученной версии Qwen3.5;
- вероятностная оценка товарных семейств.

## Запланировано

- Private result logging;
- bootstrap significance для seed ensemble;
- устойчивое к ошибкам меток обучение;
- воспроизводимая разметка спорных случаев большой открытой моделью;
- отдельный классификатор свойств легковоспламеняющихся товаров;
- второй финальный candidate с orthogonal failure mode;
- финальный full-data refit только после фиксации architecture/configuration.

## Отклонено

- direct text/VLM prompting как основная модель;
- hard regex overrides;
- ID, character-TFIDF и rare-token neighbour priors;
- aggressive clean filtering;
- Qwen3-VL multi-image LoRA;
- Gemma в primary fusion;
- logistic meta-stacker;
- OCR поверх сильной LoRA fusion;
- image-hash prior.
