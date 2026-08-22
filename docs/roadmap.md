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
- второй независимый Qwen3.5 seed и честное среднее вероятностей;
- grouped bootstrap и подробный анализ ошибок двух seed;
- точное однопроходное объединение LoRA ранга 32, отклонённое по пяти folds;
- двухпроходная сборка с проверенным Public-временем 14.23 минуты.

## В работе

- семейно-сбалансированное обучение положительных flammable-семейств;
- семейно-разнообразный отбор положительных БАД и полное покрытие отрицательных семейств БАД;
- семейно-разнообразный отбор отрицательных flammable-примеров оставлен ниже по приоритету: у текущего лидера там только 18 ложноположительных ошибок;
- выбор единственного Public-кандидата после пятифолдового решения.

## Запланировано

- Private result logging;
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
- точное rank-32 объединение двух Qwen3.5 LoRA.
