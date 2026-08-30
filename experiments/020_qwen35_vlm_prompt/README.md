# 020: Qwen3.5 direct multimodal prompting

## Гипотеза

Прямой VLM prompt сможет применить правила к тексту и изображениям.

## Протокол

Dataset: `competition_train_v1`. Основной evaluation protocol: `grouped_text_v1` из `validation/registry.toml`. Frozen grouped folds лежат в `validation/grouped_text_v1/folds.csv`; global OOF и single-fold scores помечаются как диагностические.

## Связь с SOTA и решениями победителей

Перед продолжением гипотезы агент обязан проверить `docs/research/foundations/literature-and-competitions.md`, найти сильный precedent и явно описать, почему он должен переноситься на наши данные. Цель — уровень первого места, а не локальное улучшение без независимой проверки.

## Запуск

```bash
python3 experiments/020_qwen35_vlm_prompt/preprocess.py --data /path/to/data.csv --output-dir /tmp/020_qwen35_vlm_prompt
python3 experiments/020_qwen35_vlm_prompt/run.py --data /path/to/data.csv --images /path/to/images
```

Entrypoint: `experiments/020_qwen35_vlm_prompt/submission/run.py`. Platform-specific preset должен только вызвать этот runner и хранится локально в `experiments/020_qwen35_vlm_prompt/.local/compute/`.

## Submission

Source directory: `experiments/020_qwen35_vlm_prompt/submission`. Локальный ZIP строится через `python3 experiments/020_qwen35_vlm_prompt/build_submission.py --output /tmp/submission.zip`.

## Результат

Отклонено: Public 0.468403; generative verdict нестабилен.

Подробные machine-readable результаты находятся в `results/metrics.json` и `reports/experiment-log.csv`.
