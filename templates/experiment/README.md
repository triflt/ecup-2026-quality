# XXX: Experiment title

## Гипотеза

Одно проверяемое утверждение и ожидаемый механизм улучшения.

## Данные и preprocessing

Указать `data_version`, `evaluation_version`, входы, frozen folds, предотвращение leakage и отличие train/inference paths.

## Связь с SOTA и решениями победителей

Указать статью, решение соревнования или сильный reproducible baseline и объяснить, почему механизм должен переноситься на текущие данные.

## Acceptance criterion

Заранее определить baseline, primary metric, category guardrails, runtime limit и минимально значимый gain.

## Запуск

```bash
python3 preprocess.py --data /path/to/data.csv --output-dir /tmp/experiment
python3 run.py --data /path/to/data.csv --images /path/to/images
```

## Результат и решение

Заполняется после выполнения. Ссылка на `results/metrics.json` обязательна.
