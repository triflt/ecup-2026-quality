# Руководство для исследовательских агентов

## Цель

Мы — будущие чемпионы E-CUP 2026 Quality. Цель — первое место по Private/Final Macro F1 при полной воспроизводимости и соблюдении submission runtime. Каждый агент работает на уровне DS Master / Kaggle Grandmaster: сравнивает гипотезы с SOTA, решениями бывших чемпионов и сильнейшими публичными подходами, но принимает решения только по доказательствам на наших данных.

Перед запуском новой идеи агент обязан проверить `docs/research/foundations/literature-and-competitions.md`, указать хотя бы один релевантный precedent и объяснить механизм переноса на задачу. Локальный gain без leakage audit, независимой валидации и category guardrails не считается прогрессом к первому месту.

## Иерархия доказательств

1. Private/Final leaderboard.
2. Public leaderboard для заранее сформулированной архитектурной гипотезы.
3. Frozen nested grouped CV.
4. Image-duplicate stress split.
5. Repeated 70/30 train→hidden simulations для оценки product recurrence.
6. Single fold и random OOF — только диагностика, не model selection.

## Стандарт эксперимента

Каждая папка `experiments/<id>_<slug>/` содержит `README.md`, `experiment.toml`, `run.py`, `run.sh`, `preprocess.py`, `build_submission.py`, `results/metrics.json` и `artifacts/README.md`. В `experiment.toml` и `metrics.json` обязательны `data_version` и `evaluation_version`; ссылки разрешены только на immutable records из `datasets/registry.toml` и `validation/registry.toml`.

Platform presets хранятся только локально в `experiments/<id>_<slug>/.local/compute/`. Готовые ZIP, weights и caches находятся в `artifacts/` и не публикуются.

## DS quality bar

- Разделять model selection и final refit.
- Любой threshold/weight для outer fold выбирать только на остальных folds.
- Всегда показывать F1 отдельно для двух категорий, Macro F1 и fold dispersion.
- Для редкого flammable класса проводить error cohort analysis и считать TP/FP/FN.
- Проверять идентичность train/inference preprocessing.
- Фиксировать random seed, dataset checksum, code revision и artifact checksum.
- Предпочитать простую late fusion сложному stacker, пока независимая проверка не доказала обратное.
- Не удалять отрицательные результаты: они экономят следующие итерации.
- Не предлагать очередную generic architecture без проверки бывших победителей, свежих статей и уже закрытых отрицательных веток нашего журнала.

Актуальный публичный статус находится в `docs/current-status.md`; локальные оперативные handoff-файлы не коммитятся.
