# E-CUP 2026 Quality Research

Воспроизводимый research repository для классификации качества карточек товаров в категориях `БАД` и `Легковоспламеняющиеся`. Цель команды — первое место; process и evidence quality соответствуют уровню DS Master / Kaggle Grandmaster.

Лучший подтверждённый Public результат: **0.806579 Macro F1**. Текущий основной кандидат — category-specific late fusion из TF-IDF, Qwen3-VL representation, supervised Qwen3-VL LoRA и Qwen3.5 LoRA с консервативным product-family prior.

## Навигация

- [`CODEX.md`](CODEX.md) — правила работы AI-агентов.
- [`WHATS_NEXT.md`](WHATS_NEXT.md) — точка продолжения и приоритеты.
- [`docs/current-status.md`](docs/current-status.md) — текущий model/leaderboard status.
- [`docs/research/literature-and-competitions.md`](docs/research/literature-and-competitions.md) — статьи и похожие соревнования.
- [`docs/research/data-audit.md`](docs/research/data-audit.md) — исследование данных и рисков валидации.
- [`docs/operations/experiment-cli.md`](docs/operations/experiment-cli.md) — переносимый CLI contract.
- [`docs/operations/publication.md`](docs/operations/publication.md) — безопасная публикация без legacy binaries.
- [`validation/`](validation/) — frozen fold assignment и validation basket.
- [`datasets/registry.toml`](datasets/registry.toml) — immutable версии исходных данных.
- [`validation/registry.toml`](validation/registry.toml) — immutable версии evaluation protocols.
- [`experiments/`](experiments/) — изолированные experiment packages.
- [`reports/experiment-log.csv`](reports/experiment-log.csv) — полный machine-readable журнал.
- [`reports/submissions.csv`](reports/submissions.csv) — leaderboard и готовые submission candidates.
- [`research/`](research/) — legacy research scripts и JSON-отчёты; новые эксперименты должны оформляться через `experiments/`.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python validation/build_folds.py --data /path/to/data.csv
python tools/list_experiments.py
python experiments/040_late_fusion/run.py --data /path/to/data.csv --images /path/to/images
```

Все experiment runners принимают обычные filesystem paths и не зависят от конкретной compute platform. Private runner может передать те же аргументы из локального preset в `experiments/<id>/.local/compute/`, но такие presets намеренно исключены из Git. Submission ZIP и веса лежат локально в `experiments/<id>/artifacts/`.

## Политика публикации

В репозиторий не входят исходные competition data и изображения, submission ZIP archives, model weights, adapters, embeddings, joblib artifacts, vendor-копии библиотек, credentials, internal URLs, job names и private execution presets.

Публикуются код, конфигурации экспериментов, frozen split assignments, агрегированные метрики, отчёты и submission source code без весов.
