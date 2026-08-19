# E-CUP 2026 — контроль качества карточек товаров

Text-only решение для классификации карточек в категориях `БАД` и
`Легковоспламеняющиеся`. Изображения и внешние API не используются.

## Результат

- Public leaderboard: **0.7130258381** Macro Averaged F1.
- Локальный stratified holdout 22%: **0.9043**.
- Контейнер успешно прошёл проверку формата соревнования.

Разница между локальной и публичной оценкой показывает, что случайный holdout
переоценивает качество из-за повторяющихся карточек и сдвига данных.

## Модель

- word TF-IDF: униграммы и биграммы;
- character TF-IDF: `char_wb`, n-граммы 3–5;
- отдельный `LinearSVC` для каждой категории;
- отдельный F1-порог для каждой категории;
- lookup только для повторов с непротиворечивыми train-метками;
- детерминированные объяснения длиной 50–300 символов.

`label=1` означает, что заявленная категория подтверждена, поэтому итоговый
вердикт — `не бан`. `label=0` отображается в `бан`.

## Структура

```text
submission/              минимальный архив для отправки
  metadata.json
  run.py
  strong_text.joblib
  src/
train.py                 воспроизводимое обучение
Dockerfile.train         изолированная training-среда
requirements-train.txt   зафиксированные зависимости
```

Данные и изображения намеренно не входят в репозиторий.

## Обучение

Через локальную Python-среду:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
make train TRAIN_DATA=/absolute/path/to/data.csv
```

Через Docker:

```bash
make docker-train TRAIN_DATA=/absolute/path/to/data.csv
```

## Проверка и упаковка

```bash
make smoke TRAIN_DATA=/absolute/path/to/data.csv
make package
```

Полученный `quality-text-strong-submit.zip` можно отправлять в соревнование.
