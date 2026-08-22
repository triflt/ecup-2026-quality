# 580: одно изображение из галереи при обучении Qwen3-VL

## Статус

`running_two_fold_screen`. Полный предварительный аудит прошёл; запущены только
folds `0` и `3`, по одной карте. Остальные folds и full train заблокированы до
результата замороженного экрана.

## Что именно меняется

Родитель — компонент Qwen3-VL из эксперимента 110 внутри неизменного маршрута
оценки 400. Выбор строк, повторы классов, порядок перемешивания, число эпох и
шагов, LoRA, оптимизатор, текст запроса и веса ансамбля остаются родительскими.

Меняется только изображение в обучающем предъявлении. Для каждой карточки
передаётся ровно одно настоящее изображение. Его номер детерминирован функцией
от `seed`, эпохи, идентификатора карточки и номера её появления. Повторные
появления циклически покрывают разные позиции галереи без изменения числа
предъявлений. Если изображение одно, поведение совпадает с родителем.

На holdout и в итоговом выполнении всегда используется только первое изображение
с ограничением 448 пикселей, один проход. Галерея не меняет время inference.

## Предварительный аудит без меток

Аудит читает только структуру архива и количество URL в манифесте. Он не читает
категорию, метку, fold или ошибки моделей. Проверяются все изображения:

- доля карточек с несколькими изображениями;
- полное декодирование без ошибок;
- повторяемость плана и циклическое покрытие галереи;
- побайтовое совпадение null-режима первого изображения с подготовкой родителя.

Фактический полный проход: 12 971 карточка, 49 456 изображений, 11 171 карточка
с несколькими изображениями (`86.12%`), ноль ошибок декодирования, ноль ошибок
детерминизма и покрытия, ноль побайтовых расхождений first-image null-control.

```bash
python experiments/580_qwen3vl_multiview_train_aug/audit_gallery.py \
  --images-zip /path/to/images.zip \
  --gallery-manifest research/multi_image_manifest.tsv.gz
```

## Разрешённая проверка

Разрешены только заранее выбранные folds 0 и 3, по одной GPU на задание:

```bash
python -u experiments/580_qwen3vl_multiview_train_aug/train_screen.py \
  --fold 0 \
  --gallery-manifest /work/input/multi_image_manifest.tsv.gz

python -u experiments/580_qwen3vl_multiview_train_aug/train_screen.py \
  --fold 3 \
  --gallery-manifest /work/input/multi_image_manifest.tsv.gz
```

Локальные однокарточные пресеты находятся в игнорируемом каталоге `.local`.
Имена заданий не содержат название соревнования.

## Оценка

Сначала должен пройти null-control, затем оба fold-артефакта оцениваются через
неизменный route-400:

```bash
python experiments/580_qwen3vl_multiview_train_aug/evaluate_screen.py \
  --null-control \
  --output-dir /tmp/q3vl-mview-null

python experiments/580_qwen3vl_multiview_train_aug/evaluate_screen.py \
  --fold-0 /path/fold0/lora_holdout_predictions.csv \
  --fold-3 /path/fold3/lora_holdout_predictions.csv \
  --output-dir /tmp/q3vl-mview-screen
```

Гипотеза проходит только если прирост каждого fold положителен, средний прирост
не меньше `0.001`, число false negative для flammable и safety-среза не растёт,
а исправленных ошибок больше, чем новых. Это reject-only gate: успешный screen
лишь разрешает полный пятиfoldовый цикл.

## Проверка кода

```bash
ruff check experiments/580_qwen3vl_multiview_train_aug \
  tests/test_qwen3vl_multiview_train_aug_580.py
pytest -q tests/test_qwen3vl_multiview_train_aug_580.py
```
