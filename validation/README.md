# Versioned validation sets

`registry.toml` — единственный каталог evaluation versions. Текущая frozen версия `grouped_text_v1/`: `folds.csv` содержит assignment всех train IDs в пять grouped folds, `basket.csv` — outer fold 4 для быстрых expensive-model screens, а `manifest.json` фиксирует dataset version, counts и checksums. Тексты и изображения не публикуются; IDs соединяются с локальными competition data версии `competition_train_v1`.

Воспроизведение из исходной таблицы:

```bash
python3 validation/build_folds.py --data /absolute/path/data.csv
```

Команда по умолчанию не перезаписывает существующую immutable версию. Для новой корзины задайте новые paths и `--evaluation-version`, затем добавьте запись в `registry.toml`. `--force` предназначен только для осознанного точного воспроизведения с последующей проверкой checksum.

Для точного восстановления historical folds из локального OOF cache используется `freeze_existing.py`; путь к cache не хранится в Git.

После пересборки checksums и counts в `grouped_text_v1/manifest.json` должны совпасть. Существующая версия immutable: при изменении folds создаются новый каталог и новая запись в реестре, а experiment config явно переключается на новый `evaluation_version`.

Для локальных residual/metric heads используется `component_transfer_gate_v4`.
Он сохраняет численные требования v3, но явно выбирает ровно один stability-протокол
по детерминированности модели и повторно вычисляет решения из канонических
SHA-связанных файлов. Это исключает запуск по неполному или вручную изменённому
decision JSON.
