# connected_family_guard_v2

Эта версия не меняет исторические fold ids. Она добавляет неизменяемую маску для
честного отбора локальных методов поверх уже сохранённых OOF-прогнозов.

Внутри каждой категории строки объединяются в связные компоненты по двум рёбрам:

- одинаковый `group_hash` нормализованного полного текста;
- побитово одинаковый fp16 embedding первого изображения.

Если компонента присутствует более чем в одном историческом fold, все её строки
получают `safe_for_selection=false`. Такие строки нельзя использовать в метрике,
bootstrap или как donors/prototypes/hard negatives для nearest, residual и gate-
методов. Осталось 11 207 безопасных строк из 12 971; среди них число компонент,
пересекающих folds, равно нулю.

Воспроизведение:

```bash
uv run --no-sync python validation/build_connected_guard.py \
  --folds validation/grouped_text_v1/folds.csv \
  --first-image-embeddings research/first-image-artifacts/extracted/train_embeddings_fp16.npz \
  --output-dir /tmp/connected_family_guard_v2
```

Сгенерированный `/tmp/connected_family_guard_v2/rows.csv` должен иметь ту же
SHA-256, что записана в `manifest.json`. Исторические результаты пересчитаны без
нового отбора в `existing_candidate_audit.json`; ретроспективный аудит не может
повысить статус ранее отклонённого кандидата.
