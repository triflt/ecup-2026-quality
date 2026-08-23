# 644: Qwen3.8-27B, класс + проверяемое доказательство

Статус: **prompt-gate 640 пройден; запуск ждёт checksum-locked OCR-sidecar 634
и отдельный доказанный training-runtime**.

Большая evidence-ячейка строго повторяет `643`: одинаковые training IDs и их
multiplicity, prompt, первый снимок, preprocessing, два порядка auxiliary
target, веса loss, seed, optimizer и effective batch. Единственный модельный
фактор — Qwen3.8-27B вместо Qwen3.5-4B.

Это потенциальный офлайн-учитель. Сам факт успешного обучения или лучший
aggregate F1 не разрешает дистилляцию. Нужны screen `0/3`, затем победа над
`643` минимум на `4/5` folds, отсутствие опасной просадки категорий, улучшение
на заранее определённых сложных строках и слепой аудит минимум 200 объяснений.

Координаты никогда не генерируются. Генерируется только короткая цитата и
закрытый concept; точная текстовая или OCR-область восстанавливается независимо.
Runtime требует полный checksum-locked OCR sidecar `633`, но OCR-текст не
добавляется во вход модели и поэтому не меняет честное сравнение масштаба.

```bash
python experiments/644_qwen38_27b_grounded_evidence_lora/build_runtime.py --help
python experiments/644_qwen38_27b_grounded_evidence_lora/run_fold.py --help
```

Первоисточник модели: <https://huggingface.co/Qwen/Qwen3.8-27B>.
