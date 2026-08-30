# 676 — PR-AUC и аудит обучения Qwen

## Зачем

Автор задачи подтвердил, что разметка всех частей данных получена одним процессом, но отдельно
предупредил о редком классе «Легковоспламеняющиеся» и рекомендовал контролировать PR-AUC.
Эксперимент фиксирует корректную tie-aware Average Precision, фактический training recipe и
самый дешёвый честный способ проверить, не является ли финальная точка обучения неоптимальной.

Это read-only аудит: он не запускает GPU, не меняет активные jobs, веса, порог или данные.

## Подтверждённые факты

- На каждом outer fold редкие положительные строки повторяются пять раз; train prior поэтому
  существенно выше естественной доли положительного класса в donor development.
- Текущий рецепт обучается 306 optimizer steps (одна эпоха) и сохраняет только финальный adapter.
- В отчётах нет history для loss, LR, gradient norm, промежуточных checkpoint или inner PR-AUC.
- Correct Average Precision должна объединять одинаковые BF16 scores в один threshold. Item-wise
  разрыв ties даёт оптимистичное и несовместимое число и запрещён.
- На двух завершённых screen folds 27B и фиксированный equal blend улучшают ранжирование
  flammable. Это не является разрешением менять порог или веса.

## Изменение после комментария автора

PR-AUC становится обязательной диагностикой редкого класса:

1. outer-fold PR-AUC всегда публикуется рядом с F1, FP и FN;
2. checkpoint/early stopping в следующем эксперименте выбирается только по grouped donor-inner
   PR-AUC внутри обучающей части outer fold;
3. порог и calibration не выбираются по outer validation или Public;
4. приблизительное сходство class balance не превращается в предположение о точном hidden prior.

## Следующий дешёвый screen

После терминального пятиfoldового результата frozen route проверить только момент остановки на
4B, сохранив данные, BCE objective, LoRA, LR, scheduler и preprocessing неизменными. Сохранять
steps `51/102/153/204/255/306` и оценивать их на четырёх grouped donor-inner splits.

Gate:

- non-final checkpoint выигрывает минимум 3/4 inner screens;
- средний прирост flammable AP не меньше `+0.010`;
- падение BAD AP не больше `0.002`;
- при непрохождении 27B screen не запускать.

Это проверяет один фактор. Rank/alpha, target modules и learning rate нельзя менять в том же
эксперименте. Переход на обычный SFT stack также отдельная научная гипотеза, поскольку меняет
binary BCE objective на token cross-entropy.

## Результат

Машиночитаемый результат: [`results/metrics.json`](results/metrics.json).

Решение: `PREPARE_DONOR_INNER_PR_AUC_DYNAMICS_SCREEN_WAIT_FOR_TERMINAL_659`.
