# Что делать дальше

Обновлено: 2026-08-21.

## Текущий лидер

- Подтверждённый Public: `quality-qwen3vl-late-fusion-submit.zip`, Macro F1 **0.806579**.
- Основной готовый кандидат: Dual-LoRA + shingle prior. ZIP хранится локально и не публикуется.
- Private scores пока отсутствуют.

## Немедленная следующая задача

Второй Qwen3.5 seed `31415` завершил пять outer-fold jobs и full-data training. Нужно:

1. скачать пять OOF prediction artifacts и full-data adapter в локальное artifact storage;
2. запустить `experiments/230_qwen35_second_seed/run.py`;
3. сравнить seed 42, seed 31415, mean-rank и separate-head ensemble;
4. принять модель только при улучшении nested grouped Macro F1 относительно **0.9118425206**;
5. при принятии собрать two-adapter candidate и повторить runtime/schema smoke;
6. иначе сохранить rejected result и использовать готовый Dual-LoRA + shingle prior.

## После следующего Public результата

- записать Public F1 в `reports/submissions.csv` и tracker;
- сравнить с 0.806579 без post-hoc threshold tuning;
- выбрать второй финальный кандидат с другим failure mode;
- после открытия Private заполнить Private F1 и final selection.

## Отложенные проверки

- confirmation half для soft beta-smoothed product-family prior;
- bootstrap confidence intervals для разницы двух Qwen3.5 seeds;
- calibration sensitivity только после фиксации архитектуры;
- дополнительный OCR branch — лишь как диверсифицированный кандидат.
