# Experiment registry

Папки имеют стабильный numeric ID и описывают одну проверяемую hypothesis либо группу тесно связанных ablations.

| ID | Эксперимент | Статус | Главный вывод |
|---:|---|---|---|
| 000 | Text baseline | Completed | Сильный baseline, но random holdout оптимистичен |
| 010–020 | Qwen prompting | Rejected | Direct prompting не конкурентен supervised learning |
| 030 | Qwen3-VL embedding | Completed | Image signal полезен, но недостаточен один |
| 040 | Late fusion | Accepted | Лучший подтверждённый Public 0.806579 |
| 050–090 | Stress, multi-view, interaction, tree ensemble | Mixed/Rejected | Сложные heads переобучаются на train-like duplicates |
| 100 | Targeted OCR | Rejected for primary | Помогает слабой базе, не складывается с LoRA |
| 110 | Qwen3-VL LoRA | Accepted | Честный nested gain |
| 120 | Qwen3-VL LoRA prior | Completed | Recurrence hypothesis полезна, но требует отдельной оценки |
| 130–140 | Qwen3.5 and Dual-LoRA | Accepted | Лучшая validated architecture |
| 150–160 | Gemma and multi-image LoRA | Rejected | Нестабильность и dilution visual evidence |
| 170–190 | Product-family priors | Accepted selectively | Shingle BAD prior — основной готовый candidate |
| 200–220 | Alternative priors | Rejected | Gain отсутствует либо слишком мал |
| 230 | Second Qwen3.5 seed | Accepted offline | Среднее вероятностей улучшает nested Macro F1 до 0.919237; Public runtime проверен |
| 240 | Family-balanced flammable positives | Inconclusive | Все стадии готовы, но вложенные артефакты не выгружены; повтор не нужен из-за связи двух факторов |
| 241 | Isolated family-balanced positives | Running | Одна H100; все нецелевые строки совпадают с контролем; выгрузка исправлена |
| 250 | Family-diverse flammable negatives | Prepared | 1600 разных отрицательных семейств вместо 1253–1299; наследует только принятые настройки 240 |
| 260 | Family-diverse positive BAD | Running | Независимый главный эффект относительно 230; одна H100 |
| 270 | Family-balanced negative BAD | Running | Независимый главный эффект относительно 230; одна H100 |
| 900 | Infrastructure checks | Completed | Runtime/schema/preprocessing safeguards |

Новый experiment создаётся из `templates/experiment/`, затем добавляется отдельной строкой в `reports/experiment-log.csv`.
