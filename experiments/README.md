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
| 230 | Second Qwen3.5 seed | Running | Training завершён, aggregation pending |
| 240 | Gemma-4 LoRA SFT (QC) | Running | Multi-image LoRA, loss на вердикте; zero-shot 0.34-0.38 macro |
| 241 | Qwen3-VL contrastive (QC) | Running ablations | First attempt negative; checking SVM head, hard mining, higher r before final decision |
| 242 | Fusion grid (QC) | Prepared | Фьюжн скоров треков 240/241, ждёт их завершения |
| 900 | Infrastructure checks | Completed | Runtime/schema/preprocessing safeguards |

Новый experiment создаётся из `templates/experiment/`, затем добавляется отдельной строкой в `reports/experiment-log.csv`.
