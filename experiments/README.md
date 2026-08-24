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
| 230 | Second Qwen3.5 seed | Rejected online | Nested `+0.007394`, но Public 0.863946 против 0.891924 у 190 |
| 240–241 | Family-balanced flammable positives | Rejected | Изолированный 241 дал `+0.005868` к 230 и только 3/5 побед; базовый 230 сам не переносится |
| 250 | Family-diverse flammable negatives | Superseded | Исходный рецепт смешивал факторы; чистая проверка перенесена в 410 |
| 260 | Family-diverse positive BAD | Component only | Лучший isolated-компонент: locked `+0.007204`, но точечная Public-замена 280 дала ничью |
| 270 | Family-balanced negative BAD | Rejected | Locked `−0.000216`, 3/5 побед |
| 280 | Public-190 with component 260 | Rejected online | Public 0.891924: точная ничья с 190 |
| 290–300 | MiniCPM and InternVL screens | Rejected | Независимые vision/attribute ветки не прошли заранее заданные фильтры |
| 310–390 | Soft targets, evidence, gates and metric learning | Rejected/diagnostic | Ни одна ветка не дала устойчивого production-safe улучшения; детали в карточках и журнале |
| 400 | Category-routed Qwen3.5 | Public champion | Public **0.8922900011**, `+0.0003655774` к 190; локальный масштаб эффекта был переоценён |
| 410–420 | Flammable sampling/continuation | Rejected | Дополнительные градиенты увеличили false negatives |
| 430 | One-pass adapter soup | Rejected, promising | Alpha 0.50 дала `+0.003185`, но только 2/5 побед и слабый bootstrap |
| 440–470 | R-Drop, transaction scope, embedding and seed repeat | Rejected | Независимые проверки не прошли замороженные gates |
| 480 | Fisher blockwise soup | Paused | Не запускать до новой semantic-family validation |
| 490 | Evidence-grounded explanations | Research prepared | Нужен verdict-locked span/concept baseline и слепой ручной аудит 200 строк |
| 600–631 | Новая semantic-v3 проверка Qwen | Mixed/Rejected | Лучшие component gains не пережили full-route stability или Public; детали в карточках |
| 632–635 | Exact-span reasoning repeat | Rejected in full route | Компонент воспроизвёлся, но интеграция в 140 не прошла bootstrap/stability gate |
| 633–660 | Offline OCR и строгий repair | Partial dataset only | 41 691 изображение доступны fail-closed; тяжёлый OCR запрещён в submission-runtime |
| 640–659 | Масштаб Qwen и 27B LoRA | Offline teacher accepted | 659 выиграл 5/5 для опасного класса, но 27B не deployable |
| 661 | Полный контроль Qwen3.5-4B | Accepted control | 11 118 строк, immutable five-fold input для 659 |
| 662 | Outer-train scoring Qwen3.6-27B | Controlled smoke retry | Только label-free train-targets для честной дистилляции; первый архив отвергнут по лишним macOS-метаданным |
| 679 | Qwen3.5-4B с LR 1e-4 | Rejected at screen | Один fold вырос, второй упал; PR-AUC опасного класса ухудшился |
| 680 | Flammable-only Qwen3.5-4B | Rejected at screen | Положительный fold 0 не перенёсся на fold 3; Macro и F1 редкого класса снизились |
| 681 | Outer-safe 27B→4B distillation | Prepared | Ближайший deployable перенос 27B-сигнала без second-level leakage |
| 900 | Infrastructure checks | Completed | Runtime/schema/preprocessing safeguards |

Новый experiment создаётся из `templates/experiment/`, затем добавляется отдельной строкой в `reports/experiment-log.csv`.
