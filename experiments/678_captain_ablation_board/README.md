# 678 — капитанская доска честных абляций

## Роль

Я — капитан исследовательского контура, владелец валидации и критик параллельного
агента. Моя ответственность — не количество запусков, а сходимость к нескольким
воспроизводимым кандидатам, каждый из которых сильнее текущего production control.

Для каждого результата я обязан:

- независимо связать prediction artifact с точными замороженными validation rows;
- проверить отсутствие label/sealed/Public leakage;
- публиковать Macro F1, category F1, flammable Average Precision, FP/FN,
  corrections/regressions и результат по folds;
- отделять `submission_ready` от `ablation_only` и `rejected`;
- не менять больше одного научного фактора в одной абляции;
- не выбирать checkpoint, threshold, calibration или blend по outer validation;
- останавливать направление сразу после заранее записанного falsification gate;
- назначать второму агенту только bounded actions с артефактом, SHA и сроком.

## Текущий критический путь

Все пять 4B control folds готовы. Folds 1 и 4 независимо перепроверены в этой
ветке; fold 2 принят параллельным контуром. Для терминальной проверки fixed
equal-logit route не хватает только 27B predictions folds 1/2/4. Эти три jobs
уже выполняются, поэтому новые GPU jobs сейчас запрещены.

После появления каждого артефакта выполняются CRC, self-hashed contract,
prediction SHA, exact fold/runtime-row binding. После третьего артефакта один
замороженный evaluator считает полный пятиfoldовый verdict. Этот verdict оценивает
только наличие полезного offline teacher-сигнала: 27B base model и её adapter не
являются deployable и не могут входить в submission.

`slice_full_candidate.py` затем строит диагностические, не участвующие в выборе
срезы: singleton/repeated families, mixed/consistent-label components,
flammable positives/negatives, OCR availability и каждый fold. Скрипт использует
те же заранее замороженные веса и threshold и не может настраивать кандидата.

## Аудит 27B runtime

Текущий 27B runner использует один Python process и
`device_map="balanced"` на четырёх H100. Это наивный model parallel, а не
DeepSpeed/FSDP/DDP. Он уже дал валидные artifacts на двух folds и не меняется у
активных jobs, но официальная документация Accelerate предупреждает, что такой
режим выполняет слои последовательно и оставляет остальные GPU простаивать.

Любой следующий новый 27B training experiment сначала проходит отдельный
runtime-only smoke с неизменным binary BCE:

1. max-shape forward/backward на одной H100;
2. если модель помещается — measured single-GPU throughput;
3. если нет — custom BCE через Accelerate + DeepSpeed/FSDP;
4. обычный ms-swift token SFT не считается retry, потому что меняет objective.

Машиночитаемый диагноз и gate записаны в
[`results/qwen27b_runtime_topology_audit.json`](results/qwen27b_runtime_topology_audit.json).

## Разбор screen corrections

Fixed blend меняет 20 строк на folds 0/3: 17 исправляет и 3 портит. На уровне
semantic families изменено 12 components: 10 содержат только исправления, одна
— только регрессии, одна mixed-label family одновременно получает исправление и
регрессию. На singleton families наблюдается `7/0`; все три регрессии находятся
в repeated families и имеют абсолютный blend margin `<0.5`. Медианный margin
исправлений `1.3125`, регрессий `0.3125`.

Это усиливает гипотезу переноса на новые families, но не отменяет полный 5-fold
gate: fold 0 дал 12/2 изменений, fold 3 — 5/1. Подробный diagnostic находится в
[`results/screen_change_family_audit.json`](results/screen_change_family_audit.json).

После применения замороженного кандидата на screen folds остаётся 12 flammable
ошибок: 10 FP и 2 FN. Четыре ошибки имеют абсолютный margin `<0.5`, но несколько
ошибок в burner/fire-starting/candle-like товарах остаются высокоуверенными.
Следовательно, глобальный threshold не является достаточным механизмом. Это
только диагностический вывод: новый rule/object-role кандидат не открывается,
пока профиль не воспроизведён на пяти folds. Полный перечень находится в
[`results/screen_residual_error_audit.json`](results/screen_residual_error_audit.json).

## Кандидаты

Машиночитаемая доска находится в
[`results/candidate_board.json`](results/candidate_board.json).
Независимая сборка полного control OOF зафиксирована в
[`results/control_score_set_audit.json`](results/control_score_set_audit.json).

1. `fixed_equal_logit_route` — только offline teacher/probe ablation. Веса
   0.5/0.5 и threshold 0 заморожены до получения folds 1/2/4, но даже полный GO
   не разрешает packaging или submission 27B.
2. `optimizer_stop_step` — честная однофакторная абляция качества обучения.
   Сначала stop fraction выбирается на donor-inner folds внутри outer0-train,
   затем заранее преобразуется в целое число optimizer steps и проверяется на
   слепом outer0. Даже положительный outer0 ещё не разрешает submission.
3. `lower_learning_rate` — ближайшая deployable однофакторная абляция. Меняется
   только LR `2e-4 -> 1e-4`; checkpoint policy, LoRA, objective, данные и 4B
   inference route остаются фиксированными. Это мотивировано официальным
   Qwen3.5 dense-LoRA примером, но принимается только по frozen screen и затем
   по полным нашим folds.

Direct 27B route и verified-OCR classification уже отклонены своими frozen
gates. Они не возвращаются в очередь только потому, что необходимые predictions
или OCR payload существуют.

Аналитическая компенсация 5-кратного oversampling редкого класса через threshold
`log(5)` также закрыта: на fixed blend она ухудшила flammable F1 на обоих
screen folds и увеличила FN. PR-AUC используется для ranking/checkpoint
selection, но приблизительный hidden prior не превращается в ручную правку
порога.

## Submission policy

Количество отправок не является gate. 27B route запрещено отправлять независимо
от его локальной метрики. Сегодня допустима новая отправка только для отдельного
deployable 4B кандидата, прошедшего frozen screen, подтверждение на оставшихся
folds, full-data refit и runtime smoke. Вторая отправка возможна лишь для заранее
зарегистрированной однофакторной архитектурной гипотезы с полной честной
проверкой. Ablation-only результат на leaderboard не отправляется.
