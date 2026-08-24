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
замороженный evaluator считает полный пятиfoldовый verdict.

## Кандидаты

Машиночитаемая доска находится в
[`results/candidate_board.json`](results/candidate_board.json).

1. `fixed_equal_logit_route` — единственный ближайший кандидат, который может
   стать `submission_ready` сегодня. Веса 0.5/0.5 и threshold 0 заморожены до
   получения folds 1/2/4.
2. `optimizer_stop_step` — честная однофакторная абляция качества обучения.
   Сначала stop fraction выбирается на donor-inner folds внутри outer0-train,
   затем заранее преобразуется в целое число optimizer steps и проверяется на
   слепом outer0. Даже положительный outer0 ещё не разрешает submission.
3. `lower_learning_rate` — условный следующий эксперимент, а не активный
   кандидат. Он открывается только если trajectory не показывает раннего пика.
   Тогда меняется только LR `2e-4 -> 1e-4`; checkpoint policy, LoRA и данные
   остаются фиксированными. Это мотивировано официальным Qwen3.5 dense-LoRA
   примером, но не принимается по внешнему примеру без наших folds.

Direct 27B route и verified-OCR classification уже отклонены своими frozen
gates. Они не возвращаются в очередь только потому, что необходимые predictions
или OCR payload существуют.

## Submission policy

Количество отправок не является gate. Сегодня допустима одна новая отправка,
если и только если полный fixed-route проходит все gates и затем улучшает
неизменный production pipeline при byte-identical BAD route. Вторая отправка
возможна лишь для заранее зарегистрированной однофакторной архитектурной
гипотезы с полной честной проверкой. Ablation-only результат на leaderboard не
отправляется.

