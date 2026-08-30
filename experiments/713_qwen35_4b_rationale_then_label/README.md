# 713: Qwen3.5-4B — rationale first, label last

Статус: **pipeline и teacher corpus готовы; вторичная самостоятельная гипотеза
ждёт явного user approval**.

Это прямой тест предложения пользователя: Qwen3.5-4B обучается causal CE на
всей последовательности, а метка стоит в самом конце и является последним
символом до EOS:

```text
{"reason":"included_flammable_item","evidence":{"source":"description","value":"газовый баллон входит в комплект"},"explanation":"В комплект горелки прямо включён газовый баллон, поэтому карточка содержит продаваемый горючий газ."}
LABEL=1
```

В prompt метки нет. Для каждого outer fold обучение видит только targets
outer-train. Teacher explanations holdout существуют только для аудита и в
обучение соответствующего fold не попадают.

User prompt у rationale и matched digit-control дословно одинаковый и не
заказывает конкретный output format; формат задаётся только assistant target.
Обе arms оцениваются одинаковой greedy autoregressive генерацией с
`max_new_tokens=384`, без fitted threshold. Это оставляет единственным
изменённым фактором supervised target.

Сравнение — exact Qwen3.5-4B digit-only parent на connected-safe folds 0/3.
Нельзя одновременно менять sampler, LoRA, epoch, картинки или fusion. Основная
метрика — label-last generation Macro F1 и parse rate; затем полный frozen
route. Не сравнивать по teacher-forced loss.

Поскольку unsupported teacher rows исключаются, честный causal control обязан
использовать точно тот же filtered training multiset и число steps, но старый
digit-only assistant target. Сравнение с полным solution140 parent остаётся
system reference, а не доказательством эффекта rationale objective.

Handoff builder материализует этот control отдельно как
`matched_support_digit_v5.jsonl` в том же порядке, что и rationale corpus.
Control запускается с `EXP713_OBJECTIVE=matched_support_digit`; остальные seed,
sampler, исходный image0 и его solution140 preprocessing, max length, batch,
accumulation, epochs и support должны совпадать побайтово/построчно.

Сначала построить targets через `712/build_student_targets.py --mode
rationale_then_label`, затем запустить `preflight.py`. В обучение попадают
только grounded targets, доступные student по первой картинке; честные
`not_enough_evidence` и evidence с более поздней картинки исключаются.
GPU-разрешение появляется при coverage не ниже 85% в целом и при точном
совпадении последнего label с исходным `label`. Coverage каждой
category×label ячейки обязательно публикуется как предупреждение; matched
digit-control использует ровно тот же неполный support.

`trainer.py` — runnable adapter к неизменному parent trainer. Он фильтрует
только training multiset по наличию поддержанного target, но сохраняет полный
outer holdout для генеративной оценки label. Он запрещает
full-train и folds кроме 0/3, генерирует последовательность autoregressively и
сохраняет `label_last_generations.jsonl`; label извлекается только из последней
строки. Parent rank-fusion report в этой ветке диагностический, потому что
generated label дискретный.

Так как parent изначально обучался на односимвольном answer, здесь контекст
увеличен до 2304 токенов. Каждый training batch декодирует только позиции с
loss и требует их точного совпадения с полным JSON+`LABEL`; молчаливое усечение
объяснения или финальной цифры немедленно останавливает job.

В submission JSON не выходит напрямую: parser берёт `explanation` и последний
`LABEL`, после чего общий formatter строит точную строку без закрывающих тегов:
`<комментарий>{50–300 символов}<вердикт>{бан|не бан}`. Контракт находится в
`research/explanation_submission_contract.py`.
