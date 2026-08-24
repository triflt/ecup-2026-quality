# 662: расчёт outer-train целей Qwen3.6-27B

Статус: **fail-closed runtime проверен; smoke на 8 строках выполняется**.

Эксперимент не обучает модель и не является кандидатом на отправку. Для каждого
outer fold принятый адаптер 654 оценивает только те training occurrences, на
которых он был обучен. Результат — in-sample teacher targets для отдельного
эксперимента дистилляции, а не OOF-прогнозы и не честная validation-метрика.

Runtime создаётся из неизменяемого `train.jsonl`, но удаляет метки до сборки
GPU-bundle. `validation.jsonl`, registry, selector и исходный датасет в задачу
не передаются. Контракт фиксирует SHA source train, адаптера и упорядоченных
ключей `(global_index, id, category, fold, occurrence_index)`, запрещает строки
outer validation и требует конечный raw score `logit("1") - logit("0")` без
sigmoid, температуры, порога или калибровки.

Сначала fold 0 обязан дать `8/8` корректных строк. Только после проверки ZIP,
self-hash отчёта и точной привязки к runtime разрешён полный label-free scoring.
Эти ответы можно использовать только как train targets deployable student;
outer-fold validation не участвует в обучении или выборе.

Полные runtimes всех пяти folds уже собраны и прошли локальную проверку схемы,
контрольных сумм и точного порядка occurrences. Они не разрешают массовый
запуск сами по себе: сначала скачивается и принимается отдельный smoke-архив.
