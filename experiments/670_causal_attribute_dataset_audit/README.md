# 670: аудит причинных атрибутов flammable

Статус: **отклонён до GPU по необратимо нарушенному audit gate**.

## Результат

Пакет был построен на реальных артефактах: 300 уникальных semantic components,
53 ошибки точного replay `140`, 269 flammable и 31 BAD-контроль. Он не
пересекается с audit ID экспериментов `490`, `622` и `626`; Public и sealed rows
не использовались.

Первые четыре blind-review строки дали четыре unsupported targets при заранее
зафиксированном максимуме три на всём пакете. Во всех четырёх случаях exact
слово действительно присутствовало, но parser ошибся в объекте продажи либо
отношении: аксессуар назван устройством, а входящий в комплект consumable —
`mentioned_only`. Поскольку оставшиеся 296 оценок не могут уменьшить уже
наблюдаемые четыре ошибки, scorer корректно завершил опыт досрочно:
`NO_GO_REJECT_CAUSAL_TARGETS`.

Это подтверждает механизм ошибки `635`, но опровергает дешёвый детерминированный
способ построить causal targets. Experiment `671` и любой GPU-запуск по этим
targets запрещены. Исправлять lexicon/parser по уже открытым строкам и повторять
тот же audit нельзя; допустимый новый путь должен получать объект и relation из
независимого воспроизводимого teacher либо иной новой разметки и проходить новый
непересекающийся audit.

## Что меняется

Эксперимент проверяет не новую модель, а качество новой auxiliary-разметки для
следующего кандидата. Вместо широкого `FUEL_OR_IGNITION` фиксируются три
независимых свойства:

- `sold_object`: `consumable/device/accessory/kit/unknown`;
- `regulated_substance`: `bad_marker/gas/flammable_liquid/solid_fuel/ignition_aid/none`;
- `relation`: `sold_object/included/compatible_external/mentioned_only/negated/unknown`.

Извлекатель не принимает label, fold, model score или Public feedback. Каждый
поддержанный target содержит точную surface-подстроку и offsets; неоднозначность
маскирует auxiliary loss, но не удаляет строку из обычного verdict training.

Это принципиально отличается от отклонённых `450/500/610`: target не является
keyword-классификатором или перестановкой предложений и не меняет verdict на
инференсе. Отличие от `520/570/623/632` — явное моделирование объекта продажи и
отношения к веществу, то есть конкретного механизма пяти flammable-регрессий
интеграции `635`.

Precedents: attribute extraction for compact e-commerce models (ICCVW 2025) и
faithful exact evidence из ERASER. Перенос проверяется на наших данных до GPU.

## Замороженный аудит

Builder выбирает 300 непересекающихся semantic-v3 components и исключает ID
аудитов `490`, `622` и `626`. Reviewer не видит gold label, prediction или
признак ошибки `140`. В приватном manifest эти поля нужны только для проверки,
что пакет действительно покрывает текущие ошибки.

Страты по 100 строк:

1. `transaction_scope_priority` — device/accessory/kit и явные relation cases;
2. `direct_substance_control` — прямой продаваемый consumable;
3. `coverage_and_abstention_control` — остальные supported/unknown случаи.

GO требует одновременно `282/300` строгих row-pass, `95/100` правильных
`sold_object + relation` в priority-страте и не более трёх unsupported claims.
Незаполненные поля дают `WAIT`, а не придуманные оценки.

## Воспроизведение

Сначала экспортируется точный label-free replay чемпиона 140 из checksum-locked
bundle эксперимента 635, затем строится новый пакет в пустую локальную папку.

```bash
python experiments/670_causal_attribute_dataset_audit/run.py export-baseline \
  --bundle /path/to/semantic_v3_140_replay.npz \
  --contract /path/to/replay_contract.json \
  --registry validation/semantic_family_v3/folds.csv \
  --output /new/local/path/full140_predictions.csv

python experiments/670_causal_attribute_dataset_audit/run.py build \
  --data /path/to/data.csv \
  --folds validation/semantic_family_v3/folds.csv \
  --baseline-predictions /new/local/path/full140_predictions.csv \
  --prior-audit experiments/490_evidence_grounded_explanations/analysis/manual_audit_200_v1.json \
  --prior-audit /path/to/exp622/human_audit_300.csv \
  --prior-audit /path/to/exp626/human_audit_300.csv \
  --output-dir /new/local/path/audit
```

После независимого заполнения пяти `review_*` полей:

```bash
python experiments/670_causal_attribute_dataset_audit/run.py score \
  --audit /path/to/filled/human_audit_300.csv \
  --private-manifest /path/to/audit/private_manifest.json \
  --output /new/local/path/audit_score.json
```

Reviewer может передать неизменяемому packet отдельный `--reviews` overlay.
Scorer завершает аудит досрочно только когда gate уже математически невозможно
восстановить: более трёх unsupported claims, более 18 strict failures или более
пяти object/relation failures в priority-страте. Неполный аудит без такого
терминального свидетельства остаётся в состоянии `WAIT`.

Только решение `GO_BUILD_671` разрешает модельный experiment 671.
