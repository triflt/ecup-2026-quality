# 490: детерминированные доказательства для объяснений

## Статус

`rejected_manual_audit`: реализованный Stage A text-only baseline не меняет
предсказания классификатора, но не прошёл замороженную ручную проверку 200 строк.
Вердикт и формат сохранены в `200/200`, строгая релевантность составила только
`141/200` при пороге `189/200`, обнаружено 12 критических неподтверждённых
утверждений при допустимом нуле. В финальное решение этот extractor не входит.

## Гипотеза

Короткий комментарий можно безопасно построить после уже зафиксированного
вердикта: найти точный фрагмент названия или описания, сопоставить его одному
элементу закрытого словаря правил и применить неизменяемый шаблон. Если текст не
даёт однозначного основания, система честно возвращает `NO_SAFE_EVIDENCE`.

Extractor не является классификатором и не получает gold label, score, fold или
соседние товары. Он не создаёт свободную цепочку рассуждений и не делает выводов
по изображениям.

## Связь с исследованиями

- [ERASER](https://aclanthology.org/2020.acl-main.408/) разделяет выделенное
  доказательство и правдоподобное текстовое объяснение.
- [Faithful rationalization by construction](https://aclanthology.org/2020.acl-main.409/)
  мотивирует связь решения с явно выбранным фрагментом.
- [CLARITY](https://aclanthology.org/2025.starsem-1.33/) связывает текстовый
  фрагмент с понятным концептом.

Перенос в задачу проверяемый: sidecar хранит точную surface-цитату, offsets,
закрытый concept, scope и неизменный verdict. Любая неоднозначность приводит к
отказу, а не к убедительно звучащей догадке.

## Контракт

Чистая функция:

```python
extract_evidence(
    *, row_id: str, category: str, name: str, description: str,
    frozen_prediction: int, vocabulary_version: str = "policy_concepts_v1",
) -> EvidenceResult
```

Поддерживаются только категории `БАД` и `Легковоспламеняющиеся`, predictions
`0/1` и словарь `policy_concepts_v1.json`. При `SAFE`:

- `exact_surface_span == surface[source][surface_start:surface_end]`;
- raw offsets указывают на исходный интервал до раскрытия HTML entities и
  удаления тегов;
- комментарий имеет длину 50–300 Unicode code points;
- итоговый verdict равен входному prediction.

При `NO_SAFE_EVIDENCE` offsets и concept равны `null`, а fallback явно сообщает
об отсутствии однозначного текстового основания. Такой результат не считается
успешным grounded explanation.

## CLI

Вход — CSV с колонками `id`, `category`, `name`, `description` и
`frozen_prediction`. Наличие лишней колонки `label` ничего не меняет: она не
читается.

```bash
python3 experiments/490_evidence_grounded_explanations/extract_evidence.py \
  --data /path/to/rows.csv \
  --output /path/to/evidence.jsonl \
  --submission-output /path/to/comments.csv
```

Названия колонок можно изменить аргументами `--id-column`, `--category-column`,
`--name-column`, `--description-column` и `--prediction-column`.

## Проверка до аудита

```bash
ruff check experiments/490_evidence_grounded_explanations \
  tests/test_exp490_evidence_grounded_explanations.py
pytest -q tests/test_exp490_evidence_grounded_explanations.py
```

Тесты покрывают точные offsets после HTML-нормализации, закрытость словаря,
отрицание, exclusion/inclusion, compatibility, reference-only, конфликт между
полями, bounded absence, verdict lock, длину и CLI-инвариант к лишнему `label`.

## Результат ручного аудита

Выборка была заморожена до чтения карточек и стратифицирована только по
категории, вердикту и `SAFE/NO_SAFE_EVIDENCE`; gold-метки, folds и состояние
ошибки модели не использовались. Extractor после заморозки не изменялся.

Основные причины отклонения: пропуск явных признаков flammable и отрицаний БАД,
смешение продаваемого товара с объектом применения, метафор с пиротехникой и
обрезание отрицания до противоположного смысла. Обезличенные результаты лежат в
`analysis/manual_audit_200_v1.json` и `analysis/manual_audit_200_v1.csv`.

Следующая версия должна получить новый ID, новый заранее замороженный аудит и
отдельно моделировать объект продажи, область действия отрицания и прямые
положительные/отрицательные признаки. Подгонять этот extractor по 200 увиденным
строкам запрещено.
