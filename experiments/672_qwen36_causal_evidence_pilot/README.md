# 672: свежий causal-evidence пилот 4B против 27B

## Зачем этот эксперимент

Эксперимент `670` остановил исходную H1 до GPU: уже первые четыре слепо
проверенных deterministic targets содержали четыре неподтверждённых причинных
утверждения при допустимом максимуме три на 300. Исправлять словарь на той же
выборке запрещено. Ранее `490/626/635` также показали, что наличие точного слова
о топливе или БАД не гарантирует правильного понимания объекта продажи,
комплектности и отрицания.

`672` проверяет более узкое и дешёвое утверждение: способна ли уже доступная
открытая Qwen3.6-27B строить заметно более качественное доказательное объяснение,
чем Qwen3.5-4B, когда вход, изображение, prompt, decoding и замороженный verdict
полностью одинаковы. Это explanation-only pilot. Он не меняет prediction,
BAD-route, веса, пороги или правила чемпиона `140`.

## Один изменяемый фактор

Меняется только base model: Qwen3.5-4B против Qwen3.6-27B. Для обеих моделей
фиксированы:

- ровно 40 новых semantic components, по 10 на каждую пару category × verdict;
- exact-140 semantic-v3 prediction без gold label;
- первое локальное изображение товара;
- один structured prompt, greedy decoding и `thinking=False`;
- один строгий parser и одна слепая review rubric.

В flammable strata label-blind ranking сначала ставит карточки с признаками
оборудования, внешнего топлива и комплектности. Выборка исключает компоненты
аудитов `490`, `622`, `626` и `670`. Читаются только development membership и
поля `id/category/name/description`; `label` и sealed holdout не читаются.

## Предварительно замороженный gate

Qwen3.6-27B разрешает только новый независимый 200-row explanation audit, если:

- automatic JSON contract и замороженный verdict: 40/40;
- human verdict consistency: 40/40;
- unsupported facts: 0/40;
- против 4B исправлено минимум четыре дополнительных strict-relevant случая
  (не менее 10 п.п. на выборке 40) **или** шесть дополнительных случаев с
  действительно решающим visual evidence (15 п.п.).

Даже прохождение gate не разрешает classification training, массовую teacher
разметку или distillation. Для них нужна отдельная гипотеза и новая проверка.
Public leaderboard не используется.

## Сборка локального замороженного пакета

Все выходы остаются под `.local`; существующие каталоги никогда не
перезаписываются.

```bash
python experiments/672_qwen36_causal_evidence_pilot/build_pilot.py \
  --data /absolute/path/data.csv \
  --folds validation/semantic_family_v3/folds.csv \
  --baseline-predictions experiments/670_causal_attribute_dataset_audit/.local/full140_predictions.csv \
  --images-zip /absolute/path/images.zip \
  --exclusion-manifest /absolute/path/exp490_audit.json \
  --exclusion-manifest /absolute/path/exp622_audit.csv \
  --exclusion-manifest /absolute/path/exp626_manifest.json \
  --exclusion-manifest experiments/670_causal_attribute_dataset_audit/.local/audit_v1/private_manifest.json \
  --output-dir experiments/672_qwen36_causal_evidence_pilot/.local/pilot_v1

python experiments/672_qwen36_causal_evidence_pilot/extract_images.py \
  --runtime experiments/672_qwen36_causal_evidence_pilot/.local/pilot_v1/pilot_runtime.jsonl \
  --images-zip /absolute/path/images.zip \
  --output-dir experiments/672_qwen36_causal_evidence_pilot/.local/pilot_v1/images
```

`build_bundle.py` собирает ровно runtime, 40 изображений, runner и frozen spec.
`build_presets.py` производит два private remote compute preset из уже успешных 4B/27B
шаблонов; `.local` передаётся как явный `--context-dir` и затем как
`--custom-preset-context`, поэтому bundle остаётся внутри разрешённого upload
scope. Перед отправкой каждый preset проходит `audit_preset.py` и официальный
`compute job submit --dry-run`. Обе модели используют по одной H100 и могут идти
параллельно только при свободной квоте; OCR jobs другой ветки не прерываются.

## Слепая оценка

После загрузки двух `explanations.jsonl`:

```bash
python experiments/672_qwen36_causal_evidence_pilot/build_blind_review.py \
  --runtime /absolute/path/pilot_runtime.jsonl \
  --qwen35-output /absolute/path/qwen35/explanations.jsonl \
  --qwen36-output /absolute/path/qwen36/explanations.jsonl \
  --output-dir experiments/672_qwen36_causal_evidence_pilot/.local/review_v1
```

Рецензент видит `blind_candidates_80.csv`, но заполняет только отдельный
`review_form_80.csv` значениями `yes/no`. `evaluate_review.py` проверяет SHA
неизменяемого packet, затем раскрывает model aliases и применяет frozen gate.

## Ресурсы и риски

- Максимум: две независимые задачи по одной H100 и 45 минут, суммарно не более
  1.5 GPU-часа; ожидаемое применение на 40 строк значительно короче.
- Основной риск — убедительно звучащие неподтверждённые visual claims; поэтому
  допустимый unsupported count равен нулю.
- Пилот не оценивает Macro F1 и не доказывает перенос на Private. Его назначение
  — решить, оправдан ли следующий свежий explanation audit, не расходуя десятки
  GPU-часов на teacher extraction.

## Результат

Пока `PREPARED`: локальный пакет и два GPU-прогона должны быть выполнены без
изменения frozen spec. Итог и решение фиксируются в `results/metrics.json`.
