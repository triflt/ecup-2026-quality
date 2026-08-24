# 656: детерминированный tile-based ремонт плотного OCR

Статус: **ограниченный smoke на первых восьми глобальных кандидатах запущен на
одной GPU после audit и dry-run**.

## Причина и единственный изменяемый фактор

Эксперимент 655 отклонён на ограниченном smoke: увеличение
`max_new_tokens` с 512 до 1536 не исправило плотные spotting-ответы. Среди
первых наблюдений были 666 LOC без EOS и 273/297/209 LOC с EOS, то есть ни один
из них не удовлетворял неизменяемой восьмитокенной грамматике области. Поэтому
656 не читает и не исправляет результаты 655. Он заново находит тот же
глобально отсортированный набор оборванных строк непосредственно в неизменяемых
32 артефактах 633.

Модель, revision `c5630abae1d940eafe0697512a0325494b02ab42`, prompt
`Spotting:`, preprocessing, greedy decoding и лимит 512 токенов на один decode
совпадают с 633. Единственный новый фактор — исходное изображение разбивается
на перекрывающиеся тайлы до inference.

## Детерминированный алгоритм

Число тайлов зависит только от ширины, высоты и числа LOC-токенов исходного
ответа 633. Цель — не более 160 исходных LOC на тайл, минимум два и максимум 16
тайлов. Из всех ограниченных сеток выбирается сетка с наиболее квадратным
физическим тайлом и стабильными числовыми tie-break. Каждый внутренний край
расширяется на 10% ширины или высоты своей базовой ячейки; порядок — row-major.

Каждый тайл обязан одновременно:

- закончиться `</s>`;
- иметь число LOC, кратное восьми;
- полностью разбираться: число областей равно `LOC / 8`, либо ответ равен
  каноническому пустому `</s>`.

Если хотя бы один тайл нарушает gate, вся строка помечается ошибкой, весь shard
завершается ненулевым кодом и overlay не принимается. Частичный результат не
может заменить 633.

Координаты каждой области сначала сдвигаются из тайла в пиксели исходного
изображения, затем канонически нормализуются в `0..1000` и снова переводятся в
пиксели, чтобы формат был совместим со строгим builder 634. В overlap удаляются
только области с одинаковым NFC-текстом и bbox IoU не ниже 0.75. Приоритет
детерминированный: дальше от края тайла, затем меньшие tile/detection index.
Близкие области с разным текстом или IoU ниже порога сохраняются.

## Provenance и label blindness

GPU runtime получает только label-free manifest изображений, 32 исходных
артефакта 633, публичный model ID/revision и три versioned Python-файла. Labels,
categories, folds, sealed membership и Public не читаются. `spotting.jsonl`
сохраняет геометрию overlay 633 и per-tile sequence confidence выбранной
области; top-level confidence равна `null`, потому что строка не была одной
генерацией. Следующий immutable builder обязан явно проверить эту новую
семантику confidence. Отдельный `tile_provenance.jsonl` содержит границы и
полный ответ каждого тайла, source-row SHA и причины выбора.
`report.json` фиксирует SHA manifest, всех исходных shard/report, runtime-кода,
обоих выходных JSONL, параметры tiling/dedupe и точный SHA списка global index.

Private artifact references, model-registry path, region/flavor/image и
сгенерированные presets разрешены только в игнорируемой
`experiments/656_paddleocr_tiled_repair/.local/compute/`. Builder не использует URL
доставки runtime и подключает manifest и 32 OCR shard как native artifact
inputs. В tracked файлах нет внутренних имён задач, кластеров или URL.

## Gate и запуск

Сначала разрешён ровно один smoke по первым восьми глобальным кандидатам:

```bash
python experiments/656_paddleocr_tiled_repair/build_private_presets.py \
  --base experiments/656_paddleocr_tiled_repair/.local/compute/base.yml \
  --source-artifacts-manifest \
    experiments/656_paddleocr_tiled_repair/.local/compute/source_artifacts.json \
  --output-dir experiments/656_paddleocr_tiled_repair/.local/compute/smoke \
  --mode smoke
```

Production grid строится только после `8/8` accepted, нуля failed tile/row,
проверки всех SHA и ручной сверки перевода координат на overlap-примерах:

```bash
python experiments/656_paddleocr_tiled_repair/build_private_presets.py \
  --base experiments/656_paddleocr_tiled_repair/.local/compute/base.yml \
  --source-artifacts-manifest \
    experiments/656_paddleocr_tiled_repair/.local/compute/source_artifacts.json \
  --output-dir experiments/656_paddleocr_tiled_repair/.local/compute/production \
  --mode production --num-repair-shards 32
```

Перед любым live submit каждый локальный preset должен пройти
`scripts/audit_preset.py` и `compute job submit --preset-file ... --dry-run` с
проверкой resolved private settings. Первый dry-run остановился до создания
задачи из-за платформенного лимита в 20 символов для имени выходного артефакта.
Единственное исправление — сокращение этого технического имени; код OCR,
данные, tiling, модель и gate не менялись. Повторный audit дал ноль ошибок,
dry-run прошёл, после чего запущен один smoke.

## Основание

Precedent — динамические image tiles в OCR/document VLM (в том числе
InternVL3.5) и неизменяемый structured spotting contract PaddleOCR-VL-1.6.
Механизм переносится напрямую: уменьшение визуальной плотности на один decode
снижает длину структурированного ответа, а overlap сохраняет области на швах.
Это техническое восстановление label-blind evidence, не подбор по метрике.

Главные остаточные риски: bbox IoU является приближением polygon IoU; одинаковый
текст в почти совпадающих реальных областях может быть ошибочно склеен; шов может
дать разные OCR-тексты и оставить дубликат; очень плотный тайл всё ещё может не
пройти gate. Все эти случаи fail closed на уровне синтаксиса, но геометрическое
качество требует отдельного слепого smoke-аудита до production.
