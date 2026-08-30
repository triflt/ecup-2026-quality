# 712: Qwen3.5-397B evidence teacher

Статус: **teacher synth по всем 12 971 карточке и технический аудит завершены;
11 400 принятых targets использованы для explanation-only Exp714**. Это offline
teacher, а не ветка предсказаний ансамбля и не runtime сабмита.

## Что именно запускаем

`Qwen/Qwen3.5-397B-A17B-FP8` на одном `h100-8x` через vLLM tensor parallel 8.
Зафиксирован exact HF revision. Локальный remote compute preset лежит в игнорируемом
`.local/compute/serve_qwen.yml`; project/region не входят в git.

Teacher получает все 12 971 строк, исходный train `label` и все 49 456 изображений. Для
каждой карточки отправляется вся галерея (до пяти изображений). OOF
score и verdict solution140 не передаются. На каждую карточку он возвращает
ровно четыре поля: `label`, закрытый `reason`, одно `evidence` и конкретный
`explanation`. Raw response сохраняется до parser/validator.

## Порядок исполнения

1. Поднять serving и дождаться health/model smoke.
2. Прогнать фиксированные 20 строк: по 5 на каждую пару category×label;
   принять только при 20/20 transport success и schema-valid JSON.
3. Проверить label echo, reason compatibility, exact text spans и длину 50–300.
4. При ошибке в первом JSON сделать один validator-guided repair. Почти
   совпавшую текстовую цитату разрешено заменить только на ближайший буквальный
   span исходного поля при similarity >= 0.92; действие записывается в record.
   После второй невалидной попытки использовать честный `not_enough_evidence`.
5. Только затем resume-safe полный pass по всем строкам и изображениям.
6. Не использовать teacher targets до automated gates и blind human audit.

Blind-200 artifact включает не только текстовый packet, но и все изображения
выбранных карточек. Перед копированием каждое изображение повторно сверяется с
SHA и порядком, сохранёнными в teacher record; без этого image-grounded audit
не считается выполненным.

## Результат smoke и выбор teacher

Одинаковый frozen v5 пилот был выполнен на 397B и Qwen3.8 Flash Next. Оба дали
20/20 структурно валидных JSON. 397B дал 17 содержательных объяснений и 3
честных abstain, Flash — 16 и 4. Среднее время одной строки было 3.928 с у
397B и 4.002 с у Flash. Для полного synth выбран 397B: покрытие выше, inference
не медленнее. Flash поднял API быстрее, но это одноразовая стоимость serving,
не throughput полного прохода.

Повторный student-scope разбор обнаружил у 11 из 20 содержательных ответов
лишнюю ссылку на недоступную галерею/упаковку либо служебную фразу о категории.
Фиксированный repair prompt получает только `title`, `description`, `image0` и
исходный JSON; он обязан сохранить валидные reason/evidence и переписать только
неподдержанную формулировку. Тот же проход даёт один свежий шанс исходным
`not_enough_evidence`, но только по этим student-входам и с тем же строгим
валидатором. На v3 smoke запрошено 14 ремонтов: 12 дали полезный target, один
честно сохранил abstain, один fail-closed после невалидного evidence. Итог —
18/20 поддержанных объяснений и 2/20 `not_enough_evidence`. Полный corpus
прошёл тот же selective repair, а не повторную генерацию всех строк.

Полный исходный проход дал 10 783 полезных объяснения. Из 1 151 технически
неудачных строк all-image retry с детерминированным уменьшением галереи
восстановил 764. Финальный student-scope repair запросил 6 186 строк и принял
4 615 полезных исправлений. В результате 11 400 из 12 971 строк (87.89%) имеют
объяснение, целиком доступное из `category + label + title + description +
image0`; 1 571 строка (12.11%) осталась fail-closed
`not_enough_evidence` и не должна попадать в student SFT. SHA итогового
JSONL: `69a14886c4c8e9f83d584f6d15a7c9d6ece8ce7c556ebdc9f0668238f2bfc9e8`.

Финальный audit принял все 11 400 targets: invalid JSON, недоступных evidence
refs, target truncation и tokenizer alteration — по нулям. Полный manifest SHA:
`1ca83078dbdf3ac97f12f0b47fada80445686b369c6af2303344e3b45bac0fd5`,
acceptance receipt SHA:
`ef866fce947ddfb52dc9b8ba1b521407e0b0e35f57d46803245929b9abb23031`.
Исходный image0 не копируется и не перекодируется: обучение и runtime используют
тот же `Image.open -> RGB -> thumbnail 448 LANCZOS -> processor`, что solution140.
Coverage БАД label=0 равен 67.40%; это явно сохранённое предупреждение, а не
скрытая причина дополнять корпус выдуманными объяснениями.

Пример клиента внутри одобренной remote compute среды с уже распакованными images:

```bash
python experiments/712_qwen35_397b_evidence_teacher/generate_all.py \
  --data /work/input/data.csv \
  --image-root /work/input/images \
  --api-base http://<serving-job>:8000 \
  --output /work/output/teacher_raw_and_parsed.jsonl \
  --report /work/output/report.json \
  --sample-per-cell 5
```

Для full pass убрать `--limit`. Скрипт сначала проверяет полный scope
`12971/49456`, хэширует каждое фактически отправленное изображение и умеет
возобновляться по уже записанным id.

## Короткий target

```json
{"label":1,"reason":"included_flammable_item","evidence":{"source":"description","value":"газовый баллон входит в комплект"},"explanation":"В комплект продаваемой горелки входит газовый баллон, поэтому карточка содержит включённый горючий газ."}
```

Если teacher не может честно объяснить исходный `label`, он возвращает
`not_enough_evidence` и два `null`; такие строки не становятся explanation
targets. Технические hashes, model revision и validation flags добавляет код,
а не модель.

Никаких chain-of-thought, внешнего retrieval и автоматического relabeling.
