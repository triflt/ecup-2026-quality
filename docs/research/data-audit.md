# Исследование данных

## Сводка

- 12 971 обучающий объект.
- 7 469 строк категории `БАД`.
- 5 502 строки категории `Легковоспламеняющиеся`.
- В flammable только 198 положительных меток, positive rate 3.60%.
- 49 456 изображений, от одного до пяти на товар.
- Распределение количества изображений 1/2/3/4/5: 1 800 / 1 153 / 1 460 / 1 820 / 6 738 товаров.
- Для каждого `id` изображения имеют непрерывные индексы, пропущенных и лишних товаров не найдено.

## Повторы и конфликтующие метки

- 1 868 повторяющихся normalized-name групп покрывают 5 994 строки.
- 66 name-групп содержат разные labels.
- 1 802 повторяющихся normalized-full-text групп покрывают 5 395 строк.
- 55 full-text групп содержат разные labels.

Повторы создают два разных режима качества:

1. **Novel-product generalization.** Проверяется grouped CV: весь normalized product family остаётся в одном fold.
2. **Product recurrence.** Проверяется отдельными 70/30 train→hidden simulations: prior строится только на simulated train и применяется к simulated hidden.

Эти метрики нельзя смешивать. Leave-one-out statistics, построенные до outer split, не являются честной grouped validation.

## Инсайты по категориям

### БАД

Решающим является наличие явной маркировки в тексте карточки или на упаковке. Внешний вид капсул, витаминов или спортивного питания недостаточен. Поэтому полезны word/character TF-IDF, packaging OCR, first-image VLM и осторожный wording-family prior.

### Легковоспламеняющиеся

Нужно определить, является ли сам продаваемый объект источником воспламенения/горючим веществом либо входит ли такое содержимое в комплект. Изображённый в инструкции баллон, встроенный поджиг и оборудование без топлива образуют трудные negative cases. Из-за 198 positives category F1 сильно зависит от нескольких решений.

## Проверенные гипотезы

- Random OOF переоценивает качество из-за product duplicates.
- Exact image hashes почти не добавляют качества поверх text-family prior.
- Numeric wording families и rare five-word shingles дают небольшое, но стабильное улучшение только для БАД.
- ID-neighbour, character-TFIDF neighbour и rare-token graph priors не дали надёжного выигрыша.
- Rule-based hard overrides ухудшают обе категории.
- Дополнительные gallery images разводняют first-image evidence.

## Воспроизведение

```bash
python tools/audit_dataset.py \
  --data /path/to/data.csv \
  --images /path/to/images \
  --output-dir /tmp/ecup-audit

python validation/build_folds.py \
  --data /path/to/data.csv \
  --output validation/grouped_text_v1/folds.csv \
  --basket-output validation/grouped_text_v1/basket.csv \
  --manifest-output validation/grouped_text_v1/manifest.json
```

Аудит работает только с локальными файлами и не зависит от object storage или конкретной compute platform.
