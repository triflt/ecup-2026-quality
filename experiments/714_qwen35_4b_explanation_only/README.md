# 714: отдельный Qwen3.5-4B только для объяснения

Статус: **главный submission-кандидат подготовлен и ждёт явного user approval**.

Это главный кандидат на submission. Классификатор, его score и его решение
полностью frozen. При SFT в input объясняющей
модели передаётся карточка и настоящая train-метка; assistant target не содержит
label, JSON или submission-теги:

```text
Карточка продаёт пустую канистру и прямо указывает поставку без топлива; горючее вещество в комплект не входит.
```

Эта модель не участвует в F1 и не имеет права менять verdict. Её отдельные
метрики: plain-text parse rate, 50–300 chars, teacher-grounded target lineage, blind
human relevance, unsupported claim rate и latency. Это основной production
кандидат для выполнения требования к конкретным комментариям.

Для SFT conditioning label берётся из исходного `label` только на разрешённых outer-train
строках. На holdout в то же input-поле подставляется frozen OOF verdict
solution140 — ровно deployment-схема, включая ошибки ансамбля. LoRA возвращает
только комментарий и технически не может изменить CSV verdict. Grounded
relevance относительно teacher оценивается отдельно на строках, где frozen
verdict совпадает с исходным `label`; format и соответствие выданному verdict — на всём
holdout.

Если frozen OOF verdict расходится с исходным `label`, source-label-conditioned target
не используется как объяснение противоположного решения. Такая audit-строка
получает статический fail-closed comment и отдельно считается в отчёте; качество
generated explanation измеряется только на строках с согласованным verdict.

Сначала построить targets через `712/build_student_targets.py --mode
explanation_only`, затем пройти `preflight.py`. В обучение попадают только
grounded targets, доступные student по первой картинке; required coverage — не
ниже 85% в целом, а дисбаланс category×label публикуется отдельно. Обучать full-data
до blind 200-row human audit запрещено.

`trainer.py` — runnable adapter к тому же parent trainer. Он запрещает
full-train и folds кроме 0/3, генерирует объяснения autoregressively и сохраняет
полный текст/format status. Fold screen отдельно проверяет генерацию, а
production replay обязан доказать, что verdict берётся из solution140 и не
может быть изменён объясняющей веткой. Качество текста проходит blind audit.

Контекст увеличен до 2304 токенов; для каждого training batch supervised
позиции декодируются обратно и должны точно совпасть с полным comment target.
Это запрещает молчаливое усечение длинного объяснения parent-тренером.

`submission_runtime.py` подключает explanation adapter к уже загруженному
Qwen3.5-4B `PeftModel` solution140 после расчёта всех classifier scores. Второй
экземпляр base-модели не загружается. Renderer получает только карточку и
неизменный финальный verdict, генерирует plain comment, после чего строгий
runtime contract чинит формат либо использует старый безопасный комментарий.
Первое исходное изображение выбирается тем же кодом, что в solution140, без
отдельной копии или перекодирования: `Image.open -> RGB -> thumbnail 448
LANCZOS`, затем PIL-объект напрямую передаётся тому же Qwen3.5 processor.
Отсутствующий image0 останавливает строку вместо тихой белой заглушки.
В champion package этот модуль копируется только после fold-screen и latency
smoke принятого adapter.

Финальный formatter берёт сгенерированный комментарий и неизменный verdict
ансамбля и строит требуемую строку без закрывающих тегов:
`<комментарий>{50–300 символов}<вердикт>{бан|не бан}`. Контракт находится в
`research/explanation_submission_contract.py`.
