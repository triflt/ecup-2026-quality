# 641: Qwen3.5-4B, LoRA только для класса

Статус: **folds 0/3 повторно обучаются с закреплённым быстрым kernel**.

Первый запуск дошёл до forward и подтвердил OOM в резервной PyTorch-реализации
линейного внимания. Первый технический повтор остановился до обучения из-за
слишком новой версии kernel-пакета. Текущий повтор закрепляет точную версию,
которую требует установленный Transformers; данные, модель, loss, seed,
effective batch и порог не менялись.

Это маленькая обученная ячейка честной сетки `размер × способ обучения`. Она
получает ровно тот же первый снимок, название, описание, правило, очистку текста
и atomic `0/1` prompt, что prompting-эксперимент `640`. Меняется только наличие
LoRA-обучения на донорских folds.

## Гипотеза и контроль

Контроль — `640/Qwen3.5-4B prompting`. Механизм: supervised LoRA должна изучить
локальную границу категорий без изменения входа или порога. Сравнение с `642`
измеряет только масштаб модели, а сравнение с `643` — только добавление
grounded-evidence auxiliary objective.

Используется рецепт hard selection исходного Qwen-компонента, но selection
физически выполняется до GPU только на четырёх donor folds. Validation JSONL не
содержит label или evidence target. Seed `42`, один epoch, effective batch `16`,
rsLoRA rank `16`, learning rate `2e-4`, порог `0`.

## Запуск

Сначала строится отдельный runtime для fold `0`, затем fold `3`:

```bash
python experiments/641_qwen35_4b_class_only_lora/build_runtime.py --help
python experiments/641_qwen35_4b_class_only_lora/run_fold.py --help
```

Пакет не содержит scheduler preset, URL, данные, веса или результаты. Folds
`1/2/4` разрешены только решением screen evaluator `645`.

## Предшествующие результаты

- Qwen3.5-4B model card: <https://huggingface.co/Qwen/Qwen3.5-4B>
- `600`: исходный semantic-v3 Qwen-компонент.
- `623`: verdict-primary objective показала пользу evidence-регуляризации, но
  только на `3/5` folds; поэтому class-only контроль обязателен.
