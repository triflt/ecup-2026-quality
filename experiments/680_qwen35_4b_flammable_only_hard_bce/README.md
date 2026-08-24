# 680 — Qwen3.5-4B flammable-only hard-BCE adapter

## Статус

`OPEN_SCREEN_FOLDS_0_3`. Это однофакторная проверка category specialization:
из фактического runtime 641 удаляются только BAD training occurrences. Модель,
revision, prompt, image view, BCE, LR, scheduler, LoRA, seed, одна эпоха и
inference score остаются неизменными. Число updates `306 → 143` является
детерминированным следствием одной эпохи на меньшем multiset, а не tune.

## Механизм и non-duplication

Shared adapter одновременно оптимизирует BAD и редкий flammable. Новый adapter
видит только те же 2 280 flammable occurrences каждого outer-train runtime:
1 600 отрицательных и 136 уникальных положительных товаров, повторённых пять
раз. Независимый audit подтвердил, что это не повтор experiments 260/410/420/
560/600: у них отличаются selector/multiset, suffix-CE или auxiliary loss,
image raster, LR/init либо sampling.

## Frozen screen

Folds 0/3 обучаются параллельно на одной H100 каждый. Primary flammable metric —
tie-aware Average Precision. Threshold `0` не меняется и не подбирается.
Confirmation folds открываются только если на обоих folds:

- Average Precision и production Macro F1 строго выше control 641;
- mean AP gain `>= +0.005`, mean production Macro gain `>= +0.0015`;
- flammable F1 не падает, FN не растут;
- corrections/regressions `>= 1.5`;
- BAD production route побайтово неизменён.

До полного 5-fold GO запрещены full-data refit, Public и изменение порога.

## Deployability

Production precedent уже загружает один Qwen3.5-4B base и два совместимых LoRA,
группирует строки по category и вызывает `set_adapter`, поэтому каждая строка
имеет ровно один forward. Дополнительный adapter занимает около 12.6 MB.
Перед submission обязателен null-route parity smoke и train-identical image
preprocessing: нельзя молча подменять area-cap 262144 на старый 448 thumbnail.

