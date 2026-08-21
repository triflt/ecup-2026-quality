# 241: Qwen3-VL-Embedding-2B Contrastive (Track B, Максим/QC)

## Гипотеза

Контрастивное дообучение Qwen3-VL-Embedding-2B (sigmoid / triplet loss) сблизит
эмбеддинги товаров одного вердикта (бан/не бан) и разведёт чужие, что поднимет
качество image-embedding head по сравнению с zero-shot эмбеддером из 030
и даст независимый сигнал для фьюжна с текстовым/LoRA-ветками.

## Данные и preprocessing

- Тот же сплит, что и 240: stratified 90/10 по (category,label), seed=42
  (`tmp/QC/split.json`), train 11673 / val 1298.
- Маппинг метки как везде: raw `1` = «бан», raw `0` = «не бан».
- На товар берётся до 2 изображений (`n_images=2`), caption = title + description
  (truncated) + OCR-кэш при наличии.
- Батчинг: `batch_size=8` как 2 негатива × 4 класса-семпла, `grad_accum=4`
  (эффективный 32). Это ответ на OOM: с маленьким батчем в-batch negatives
  почти нет, поэтому структура батча фиксирована (P/K-сэмплинг), а не random.
- Loss: sigmoid (multi-positive) и triplet-вариант как абляция; конфиг в
  `tmp/QC/exps/expB01|expB02/config.json`.

## Связь с SOTA

- Precedent: 030_qwen3vl_embedding (zero-shot embedding SVM: grouped CV
  0.8581 macro, «image signal полезен, но недостаточен один») и
  060_multiview_fusion (first-image-only как независимый сигнал).
- Контрастивная доводка — попытка поднять именно этот head, не трогая
  validated late fusion из 040/090.

## Acceptance criterion

- Primary: macro-F1 эмбеддер-head (SVM поверх эмбеддингов) на val (diagnostic),
  далее пересчёт на grouped folds.
- Guardrails: gain к zero-shot эмбеддеру > +0.01 macro, иначе не идёт в фьюжн.

## Инженерные грабли

1. OOM при naive all-pairs batch → фиксированный P/K-сэмплинг батча и
   `grad_accum=4`.
2. Zero-shot эмбеддер-эвал гоняется отдельным скриптом (`eval_embedder.py`),
   т.к. инференс эмбеддера отличается от генеративного пайплайна.
3. GPU: только 0-5 (6-7 чужие).

## Запуск

```bash
bash tmp/QC/run_wave1.sh   # expB01/expB02 на GPU 4-5
```

Волна 1 (Track B): expB01 sigmoid · expB02 triplet.

## Результат и решение

Обучение/эвал идут. См. `results/metrics.json` и
`docs/status/2026-08-21-maksimcrewceo-qc-tracks.md`.
