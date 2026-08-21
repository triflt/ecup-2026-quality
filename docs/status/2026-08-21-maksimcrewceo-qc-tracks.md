# Статус ветки maksimcrewceo (QC-треки) — 2026-08-21

Автор: Максим (агент Kimi, рабочая папка `tmp/QC`, отдельно от `main` Данека).
GPU-политика: только GPU 0-5; GPU 6-7 заняты чужими процессами, не трогаем.

## TL;DR

- Запущены 3 трека: **240** (Gemma-4-E4B LoRA SFT, Track A), **241**
  (Qwen3-VL-Embedding-2B contrastive, Track B), **242** (fusion grid, Track C).
- B01 (sigmoid) и B02 (triplet) **обучены** (~26 мин каждый, адаптеры сохранены),
  эмбеддер-эвалы идут.
- A01-A04 (LoRA, 730 шагов) на 10-17% (~3 ч до конца), NaN-guard работает
  (7 пропущенных батчей на A04).
- Auto-eval по крону каждые 10 мин подхватывает готовые чекпоинты.

## Ключевые факты, которые нужно знать команде

1. **Маппинг метки в train CSV инвертирован относительно примера спеки:**
   raw `1` = «бан», raw `0` = «не бан». Подтверждено визуально и поведением
   zero-shot. Все мои эвалы используют этот маппинг — при совместном фьюжне
   сверяйте знак скоров!
2. Мой сплит — diagnostic stratified 90/10 по (category,label), seed=42
   (`tmp/QC/split.json`, train 11673 / val 1298). Это НЕ grouped folds из
   `validation/grouped_text_v1/`; перед любым сабмитом победитель должен быть
   пересчитан на ваших folds.
3. Zero-shot Gemma-4-E4B-it на этом val: macro-F1 **0.3379** (std) /
   **0.3776** (noocr). Низко — большой запас для SFT.

## Прогресс по трекам

| Трек | Эксп | Конфиг | Статус |
|---|---|---|---|
| 240 | A01 | noocr, r64, lr1e-4 | running 17% |
| 240 | A02 | noocr, r32 | running 17% |
| 240 | A03 | ocrlong, r64 | running 14% |
| 240 | A04 | std, r64 | running 11% |
| 241 | B01 | sigmoid loss, bs8, ga4 | train done, eval running |
| 241 | B02 | triplet loss, bs8, ga4 | train done, eval queued |
| 242 | — | fusion grid поверх скоров A/B | prepared, ждёт A/B |

## Инженерные находки (чтобы не наступать дважды)

- **NaN в backward на Gemma-4**: forward чистый, NaN-грады возникают в backward
  по gradient checkpointing на редких парах картинок (триггер id 10210×10519).
  Лечится `NaNProofAdamW` (зануление нефинитных градов, skip step).
  См. `experiments/240_gemma4_lora_sft/src/train_lora.py`.
- peft 0.20 не оборачивает `Gemma4ClippableLinear` → LoRA только на
  `language_model.*` (258 модулей).
- Loss считается только по токенам вердикта («бан»/«не бан»), не по всему
  assistant-ответу.
- Contrastive: OOM лечится фиксированным P/K-сэмплингом батча (8 = 2×4) +
  grad_accum 4; sigmoid-лосс с multi-positive маской.

## Что взято из main (Данек) и как стыкуемся

- Лучший Public команды — **0.8919** (`190_shingle_neighbor_prior`,
  dual-LoRA + shingle prior). Архитектура Данека: Qwen3-VL / Qwen3.5 rsLoRA
  first-image + robust-base fusion + product-family priors.
- Мой план стыковки: контрастивный эмбеддер (241) как дополнительный head к
  late fusion; Gemma multi-image LoRA (240) как независимая генеративная ветка.
  Если 240/241 на grouped folds не дадут +0.01 к 0.9274 (текущий best OOF из
  190) — в сабмит не идут, остаются как отрицательный результат.

## Артефакты

- Исходники скопированы в `experiments/24{0,1,2}_*/src/`.
- Конфиги A01-A04 и B01-B02 — в `experiments/240_gemma4_lora_sft/artifacts/configs/`
  и `experiments/241_qwen3vl_contrastive/artifacts/configs/`.
- Локальные чекпоинты/логи: `/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/exps/`
  (в репо не коммитим — большие).
- OCR-кэш: `tmp/QC/ocr_cache.parquet` (покрытие 100%, пустой OCR 2.12%).

## Карта «моя папка → репо Данека»

| Моя сущность | Путь в `tmp/QC` | Перенесено в `ecup-2026-quality` |
|---|---|---|
| LoRA train/eval scripts | `train_lora.py`, `eval_val.py`, `find_bad_batch2.py` | `experiments/240_gemma4_lora_sft/src/` |
| A01-A04 configs | `exps/expA0{1-4}/config.yaml` | `experiments/240_gemma4_lora_sft/artifacts/configs/` |
| Contrastive train/eval scripts | `train_contrastive.py`, `eval_embedder.py` | `experiments/241_qwen3vl_contrastive/src/` |
| B01-B02 configs | `exps/expB0{1,2}/config.yaml` | `experiments/241_qwen3vl_contrastive/artifacts/configs/` |
| Fusion scripts | `run_fusion_grid.py`, `eval_fusion.py` | `experiments/242_fusion_grid/src/` |
| Сводный статус | — | `docs/status/2026-08-21-maksimcrewceo-qc-tracks.md` |
| Регистры | — | `experiments/README.md`, `reports/hypothesis-board.csv`, `reports/experiment-log.csv` |

## Инженерный фикс (2026-08-21 15:45)

`eval_embedder.py` падал на этапе `evaluate` из-за перебора по словарю `results`,
в который уже были добавлены скаляры `macro_f1_lr`/`macro_f1_knn`.
Пофиксил — теперь macro-F1 считается только по категориям.
Перезапущены zero_embed, B01 и B02 eval-ы; логи `.bak` сохранены.

## Update после B-эвалов (2026-08-21 17:05)

| Config | BAD F1 ban | Flammable F1 ban | Macro-F1 LR | Macro-F1 KNN |
|---|---|---|---|---|
| zero_embed | 0.9130 | 0.5672 | **0.7401** | **0.8772** |
| B01 sigmoid | 0.8413 | 0.2466 | **0.5439** | **0.8444** |
| B02 triplet | 0.8702 | 0.2857 | **0.5780** | **0.8473** |

**Решение по Track B:** контрастивное дообучение **ухудшило** embedding-head vs zero-shot на diagnostic split:
- LR macro-F1: -0.1961 (B01) и -0.1621 (B02).
- KNN macro-F1: -0.0328 (B01) и -0.0299 (B02).

Гипотеза H007 **отклонена** при текущих настройках (r16, lr1e-4, 3 эпохи, n_images=2). Эмбеддер не идёт в Track C fusion. Адаптеры и метрики сохранены как отрицательный результат.

**Следующий шаг:** дождаться A01-A04 LoRA (еще ~2 ч), оценить их на val и решить, есть ли смысл продолжать Track A/fusion.
