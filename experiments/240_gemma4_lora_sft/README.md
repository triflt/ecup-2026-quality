# 240: Gemma-4-E4B-it LoRA SFT (Track A, Максим/QC)

## Гипотеза

Supervised LoRA поверх multimodal Gemma-4-E4B-it (5 изображений + OCR-текст)
переобучит zero-shot бейзлайн (local val macro-F1 0.3379/0.3776) до уровня,
конкурентного с Qwen-ветками, и даст дешёвый одностадийный инференс.

## Данные и preprocessing

- Данные: `competition_train_v1` (12 971 товар; БАД 7469 / Легковоспламеняющиеся 5502).
- **КРИТИЧНО: маппинг метки в CSV — raw `1` = «бан», raw `0` = «не бан»**
  (инвертирован относительно примера спеки; подтверждено глазами + поведением
  zero-shot). Все локальные train/eval используют этот маппинг.
- Сплит: stratified 90/10 по (category, label), seed=42 → train 11673 / val 1298
  (`tmp/QC/split.json`). Это диагностический random-style split, НЕ grouped folds
  из `validation/grouped_text_v1/` — для финального выбора нужен пересчёт на folds.
- OCR-кэш: `tmp/QC/ocr_cache.parquet`, покрытие 100%, пустой OCR 2.12%.
- Препроцессинг скрипты лежат в `src/preprocess/` (скопированы из `tmp/QC`).
- Три SFT-датасета (абляция OCR): `sft_std` (≤700 chars), `sft_noocr` (без OCR),
  `sft_ocrlong` (≤1500). Промпт = `_messages` из `submit/src/gemma_stage.py`
  (картинки перед текстом, ≤5 изображений, 768px, thinking off).
- **Лосс только по токенам вердикта** («бан»/«не бан»), а не по всей строке
  ассистента — иначе обучение тратит capacity на boilerplate.

## Связь с SOTA

Precedent: 150_gemma_lora (Gemma E4B rsLoRA, first-image, hard mining) дал
0.9342 macro на fold 4. Отличие этой ветки: все 5 изображений, стандартный LoRA
(не rsLoRA), loss на таргете, OCR-абляция. Также связано с 110/130
(assistant-suffix-only loss) — тот же принцип узкого таргета.

## Acceptance criterion

- Primary: macro-F1 на val (diagnostic), далее обязателен пересчёт на grouped folds.
- Guardrails: F1 по обеим категориям; parse_fail_rate < 2%.
- Шум: прирост < +0.03 к zero-shot — эксп не идёт в сабмит.

## Инженерные грабли (важно для повторения)

1. **NaN в backward**: forward чистый, но backward по gradient checkpointing даёт
   NaN-грады на редких парах изображений (триггер: пара id 10210 × 10519).
   Лечение: `NaNProofAdamW` — зануление нефинитных градов перед step
   (плохой микробатч просто пропускается, ~1-3 батча на эпоху).
2. `Gemma4ClippableLinear` в vision/audio tower не поддерживается peft 0.20 →
   LoRA только на `language_model.*` (258 модулей, q/k/v/o/gate/up/down).
3. Vision tower требует `mm_token_type_ids` и `image_position_ids` в input —
   collate_fn их сохраняет/конкатенирует; pixel_values конкатенируются по батчу.
4. `processor(truncation=True)` ломается на image-токенах → truncation=False,
   лимит длины вручную (3072, left-truncate).
5. GPU: только 0-5 (6-7 заняты чужими задачами).

## Запуск

```bash
bash tmp/QC/run_wave1.sh   # 4× LoRA (GPU 0-3) + 2× contrastive (GPU 4-5)
```

Волна 1 (Track A): expA01 noocr r64 lr1e-4 · expA02 noocr r32 · expA03 ocrlong r64 · expA04 std r64.

## Результат и решение

Обучение идёт. Zero-shot точки (val, n=1298): std 0.3379 / noocr 0.3776 macro-F1.
См. `results/metrics.json` и `docs/status/2026-08-21-maksimcrewceo-qc-tracks.md`.
