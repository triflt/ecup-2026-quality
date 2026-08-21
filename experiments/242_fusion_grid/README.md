# 242: Fusion Grid (Track C, Максим/QC)

## Гипотеза

Простой weighted-fusion поверх вероятностей из Track A (Gemma LoRA SFT) и
Track B (контрастивный эмбеддер) с пороговым тюнингом по val даст прирост
к лучшей одиночной ветке за счёт независимости ошибок генеративной и
эмбеддинговой моделей.

## Данные и preprocessing

- Val-сплит как в 240/241 (`tmp/QC/split.json`, n=1298).
- Входы: per-item probability/score от каждого обученного чекпоинта
  (A01-A04, B01-B02) + zero-shot базы.
- `run_fusion_grid.py` перебирает сетку весов и порогов; `eval_fusion.py`
  считает macro-F1 по категориям.

## Связь с SOTA

- Precedent: 040_late_fusion (лучший ранний Public 0.8066), 070/080/090
  (category-mixed fusion — best grouped CV 0.9072), 140_dual_lora_fusion.
- Отличие: фьюжн с моими QC-ветками (Gemma multi-image + contrastive),
  а не с Qwen-адаптерами Данека.

## Acceptance criterion

- Primary: macro-F1 на val > лучшая одиночная ветка + 0.01.
- Обязателен пересчёт победителя на grouped folds перед сабмитом.

## Результат и решение

Ожидает завершения Track A/B (нужны скоры с чекпоинтов). См.
`docs/status/2026-08-21-maksimcrewceo-qc-tracks.md`.
