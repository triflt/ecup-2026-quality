# 723: Embedder tuning for the top solution (former 243-QC)

## Гипотеза

Текущий чемпион (`190_shingle_neighbor_prior`, Public 0.8919) строится на
rank-fusion из robust text base, Qwen3-VL rsLoRA и Qwen3.5 rsLoRA, где каждая
модель даёт вероятности по категориям. Мы гипотезируем, что:

1. Дообучение **Qwen3-VL-Embedding-2B** на нашем трейне с hard-mined negatives
   даст лучший image-embedding head, чем zero-shot.
2. Разделение адаптеров по категориям (**BAD-only** vs **flammable-only**)
   снимет конфликт сигналов между классами и поднимет редкий класс
   «Легковоспламеняющиеся».
3. Когда дообученный эмбеддер оценивается через **LinearSVC + rank fusion**
   (как у Данека), а не через LR/KNN, полученный head можно подставить в
   существующую чемпионскую схему и получить прирост на nested grouped CV.

## Почему это должно сработать

- Precedent: `030_qwen3vl_embedding` показал, что image signal полезен.
- Precedent: `060_multiview_fusion` показал, что отдельный first-image-only
  embedding даёт независимый сигнал.
- Champion (`190`) уже использует Qwen3-VL/Qwen3.5 эмбеддеры как ключевой
  компонент. Если дообучение под наш датасет улучшит этот компонент,
  gain должен перенестись на всю систему.
- Первая попытка (`721`) дала негативный результат, но она использовала
  LR/KNN head, r16/alpha16, 3 эпохи, no hard mining — то есть **не ту же
  постановку**, что у Данека.

## Acceptance criterion

- Primary: nested grouped CV macro-F1 лучше текущего чемпиона
  `0.9118425206` (или хотя бы лучше baseline без embedding tuning) с gain
  не менее **+0.005**.
- Guardrails: категория BAD не падает более чем на 0.005; flammable не падает
  более чем на 0.01.
- Runtime: дообучение + инференс эмбеддеров не превышает лимит сабмита
  (projected Private < 60 мин).

## План абляций (Wave 1.5)

1. **Один адаптер, LinearSVC head** (baseline ablation):
   - r=32/64, alpha=r, lr=1e-4, 5-10 эпох.
   - Sigmoid contrastive loss с hard/semi-hard negatives.
   - Оценка: LinearSVC на дообученных эмбеддингах; сравнение с zero-shot
     на том же сплите.
2. **Два категориальных адаптера**:
   - Adapter BAD: обучаем только на BAD-примерах.
   - Adapter flammable: обучаем только на flammable-примерах.
   - Для каждой категории свой эмбеддинг, затем отдельный LinearSVC.
3. **Интеграция в чемпионскую схему**:
   - Заменить zero-shot Qwen3-VL embedding head на дообученный.
   - Повторить rank fusion + shingle prior на nested grouped folds.
   - Если gain ≥ +0.005 — готовим сабмит и official-image smoke.
4. **Запасные варианты**:
   - Triplet loss с hard mining.
   - n_images=5 vs n_images=2.
   - Добавить OCR-текст в input эмбеддера.

## Данные

- `competition_train_v1`, frozen `grouped_text_v1` folds.
- Label mapping: raw `1` = «бан», raw `0` = «не бан» (см. статус Максима).
- Изображения: до 5 штук; для baseline ablation n_images=2, для финальной
  интеграции — 5 (как у Данека).

## Запуск

Локальные QC-скрипты лежат в `experiments/721_qwen3vl_contrastive/src/`:
- `train_contrastive.py` — дообучение адаптера.
- `eval_embedder.py` — нужно дополнить LinearSVC-эвал (см. TODO ниже).

```bash
# Пример одного конфига
python3 experiments/721_qwen3vl_contrastive/src/train_contrastive.py \
  --exp_id expB03 --r 64 --alpha 64 --lr 1e-4 --epochs 5 \
  --batch_size 8 --grad_accum 4 --samples_per_class 2 \
  --loss_type sigmoid --hard_mining --n_images 2 --gpu 0

# Эвал (после добавления LinearSVC в eval_embedder)
python3 experiments/721_qwen3vl_contrastive/src/eval_embedder.py \
  --adapter exps/expB03/adapter_embed --out exps/expB03/eval \
  --gpu 0 --batch_size 8 --n_images 2 --head linear_svc
```

## TODO / Что нужно сделать

- [ ] Дописать `eval_embedder.py`: добавить LinearSVC head с
      class_weight='balanced' и подбор C по внутреннему val.
- [ ] Реализовать hard negative mining в `train_contrastive.py`.
- [ ] Разделить датасет на BAD-only / flammable-only для двух адаптеров.
- [ ] Подготовить скрипт интеграции в чемпионскую rank-fusion схему.
- [ ] Запустить на nested grouped folds и сравнить с champion `190`.

## Связь с SOTA и чемпионом

- Champion: `190_shingle_neighbor_prior`, Public 0.8919, nested macro 0.9118.
- Отличие: дообучение эмбеддера + категориальные адаптеры.
- Если не заходит — честно фиксируем отрицательный результат, не ломаем
  действующего чемпиона.
