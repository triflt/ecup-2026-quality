# 701: fast dual-LoRA epoch sweep

One-factor diagnostic over solution140 training: only the number of epochs changes.
Qwen3.5-4B and Qwen3-VL-2B are evaluated independently on frozen fold 0 at
epochs 1/2/3/4/5. Sampling, seed, LR, preprocessing, rsLoRA configuration and
validation membership remain the historical 140 recipe.

Runs are independent rather than warm-started checkpoints, so each epoch count
has the exact cosine schedule corresponding to its declared horizon. Eight H100s
run the first eight arms in parallel; the remaining two Qwen3-VL arms start when
slots free. No full5, package or Public is authorized by this diagnostic.

Promotion requires a meaningful final solution140 replay gain, no category F1
drop, no flammable FN increase and confirmation of one selected epoch on fold 3.

