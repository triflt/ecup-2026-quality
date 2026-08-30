# 702: doubled-capacity dual-LoRA epoch sweep

Exact paired capacity ablation against experiment 701. The only changed factor
is rsLoRA capacity: `r=16, alpha=32` becomes `r=32, alpha=64`; `alpha/r` remains
constant. Both Qwen3.5-4B and Qwen3-VL-2B run epochs 1/2/3/4/5 on frozen fold 0.

Each arm starts on the same GPU immediately after its corresponding rank-16
queue is terminal, keeping total owned use at at most eight H100s. Sampling,
seed, LR, data, images, preprocessing, targets and evaluation remain identical.
No full5, refit, package or Public is authorized.

