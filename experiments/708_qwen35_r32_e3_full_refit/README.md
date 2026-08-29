# Experiment 708 — Qwen3.5-4B r32 epoch3 full refit

Full-data refit of the winning exp703 screen checkpoint recipe. The only changes
relative to the solution140 Qwen3.5 leg are rsLoRA rank/alpha `16/32 -> 32/64`
and training horizon `1 -> 3` epochs. Seed, hard sampler, LR, optimizer, image
view, base model and target modules remain frozen. This run produces the adapter
used to replace only the Qwen3.5 leg in solution140.
