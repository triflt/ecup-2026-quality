# Experiment 709 — Gemma 4 LoRA epoch/rank screen

Fold-0 screen replacing only the Qwen3-VL-2B leg of frozen solution140.
Two public Gemma 4 multimodal backbones (`E2B-it`, `E4B-it`) are crossed with
rsLoRA ranks 16/32/64 and trained continuously for five epochs. Every epoch is
evaluated standalone and later inside the unchanged solution140 route. Data,
sampler, seed, optimizer, image cache and checkpoint schedule match exp703.
