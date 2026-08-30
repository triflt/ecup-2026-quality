# Experiment 709 — Gemma 4 LoRA epoch/rank screen

Fold-0 screen replacing only the Qwen3-VL-2B leg of frozen solution140.
Two public Gemma 4 multimodal backbones (`E2B-it`, `E4B-it`) are crossed with
rsLoRA ranks 16/32/64 and trained continuously for five epochs. Every epoch is
evaluated standalone and later inside the unchanged solution140 route. Data,
sampler, seed, optimizer, image cache and checkpoint schedule match exp703.

## Acceptance criterion

Selection is based only on the frozen fold-0 solution140 replay. A checkpoint
must improve Macro F1 without an unacceptable rare-class FP/FN trade-off and
must then be confirmed independently before refit, packaging or Public use.

## Result and decision

All 30 epoch checkpoints were evaluated. `gemma4_e4b_r16` epoch 4 led fold 0
with Macro delta `+0.0069396988`, 16 corrections, 9 regressions and flammable
F1 `0.939759` (`FP=4`, `FN=1`). This is a screen winner, not a production
winner: confirmation remains required. See `results/metrics.json` and
`results/terminal_summary.json`.

Private execution presets and absolute infrastructure paths are intentionally
excluded from the repository.
