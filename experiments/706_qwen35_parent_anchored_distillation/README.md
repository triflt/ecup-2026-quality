# 706: Parent-anchored Qwen3.5 distillation

> **Final status: REJECTED.** The strict-OOF final package scored
> `0.8217663209` public macro-F1 versus `0.8923976821` for experiment 140.
> Do not promote the final package, its continuation, or any earlier routed
> probe. See [RESULTS.md](RESULTS.md) for the complete record and postmortem.

The production Qwen3.5 adapter from experiment 140 is immutable for BAD. A
distilled adapter may affect only the flammable route. Immediate routed probes
blend category-local percentile scores conservatively, while the training lane
continues from the production adapter with an explicit parent-logit anchor.

The experiment was opened after the scratch full-refit arms from experiment 705
scored `0.82514`, `0.80051`, and `0.79685` against production `0.89240`.
Those results reject scratch replacement and do not reject conservative
parent-anchored residual transfer.

The completed strict five-fold experiment did reject that residual-transfer
hypothesis. The teacher itself underperformed experiment 140 on full OOF, and
the adapter-selection metric did not validate the exact production fusion.
