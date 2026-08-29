# Evaluation of solution 140

Solution 140 has two different local scores because they answer different
questions. They must never be presented as one interchangeable CV number.

| Protocol | Macro F1 | What it estimates |
|---|---:|---|
| Nested grouped outer-fold fusion | `0.9118425206` | Generalisation to products whose normalized text group is held out; fusion is selected only on the other four folds |
| Nested recurrence simulation | `0.942878` | Full submitted recipe including exact/name donor memory under a recurrence-oriented simulation |
| Repeated 70/30 recurrence simulation | `~0.94109` | Stability of the donor-memory benefit under repeated train/test simulations |
| Public leaderboard | `0.8923976821` | Actual score of the immutable submitted solution-140 archive |

The first score is the architecture-selection result recorded in
`experiments/140_dual_lora_fusion/results/metrics.json`. The recurrence scores
describe the additional train-donor memory used by the submitted recipe and are
recorded in `reports/champion.json` and `reports/submissions.csv`.

## Current evaluation policy

- `grouped_text_v1` / nested CV preserves comparability with the historical
  solution-140 programme.
- `semantic_family_v3` is the stricter architecture-promotion guard for new
  components. Related product families do not cross development folds.
- A one- or two-fold result is a screen, never a full leaderboard result.
- The canonical local leaderboard is
  [`reports/semantic-v3-leaderboard.csv`](../../../reports/semantic-v3-leaderboard.csv).
- Public is used to validate an immutable submission, not to retune it.

Solution 140 predates the full semantic-family programme. Its checksum-locked
replay appears as experiment 635's `original_qwen35` reference row on the local
leaderboard; later components are promoted only if they beat that route under
the frozen gates.
