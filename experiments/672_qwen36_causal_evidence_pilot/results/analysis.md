# Experiment 672 result

## Decision

`NO_GO_REJECT_27B_EXPLANATION_SCALE_UP`.

The frozen paired pilot completed all 40 rows for both candidates with exact runtime IDs,
matching runtime and output hashes, and zero reads of labels or sealed rows. The 4B control
passed the strict output contract on 26/40 rows. Qwen3.6-27B passed on 38/40 rows, below the
preregistered 40/40 gate.

The two 27B failures were one non-exact name quote and one non-exact description quote.
They were not infrastructure, sample-integrity, or verdict-lock failures. Because human
ratings cannot reverse the failed automatic gate, the 80-candidate human review was skipped.

## Runtime and infrastructure

- Both first registered jobs failed before model loading because the base environment
  enforces PEP 668.
- The controlled retry changed only the install command to the already proven
  `pip --break-system-packages` contract; both model runs then succeeded.
- Measured generation time was 161.46 seconds for 4B and 249.73 seconds for 27B.
- No classification prediction changed and no Public submission was made.

## Consequence

This experiment does not authorize a 200-row explanation audit, classification training,
full teacher extraction, distillation, or a Public submission. The next cheap independent
step is to consume the accepted fail-closed OCR availability manifest from experiment 660;
unverified OCR remains explicitly unavailable.
