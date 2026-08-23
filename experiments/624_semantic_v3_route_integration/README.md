# Experiment 624 — semantic-v3 route integration

Status: `blocked_by_dependency`.

This stage integrates **one** component only after experiment 621, 622, or 623
passes its frozen development gates. The reference-route weights remain those
of original route 603. Any new threshold must be calibrated donor-only; a
second-level router trained on historical OOF scores is forbidden.

If none of 621/622/623 passes, this stage records original route 603 as the
cycle winner and performs no speculative fusion. No metric or artifact is
claimed while the upstream decision is unresolved.
