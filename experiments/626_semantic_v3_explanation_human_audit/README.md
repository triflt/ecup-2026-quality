# Experiment 626 — independent explanation audit

Status: `blocked_by_dependency`.

This stage prepares a new blind audit of at least 300 rows for the explanation
renderer selected by experiment 624 and reproduced by experiment 625. It must
exclude the 200 legacy experiment-490 audit rows and review exact evidence
relevance, object of sale, negation, composition or completeness, verdict
agreement, and unsupported facts.

Human ratings must remain absent until a real reviewer supplies them. An
automated structural check cannot substitute for the human gate. Therefore no
audit score or acceptance is claimed in this card.

`build_human_audit.py` creates a development-only, 300-component packet and a
frozen manifest below `.local`. It requires explicit experiment-490 and
experiment-622 exclusion manifests plus all other known prior audit/review ID
manifests. `evaluate_human_audit.py` accepts only a separate completed copy and
fails closed on any change to the sample, order, source card, span, concept,
verdict, or explanation. See `human_audit_protocol.md` for the review rules.
