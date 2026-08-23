# Experiment 630 — production submission assembly

Status: `blocked_by_dependency`.

A production ZIP may be assembled only from a successful experiment-629
full-data refit. The archive must pass SHA-256 and ZIP integrity checks, output
schema and unique-ID checks, cover both categories without NaN, satisfy runtime
limits with margin, and render a concrete product feature rather than a generic
violation statement.

No ZIP, checksum, metric, or component provenance is claimed while experiment
629 is incomplete. If upstream acceptance fails, this stage records the Public
champion fallback instead of creating a fictitious submission.

Acceptance additionally requires exact equality with the expected ID set and
row count, an allowlisted archive, finite outputs, measured Public and Private
runtime margins, and the closed reasoning chain: card → exact substring or
safe `NO_EVIDENCE` → concept → verdict → short explanation containing the
quote. Generic violation text and unsupported visual or absence claims fail.
