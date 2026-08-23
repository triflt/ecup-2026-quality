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
