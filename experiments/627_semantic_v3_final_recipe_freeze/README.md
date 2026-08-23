# Experiment 627 — freeze one final recipe

Status: `skipped_by_gate`.

No new recipe passed route integration, independent reproduction, and the real
human explanation audit. Consequently no final recipe manifest was frozen and
the sealed holdout remains closed.

Exactly one recipe may be frozen after development validation, independent-seed
reproduction, safety checks, and the required explanation audit are resolved.
The freeze must bind code, component weights, route weights, thresholds,
renderer, seed, data selection, and checksums in a fail-closed manifest.
`manifest_schema_v1.json` enumerates the required hashes and
`verify_manifest.py` refuses an incomplete, multi-recipe, or post-sealed
manifest. The sealed comparator and numerical acceptance policy are part of
that frozen contract rather than a decision made after reveal.

No recipe has been selected yet. The sealed holdout remains closed, and no
manifest or artifact is represented as complete by this placeholder card.
