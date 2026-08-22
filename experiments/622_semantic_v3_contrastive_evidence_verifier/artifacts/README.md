# Artifact policy

This directory contains no generated candidate banks or audit rows.

Candidate banks, donor targets and the 300-row review sheet contain row-level
data and must be written only to an ignored private execution directory. A
publishable preflight report may contain counts, rates, frozen configuration
hashes and pass/fail states only. It must not contain row IDs, candidate IDs,
source text, spans, offsets, semantic-component identifiers or reviewer notes.
