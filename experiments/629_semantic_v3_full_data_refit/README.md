# Experiment 629 — accepted-recipe full-data refit

Status: `blocked_by_dependency`.

Full-data training is allowed only if the one-time experiment-628 evaluation
accepts the frozen recipe. Each component receives one fit on all permitted
labeled data, with no seed or hyperparameter search. Checkpoints must be
integrity-checked and stored only in approved private locations.

No training or checkpoint is claimed before experiment 628 is accepted. If it
is not accepted, this stage is skipped rather than manufacturing an artifact.

`full_data_contract.py plan` creates a deterministic one-fit-per-component plan
only after checking the accepted sealed result against the exact frozen recipe
and confirming that every training manifest contains zero sealed rows.
`full_data_contract.py verify` then checks the resulting component manifest,
including immutable input/config hashes, one GPU and one fit per component,
artifact size and checksum, integrity status, runtime evidence, and provenance.
It does not launch training or infer acceptance from the presence of a file.
