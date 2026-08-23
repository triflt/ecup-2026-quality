# Experiment 629 — accepted-recipe full-data refit

Status: `blocked_by_dependency`.

Full-data training is allowed only if the one-time experiment-628 evaluation
accepts the frozen recipe. Each component receives one fit on all permitted
labeled data, with no seed or hyperparameter search. Checkpoints must be
integrity-checked and stored only in approved private locations.

No training or checkpoint is claimed before experiment 628 is accepted. If it
is not accepted, this stage is skipped rather than manufacturing an artifact.
