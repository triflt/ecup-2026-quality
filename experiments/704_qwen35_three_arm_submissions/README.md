# Experiment 704: three-arm submission ladder

This auxiliary experiment makes the three experiment-698 arms independently
submittable without changing the immutable experiment-719 final selector.

Two artifact tiers are produced:

- `results/provisional/`: fold-3 adapters trained on the same outer-train split.
  These are early leaderboard probes, not final full-data models.
- `results/full/`: explicit full-data refits of `gold_control`,
  `hardneg_candidate`, and `rank_candidate` using the frozen experiment-715
  runtime and strict OOF teacher targets.

Every ZIP is the frozen allowed solution-140 base with only `adapter_qwen35/`
and metadata replaced. The 27B teacher and the 4B base weights are never
included in a submission.

All three full-data ZIPs receive a three-row one-H100 runtime smoke under the
same advisory GPU lock used by experiments 716 and 718. Static ZIP validation
is completed before publication, so provisional ZIPs can be submitted while
the longer full-data lane continues.

The normal experiment-715 winner refit and experiment-719 final selection keep
their original gate semantics. Experiment 704 is an auxiliary submission
ladder and cannot promote itself into experiment 719.
