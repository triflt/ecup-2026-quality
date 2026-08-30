# Shared Qwen training contract

This directory is retained as a compatibility layer for the later Qwen
experiments merged into `main`. Those experiments import the frozen grid,
runtime, training, and artifact-verification contracts from here.

The original standalone grid launch package is intentionally not restored.
Only the shared, reproducibility-critical source files remain; operational
presets and platform-specific launch wrappers stay excluded.
