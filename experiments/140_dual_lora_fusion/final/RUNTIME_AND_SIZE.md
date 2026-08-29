# Offline, runtime and size

These are submission-operability requirements, not model-quality metrics.

## Offline

During judging the container has no internet. Solution 140 therefore loads base
models from the competition-provided `SHARED_MODELS_PATH` and all trained LoRA
and classifier files from the submission archive. The production runner has no
top-level HTTP client import and does not download anything. It enumerates images
only below the input CSV's sibling `images/` directory and passes resolved
`file://` URIs to the vendored vision utility. That utility retains generic URL
support from upstream, but the solution-140 production call path never supplies
an HTTP(S) URI.

## Runtime

The official limits are 3 minutes for Check, 20 minutes for Public and 40
minutes for Private on one H100 80 GB. The recorded solution-140 smoke processed
600 products / 2,274 images in `228.76 s`, projecting approximately `10.17 min`
for Public and `24.15 min` for Private. This is passing historical evidence,
not a guarantee for a rebuilt archive: the exact published-weight package must
repeat the smoke before final submission.

## Size

The submission ZIP must be at most 5 GB and a custom compressed Docker image at
most 15 GB. Solution 140 relies on competition-mounted base models, so only the
two LoRA adapters, fitted classifier bundles and Python source are packaged.
The historical submitted archive SHA-256 is recorded in
`reports/champion.json`; per-file size and hashes will be filled in
`artifact-contract.json` when the weights are published.
