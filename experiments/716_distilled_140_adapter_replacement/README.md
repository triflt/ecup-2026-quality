# 716: fixed-architecture distilled adapter replacement

This stage changes exactly one model component in the frozen, integrity-checked
allowed-model base tree from `QC/step1_140ocr/submission_ocr`: `adapter_qwen35`.
The source tree is read-only and bound by a per-file canonical SHA-256 manifest.
`run.py`, the Qwen3-VL adapter, text/OCR
embedding leg, priors and fusion constants remain byte-identical. Metadata is
updated only to record the distilled adapter provenance.

The replacement is accepted only from a non-smoke experiment-715 full-refit
contract with a valid self-hash, a dual-gate promoted distillation method, the
expected r16/alpha32/qkvo rsLoRA configuration and an exact adapter checksum.
The builder rejects unsafe or duplicate ZIP paths, checks every unchanged member
by content hash, verifies the resulting ZIP, forbids bundled base weights and
enforces the 5 GiB limit.

The adjacent Giga-Embeddings archive is explicitly not used: that backbone is
not in the competition allow-list supplied for deployment. The frozen base
runtime is checked to reference only Qwen3-VL-Embedding-2B,
Qwen3-VL-2B-Instruct and Qwen3.5-4B, with base weights supplied by platform
mounts.

This proves packaging and single-factor architecture isolation. It does not
claim deployment completion until real one-H100 smoke reports are bound into
the immutable `runtime_acceptance.json` verdict with measured runtime/VRAM
evidence.

The package builder also requires the immutable experiment-717 incumbent
evidence audit. Because the exact solution-140 component OOF files are absent,
the audit fail-closes on frozen fusion weights and thresholds and permits only
the isolated `adapter_qwen35` replacement. The package metadata and report bind
the audit checksum and explicitly do not claim a measured fixed-fusion OOF
delta.

The builder additionally requires the full-refit adapter to declare
`solution140_first_image_thumbnail_448_lanczos_v1`. This binds training and OOF
to the unchanged Qwen3.5 preprocessing already present in solution 140 and
prevents packaging an adapter evaluated at a different image resolution.

`runtime_smoke_wait.sh` first executes a three-row schema smoke and then a
600-row timing run with all three real allow-listed model mounts on exactly one
visible H100. It records output validity, peak VRAM and projected Public/Private
minutes. The finalizer rejects missing, failed or checksum-mismatched smoke
reports. The waiter never installs dependencies automatically; a missing
package in the development environment is reported rather than mutating another
environment.

Before either run, `runtime_preflight.py` pins and verifies the actual hackathon
environment (`torch 2.9.0`, GPU Paddle 3.3.0, PaddleOCR 3.7.0, PaddleX 3.7.2 and
Qwen VL Utils 0.0.14), requires exactly one visible GPU for both frameworks and
executes a CUDA matmul through each stack. This catches dependency or CUDA ABI
breakage before loading the three competition models.

The exp716 and conditional exp718 runtime waiters share one advisory `flock`
for physical GPU0. Each holds it across preflight and both timed smokes, so peak
VRAM, elapsed time and OOM behavior cannot be corrupted by the other candidate.
