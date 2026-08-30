#!/usr/bin/env bash
set -euo pipefail

E=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution
X="$E/experiments/706_qwen35_parent_anchored_distillation"
P=/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python
RESULTS="$X/results"
LOCAL="$X/.local"
FULL_OOF="$RESULTS/full_oof"
MAIN="$RESULTS/full_refit_coverage"
CONT="$RESULTS/full_refit_continuation20"
SUBMISSIONS="$RESULTS/submissions"
BASE="$E/experiments/716_distilled_140_adapter_replacement/.local/allowed-140-base.zip"

mkdir -p "$RESULTS" "$LOCAL" "$SUBMISSIONS"
exec 9>"$LOCAL/continuation-wait.lock"
if ! flock -n 9; then
  echo "another exp706 continuation waiter owns the lock"
  exit 0
fi

while [[ ! -s "$MAIN/output_contract.json" ]]; do
  echo "$(date -Is) waiting_for_selected_update40"
  sleep 60
done

# The selected trainer publishes its contract immediately before process exit.
# Continue on an otherwise idle physical GPU4 so formal fold jobs cannot block it.
while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=4 | tr -d ' ')
  if (( used < 5000 )); then
    break
  fi
  echo "$(date -Is) waiting_for_gpu1_release memory_mib=$used"
  sleep 15
done

if [[ ! -s "$CONT/output_contract.json" ]]; then
  CUDA_VISIBLE_DEVICES=4 "$P" "$X/continue_parent_anchored_full.py" \
    --runtime-dir "$FULL_OOF/runtime" \
    --teacher-target-dir "$FULL_OOF/targets" \
    --original-parent-adapter "$X/.local/parent-adapter" \
    --initial-adapter "$MAIN/checkpoints/update_0040/adapter" \
    --prior-contract "$MAIN/output_contract.json" \
    --parent-predictions \
      "$RESULTS/full_parent_scores/f0/predictions.jsonl" \
      "$RESULTS/full_parent_scores/f1/predictions.jsonl" \
      "$RESULTS/full_parent_scores/f2/predictions.jsonl" \
      "$RESULTS/full_parent_scores/f3/predictions.jsonl" \
      "$RESULTS/holdout_eval/parent/predictions.jsonl" \
    --output-dir "$CONT" \
    --additional-updates 20 \
    --pairs-per-update 133 \
    --pairs-per-microbatch 8 \
    --checkpoint-every 5 \
    --learning-rate 3e-6 \
    --anchor-weight 4.0 \
    --hard-weight 0.10 \
    --rank-weight 0.05 \
    > "$LOCAL/full-refit-continuation20.log" 2>&1
fi

ADAPTER="$CONT/checkpoints/update_0020/adapter"
ZIP="$SUBMISSIONS/exp706-strict-oof-cont20-submit.zip"
PACKAGE="$SUBMISSIONS/exp706-strict-oof-cont20-package.json"
if [[ ! -s "$ZIP" || ! -s "$PACKAGE" ]]; then
  if [[ -e "$ZIP" || -e "$PACKAGE" ]]; then
    echo "partial continuation package requires diagnosis"
    exit 1
  fi
  "$P" "$X/build_final_submission.py" \
    --base-submission "$BASE" \
    --distill-adapter "$ADAPTER" \
    --full-refit-contract "$CONT/output_contract.json" \
    --allow-continuation \
    --alpha 1.0 \
    --output "$ZIP" \
    --report "$PACKAGE" \
    > "$LOCAL/build-continuation20-package.log" 2>&1
fi

COMPAT="$SUBMISSIONS/cont20-runtime-package-compat.json"
if [[ ! -s "$COMPAT" ]]; then
  "$P" - "$PACKAGE" "$ZIP" "$COMPAT" <<'PY'
import hashlib
import json
import sys

package = json.load(open(sys.argv[1], encoding="utf-8"))
digest = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
if package.get("schema_version") != "exp706_final_submission_v1":
    raise SystemExit("unexpected continuation package report")
if package.get("selection_status") != "SPECULATIVE_NO_NEW_HOLDOUT":
    raise SystemExit("continuation package lost its speculative marker")
if package.get("output_sha256") != digest:
    raise SystemExit("continuation ZIP checksum mismatch")
value = {
    "schema_version": "exp716_submission_package_v1",
    "output_sha256": digest,
    "teacher_in_submission": False,
    "under_5_gib": bool(package.get("under_5_gib")),
    "source_exp706_package_report": sys.argv[1],
}
open(sys.argv[3], "w", encoding="utf-8").write(
    json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
PY
fi

SMOKE3="$SUBMISSIONS/cont20-runtime-smoke-3.json"
SMOKE600="$SUBMISSIONS/cont20-runtime-smoke-600.json"
ACCEPTANCE="$SUBMISSIONS/cont20-runtime-acceptance.json"
RUNTIME_TOOL="$E/experiments/716_distilled_140_adapter_replacement/runtime_smoke.py"
exec 8>"$E/.local/mlitvinov_q38_runtime_gpu3.lock"
flock 8
common=(
  --submission "$ZIP"
  --package-report "$COMPAT"
  --data /home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/data/data.csv
  --images /home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/data/images/images
  --python "$P"
  --qwen-embed-model /home/jovyan/shares/SR008.fs2/litvinov/models/Qwen3-VL-Embedding-2B
  --qwen-instruct-model /home/jovyan/shares/SR008.fs2/litvinov/models/Qwen3-VL-2B-Instruct
  --qwen35-model /home/jovyan/shares/SR008.fs2/litvinov/models/Qwen3.5-4B
  --device 3
)
if [[ ! -s "$SMOKE3" ]]; then
  CUDA_VISIBLE_DEVICES=3 "$P" "$RUNTIME_TOOL" "${common[@]}" \
    --rows 3 --work-dir "$LOCAL/cont20-runtime-smoke-3" --report "$SMOKE3" \
    > "$LOCAL/cont20-runtime-smoke-3.log" 2>&1
fi
if [[ ! -s "$SMOKE600" ]]; then
  CUDA_VISIBLE_DEVICES=3 "$P" "$RUNTIME_TOOL" "${common[@]}" \
    --rows 600 --work-dir "$LOCAL/cont20-runtime-smoke-600" --report "$SMOKE600" \
    > "$LOCAL/cont20-runtime-smoke-600.log" 2>&1
fi
if [[ ! -s "$ACCEPTANCE" ]]; then
  "$P" - "$PACKAGE" "$SMOKE3" "$SMOKE600" "$ACCEPTANCE" <<'PY'
import json
import sys

package = json.load(open(sys.argv[1], encoding="utf-8"))
smoke3 = json.load(open(sys.argv[2], encoding="utf-8"))
smoke600 = json.load(open(sys.argv[3], encoding="utf-8"))
if smoke3.get("decision") != "RUNTIME_SMOKE_PASS" or smoke600.get("decision") != "RUNTIME_SMOKE_PASS":
    raise SystemExit("continuation runtime smoke failed")
if smoke600.get("projected_private_minutes_3800", 1e9) >= 40:
    raise SystemExit("continuation projected runtime exceeds limit")
value = {
    "schema_version": "exp706_continuation_runtime_acceptance_v1",
    "decision": "RUNTIME_ACCEPTED_SPECULATIVE_QUALITY",
    "submission_sha256": package["output_sha256"],
    "projected_public_minutes_1600": smoke600.get("projected_public_minutes_1600"),
    "projected_private_minutes_3800": smoke600.get("projected_private_minutes_3800"),
    "peak_gpu_memory_mib": smoke600.get("peak_gpu_memory_mib"),
    "one_h100": True,
    "selection_status": "SPECULATIVE_NO_NEW_HOLDOUT",
}
open(sys.argv[4], "w", encoding="utf-8").write(
    json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
PY
fi

sha256sum "$ZIP"
echo "$(date -Is) continuation20 package and runtime accepted: $ZIP"
