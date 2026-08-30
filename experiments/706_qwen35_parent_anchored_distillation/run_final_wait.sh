#!/usr/bin/env bash
set -euo pipefail

E=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-distill-execution
T=/home/jovyan/shares/SR008.fs2/litvinov/tmp/ecup-qwen38-teacher-distill/experiments/697_qwen38_27b_grouped_teacher
X="$E/experiments/706_qwen35_parent_anchored_distillation"
P=/home/jovyan/shares/SR008.fs2/me/envs/mlitvinov_vlm/bin/python
BASE="$E/experiments/716_distilled_140_adapter_replacement/.local/allowed-140-base.zip"
RESULTS="$X/results"
LOCAL="$X/.local"
FULL_OOF="$RESULTS/full_oof"
FULL_REFIT="$RESULTS/full_refit_coverage"
SUBMISSIONS="$RESULTS/submissions"

mkdir -p "$RESULTS" "$LOCAL" "$SUBMISSIONS"
exec 9>"$LOCAL/final-wait.lock"
if ! flock -n 9; then
  echo "another exp706 final waiter owns the lock"
  exit 0
fi

required=(
  "$T/.local/output-f4/output_contract.json"
  "$RESULTS/holdout_selection.json"
  "$RESULTS/full_parent_scores/f0/metrics.json"
  "$RESULTS/full_parent_scores/f1/metrics.json"
  "$RESULTS/full_parent_scores/f2/metrics.json"
  "$RESULTS/full_parent_scores/f3/metrics.json"
  "$LOCAL/full-coverage-smoke/output_contract.json"
)
while true; do
  missing=0
  for path in "${required[@]}"; do
    if [[ ! -s "$path" ]]; then
      missing=$((missing + 1))
    fi
  done
  if (( missing == 0 )); then
    break
  fi
  echo "$(date -Is) waiting_for_contracts=$missing"
  sleep 60
done

"$P" - "$RESULTS/holdout_selection.json" "$LOCAL/full-coverage-smoke/output_contract.json" <<'PY'
import json
import sys

selection = json.load(open(sys.argv[1], encoding="utf-8"))
smoke = json.load(open(sys.argv[2], encoding="utf-8"))
if selection.get("decision") != "PROMOTE_TO_FULL_OOF_REFIT":
    raise SystemExit("strict holdout did not promote the candidate")
if selection.get("selected_update") != 40:
    raise SystemExit("strict holdout selected an unexpected update")
if smoke.get("technical_smoke") is not True:
    raise SystemExit("coverage-aware trainer smoke contract is invalid")
if smoke.get("unique_flammable_rows_covered") != smoke.get("unique_flammable_rows"):
    raise SystemExit("coverage-aware trainer smoke omitted rows")
PY

if [[ ! -s "$FULL_OOF/targets/teacher_target_contract.json" ]]; then
  if [[ -e "$FULL_OOF" ]]; then
    echo "partial full OOF directory requires diagnosis: $FULL_OOF"
    exit 1
  fi
  temporary=$(mktemp -d "$RESULTS/.full_oof.XXXXXX")
  "$P" "$X/build_oof_runtime.py" \
    --teacher-experiment "$T" \
    --folds 0 1 2 3 4 \
    --output-dir "$temporary" \
    > "$LOCAL/build-full-oof.log" 2>&1
  mv "$temporary" "$FULL_OOF"
fi

"$P" - "$FULL_OOF/targets/teacher_target_contract.json" <<'PY'
import json
import sys

contract = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "folds": [0, 1, 2, 3, 4],
    "strict_oof": True,
    "teacher_trained_without_each_target_row": True,
}
for key, value in expected.items():
    if contract.get(key) != value:
        raise SystemExit(f"full OOF contract mismatch for {key}")
if int(contract.get("flammable_rows", 0)) < 5000:
    raise SystemExit("full OOF flammable coverage is incomplete")
PY

if [[ ! -s "$FULL_REFIT/output_contract.json" ]]; then
  CUDA_VISIBLE_DEVICES=1 "$P" "$X/train_parent_anchored_full.py" \
    --runtime-dir "$FULL_OOF/runtime" \
    --teacher-target-dir "$FULL_OOF/targets" \
    --parent-adapter "$X/.local/parent-adapter" \
    --parent-predictions \
      "$RESULTS/full_parent_scores/f0/predictions.jsonl" \
      "$RESULTS/full_parent_scores/f1/predictions.jsonl" \
      "$RESULTS/full_parent_scores/f2/predictions.jsonl" \
      "$RESULTS/full_parent_scores/f3/predictions.jsonl" \
      "$RESULTS/holdout_eval/parent/predictions.jsonl" \
    --output-dir "$FULL_REFIT" \
    --optimizer-updates 40 \
    --pairs-per-microbatch 8 \
    --checkpoint-every 5 \
    > "$LOCAL/full-refit-coverage.log" 2>&1
fi

FINAL_ADAPTER="$FULL_REFIT/checkpoints/update_0040/adapter"
FINAL_ZIP="$SUBMISSIONS/exp706-strict-oof-final-submit.zip"
FINAL_REPORT="$SUBMISSIONS/exp706-strict-oof-final-package.json"
if [[ ! -s "$FINAL_ZIP" || ! -s "$FINAL_REPORT" ]]; then
  if [[ -e "$FINAL_ZIP" || -e "$FINAL_REPORT" ]]; then
    echo "partial final package requires diagnosis"
    exit 1
  fi
  "$P" "$X/build_final_submission.py" \
    --base-submission "$BASE" \
    --distill-adapter "$FINAL_ADAPTER" \
    --full-refit-contract "$FULL_REFIT/output_contract.json" \
    --alpha 1.0 \
    --output "$FINAL_ZIP" \
    --report "$FINAL_REPORT" \
    > "$LOCAL/build-final-package.log" 2>&1
fi

sha256sum "$FINAL_ZIP"

COMPAT_REPORT="$SUBMISSIONS/runtime-package-compat.json"
SMOKE3="$SUBMISSIONS/runtime-smoke-3.json"
SMOKE600="$SUBMISSIONS/runtime-smoke-600.json"
RUNTIME_ACCEPTANCE="$SUBMISSIONS/runtime-acceptance.json"
if [[ ! -s "$COMPAT_REPORT" ]]; then
  "$P" - "$FINAL_REPORT" "$FINAL_ZIP" "$COMPAT_REPORT" <<'PY'
import hashlib
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
digest = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
if report.get("schema_version") != "exp706_final_submission_v1":
    raise SystemExit("unexpected final package report")
if report.get("output_sha256") != digest:
    raise SystemExit("final package checksum mismatch")
compat = {
    "schema_version": "exp716_submission_package_v1",
    "output_sha256": digest,
    "teacher_in_submission": False,
    "under_5_gib": bool(report.get("under_5_gib")),
    "source_exp706_package_report": sys.argv[1],
}
open(sys.argv[3], "w", encoding="utf-8").write(
    json.dumps(compat, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
PY
fi

RUNTIME_TOOL="$E/experiments/716_distilled_140_adapter_replacement/runtime_smoke.py"
DATA=/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/data/data.csv
IMAGES=/home/jovyan/shares/SR008.fs2/litvinov/tmp/QC/data/images/images
EMBED=/home/jovyan/shares/SR008.fs2/litvinov/models/Qwen3-VL-Embedding-2B
INSTRUCT=/home/jovyan/shares/SR008.fs2/litvinov/models/Qwen3-VL-2B-Instruct
QWEN35=/home/jovyan/shares/SR008.fs2/litvinov/models/Qwen3.5-4B
exec 8>"$E/.local/mlitvinov_q38_runtime_gpu3.lock"
flock 8
common=(
  --submission "$FINAL_ZIP"
  --package-report "$COMPAT_REPORT"
  --data "$DATA"
  --images "$IMAGES"
  --python "$P"
  --qwen-embed-model "$EMBED"
  --qwen-instruct-model "$INSTRUCT"
  --qwen35-model "$QWEN35"
  --device 3
)
if [[ ! -s "$SMOKE3" ]]; then
  CUDA_VISIBLE_DEVICES=3 "$P" "$RUNTIME_TOOL" "${common[@]}" \
    --rows 3 \
    --work-dir "$LOCAL/final-runtime-smoke-3" \
    --report "$SMOKE3" \
    > "$LOCAL/final-runtime-smoke-3.log" 2>&1
fi
if [[ ! -s "$SMOKE600" ]]; then
  CUDA_VISIBLE_DEVICES=3 "$P" "$RUNTIME_TOOL" "${common[@]}" \
    --rows 600 \
    --work-dir "$LOCAL/final-runtime-smoke-600" \
    --report "$SMOKE600" \
    > "$LOCAL/final-runtime-smoke-600.log" 2>&1
fi
if [[ ! -s "$RUNTIME_ACCEPTANCE" ]]; then
  "$P" - "$FINAL_REPORT" "$SMOKE3" "$SMOKE600" "$RUNTIME_ACCEPTANCE" <<'PY'
import hashlib
import json
import sys

package = json.load(open(sys.argv[1], encoding="utf-8"))
smoke3 = json.load(open(sys.argv[2], encoding="utf-8"))
smoke600 = json.load(open(sys.argv[3], encoding="utf-8"))
if smoke3.get("decision") != "RUNTIME_SMOKE_PASS":
    raise SystemExit("3-row runtime smoke failed")
if smoke600.get("decision") != "RUNTIME_SMOKE_PASS":
    raise SystemExit("600-row runtime smoke failed")
if smoke600.get("projected_private_minutes_3800", 1e9) >= 40:
    raise SystemExit("projected private runtime exceeds 40 minutes")
result = {
    "schema_version": "exp706_runtime_acceptance_v1",
    "decision": "RUNTIME_ACCEPTED",
    "submission_sha256": package["output_sha256"],
    "package_report_sha256": hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest(),
    "smoke_3_sha256": hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest(),
    "smoke_600_sha256": hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest(),
    "projected_public_minutes_1600": smoke600.get("projected_public_minutes_1600"),
    "projected_private_minutes_3800": smoke600.get("projected_private_minutes_3800"),
    "peak_gpu_memory_mib": smoke600.get("peak_gpu_memory_mib"),
    "one_h100": True,
    "under_5_gib": bool(package.get("under_5_gib")),
    "ocr_runtime_absent": bool(package.get("ocr_runtime_absent")),
}
open(sys.argv[4], "w", encoding="utf-8").write(
    json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
)
PY
fi

echo "$(date -Is) exp706 final package and runtime accepted: $FINAL_ZIP"
