from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DECISION = ROOT / "experiments/310_qwen35_crossfit_soft_targets/results/post310_decision.json"

if not DECISION.exists():
    raise SystemExit(
        "Experiment 320 training is gated on the completed experiment 310 audit: "
        f"missing {DECISION}."
    )
decision = json.loads(DECISION.read_text(encoding="utf-8"))
required = {
    "audit_complete",
    "offline_core_gate_passed",
    "sports_errors_fixed_out_of_243",
    "guardrail_breach",
    "launch_experiment_320",
    "evidence",
}
if not required.issubset(decision):
    raise SystemExit(f"Experiment 310 decision is incomplete; required fields: {sorted(required)}")
fixed = decision["sports_errors_fixed_out_of_243"]
for field in (
    "audit_complete",
    "offline_core_gate_passed",
    "guardrail_breach",
    "launch_experiment_320",
):
    if type(decision[field]) is not bool:
        raise SystemExit(f"{field} must be a JSON boolean")
if type(fixed) is not int or not 0 <= fixed <= 243:
    raise SystemExit("sports_errors_fixed_out_of_243 must be an integer in [0, 243]")
if decision["audit_complete"] is not True or decision["launch_experiment_320"] is not True:
    raise SystemExit("Recorded experiment 310 decision does not authorize experiment 320")
expected_evidence = {
    "experiments/310_qwen35_crossfit_soft_targets/results/acceptance_audit.json",
    "experiments/310_qwen35_crossfit_soft_targets/results/connected_safe_audit.json",
    "experiments/310_qwen35_crossfit_soft_targets/results/prior_replay.json",
    "experiments/310_qwen35_crossfit_soft_targets/results/sports_error_audit.json",
}
if not isinstance(decision["evidence"], list) or len(decision["evidence"]) != len(
    expected_evidence
):
    raise SystemExit(
        "Experiment 310 decision must bind exactly the four frozen audit evidence files"
    )
recorded_evidence: set[str] = set()
for item in decision["evidence"]:
    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
        raise SystemExit("Each evidence item must contain exactly path and sha256")
    if not isinstance(item["path"], str) or not isinstance(item["sha256"], str):
        raise SystemExit("Experiment 310 evidence path and sha256 must be strings")
    if len(item["sha256"]) != 64 or any(
        character not in "0123456789abcdef" for character in item["sha256"]
    ):
        raise SystemExit("Experiment 310 evidence sha256 must be lowercase hexadecimal")
    recorded_evidence.add(item["path"])
    evidence_path = (ROOT / item["path"]).resolve()
    if ROOT not in evidence_path.parents or not evidence_path.is_file():
        raise SystemExit(f"Invalid experiment 310 evidence path: {item['path']}")
    digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    if digest != item["sha256"]:
        raise SystemExit(f"Experiment 310 evidence hash mismatch: {item['path']}")
if recorded_evidence != expected_evidence:
    raise SystemExit(
        "Experiment 310 evidence paths do not match the four frozen audit outputs"
    )

# Hashes alone only prove that the evidence files were not modified. Recompute the
# complete decision from those canonical files so that branch fields cannot be
# edited independently of their evidence.
with tempfile.TemporaryDirectory(prefix="post310-replay-") as temporary_directory:
    replay_output = Path(temporary_directory) / "post310_decision.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "research/finalize_qwen_soft_target.py"),
            "--locked",
            "experiments/310_qwen35_crossfit_soft_targets/results/acceptance_audit.json",
            "--connected",
            "experiments/310_qwen35_crossfit_soft_targets/results/connected_safe_audit.json",
            "--priors",
            "experiments/310_qwen35_crossfit_soft_targets/results/prior_replay.json",
            "--sports",
            "experiments/310_qwen35_crossfit_soft_targets/results/sports_error_audit.json",
            "--output",
            str(replay_output),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    replayed_decision = json.loads(replay_output.read_text(encoding="utf-8"))
if replayed_decision != decision:
    raise SystemExit(
        "Experiment 310 decision is not the canonical result of its bound evidence"
    )
if fixed >= 49 and decision["guardrail_breach"] is False:
    raise SystemExit(
        "Experiment 310 fixed at least 49 sports errors without a guardrail breach; "
        "the frozen branching rule forbids launching experiment 320 first"
    )

subprocess.run(
    [
        sys.executable,
        str(ROOT / "research/bad_regulatory_residual_cv.py"),
        "--topologies",
        "historical",
        "repeat_0",
        "repeat_1",
        "repeat_2",
        "--output",
        str(ROOT / "experiments/320_bad_regulatory_evidence/results/residual_audit.json"),
        "--output-npz",
        str(ROOT / "experiments/320_bad_regulatory_evidence/results/residual_predictions.npz"),
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        str(ROOT / "research/audit_component_decision_survival.py"),
        "--data",
        str(ROOT / "research/data.csv"),
        "--predictions",
        str(ROOT / "experiments/320_bad_regulatory_evidence/results/residual_predictions.npz"),
        "--candidate",
        "exp320",
        "--output",
        str(ROOT / "experiments/320_bad_regulatory_evidence/results/prior_replay.json"),
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        str(ROOT / "research/finalize_bad_regulatory_gate.py"),
        "--residual-audit",
        str(ROOT / "experiments/320_bad_regulatory_evidence/results/residual_audit.json"),
        "--prior-replay",
        str(ROOT / "experiments/320_bad_regulatory_evidence/results/prior_replay.json"),
        "--output",
        str(ROOT / "experiments/320_bad_regulatory_evidence/results/final_gate.json"),
    ],
    check=True,
)
