from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODES = ("causal_candidate",)


def checked_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def commit_fold_output(staging: Path, final: Path) -> None:
    if final.exists():
        raise FileExistsError(f"refusing to replace committed fold output: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)


def validate_memory_smoke_contract(contract: dict[str, object]) -> int:
    body = dict(contract)
    declared = body.pop("contract_sha256", None)
    if (
        declared
        != hashlib.sha256(
            json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise ValueError("structured candidate memory smoke self-hash failed")
    peak = contract.get("peak_gpu_memory_bytes")
    factor = contract.get("changed_factor_smoke")
    if (
        contract.get("mode") != "causal_candidate"
        or contract.get("technical_smoke") is not True
        or contract.get("decision") != "TECHNICAL_SMOKE_ONLY"
        or isinstance(peak, bool)
        or not isinstance(peak, int)
        or not 0 < peak < 75 * 1024**3
        or not isinstance(factor, dict)
        or int(factor.get("eligible_grounded_rows", 0)) < 1
        or set(factor.get("auxiliary_losses", {}))
        != {"sold_object", "substance", "relation", "evidence_pointer"}
    ):
        raise ValueError("structured candidate memory smoke contract failed")
    return peak


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One job: changed-factor smoke, then candidate train+eval for five folds."
    )
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--teacher-acceptance", type=Path, required=True)
    parser.add_argument("--teacher-acceptance-sha256", required=True)
    parser.add_argument("--teacher-winner", type=Path, required=True)
    parser.add_argument("--teacher-winner-sha256", required=True)
    parser.add_argument("--runtime-bundle-sha256", required=True)
    parser.add_argument("--baseline-bundle-sha256", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--runtime-backend", choices=("verified_fast_path", "legacy_eager"), default="legacy_eager"
    )
    parser.add_argument("--technical-smoke", action="store_true")
    args = parser.parse_args()
    checked_empty(args.output_root)

    def training_command(fold: int, mode: str, output: Path, *, smoke: bool) -> list[str]:
        command = [
            sys.executable,
            str(HERE / "train_fold.py"),
            "--fold",
            str(fold),
            "--runtime-dir",
            str(args.runtime_root / f"fold{fold}"),
            "--teacher-root",
            str(args.teacher_root),
            "--teacher-acceptance",
            str(args.teacher_acceptance),
            "--teacher-acceptance-sha256",
            args.teacher_acceptance_sha256,
            "--teacher-winner",
            str(args.teacher_winner),
            "--teacher-winner-sha256",
            args.teacher_winner_sha256,
            "--images",
            str(args.images),
            "--model-root",
            str(args.model_root),
            "--model-revision",
            args.model_revision,
            "--vendor",
            str(args.vendor),
            "--output-dir",
            str(output),
            "--runtime-backend",
            args.runtime_backend,
            "--micro-batch-size-override",
            "2",
            "--mode",
            mode,
        ]
        if smoke:
            command.append("--technical-smoke")
        return command

    if not args.technical_smoke:
        smoke_output = args.output_root / "memory_smoke"
        subprocess.run(
            training_command(0, "causal_candidate", smoke_output, smoke=True), check=True
        )
        smoke_contract_path = smoke_output / "output_contract.json"
        smoke_contract = json.loads(smoke_contract_path.read_text(encoding="utf-8"))
        peak = validate_memory_smoke_contract(smoke_contract)
        smoke_acceptance = {
            "schema_version": "exp693_structured_memory_smoke_v1",
            "fold": 0,
            "peak_gpu_memory_bytes": peak,
            "limit_bytes": 75 * 1024**3,
            "output_contract_file_sha256": hashlib.sha256(
                smoke_contract_path.read_bytes()
            ).hexdigest(),
            "output_contract_self_sha256": smoke_contract["contract_sha256"],
            "decision": "OPEN_FIVEFOLD_CANDIDATE_IN_SAME_JOB",
        }
        smoke_acceptance["acceptance_sha256"] = hashlib.sha256(
            json.dumps(
                smoke_acceptance,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        (args.output_root / "memory_smoke_acceptance.json").write_text(
            json.dumps(smoke_acceptance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    for fold in range(5):
        for mode in MODES:
            arm = "candidate"
            final_output = args.output_root / arm / f"fold{fold}"
            staging_output = (
                args.output_root.parent
                / ".qwen4_staging"
                / args.output_root.name
                / arm
                / f"fold{fold}"
            )
            subprocess.run(
                training_command(fold, mode, staging_output, smoke=args.technical_smoke),
                check=True,
            )
            commit_fold_output(staging_output, final_output)
    evaluation_command = [
        sys.executable,
        str(HERE / "evaluate.py"),
        "--runtime-root",
        str(args.runtime_root),
        "--teacher-root",
        str(args.teacher_root),
        "--teacher-acceptance",
        str(args.teacher_acceptance),
        "--teacher-acceptance-sha256",
        args.teacher_acceptance_sha256,
        "--teacher-winner",
        str(args.teacher_winner),
        "--teacher-winner-sha256",
        args.teacher_winner_sha256,
        "--runtime-bundle-sha256",
        args.runtime_bundle_sha256,
        "--baseline-bundle-sha256",
        args.baseline_bundle_sha256,
        "--baseline-root",
        str(args.baseline_root),
        "--candidate-root",
        str(args.output_root / "candidate"),
        "--output",
        str(args.output_root / "evaluation.json"),
    ]
    subprocess.run(evaluation_command, check=True)


if __name__ == "__main__":
    main()
