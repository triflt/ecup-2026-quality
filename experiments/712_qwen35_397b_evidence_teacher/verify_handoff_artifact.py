from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from PIL import Image
from transformers import AutoProcessor


EXPECTED_ROWS = 12_971
EXPECTED_MODEL = "Qwen/Qwen3.5-4B"
EXPECTED_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODES = ("rationale_then_label", "explanation_only")


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line in {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL row in {path}:{line_number}")
            rows.append(row)
    return rows


def find_model_root(root: Path) -> Path:
    candidates = [root] + sorted(path.parent for path in root.rglob("config.json"))
    for candidate in candidates:
        if (candidate / "config.json").is_file() and any(
            (candidate / name).is_file()
            for name in ("processor_config.json", "preprocessor_config.json", "tokenizer_config.json")
        ):
            return candidate
    raise FileNotFoundError(f"cannot locate processor root below {root}")


def config_bundle(root: Path) -> dict[str, Any]:
    allowed_names = {
        "added_tokens.json", "chat_template.json", "chat_template.jinja", "config.json",
        "generation_config.json", "merges.txt", "preprocessor_config.json",
        "processor_config.json", "special_tokens_map.json", "tokenizer.json",
        "tokenizer.model", "tokenizer_config.json", "vocab.json",
    }
    files = sorted(
        path for path in root.iterdir()
        if path.is_file() and (path.name in allowed_names or path.suffix == ".tiktoken")
    )
    if not files:
        raise ValueError("no processor/tokenizer configuration files found")
    rows = [
        {
            "path": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(compact_json(row).encode("utf-8"))
        digest.update(b"\n")
    return {"files": rows, "bundle_sha256": digest.hexdigest()}


def quantiles(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {"min": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)

    def pick(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]

    return {"min": ordered[0], "p50": pick(0.5), "p95": pick(0.95), "max": ordered[-1]}


def apply_chat(processor, conversations):
    kwargs = {
        "add_generation_prompt": False,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "padding": True,
        "truncation": False,
    }
    try:
        return processor.apply_chat_template(conversations, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(conversations, **kwargs)


def apply_prompt_chat(processor, conversations):
    kwargs = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
        "padding": True,
        "truncation": False,
    }
    try:
        return processor.apply_chat_template(conversations, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(conversations, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2304)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    rationale_contract = load_module(
        "receipt_rationale_contract", args.code_root / "713" / "target_contract.py"
    )
    explanation_contract = load_module(
        "receipt_explanation_contract", args.code_root / "714" / "target_contract.py"
    )
    contracts = {
        "rationale_then_label": rationale_contract,
        "explanation_only": explanation_contract,
    }

    with args.data.open(encoding="utf-8", newline="") as stream:
        data_rows = list(csv.DictReader(stream))
    data = {str(row["id"]): row for row in data_rows}
    if len(data_rows) != EXPECTED_ROWS or len(data) != EXPECTED_ROWS:
        raise ValueError("source data row/id count mismatch")

    manifest_path = args.handoff_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for object_name, binding in manifest["outputs"].items():
        path = args.handoff_root / str(binding["path"])
        if not path.is_file():
            raise FileNotFoundError(f"missing bound object {object_name}: {path}")
        if sha256_file(path) != binding["sha256"] or path.stat().st_size != int(
            binding["size_bytes"]
        ):
            raise ValueError(f"bound object SHA/size mismatch: {object_name}")
        if path.suffix == ".jsonl" and "rows" in binding:
            with path.open(encoding="utf-8") as stream:
                rows = sum(1 for line in stream if line.strip())
            if rows != int(binding["rows"]):
                raise ValueError(f"bound object row count mismatch: {object_name}")
    accepted_rows = read_jsonl(args.handoff_root / "accepted_rows.jsonl")
    accepted_by_id = {str(row["id"]): row for row in accepted_rows}
    if len(accepted_by_id) != len(accepted_rows):
        raise ValueError("duplicate accepted row ids")

    targets: dict[str, list[dict[str, Any]]] = {}
    for mode in MODES:
        path = args.handoff_root / f"{mode}_v5.jsonl"
        expected = manifest["outputs"][mode]
        if sha256_file(path) != expected["sha256"] or path.stat().st_size != expected["size_bytes"]:
            raise ValueError(f"{mode} object binding mismatch")
        rows = read_jsonl(path)
        if len(rows) != expected["rows"]:
            raise ValueError(f"{mode} row count mismatch")
        if any(set(row) != {"id", "label", "target"} for row in rows):
            raise ValueError(f"{mode} schema must be exactly id/label/target")
        ids = [str(row["id"]) for row in rows]
        if ids != [str(row["id"]) for row in accepted_rows]:
            raise ValueError(f"{mode} order/support differs from accepted_rows")
        for row in rows:
            item_id = str(row["id"])
            if int(row["label"]) != int(data[item_id]["label"]):
                raise ValueError(f"{mode} source label mismatch for id={item_id}")
            parsed = contracts[mode].parse_target(str(row["target"]))
            if mode == "rationale_then_label" and int(parsed["label"]) != int(row["label"]):
                raise ValueError(f"{mode} terminal label mismatch for id={item_id}")
            if not accepted_by_id[item_id]["mode_eligibility"][mode]:
                raise ValueError(f"{mode} row is not eligible: id={item_id}")
        targets[mode] = rows

    matched_digit_rows = read_jsonl(args.handoff_root / "matched_support_digit_v5.jsonl")
    if [str(row["id"]) for row in matched_digit_rows] != [
        str(row["id"]) for row in accepted_rows
    ]:
        raise ValueError("matched digit control order/support differs from accepted_rows")
    for row in matched_digit_rows:
        if (
            set(row) != {"id", "label", "target"}
            or int(row["label"]) != int(data[str(row["id"])]["label"])
            or str(row["target"]) != str(int(row["label"]))
        ):
            raise ValueError(f"invalid matched digit control row id={row.get('id')}")

    image_manifest_path = args.handoff_root / "ensemble_image0_manifest.jsonl"
    image_rows = read_jsonl(image_manifest_path)
    image_by_id = {str(row["id"]): row for row in image_rows}
    for item_id, row in accepted_by_id.items():
        image_row = image_by_id[item_id]
        image_path = args.image_root / str(image_row["path"])
        if (
            sha256_file(image_path) != image_row["sha256"]
            or image_path.stat().st_size != int(image_row["size_bytes"])
        ):
            raise ValueError(f"source ensemble image0 binding mismatch for id={item_id}")
        with Image.open(image_path) as image:
            if tuple(image.size) != (
                int(image_row["width"]), int(image_row["height"])
            ):
                raise ValueError(f"source ensemble image0 pixel dimensions mismatch for id={item_id}")

    resolved_model_root = find_model_root(args.model_root)
    processor = AutoProcessor.from_pretrained(
        resolved_model_root, trust_remote_code=True, local_files_only=True
    )
    processor_bundle = config_bundle(resolved_model_root)

    mode_stats: dict[str, Any] = {}
    for mode in MODES:
        target_tokens: list[int] = []
        full_tokens: list[int] = []
        truncations: list[str] = []
        alterations: list[str] = []
        rows = targets[mode]
        for start in range(0, len(rows), args.batch_size):
            local = rows[start:start + args.batch_size]
            images = []
            for row in local:
                with Image.open(
                    args.image_root / image_by_id[str(row["id"])]["path"]
                ) as opened:
                    image = opened.convert("RGB")
                    image.thumbnail((448, 448), Image.Resampling.LANCZOS)
                    image.load()
                images.append(image)
            full_conversations = []
            prompt_conversations = []
            for row, image in zip(local, images, strict=True):
                source = data[str(row["id"])]
                item = SimpleNamespace(
                    id=str(row["id"]),
                    category=str(source["category"]),
                    name=str(source["name"]),
                    description=str(source["description"]),
                    target=str(row["target"]),
                )
                if mode == "rationale_then_label":
                    full_conversations.append(contracts[mode].messages(item, with_answer=True, image=image))
                    prompt_conversations.append(contracts[mode].messages(item, with_answer=False, image=image))
                else:
                    full_conversations.append(contracts[mode].messages(
                        item, verdict=int(row["label"]), with_answer=True, image=image
                    ))
                    prompt_conversations.append(contracts[mode].messages(
                        item, verdict=int(row["label"]), with_answer=False, image=image
                    ))
            full_batch = apply_chat(processor, full_conversations)
            prompt_batch = apply_prompt_chat(processor, prompt_conversations)
            for image in images:
                image.close()
            for row_index, row in enumerate(local):
                full_positions = full_batch["attention_mask"][row_index].nonzero().flatten()
                prompt_positions = prompt_batch["attention_mask"][row_index].nonzero().flatten()
                full_ids = full_batch["input_ids"][row_index, full_positions]
                prompt_ids = prompt_batch["input_ids"][row_index, prompt_positions]
                full_tokens.append(len(full_ids))
                target_tokens.append(len(processor.tokenizer.encode(
                    str(row["target"]), add_special_tokens=False
                )))
                if len(full_ids) > args.max_length:
                    truncations.append(str(row["id"]))
                    continue
                limit = min(len(full_ids), len(prompt_ids))
                mismatch = (full_ids[:limit] != prompt_ids[:limit]).nonzero().flatten()
                common = int(mismatch[0]) if len(mismatch) else limit
                if common < len(prompt_ids) - 2 or common >= len(full_ids):
                    alterations.append(str(row["id"]))
                    continue
                decoded = processor.tokenizer.decode(
                    full_ids[common:].tolist(), skip_special_tokens=True
                ).strip()
                if decoded != str(row["target"]).strip():
                    alterations.append(str(row["id"]))
            done = min(start + len(local), len(rows))
            if done % 500 < args.batch_size or done == len(rows):
                print(f"processor_receipt mode={mode} rows={done}/{len(rows)}", flush=True)
        mode_stats[mode] = {
            "rows": len(rows),
            "target_tokens": quantiles(target_tokens),
            "full_sequence_tokens": quantiles(full_tokens),
            "target_truncation": len(truncations),
            "target_alteration": len(alterations),
            "first_truncation_ids": truncations[:20],
            "first_alteration_ids": alterations[:20],
        }

    overall_coverage_ok = True
    cell_coverage_ok = True
    conservative_overall_coverage_ok = True
    conservative_cell_coverage_ok = True
    for mode in MODES:
        audit = manifest["mode_audits"][mode]
        coverage = audit["coverage"]
        cell_keys = [key for key in coverage if key.startswith("cell:")]
        cell_keys.extend(
            key for key in coverage
            if key.startswith("fold_cell:0|") or key.startswith("fold_cell:3|")
        )
        if float(coverage["all"]["coverage"]) < 0.85:
            overall_coverage_ok = False
        if any(float(coverage[key]["coverage"]) < 0.85 for key in set(cell_keys)):
            cell_coverage_ok = False
        conservative = audit["conservative_coverage_excluding_nee_recovery"]
        if float(conservative["all"]["coverage"]) < 0.85:
            conservative_overall_coverage_ok = False
        if any(float(conservative[key]["coverage"]) < 0.85 for key in set(cell_keys)):
            conservative_cell_coverage_ok = False

    args.output_root.mkdir(parents=True, exist_ok=True)
    processor_audits: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        audit_path = args.output_root / f"{mode}_v5.processor_audit.json"
        audit = {
            "schema_version": "exact_processor_target_audit_v1",
            "mode": mode,
            "model": EXPECTED_MODEL,
            "model_revision": EXPECTED_MODEL_REVISION,
            "max_sequence_length": args.max_length,
            **mode_stats[mode],
        }
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        processor_audits[mode] = {
            "path": audit_path.name,
            "sha256": sha256_file(audit_path),
            "size_bytes": audit_path.stat().st_size,
        }

    technical_accept = (
        overall_coverage_ok
        and all(mode_stats[mode]["target_truncation"] == 0 for mode in MODES)
        and all(mode_stats[mode]["target_alteration"] == 0 for mode in MODES)
        and all(manifest["mode_audits"][mode]["invalid_json"] == 0 for mode in MODES)
        and all(
            manifest["mode_audits"][mode]["accepted_unsupported_evidence"] == 0
            for mode in MODES
        )
        and all(
            manifest["mode_audits"][mode]["accepted_secondary_only_evidence"] == 0
            for mode in MODES
        )
        and all(
            manifest["mode_audits"][mode]["accepted_student_scope_errors"] == 0
            for mode in MODES
        )
    )
    receipt = {
        "schema_version": "explanation_student_handoff_receipt_v1",
        "decision": "TECHNICAL_ACCEPT_PENDING_USER_APPROVAL" if technical_accept else "REJECT",
        "training_authorized": False,
        "public_submissions": 0,
        "manifest_sha256": sha256_file(manifest_path),
        "accepted_rows_sha256": sha256_file(args.handoff_root / "accepted_rows.jsonl"),
        "accepted_rows": len(accepted_rows),
        "accepted_recovered_from_not_enough_evidence": {
            mode: manifest["mode_audits"][mode][
                "accepted_recovered_from_not_enough_evidence"
            ]
            for mode in MODES
        },
        "expanded_overall_coverage_gate_passed": overall_coverage_ok,
        "expanded_cell_coverage_gate_passed": cell_coverage_ok,
        "conservative_overall_coverage_excluding_nee_recovery_gate_passed": (
            conservative_overall_coverage_ok
        ),
        "conservative_cell_coverage_excluding_nee_recovery_gate_passed": (
            conservative_cell_coverage_ok
        ),
        "coverage_warning": None if cell_coverage_ok else (
            "At least one category-label cell is below 85%; retain exact-support control "
            "and report the affected cells, but do not reject an otherwise valid corpus "
            "whose overall coverage exceeds 85%."
        ),
        "exact_support_and_order": True,
        "model": EXPECTED_MODEL,
        "model_revision": EXPECTED_MODEL_REVISION,
        "processor_root_name": resolved_model_root.name,
        "processor_tokenizer_bundle": processor_bundle,
        "verifier_sha256": sha256_file(Path(__file__)),
        "target_contract_sha256": {
            "rationale_then_label": sha256_file(args.code_root / "713" / "target_contract.py"),
            "explanation_only": sha256_file(args.code_root / "714" / "target_contract.py"),
        },
        "container_image": "odsai/ecup26-quality-baseline:1.0",
        "preprocessing_contract": manifest["ensemble_image0_preprocessing"],
        "training_contract": {
            "fold0_screen_then_fold3_confirmation": True,
            "epochs": 1,
            "seed": 42,
            "micro_batch_size": 2,
            "gradient_accumulation": 8,
            "effective_batch_size": 16,
            "max_sequence_length": args.max_length,
            "matched_digit_control_same_support_order_seed_steps_token_budget": True,
            "matched_digit_control_path": "/work/handoff/matched_support_digit_v5.jsonl",
            "image_directory": "/work/input/images",
            "image_manifest_path": "/work/handoff/ensemble_image0_manifest.jsonl",
            "image_download_during_training": False,
            "required_environment": {
                "ECUP_SOURCE_IMAGES": "/work/input/images",
                "ECUP_IMAGE0_MANIFEST": "/work/handoff/ensemble_image0_manifest.jsonl",
            },
            "rationale_max_generation_tokens": 384,
            "explanation_only_max_generation_tokens": 192,
        },
        "output_contract": {
            "result": "<комментарий>{50-300 chars}<вердикт>{бан|не бан}",
            "label_0_verdict": "бан",
            "label_1_verdict": "не бан",
            "closing_tags_forbidden": True,
        },
        "mode_stats": mode_stats,
        "processor_audits": processor_audits,
        "technical_checks_passed": technical_accept,
        "technical_check_scope": (
            "schema, provenance, support/order, source accessibility, exact text spans, "
            "image bytes, processor tokens and truncation; not a claim of exhaustive "
            "human semantic grounding"
        ),
        "semantic_grounding_audit": "PENDING_USER_REVIEW_AND_BLIND_AUDIT",
        "approval_required_from": "user",
    }
    receipt_path = args.output_root / "acceptance.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt_sha = sha256_file(receipt_path)
    (args.output_root / "acceptance.json.sha256").write_text(
        receipt_sha + "  acceptance.json\n", encoding="utf-8"
    )
    print("HANDOFF_RECEIPT=" + compact_json({
        "decision": receipt["decision"],
        "sha256": receipt_sha,
        "accepted_rows": len(accepted_rows),
    }), flush=True)
    if not technical_accept:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
