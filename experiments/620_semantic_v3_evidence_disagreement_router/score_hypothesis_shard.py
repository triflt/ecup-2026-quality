from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from contract import (
    EXPECTED_DEVELOPMENT_IDS_SHA256,
    EXPECTED_DEVELOPMENT_ROWS,
    PROTOCOL_VERSION,
    canonical_sha256,
    load_development_registry,
    load_json,
    sha256_file,
)
from smoke_evidence_head import metadata_checksums, resolve_model_dir

SAFE_DATA_COLUMNS = ("id", "category", "name", "description")
DESCRIPTION_LIMIT = 1800
MAX_LENGTH = 1536
DEFAULT_BATCH_SIZE = 16


def validate_runtime_inputs(
    *,
    audit_path: Path,
    data_path: Path,
    folds_path: Path,
    selector_path: Path,
    evidence_path: Path,
    enforce_frozen: bool = True,
) -> dict[str, Any]:
    audit = load_json(audit_path)
    audit_without_hash = dict(audit)
    supplied_audit_hash = audit_without_hash.pop("audit_sha256", None)
    if supplied_audit_hash != canonical_sha256(audit_without_hash):
        raise ValueError("feature runtime audit self-hash mismatch")
    required = {
        "status": "complete",
        "labels_present": False,
        "sealed_rows_present": 0,
        "competition_trained_adapters_present": False,
    }
    if enforce_frozen:
        required.update(
            development_rows=EXPECTED_DEVELOPMENT_ROWS,
            development_ids_sha256=EXPECTED_DEVELOPMENT_IDS_SHA256,
        )
    for key, expected in required.items():
        if audit.get(key) != expected:
            raise ValueError(f"feature runtime audit mismatch for {key}")
    checks = {
        "data_sha256": data_path,
        "folds_sha256": folds_path,
        "selector_components_sha256": selector_path,
        "evidence_manifest_sha256": evidence_path,
    }
    for key, path in checks.items():
        if audit.get(key) != sha256_file(path):
            raise ValueError(f"feature runtime checksum mismatch for {key}")
    return audit


def compact_text(value: Any, limit: int) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    return text[:head].rstrip() + " … " + text[-(limit - head) :].lstrip()


def load_category_rows(
    *,
    data_path: Path,
    folds_path: Path,
    category: str,
    enforce_frozen: bool = True,
    expected_rows: int = EXPECTED_DEVELOPMENT_ROWS,
) -> pd.DataFrame:
    registry = load_development_registry(
        folds_path,
        enforce_frozen=enforce_frozen,
        expected_rows=expected_rows,
    )
    data = pd.read_csv(data_path, usecols=list(SAFE_DATA_COLUMNS), dtype={"id": str})
    if set(data.columns) != set(SAFE_DATA_COLUMNS):
        raise ValueError("development feature data schema mismatch")
    data = data.loc[:, list(SAFE_DATA_COLUMNS)]
    if len(data) != expected_rows or data["id"].duplicated().any():
        raise ValueError("development feature data row count or ID uniqueness mismatch")
    if data["id"].astype(str).tolist() != registry["id"].astype(str).tolist():
        raise ValueError("feature data order differs from the frozen registry")
    if not data["category"].astype(str).equals(registry["category"].astype(str)):
        raise ValueError("feature data categories disagree with the frozen registry")
    rows = data.loc[data["category"].astype(str).eq(category)].copy()
    if rows.empty:
        raise ValueError(f"no rows for category {category!r}")
    return rows.reset_index(drop=True)


def load_selector_components(path: Path) -> dict[str, np.ndarray]:
    required = {
        "ids",
        "categories",
        "robust_base_score",
        "qwen3vl_score",
        "baseline_predictions",
    }
    forbidden = {"labels", "gold", "targets", "sealed_ids"}
    with np.load(path, allow_pickle=False) as bundle:
        if set(bundle.files) != required or forbidden & set(bundle.files):
            raise ValueError("selector component bundle schema mismatch")
        values = {name: bundle[name].copy() for name in required}
    if not np.isfinite(values["robust_base_score"]).all() or not np.isfinite(
        values["qwen3vl_score"]
    ).all():
        raise ValueError("selector component scores are non-finite")
    if not np.isin(values["baseline_predictions"], [0, 1]).all():
        raise ValueError("selector baseline predictions are not binary")
    return values


def load_evidence_manifest(path: Path) -> list[dict[str, Any]]:
    forbidden = {"label", "labels", "gold", "target", "targets"}
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            if forbidden & set(row):
                raise ValueError(f"evidence row {line_number} contains supervision")
            rows.append(row)
    return rows


def ordered_hypotheses(spec: dict[str, Any]) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    grouped = spec.get("hypotheses")
    if not isinstance(grouped, dict) or set(grouped) != {"БАД", "Легковоспламеняющиеся"}:
        raise ValueError("frozen hypothesis categories mismatch")
    ids: list[str] = []
    for category in ("БАД", "Легковоспламеняющиеся"):
        items = grouped[category]
        if len(items) != 5:
            raise ValueError(f"{category} must have exactly five frozen hypotheses")
        ids.extend(str(item["id"]) for item in items)
    if len(set(ids)) != 10:
        raise ValueError("frozen hypothesis IDs are not unique")
    return ids, grouped


def select_hypothesis(spec: dict[str, Any], hypothesis_id: str) -> tuple[str, dict[str, Any]]:
    _, grouped = ordered_hypotheses(spec)
    matches = [
        (category, item)
        for category, items in grouped.items()
        for item in items
        if item["id"] == hypothesis_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate hypothesis ID: {hypothesis_id}")
    return matches[0]


def hypothesis_prompt(row: pd.Series, hypothesis: str) -> str:
    return (
        f"Категория: {row.category}\n"
        f"Название: {compact_text(row['name'], 320)}\n"
        f"Описание: {compact_text(row.description, DESCRIPTION_LIMIT)}\n"
        f"Утверждение: {hypothesis}\n"
        "Подтверждает ли карточка утверждение прямым смыслом? "
        "Ответь только одной цифрой: 1 — подтверждает, 0 — не подтверждает."
    )


def select_candidate_rows(
    *,
    rows: pd.DataFrame,
    selector: dict[str, np.ndarray],
    evidence_rows: list[dict[str, Any]],
    hypothesis: dict[str, Any],
    compatible_concepts: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    ids = selector["ids"].astype(str)
    categories = selector["categories"].astype(str)
    if len(evidence_rows) != len(ids):
        raise ValueError("evidence and selector row counts differ")
    evidence_ids = np.asarray([str(row["id"]) for row in evidence_rows])
    if not np.array_equal(evidence_ids, ids):
        raise ValueError("evidence and selector IDs/order differ")
    by_id = {str(row.id): row for row in rows.itertuples(index=False)}
    proposal = int(hypothesis["supports_verdict"])
    selected: list[dict[str, Any]] = []
    counts = {
        "category_rows": 0,
        "baseline_disagreements": 0,
        "independent_vote_support": 0,
        "compatible_exact_evidence": 0,
        "span_retained_in_prompt": 0,
    }
    compatible = set(compatible_concepts)
    for position, evidence in enumerate(evidence_rows):
        if categories[position] != str(rows.iloc[0].category):
            continue
        counts["category_rows"] += 1
        if int(selector["baseline_predictions"][position]) == proposal:
            continue
        counts["baseline_disagreements"] += 1
        robust_vote = int(float(selector["robust_base_score"][position]) >= 0.5)
        visual_vote = int(float(selector["qwen3vl_score"][position]) >= 0.0)
        if proposal not in (robust_vote, visual_vote):
            continue
        counts["independent_vote_support"] += 1
        candidate = evidence.get(f"candidate_for_{proposal}")
        if not candidate or candidate.get("concept") not in compatible:
            continue
        counts["compatible_exact_evidence"] += 1
        row = by_id.get(ids[position])
        if row is None:
            raise ValueError("selector ID is absent from feature data")
        row_series = pd.Series(row._asdict())
        prompt = hypothesis_prompt(row_series, str(hypothesis["text"]))
        surface = str(candidate.get("exact_surface_span") or "").strip()
        if not surface or surface.casefold() not in prompt.casefold():
            continue
        counts["span_retained_in_prompt"] += 1
        selected.append(row_series.to_dict())
    return pd.DataFrame(selected, columns=list(SAFE_DATA_COLUMNS)), counts


def render_chat(processor: Any, text: str) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def answer_token_ids(processor: Any) -> tuple[int, int]:
    zero_ids = processor.tokenizer.encode("0", add_special_tokens=False)
    one_ids = processor.tokenizer.encode("1", add_special_tokens=False)
    if len(zero_ids) != 1 or len(one_ids) != 1 or zero_ids[0] == one_ids[0]:
        raise ValueError("digit answer tokens are not distinct atomic tokens")
    return int(zero_ids[0]), int(one_ids[0])


def score_hypothesis(
    *,
    rows: pd.DataFrame,
    hypothesis: dict[str, Any],
    model_root: Path,
    expected_model_metadata: dict[str, str],
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    model_dir = resolve_model_dir(model_root)
    actual_metadata = metadata_checksums(model_dir)
    if actual_metadata != expected_model_metadata:
        raise ValueError("mounted model metadata differs from accepted backbone decision")
    processor = AutoProcessor.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=True
    )
    processor.tokenizer.padding_side = "left"
    zero_id, one_id = answer_token_ids(processor)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()

    scores = np.empty(len(rows), dtype=np.float32)
    forward_batches = 0
    started = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        prompts = [
            render_chat(processor, hypothesis_prompt(rows.iloc[position], hypothesis["text"]))
            for position in range(start, stop)
        ]
        batch = processor.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        batch = {name: value.to("cuda") for name, value in batch.items()}
        with torch.inference_mode():
            logits = model(**batch, return_dict=True, use_cache=False).logits[:, -1, :]
            pair = logits[:, [zero_id, one_id]].float()
            probabilities = torch.softmax(pair, dim=-1)[:, 1]
        scores[start:stop] = probabilities.cpu().numpy().astype(np.float32)
        forward_batches += 1
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("hypothesis scores are not finite binary-normalized scores")
    return scores, {
        "forward_batches": forward_batches,
        "scoring_seconds": round(time.perf_counter() - started, 6),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "zero_token_id": zero_id,
        "one_token_id": one_id,
    }


def write_outputs(
    *,
    output_dir: Path,
    rows: pd.DataFrame,
    hypothesis: dict[str, Any],
    category: str,
    scores: np.ndarray,
    spec_path: Path,
    runtime: dict[str, Any],
    backbone_decision: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    hypothesis_id = str(hypothesis["id"])
    scorer_sha256 = sha256_file(Path(__file__).resolve())
    bundle_path = output_dir / "hypothesis_score_shard.npz"
    np.savez_compressed(
        bundle_path,
        ids=np.asarray(rows["id"].astype(str).tolist(), dtype=str),
        categories=np.asarray(rows["category"].astype(str).tolist(), dtype=str),
        hypothesis_id=np.asarray(hypothesis_id),
        supports_verdict=np.asarray(int(hypothesis["supports_verdict"]), dtype=np.int8),
        hypothesis_scores=scores.astype(np.float32),
        protocol_version=np.asarray(PROTOCOL_VERSION),
        spec_sha256=np.asarray(sha256_file(spec_path)),
        backbone_decision_sha256=np.asarray(canonical_sha256(backbone_decision)),
        scorer_sha256=np.asarray(scorer_sha256),
    )
    report = {
        "status": "complete",
        "experiment_id": "620",
        "protocol_version": PROTOCOL_VERSION,
        "stage": "label_independent_hypothesis_scoring",
        "hypothesis_id": hypothesis_id,
        "category": category,
        "supports_verdict": int(hypothesis["supports_verdict"]),
        "rows": len(rows),
        "labels_loaded": False,
        "sealed_rows_loaded": 0,
        "competition_trained_adapter_used": False,
        "score_semantics": "binary_normalized_support_score_not_calibrated_probability",
        "spec_sha256": sha256_file(spec_path),
        "backbone_decision_sha256": canonical_sha256(backbone_decision),
        "model_id": backbone_decision["model_id"],
        "model_revision": backbone_decision["revision"],
        "model_metadata_sha256": backbone_decision["files_sha256"],
        "scorer_sha256": scorer_sha256,
        "bundle_sha256": sha256_file(bundle_path),
        **runtime,
    }
    report["report_sha256"] = canonical_sha256(report)
    (output_dir / "hypothesis_score_report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothesis-id", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--selector-components", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--runtime-audit", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=HERE / "frozen_router_spec.json")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--backbone-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    spec = load_json(args.spec)
    backbone_decision = load_json(args.backbone_decision)
    if (
        backbone_decision.get("status") != "accepted"
        or backbone_decision.get("compatibility_passed") is not True
    ):
        raise ValueError("backbone decision is not accepted")
    category, hypothesis = select_hypothesis(spec, args.hypothesis_id)
    runtime_audit = validate_runtime_inputs(
        audit_path=args.runtime_audit,
        data_path=args.data,
        folds_path=args.folds,
        selector_path=args.selector_components,
        evidence_path=args.evidence_manifest,
    )
    rows = load_category_rows(
        data_path=args.data,
        folds_path=args.folds,
        category=category,
        enforce_frozen=False,
    )
    compatible = spec["hypothesis_concept_compatibility"][args.hypothesis_id]
    selector = load_selector_components(args.selector_components)
    evidence_rows = load_evidence_manifest(args.evidence_manifest)
    rows, selection_audit = select_candidate_rows(
        rows=rows,
        selector=selector,
        evidence_rows=evidence_rows,
        hypothesis=hypothesis,
        compatible_concepts=compatible,
    )
    scores, runtime = score_hypothesis(
        rows=rows,
        hypothesis=hypothesis,
        model_root=args.model_root,
        expected_model_metadata=backbone_decision["files_sha256"],
        batch_size=args.batch_size,
    )
    report = write_outputs(
        output_dir=args.output_dir,
        rows=rows,
        hypothesis=hypothesis,
        category=category,
        scores=scores,
        spec_path=args.spec,
        runtime={
            "runtime_audit_sha256": canonical_sha256(runtime_audit),
            **selection_audit,
            **runtime,
        },
        backbone_decision=backbone_decision,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
