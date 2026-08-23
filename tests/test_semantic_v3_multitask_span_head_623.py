from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/623_semantic_v3_multitask_span_head"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load("exp623_protocol_test", "protocol.py")
head = _load("exp623_head_test", "multitask_head.py")
renderer = _load("exp623_renderer_test", "renderer.py")
model_module = _load("exp623_model_test", "model.py")
preflight = _load("exp623_preflight_test", "preflight.py")
alignment = _load("exp623_alignment_test", "alignment.py")
runner = _load("exp623_runner_test", "run_fold.py")

# The experiment is also runnable as a standalone script and therefore uses
# local absolute imports.  Do not leak those generic module names into other
# experiment test modules during pytest collection.
for _local_module in ("protocol", "multitask_head", "renderer", "model"):
    sys.modules.pop(_local_module, None)
if str(EXP) in sys.path:
    sys.path.remove(str(EXP))


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    ids = [f"dev-{index}" for index in range(5)] + ["sealed"]
    data = pd.DataFrame(
        {
            "id": ids,
            "category": ["БАД", "БАД", "БАД", "Легковоспламеняющиеся", "БАД", "БАД"],
            "label": [1, 0, 1, 0, 1, 0],
            "name": ["БАД комплекс", "чай", "витамины", "горелка", "БАД", "secret"],
            "description": ["добавка", "напиток", "БАД", "без топлива", "капсулы", "sealed"],
        }
    )
    folds = pd.DataFrame(
        {
            "id": ids,
            "category": data["category"],
            "label": data["label"],
            "semantic_component": [f"c-{index}" for index in range(6)],
            "component_size": [1] * 6,
            "split": ["development"] * 5 + ["sealed_holdout"],
            "development_fold": [0, 1, 2, 3, 4, -1],
        }
    )
    manifest = []
    for index in range(5):
        positive = None
        if index == 0:
            positive = {
                "source": "name",
                "surface_start": 0,
                "surface_end": 3,
                "exact_surface_span": "БАД",
                "concept": "BAD_EXPLICIT_MARKING",
                "verdict": 1,
            }
        manifest.append(
            {
                "id": ids[index],
                "category": data.loc[index, "category"],
                "development_fold": index,
                "semantic_component": f"c-{index}",
                "candidate_for_0": None,
                "candidate_for_1": positive,
            }
        )
    data_path, folds_path, manifest_path, selector_path, image_path = (
        tmp_path / "data.csv",
        tmp_path / "folds.csv",
        tmp_path / "evidence.jsonl",
        tmp_path / "selector.npz",
        tmp_path / "images.tsv.gz",
    )
    data.to_csv(data_path, index=False)
    folds.to_csv(folds_path, index=False)
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest), encoding="utf-8"
    )
    np.savez_compressed(
        selector_path,
        ids=np.asarray(ids[:5]),
        fold_ids=np.arange(5, dtype=np.int8),
        fused_scores=np.linspace(0.1, 0.9, 5, dtype=np.float32),
    )
    with gzip.open(image_path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "image_url"], delimiter="\t")
        writer.writeheader()
        for item_id in ids[:5]:
            writer.writerow({"id": item_id, "image_url": f"https://example.invalid/{item_id}"})
    return data_path, folds_path, manifest_path, selector_path, image_path


def test_runtime_physically_isolates_outer_labels_and_sealed(tmp_path: Path) -> None:
    data, folds, evidence, selector, images = _inputs(tmp_path)
    output = tmp_path / "runtime"
    report = protocol.build_fold_runtime(
        data_path=data,
        folds_path=folds,
        evidence_manifest_path=evidence,
        selector_path=selector,
        image_manifest_path=images,
        outer_fold=3,
        output_dir=output,
        enforce_frozen_hashes=False,
    )
    train = protocol.read_jsonl(output / "train.jsonl")
    validation = protocol.read_jsonl(output / "validation.jsonl")
    assert report["decision"] == "GO"
    assert {row["id"] for row in train} == {"dev-0", "dev-1", "dev-2", "dev-4"}
    assert [row["id"] for row in validation] == ["dev-3"]
    assert "label" not in validation[0]
    assert "rationale" not in validation[0]
    assert "sealed" not in (output / "train.jsonl").read_text(encoding="utf-8")
    assert "sealed" not in (output / "validation.jsonl").read_text(encoding="utf-8")
    assert train[0]["rationale"]["exact_span"] == "БАД"


def test_manifest_with_sealed_membership_is_rejected(tmp_path: Path) -> None:
    data, folds, evidence, selector, images = _inputs(tmp_path)
    with evidence.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"id": "sealed"}) + "\n")
    with pytest.raises(ValueError, match="exactly development ids"):
        protocol.build_fold_runtime(
            data_path=data,
            folds_path=folds,
            evidence_manifest_path=evidence,
            selector_path=selector,
            image_manifest_path=images,
            outer_fold=0,
            output_dir=tmp_path / "out",
            enforce_frozen_hashes=False,
        )


def test_runtime_fails_closed_on_unfrozen_inputs(tmp_path: Path) -> None:
    data, folds, evidence, selector, images = _inputs(tmp_path)
    with pytest.raises(ValueError, match="frozen parent/runtime input checksum mismatch"):
        protocol.build_fold_runtime(
            data_path=data,
            folds_path=folds,
            evidence_manifest_path=evidence,
            selector_path=selector,
            image_manifest_path=images,
            outer_fold=0,
            output_dir=tmp_path / "out",
        )


def test_runner_rejects_runtime_built_without_frozen_enforcement(tmp_path: Path) -> None:
    data, folds, evidence, selector, images = _inputs(tmp_path)
    output = tmp_path / "runtime"
    protocol.build_fold_runtime(
        data_path=data,
        folds_path=folds,
        evidence_manifest_path=evidence,
        selector_path=selector,
        image_manifest_path=images,
        outer_fold=0,
        output_dir=output,
        enforce_frozen_hashes=False,
    )
    with pytest.raises(ValueError, match="frozen input enforcement"):
        runner._read_runtime(output, 0)


def test_bad_exact_offset_is_masked_instead_of_weakly_supervised() -> None:
    row = {"name": "БАД", "description": ""}
    candidate = {
        "source": "name",
        "surface_start": 1,
        "surface_end": 3,
        "exact_surface_span": "БАД",
        "concept": "BAD_EXPLICIT_MARKING",
    }
    target = protocol.rationale_target(row, candidate)
    assert target["has_evidence"] is False
    assert target["quality_weight"] == 0.0


def test_frozen_loss_is_verdict_primary_and_masks_unsafe_rows() -> None:
    torch.manual_seed(1)
    auxiliary = {
        "start_logits": torch.randn(2, 6, requires_grad=True),
        "end_logits": torch.randn(2, 6, requires_grad=True),
        "concept_logits": torch.randn(2, 5, requires_grad=True),
    }
    verdict_logits = torch.randn(2, requires_grad=True)
    total, parts = head.multitask_loss(
        verdict_logits=verdict_logits,
        auxiliary=auxiliary,
        verdict_targets=torch.tensor([1, 0]),
        start_targets=torch.tensor([2, 5]),
        end_targets=torch.tensor([3, 5]),
        concept_targets=torch.tensor([1, -1]),
        quality_weights=torch.tensor([1.0, 0.0]),
    )
    expected = (
        parts["verdict"]
        + 0.10 * parts["span"]
        + 0.05 * parts["concept"]
    )
    assert torch.allclose(total.detach(), expected)
    total.backward()
    assert verdict_logits.grad is not None


def test_exact_subsequence_and_character_alignment_fail_closed() -> None:
    start, end, mask = head.attach_span_targets(
        full_input_ids=[9, 1, 2, 3, 8],
        canonical_input_ids=[1, 2, 3],
        canonical_offsets=[(0, 2), (2, 5), (5, 8)],
        rationale={"has_evidence": True, "char_start": 2, "char_end": 8},
    )
    assert (start, end) == (2, 3)
    assert mask == [False, True, True, True, False]
    with pytest.raises(ValueError, match="exactly once"):
        head.locate_unique_subsequence([1, 2, 1, 2], [1, 2])


class _CharacterTokenizer:
    def __call__(self, text, **_):
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def test_parent_prompt_alignment_and_prediction_mapping_are_exact() -> None:
    prompt = "Категория: БАД\nНазвание: БАД комплекс"
    prefix = [1, 2]
    full = prefix + [ord(character) for character in prompt] + [3]
    result = alignment.align_parent_prompt(
        tokenizer=_CharacterTokenizer(),
        full_input_ids=full,
        parent_user_text=prompt,
        rationale={
            "has_evidence": True,
            "exact_span": "БАД комплекс",
            "concept": "OBJECT_OF_SALE",
            "quality_weight": 1.0,
        },
    )
    assert result.quality_weight == 1.0
    assert result.start_target == len(prefix) + prompt.index("БАД комплекс")
    canonical_start, canonical_end = alignment.predicted_token_span_to_canonical_offsets(
        start_token=result.start_target,
        end_token=result.end_target,
        alignment=result,
        parent_user_text=prompt,
        name="БАД комплекс",
        description="",
    )
    assert (canonical_start, canonical_end) == (10, 22)


def test_parent_prompt_alignment_masks_ambiguous_exact_quote() -> None:
    prompt = "БАД и БАД"
    result = alignment.align_parent_prompt(
        tokenizer=_CharacterTokenizer(),
        full_input_ids=[ord(character) for character in prompt],
        parent_user_text=prompt,
        rationale={
            "has_evidence": True,
            "exact_span": "БАД",
            "concept": "OBJECT_OF_SALE",
            "quality_weight": 1.0,
        },
    )
    assert result.quality_weight == 0.0
    assert result.start_target == len(prompt)


def test_renderer_never_generates_free_text_or_repairs_offsets() -> None:
    text, _ = protocol.canonical_text("Горелка", "без топлива")
    start = text.index("без топлива")
    result = renderer.render_explanation(
        name="Горелка",
        description="без топлива",
        char_start=start,
        char_end=start + len("без топлива"),
        concept="NEGATION",
    )
    assert result["evidence"] == "без топлива"
    assert "без топлива" in result["explanation"]
    assert (
        renderer.render_explanation(
            name="x", description="y", char_start=-1, char_end=2, concept="NEGATION"
        )["evidence"]
        == "NO_EVIDENCE"
    )


def test_label_free_prediction_contract_checks_exact_grounding(tmp_path: Path) -> None:
    validation = tmp_path / "validation.jsonl"
    validation.write_text(
        json.dumps(
            {
                "id": "x",
                "category": "БАД",
                "name": "БАД комплекс",
                "description": "",
                "development_fold": 3,
                "row_index": 0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        [{
            "id": "x",
            "category": "БАД",
            "fold": 3,
            "lora_score": 1.0986123,
            "verdict_probability": 0.75,
            "evidence": "БАД комплекс",
            "concept": "OBJECT_OF_SALE",
            "explanation": "Решение основано на том, что именно продаётся: «БАД комплекс».",
            "char_start": 10,
            "char_end": 22,
        }]
    ).to_csv(predictions, index=False)
    report = protocol.validate_label_free_predictions(
        prediction_path=predictions,
        validation_path=validation,
        outer_fold=3,
    )
    assert report["decision"] == "GO"
    frame = pd.read_csv(predictions)
    frame.loc[0, "evidence"] = "комплекс БАД"
    frame.to_csv(predictions, index=False)
    with pytest.raises(ValueError, match="exact substring"):
        protocol.validate_label_free_predictions(
            prediction_path=predictions,
            validation_path=validation,
            outer_fold=3,
        )


def test_qwen_wrapper_uses_parent_digit_logit_and_one_hidden_pass() -> None:
    backbone = model_module.TinyBackbone(vocab_size=8, hidden_size=12)
    wrapped = model_module.Qwen35VerdictSpanModel(
        backbone, hidden_size=12, concept_count=5, token_zero=0, token_one=1
    )
    output = wrapped(
        input_ids=torch.tensor([[2, 3, 4], [5, 6, 0]]),
        attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
        text_token_mask=torch.tensor([[0, 1, 1], [1, 1, 0]]),
    )
    assert output["verdict_logits"].shape == (2,)
    assert output["start_logits"].shape == (2, 4)
    assert output["concept_logits"].shape == (2, 5)


def test_cpu_preflight_go() -> None:
    report = preflight.run_preflight()
    assert report["decision"] == "GO"
    assert all(report["checks"].values())


def test_frozen_spec_matches_implementation_weights() -> None:
    spec = json.loads((EXP / "frozen_spec.json").read_text(encoding="utf-8"))
    weights = head.LossWeights()
    assert spec["loss"]["verdict_ce_weight"] == weights.verdict
    assert spec["loss"]["span_ce_weight"] == weights.span
    assert spec["loss"]["concept_ce_weight"] == weights.concept
    assert spec["frozen_input_sha256"] == protocol.FROZEN_INPUT_SHA256
