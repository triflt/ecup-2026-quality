from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

MODEL_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
)
RUSSIAN_PROBE = "Биологически активная добавка, 30 капсул."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_model_dir(root: Path) -> Path:
    if (root / "config.json").is_file():
        return root
    matches = sorted(root.rglob("config.json"))
    if len(matches) != 1:
        raise ValueError(f"expected one model config under model root, found {len(matches)}")
    return matches[0].parent


def validate_offsets(text: str, offsets: list[list[int]] | list[tuple[int, int]]) -> None:
    if not offsets:
        raise ValueError("tokenizer returned no Russian offsets")
    previous_end = 0
    covered = 0
    for start, end in offsets:
        start, end = int(start), int(end)
        if start < 0 or end < start or end > len(text) or start < previous_end:
            raise ValueError("tokenizer returned invalid or non-monotonic offsets")
        if end > start:
            covered += end - start
            previous_end = end
    if covered == 0:
        raise ValueError("Russian offsets cover no characters")


def metadata_checksums(model_dir: Path) -> dict[str, str]:
    checksums = {
        name: sha256_file(model_dir / name)
        for name in MODEL_METADATA_FILES
        if (model_dir / name).is_file()
    }
    if "config.json" not in checksums or "tokenizer_config.json" not in checksums:
        raise FileNotFoundError("model metadata lacks config or tokenizer_config")
    return checksums


def write_report(output_dir: Path, report: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "compatibility_probe_report.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def run_probe(*, model_root: Path, model_id: str, revision: str) -> dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    model_dir = resolve_model_dir(model_root)
    metadata = metadata_checksums(model_dir)
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForMultimodalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()
    load_seconds = time.perf_counter() - started

    encoded = processor.tokenizer(
        RUSSIAN_PROBE,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    validate_offsets(RUSSIAN_PROBE, offsets)

    image = Image.new("RGB", (64, 64), (220, 220, 220))
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": (
                    "Это синтетическая карточка товара. Найди точную фразу-доказательство: "
                    + RUSSIAN_PROBE
                ),
            },
        ],
    }]
    template_kwargs = {
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    try:
        inputs = processor.apply_chat_template(messages, enable_thinking=False, **template_kwargs)
    except TypeError:
        inputs = processor.apply_chat_template(messages, **template_kwargs)
    inputs = {name: value.to("cuda") for name, value in inputs.items()}

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]
        head = torch.nn.Linear(hidden.shape[-1], 1, bias=True).to(
            device=hidden.device,
            dtype=hidden.dtype,
        )
        scores = head(hidden)
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started
    if outputs.logits.ndim != 3 or hidden.ndim != 3 or scores.shape[-1] != 1:
        raise ValueError("unexpected model or evidence-head tensor shape")
    if not torch.isfinite(outputs.logits).all() or not torch.isfinite(hidden).all():
        raise ValueError("model returned non-finite logits or hidden states")
    if not torch.isfinite(scores).all():
        raise ValueError("evidence head returned non-finite scores")

    return {
        "status": "passed",
        "probe_version": "semantic_v3_evidence_head_synthetic_v1",
        "model_id": model_id,
        "revision": revision,
        "model_metadata_sha256": metadata,
        "synthetic_input_only": True,
        "competition_rows_loaded": 0,
        "labels_loaded": False,
        "sealed_rows_loaded": 0,
        "rationale_generated": False,
        "logits_finite": True,
        "hidden_states_finite": True,
        "evidence_scores_finite": True,
        "russian_offsets_valid": True,
        "russian_tokens": len(encoded["input_ids"]),
        "sequence_length": int(hidden.shape[1]),
        "hidden_size": int(hidden.shape[2]),
        "load_seconds": round(load_seconds, 6),
        "forward_seconds": round(forward_seconds, 6),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_probe(
        model_root=args.model_root,
        model_id=args.model_id,
        revision=args.revision,
    )
    output = write_report(args.output_dir, report)
    print(json.dumps({"report": str(output), **report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
