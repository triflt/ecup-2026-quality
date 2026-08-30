from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from explanation_contract import messages, normalize_generated_comment
from PIL import Image
from src.output import comment_for

ADAPTER_NAME = "explanation_renderer"


def attach_explanation_adapter(model: Any, adapter_path: Path) -> None:
    """Attach the renderer after all solution140 classifier scores are frozen."""
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"explanation adapter not found: {adapter_path}")
    model.load_adapter(str(adapter_path), adapter_name=ADAPTER_NAME, is_trainable=False)
    model.set_adapter(ADAPTER_NAME)
    model.eval()


def _fallback(row: Any, verdict: int) -> str:
    return comment_for(
        str(row.category), int(verdict), str(row.name), str(row.description)
    )


def _open_image0(row: Any) -> Image.Image:
    with Image.open(row.image_paths[0]) as opened:
        image = opened.convert("RGB")
        image.thumbnail((448, 448), Image.Resampling.LANCZOS)
        image.load()
    return image


@torch.inference_mode()
def generate_explanations(
    model: Any,
    processor: Any,
    frame: Any,
    frozen_verdicts: Any,
    *,
    batch_size: int,
) -> tuple[list[str], list[str]]:
    """Generate comments only; frozen_verdicts are inputs and are never returned."""
    if len(frame) != len(frozen_verdicts):
        raise ValueError("frame/verdict length mismatch")
    if batch_size < 1:
        raise ValueError("explanation batch size must be positive")
    model.set_adapter(ADAPTER_NAME)
    model.eval()
    comments: list[str | None] = [None] * len(frame)
    statuses: list[str | None] = [None] * len(frame)
    started = time.monotonic()

    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        positions = list(range(start, stop))
        active_positions = [
            position for position in positions if frame.iloc[position].image_paths
        ]
        for position in positions:
            if position in active_positions:
                continue
            row = frame.iloc[position]
            verdict = int(frozen_verdicts[position])
            fallback = _fallback(row, verdict)
            comment, _ = normalize_generated_comment(
                "",
                fallback,
                category=str(row.category),
                verdict=verdict,
            )
            comments[position] = comment
            statuses[position] = "fallback_missing_image0"

        rows = [frame.iloc[position] for position in active_positions]
        verdicts = [int(frozen_verdicts[position]) for position in active_positions]
        images = [_open_image0(row) for row in rows]
        if rows:
            try:
                conversations = [
                    messages(row, verdict=verdict, image=image)
                    for row, verdict, image in zip(rows, verdicts, images, strict=True)
                ]
                kwargs = {
                    "add_generation_prompt": True,
                    "tokenize": True,
                    "return_dict": True,
                    "return_tensors": "pt",
                    "padding": True,
                    "truncation": True,
                    "max_length": 2304,
                }
                try:
                    batch = processor.apply_chat_template(
                        conversations, enable_thinking=False, **kwargs
                    )
                except TypeError:
                    batch = processor.apply_chat_template(conversations, **kwargs)
            finally:
                for image in images:
                    image.close()
            batch = {key: value.to("cuda") for key, value in batch.items()}
            prompt_length = batch["input_ids"].shape[1]
            generated = model.generate(
                **batch,
                max_new_tokens=192,
                do_sample=False,
                use_cache=True,
            )
            texts = processor.tokenizer.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True
            )
            for position, row, verdict, raw in zip(
                active_positions, rows, verdicts, texts, strict=True
            ):
                comment, status = normalize_generated_comment(
                    raw,
                    _fallback(row, verdict),
                    category=str(row.category),
                    verdict=verdict,
                )
                comments[position] = comment
                statuses[position] = status

        if stop % 200 < batch_size or stop == len(frame):
            print(
                f"explanations_generated={stop}/{len(frame)} "
                f"elapsed_min={(time.monotonic() - started) / 60:.1f}",
                flush=True,
            )

    if any(value is None for value in comments + statuses):
        raise AssertionError("explanation output binding is incomplete")
    final_comments = [str(value) for value in comments]
    final_statuses = [str(value) for value in statuses]
    print(f"explanation_statuses={dict(Counter(final_statuses))}", flush=True)
    return final_comments, final_statuses
