from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image

from runtime_contract import normalize_generated_comment
from target_contract import messages


ADAPTER_NAME = "explanation_renderer"


def attach_adapter(model: Any, adapter_path: Path) -> None:
    """Attach the renderer to the already-loaded solution140 Qwen3.5 base."""
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"explanation adapter not found: {adapter_path}")
    model.load_adapter(str(adapter_path), adapter_name=ADAPTER_NAME, is_trainable=False)
    model.set_adapter(ADAPTER_NAME)
    model.eval()


def _open_first_image(row: Any) -> Image.Image:
    paths = sorted(Path(path) for path in row.image_paths)
    if not paths:
        raise ValueError(f"missing image0 for id={row.id}")
    with Image.open(paths[0]) as opened:
        image = opened.convert("RGB")
        image.thumbnail((448, 448), Image.Resampling.LANCZOS)
        image.load()
    return image


@torch.inference_mode()
def generate_comments(
    model: Any,
    processor: Any,
    frame: Any,
    verdicts: Any,
    *,
    fallback: Callable[[str, int], str],
    batch_size: int = 2,
) -> tuple[list[str], list[str]]:
    """Generate comments only; verdicts are immutable inputs and never returned by the model."""
    if len(frame) != len(verdicts):
        raise ValueError("frame/verdict length mismatch")
    model.set_adapter(ADAPTER_NAME)
    model.eval()
    comments: list[str] = []
    statuses: list[str] = []
    started = time.monotonic()
    for start in range(0, len(frame), batch_size):
        rows = list(frame.iloc[start:start + batch_size].itertuples(index=False))
        local_verdicts = [int(value) for value in verdicts[start:start + batch_size]]
        images = [_open_first_image(row) for row in rows]
        try:
            conversations = [
                messages(row, verdict=verdict, with_answer=False, image=image)
                for row, verdict, image in zip(rows, local_verdicts, images, strict=True)
            ]
            kwargs = dict(
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2304,
            )
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
        for row, verdict, raw in zip(rows, local_verdicts, texts, strict=True):
            comment, status = normalize_generated_comment(
                raw,
                fallback(str(row.category), verdict),
                category=str(row.category),
                verdict=verdict,
            )
            comments.append(comment)
            statuses.append(status)
        done = min(start + batch_size, len(frame))
        if done % 200 < batch_size or done == len(frame):
            print(
                f"explanations_generated={done}/{len(frame)} "
                f"elapsed_min={(time.monotonic() - started) / 60:.1f}",
                flush=True,
            )
    return comments, statuses
