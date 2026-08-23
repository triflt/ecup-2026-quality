from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
PROTOTYPES = {
    "БАД|0": "Товар не является биологически активной добавкой; явной маркировки БАД или dietary supplement нет.",
    "БАД|1": "На товаре прямо указано, что это БАД, биологически активная добавка или dietary supplement.",
    "Легковоспламеняющиеся|0": "Топливо, газ или источник огня не продаются и не входят в комплект; это пустое оборудование, насадка или негорючий аксессуар.",
    "Легковоспламеняющиеся|1": "Продаётся самостоятельное топливо, газ, горючее вещество или источник огня либо они явно входят в комплект.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    expected = list(range(len(rows)))
    if [int(row["global_index"]) for row in rows] != expected:
        raise ValueError("runtime global indices are not complete and ordered")
    forbidden = {"label", "semantic_component", "split"}
    if any(forbidden & set(row) for row in rows):
        raise ValueError("GPU runtime contains forbidden supervision or split metadata")
    return rows


def render(row: dict[str, Any]) -> str:
    return f"Категория: {row['category']}\nНазвание: {row['name']}\nОписание: {row['description']}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.model_revision != MODEL_REVISION:
        raise ValueError("model revision differs from the frozen contract")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite nonempty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from torch.nn import functional
    from transformers import AutoModel, AutoTokenizer

    rows = load_rows(args.runtime)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, padding_side="left", local_files_only=True)
    model = AutoModel.from_pretrained(args.model_root, torch_dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()

    def encode(texts: list[str]) -> np.ndarray:
        parts = []
        for start in range(0, len(texts), args.batch_size):
            batch = tokenizer(
                texts[start : start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                hidden = model(**batch).last_hidden_state
                if int(batch["attention_mask"][:, -1].sum()) == hidden.shape[0]:
                    pooled = hidden[:, -1]
                else:
                    lengths = batch["attention_mask"].sum(dim=1) - 1
                    pooled = hidden[
                        torch.arange(hidden.shape[0], device=hidden.device), lengths
                    ]
                pooled = functional.normalize(pooled.float(), p=2, dim=1)
            parts.append(pooled.cpu().numpy())
        return np.concatenate(parts)

    started = time.monotonic()
    embeddings = encode([render(row) for row in rows])
    prototype_names = sorted(PROTOTYPES)
    prototype_embeddings = encode([PROTOTYPES[name] for name in prototype_names])
    output_path = args.output_dir / "embeddings.npz"
    np.savez_compressed(
        output_path,
        ids=np.asarray([str(row["id"]) for row in rows], dtype=str),
        global_indices=np.arange(len(rows), dtype=np.int64),
        embeddings=embeddings.astype(np.float16),
        prototype_names=np.asarray(prototype_names, dtype=str),
        prototype_embeddings=prototype_embeddings.astype(np.float16),
    )
    elapsed = time.monotonic() - started
    report = {
        "schema_version": 1,
        "experiment_id": "650",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "runtime_sha256": sha256_file(args.runtime),
        "output_sha256": sha256_file(output_path),
        "rows": len(rows),
        "dimension": int(embeddings.shape[1]),
        "finite": bool(np.isfinite(embeddings).all()),
        "norm_min": float(np.linalg.norm(embeddings, axis=1).min()),
        "norm_max": float(np.linalg.norm(embeddings, axis=1).max()),
        "elapsed_seconds": elapsed,
        "rows_per_second": len(rows) / elapsed if elapsed else 0.0,
        "labels_read": 0,
        "sealed_rows": 0,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
