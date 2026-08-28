from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-packet", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("multimodal", "text_only"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    source = np.load(args.base_packet, allow_pickle=False)
    ids = source["ids"].astype(str)
    frame = pd.read_csv(args.data, dtype={"id": str}).set_index("id")
    if any(item not in frame.index for item in ids):
        raise ValueError("data does not cover packet ids")
    rows = frame.loc[ids]
    texts = (
        rows["name"].fillna("").astype(str)
        + "\n"
        + rows["description"].fillna("").astype(str).str.slice(0, 1200)
    ).tolist()

    processor = CLIPProcessor.from_pretrained(args.model, local_files_only=True)
    model = CLIPModel.from_pretrained(args.model, local_files_only=True).cuda().eval()
    text_vectors = []
    with torch.inference_mode():
        for start in range(0, len(ids), 256):
            batch = processor(
                text=texts[start : start + 256],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = model.text_model(**{k: v.cuda() for k, v in batch.items()})
            values = model.text_projection(encoded.pooler_output)
            values = torch.nn.functional.normalize(values.float(), dim=1)
            text_vectors.append(values.cpu().numpy())
    text = np.concatenate(text_vectors).astype(np.float32)

    image = np.zeros_like(text)
    if args.mode == "multimodal":
        positions, images = [], []
        with torch.inference_mode():
            for index, item in enumerate(ids):
                path = args.image_cache / f"{item}.jpg"
                if path.exists():
                    with Image.open(path) as handle:
                        images.append(handle.convert("RGB"))
                    positions.append(index)
                if len(images) == 128 or (index + 1 == len(ids) and images):
                    batch = processor(images=images, return_tensors="pt")
                    encoded = model.vision_model(
                        **{k: v.cuda() for k, v in batch.items()}
                    )
                    values = model.visual_projection(encoded.pooler_output)
                    values = torch.nn.functional.normalize(values.float(), dim=1)
                    image[np.asarray(positions)] = values.cpu().numpy()
                    images, positions = [], []
        embeddings = np.concatenate(
            (text, image, text * image, np.abs(text - image)), axis=1
        )
    else:
        embeddings = np.concatenate(
            (text, np.zeros((len(text), 1536), dtype=np.float32)), axis=1
        )
    if embeddings.shape != (len(ids), 2048) or not np.isfinite(embeddings).all():
        raise ValueError("invalid CLIP embeddings")
    arrays = {name: source[name] for name in source.files}
    arrays["embeddings"] = embeddings.astype(np.float16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print({"mode": args.mode, "rows": len(ids), "images": int(np.any(image, axis=1).sum())})


if __name__ == "__main__":
    main()
