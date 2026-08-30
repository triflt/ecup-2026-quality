from __future__ import annotations

import argparse
import json
from pathlib import Path

from gallery_augmentation import audit_archive

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "analysis/label_blind_gallery_audit_v1.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-zip", type=Path, required=True)
    parser.add_argument("--gallery-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = audit_archive(
        images_zip=args.images_zip,
        gallery_manifest=args.gallery_manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
