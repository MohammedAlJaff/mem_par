#!/usr/bin/env python3
"""
Create a small contact sheet of the first N FFHQ thumbnails.

This is a lightweight viewer helper for quick sanity checks without importing
the full training stack (no torch/matplotlib required).
"""

from pathlib import Path

from PIL import Image
import argparse


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a contact sheet from the first N FFHQ images."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/ffhq/thumbnails128x128"),
        help="Path to FFHQ thumbnail root directory.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=4,
        help="How many images to include from the start of the sorted file list.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Image crop/resize size for each tile.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding (pixels) between tiles and outer border.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/ffhq_first4_view.jpg"),
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p
        for p in Path(args.root).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        raise FileNotFoundError(f"No images found in {args.root}")

    selected = files[: args.n]
    if len(selected) == 0:
        raise ValueError(f"Requested n={args.n} but no files were found.")

    print("FFHQ first images:")
    for i, file_path in enumerate(selected, start=1):
        print(f"  {i}: {file_path}")

    images = []
    for file_path in selected:
        img = Image.open(file_path).convert("RGB")
        img = img.resize((args.size, args.size), Image.BICUBIC)
        images.append(img)

    width = len(selected) * args.size + (len(selected) + 1) * args.padding
    height = args.size + 2 * args.padding
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    for i, img in enumerate(images):
        x = args.padding + i * (args.size + args.padding)
        y = args.padding
        canvas.paste(img, (x, y))

    canvas.save(args.out)
    print(f"saved: {args.out.resolve()}")


if __name__ == "__main__":
    main()
