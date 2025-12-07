import argparse
import os
from typing import List

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop lower half of each frame image and resize to 512x320 "
            "while preserving the sentence/frames folder structure."
        )
    )
    parser.add_argument(
        "--in-root",
        type=str,
        required=True,
        help=(
            "Root directory of input frames, e.g. output/frames. "
            "Expected layout: <in-root>/<sentence_id>/*.jpg"
        ),
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help=(
            "Root directory to save processed frames. "
            "Folder structure <sentence_id>/ will be mirrored here."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Target width after resize (default: 512).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=320,
        help="Target height after resize (default: 320).",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_image_file(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png")


def process_image(
    in_path: str,
    out_path: str,
    target_width: int,
    target_height: int,
) -> None:
    img = cv2.imread(in_path)
    if img is None:
        print(f"[WARN] Failed to read image: {in_path}")
        return

    h, w = img.shape[:2]
    if h <= 1:
        print(f"[WARN] Image height too small to crop: {in_path}")
        return

    # Keep upper half (roughly upper body), drop lower half.
    crop_h = h // 2
    cropped = img[:crop_h, :]

    resized = cv2.resize(cropped, (target_width, target_height))

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, resized)


def main() -> None:
    args = parse_args()

    in_root = os.path.abspath(args.in_root)
    out_root = os.path.abspath(args.out_root)

    if not os.path.isdir(in_root):
        raise NotADirectoryError(f"in-root not found: {in_root}")
    ensure_dir(out_root)

    total = 0
    for sentence_id in os.listdir(in_root):
        in_sentence_dir = os.path.join(in_root, sentence_id)
        if not os.path.isdir(in_sentence_dir):
            continue

        out_sentence_dir = os.path.join(out_root, sentence_id)

        for name in os.listdir(in_sentence_dir):
            if not is_image_file(name):
                continue

            in_path = os.path.join(in_sentence_dir, name)
            out_path = os.path.join(out_sentence_dir, name)

            process_image(
                in_path,
                out_path,
                target_width=args.width,
                target_height=args.height,
            )
            total += 1

    print(
        f"Done. Processed {total} images from '{in_root}' into '{out_root}' "
        f"with size {args.width}x{args.height}."
    )


if __name__ == "__main__":
    main()

# python scripts/sentence/crop_resize_frames.py --in-root output/sentence_level/frames --out-root output/sentence_level/frames_512x320

