import argparse
import os
from typing import List

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resize each frame image to specified resolution "
            "while preserving the sentence/frames folder structure. "
            "No cropping, just direct resize."
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
    parser.add_argument(
        "--interpolation",
        type=str,
        default="lanczos",
        choices=["nearest", "linear", "cubic", "lanczos"],
        help="Interpolation method for resizing (default: lanczos).",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_image_file(name: str) -> bool:
    lower = name.lower()
    return lower.endswith((".jpg", ".jpeg", ".png"))


def get_interpolation_flag(method: str) -> int:
    """Convert interpolation method string to OpenCV flag."""
    method_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    return method_map.get(method, cv2.INTER_LANCZOS4)


def process_image(
    in_path: str,
    out_path: str,
    target_width: int,
    target_height: int,
    interpolation: int,
) -> None:
    img = cv2.imread(in_path)
    if img is None:
        print(f"[WARN] Failed to read image: {in_path}")
        return

    # Direct resize without cropping
    resized = cv2.resize(img, (target_width, target_height), interpolation=interpolation)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, resized)


def main() -> None:
    args = parse_args()

    in_root = os.path.abspath(args.in_root)
    out_root = os.path.abspath(args.out_root)

    if not os.path.isdir(in_root):
        raise NotADirectoryError(f"in-root not found: {in_root}")
    ensure_dir(out_root)

    interpolation_flag = get_interpolation_flag(args.interpolation)

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
                interpolation=interpolation_flag,
            )
            total += 1

        if total % 500 == 0 and total > 0:
            print(f"  Processed {total} images...")

    print(
        f"Done. Resized {total} images from '{in_root}' to '{out_root}' "
        f"with size {args.width}x{args.height} (interpolation: {args.interpolation})."
    )


if __name__ == "__main__":
    main()

# Example usage:
# 1. Basic resize to 512x320:
#    python scripts/sentence/resize_frames.py \
#        --in-root output/sentence_level/frames \
#        --out-root output/sentence_level/frames_512x320_resized
#
# 2. Resize with different interpolation:
#    python scripts/sentence/resize_frames.py \
#        --in-root output/sentence_level/frames \
#        --out-root output/sentence_level/frames_512x320_resized \
#        --interpolation cubic
#
# 3. Resize to custom resolution:
#    python scripts/sentence/resize_frames.py \
#        --in-root output/sentence_level/frames \
#        --out-root output/sentence_level/frames_768x480 \
#        --width 768 --height 480
