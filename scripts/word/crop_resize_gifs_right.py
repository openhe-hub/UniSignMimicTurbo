import argparse
import os
from typing import List

from PIL import Image, ImageSequence

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crop the right side of each GIF in output/interp-style folders "
            "and resize to a fixed resolution (default 512x320). "
            "Folder structure is preserved."
        )
    )
    parser.add_argument(
        "--in-root",
        type=str,
        required=True,
        help=(
            "Root directory of input GIFs, e.g. output/interp. "
            "Expected layout: <in-root>/<sentence_id>/*.gif"
        ),
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help=(
            "Root directory to save processed GIFs. "
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


def is_gif(name: str) -> bool:
    return name.lower().endswith(".gif")


def process_gif(
    in_path: str,
    out_path: str,
    target_width: int,
    target_height: int,
) -> None:
    try:
        im = Image.open(in_path)
    except Exception as e:
        print(f"[WARN] Failed to open GIF {in_path}: {e}")
        return

    frames: List[Image.Image] = []
    durations: List[int] = []

    for frame in ImageSequence.Iterator(im):
        frame = frame.convert("RGBA")
        w, h = frame.size
        if w <= 0 or h <= 0:
            continue

        # Keep the right side: crop a vertical strip from the right.
        crop_width = min(target_width, w)
        left = max(w - crop_width, 0)
        box = (left, 0, w, h)
        cropped = frame.crop(box)

        resized = cropped.resize((target_width, target_height), Image.LANCZOS)

        # Convert back to palette mode for GIF.
        pal_frame = resized.convert("P", palette=Image.ADAPTIVE)
        frames.append(pal_frame)

        duration = frame.info.get("duration", im.info.get("duration", 40))
        durations.append(duration)

    if not frames:
        print(f"[WARN] No valid frames in GIF {in_path}")
        return

    ensure_dir(os.path.dirname(out_path))

    first, rest = frames[0], frames[1:]
    save_kwargs = {
        "save_all": True,
        "append_images": rest,
        "loop": im.info.get("loop", 0),
        "duration": durations,
        "disposal": im.info.get("disposal", 2),
    }

    try:
        first.save(out_path, format="GIF", **save_kwargs)
    except Exception as e:
        print(f"[WARN] Failed to save GIF {out_path}: {e}")


def main() -> None:
    args = parse_args()

    in_root = os.path.abspath(args.in_root)
    out_root = os.path.abspath(args.out_root)

    if not os.path.isdir(in_root):
        raise NotADirectoryError(f"in-root not found: {in_root}")
    ensure_dir(out_root)

    total = 0
    for dirpath, _, filenames in os.walk(in_root):
        rel = os.path.relpath(dirpath, in_root)
        out_dir = os.path.join(out_root, rel) if rel != "." else out_root

        for name in filenames:
            if not is_gif(name):
                continue

            in_path = os.path.join(dirpath, name)
            out_path = os.path.join(out_dir, name)

            process_gif(
                in_path,
                out_path,
                target_width=args.width,
                target_height=args.height,
            )
            total += 1

    print(
        f"Done. Processed {total} GIFs from '{in_root}' into '{out_root}' "
        f"with size {args.width}x{args.height}."
    )


if __name__ == "__main__":
    main()

# python scripts/word/crop_resize_gifs_right.py --in-root output/word_level/interp --out-root output/word_level/interp_512x320

