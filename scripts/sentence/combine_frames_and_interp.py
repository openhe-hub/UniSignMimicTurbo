import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
from PIL import Image, ImageSequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine per-word JPG frames and interpolation GIFs into a single MP4 "
            "for each sentence. Word order is inferred from frame_id within "
            "each sentence directory."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help=(
            "Path to sentences CSV (e.g., assets/sentence/test.csv). "
            "Only used to know which sentence_ids to process."
        ),
    )
    parser.add_argument(
        "--frames-root",
        type=str,
        required=True,
        help=(
            "Root directory of per-frame JPGs, e.g. output/frames_512x320. "
            "Expected layout: <frames-root>/<sentence_id>/<frame_id>_{ref_id}.jpg"
        ),
    )
    parser.add_argument(
        "--interp-root",
        type=str,
        required=True,
        help=(
            "Root directory of interpolation GIFs, e.g. output/interp_right_512x320. "
            "Expected layout: <interp-root>/<sentence_id>/vis_gif{i}.gif"
        ),
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help=(
            "Root directory to save combined MP4s. "
            "Videos will be saved as <out-root>/<sentence_id>/<sentence_id>.mp4"
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Frames per second for the output MP4 (default: 25).",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=None,
        help="Optional: only process the first N sentences in the CSV.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_field(row: Dict[str, str], name: str) -> str:
    """
    Robustly fetch a field from a CSV row, ignoring whitespace around keys.
    """
    for k, v in row.items():
        if k is None:
            continue
        if k.strip() == name:
            return v
    return ""


def collect_frames_for_sentence(
    frames_sentence_dir: str,
) -> Dict[str, List[Tuple[int, str]]]:
    """
    Scan a sentence frames directory and collect:
        {ref_id: [(frame_id, path), ...]}

    Files are expected to be named:
        <frame_id>_{ref_id}.jpg
    """
    by_ref: Dict[str, List[Tuple[int, str]]] = {}

    if not os.path.isdir(frames_sentence_dir):
        return by_ref

    for name in os.listdir(frames_sentence_dir):
        if not name.lower().endswith(".jpg"):
            continue
        stem, _ = os.path.splitext(name)
        try:
            frame_id_str, ref_id = stem.split("_", 1)
            frame_id = int(frame_id_str)
        except ValueError:
            # Unexpected naming, skip.
            continue

        path = os.path.join(frames_sentence_dir, name)
        by_ref.setdefault(ref_id, []).append((frame_id, path))

    # Sort frames for each ref_id by frame_id
    for ref_id, lst in by_ref.items():
        lst.sort(key=lambda x: x[0])

    return by_ref


def collect_interp_gifs_for_sentence(
    interp_sentence_dir: str,
) -> Dict[int, str]:
    """
    Scan a sentence interpolation directory and collect:
        {index: path}

    Files are expected to be named like:
        vis_gif1.gif, vis_gif2.gif, ... (format 1)
        OR
        0.gif, 1.gif, 2.gif, ... (format 2)

    We interpret vis_gif{i}.gif or {i-1}.gif as the interpolation between
    ref_ids[i-1] and ref_ids[i] (1-based index).
    """
    mapping: Dict[int, str] = {}

    if not os.path.isdir(interp_sentence_dir):
        return mapping

    # Try format 1: vis_gif1.gif, vis_gif2.gif, ...
    pattern1 = re.compile(r"vis_gif(\d+)\.gif$", re.IGNORECASE)
    # Try format 2: 0.gif, 1.gif, 2.gif, ...
    pattern2 = re.compile(r"^(\d+)\.gif$")

    for name in os.listdir(interp_sentence_dir):
        m = pattern1.match(name)
        if m:
            # Format 1: vis_gif{i}.gif -> index i
            idx = int(m.group(1))
            path = os.path.join(interp_sentence_dir, name)
            mapping[idx] = path
        else:
            m = pattern2.match(name)
            if m:
                # Format 2: {i}.gif -> index i+1 (convert 0-based to 1-based)
                idx = int(m.group(1)) + 1
                path = os.path.join(interp_sentence_dir, name)
                mapping[idx] = path

    return mapping


def iter_gif_frames(path: str, target_size: Optional[Tuple[int, int]] = None):
    """
    Yield GIF frames as BGR numpy arrays, optionally resized to target_size (w, h).
    """
    im = Image.open(path)
    for frame in ImageSequence.Iterator(im):
        frame = frame.convert("RGB")
        if target_size is not None:
            frame = frame.resize(target_size, Image.LANCZOS)
        # convert to OpenCV BGR
        import numpy as np

        arr = np.array(frame)
        bgr = arr[:, :, ::-1]
        yield bgr


def combine_sentence(
    sentence_id: str,
    frames_root: str,
    interp_root: str,
    out_root: str,
    fps: int,
) -> None:
    """
    Combine JPG frames and interpolation GIFs for a single sentence into an MP4.

    Word order is inferred from frame_id across JPGs within the sentence:
      - frames are named <frame_id>_{ref_id}.jpg
      - sorting by frame_id gives the temporal order;
        the order of distinct ref_ids encountered yields the word order.

    Sequence:
        word_1 frames,
        vis_gif1 (if exists),
        word_2 frames,
        vis_gif2,
        ...
    """
    frames_sentence_dir = os.path.join(frames_root, sentence_id)
    interp_sentence_dir = os.path.join(interp_root, sentence_id)

    frames_by_ref = collect_frames_for_sentence(frames_sentence_dir)
    interp_by_idx = collect_interp_gifs_for_sentence(interp_sentence_dir)

    if not frames_by_ref:
        print(f"[WARN] No frames found for sentence_id={sentence_id}")
        return

    # Infer word order from global frame order.
    global_frames: List[Tuple[int, str]] = []
    for ref_id, lst in frames_by_ref.items():
        for frame_id, _ in lst:
            global_frames.append((frame_id, ref_id))
    if not global_frames:
        print(f"[WARN] No frame metadata for sentence_id={sentence_id}")
        return

    global_frames.sort(key=lambda x: x[0])
    ordered_ref_ids: List[str] = []
    last_ref: Optional[str] = None
    for _, ref_id in global_frames:
        if ref_id != last_ref:
            ordered_ref_ids.append(ref_id)
            last_ref = ref_id

    out_dir = os.path.join(out_root, sentence_id)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, f"{sentence_id}.mp4")

    writer: Optional[cv2.VideoWriter] = None
    video_size: Optional[Tuple[int, int]] = None

    def write_frame(frame):
        nonlocal writer, video_size
        if frame is None:
            return
        h, w = frame.shape[:2]
        if video_size is None:
            video_size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, video_size)
        else:
            if (w, h) != video_size:
                frame = cv2.resize(frame, video_size)
        assert writer is not None
        writer.write(frame)

    # Build sequence
    for idx, ref_id in enumerate(ordered_ref_ids):
        # Word frames
        word_frames = frames_by_ref.get(ref_id, [])
        if not word_frames:
            print(
                f"[WARN] No JPG frames for ref_id={ref_id} in sentence_id={sentence_id}"
            )
        for _, jpg_path in word_frames:
            img = cv2.imread(jpg_path)
            if img is None:
                print(f"[WARN] Failed to read {jpg_path}")
                continue
            write_frame(img)

        # Interpolation between this and next word: vis_gif{i}.gif (1-based)
        if idx < len(ordered_ref_ids) - 1:
            gif_idx = idx + 1
            gif_path = interp_by_idx.get(gif_idx)
            if gif_path is None:
                continue
            for frame in iter_gif_frames(gif_path, target_size=video_size):
                write_frame(frame)

    if writer is not None:
        writer.release()
        print(f"[INFO] Saved {out_path}")
    else:
        print(f"[WARN] No frames written for sentence_id={sentence_id}")


def main() -> None:
    args = parse_args()

    csv_path = os.path.abspath(args.csv)
    frames_root = os.path.abspath(args.frames_root)
    interp_root = os.path.abspath(args.interp_root)
    out_root = os.path.abspath(args.out_root)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isdir(frames_root):
        raise NotADirectoryError(f"frames-root not found: {frames_root}")
    if not os.path.isdir(interp_root):
        raise NotADirectoryError(f"interp-root not found: {interp_root}")
    ensure_dir(out_root)

    processed = 0
    # Use utf-8-sig to handle possible BOM.
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.num is not None and processed >= args.num:
                break

            sentence_raw = get_field(row, "sentence_id")
            sentence_id = (sentence_raw or "").strip()
            if not sentence_id:
                continue

            combine_sentence(
                sentence_id,
                frames_root,
                interp_root,
                out_root,
                fps=args.fps,
            )
            processed += 1

    print(f"Done. Processed {processed} sentences into '{out_root}'.")


if __name__ == "__main__":
    main()
