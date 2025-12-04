import argparse
import csv
import os
from typing import List, Optional

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each sentence folder of processed mp4s, "
            "follow CSV ref_ids order (e.g. a,b,c) and extract "
            "a_end, b_start, b_end, c_start frames as jpg."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to sentences.csv.",
    )
    parser.add_argument(
        "--mp4-root",
        type=str,
        required=True,
        help=(
            "Root directory of processed mp4s. "
            "Expected layout: <mp4-root>/<sentence_id>/{id}_{timestamp}.mp4"
        ),
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help=(
            "Root directory to save extracted frames. "
            "Frames will be saved under <out-root>/<sentence_id>/."
        ),
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


def parse_ref_ids(raw: str) -> List[str]:
    """
    Parse the ref_ids field into a list of ids.
    Skip placeholder tokens like 'NO_REF_xxx'.
    """
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p and not p.startswith("NO_REF")]


def find_mp4_for_id(sentence_dir: str, ref_id: str) -> Optional[str]:
    """
    Find the mp4 file for a given ref_id within a sentence folder.

    We expect filenames like '{ref_id}_{timestamp}.mp4'.
    If multiple candidates exist, the lexicographically smallest is chosen.
    """
    if not os.path.isdir(sentence_dir):
        return None

    candidates: List[str] = []
    for name in os.listdir(sentence_dir):
        if not name.lower().endswith(".mp4"):
            continue
        if not name.startswith(ref_id + "_"):
            continue
        candidates.append(name)

    if not candidates:
        return None

    candidates.sort()
    return os.path.join(sentence_dir, candidates[0])


def extract_frame(
    video_path: str,
    which: str,
) -> Optional[any]:
    """
    Extract first or last frame from a video.

    which: 'start' or 'end'
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Failed to open video: {video_path}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        print(f"[WARN] Video has no frames: {video_path}")
        cap.release()
        return None

    if which == "start":
        target_idx = 0
    elif which == "end":
        target_idx = max(frame_count - 1, 0)
    else:
        cap.release()
        raise ValueError(f"Invalid 'which' value: {which}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print(f"[WARN] Failed to read frame {target_idx} from {video_path}")
        return None

    return frame


def process_sentence(
    sentence_id: str,
    ref_ids: List[str],
    mp4_root: str,
    out_root: str,
) -> None:
    """
    For a given sentence and its ordered ref_ids list (e.g., [a, b, c]),
    extract frames:
      - a_end
      - b_start, b_end
      - c_start
    and save them as jpgs in <out-root>/<sentence_id>/.
    """
    if not ref_ids:
        return

    sentence_dir = os.path.join(mp4_root, sentence_id)
    out_dir = os.path.join(out_root, sentence_id)
    ensure_dir(out_dir)

    # Map ref_id -> its mp4 path (if exists)
    video_paths: List[Optional[str]] = []
    for ref_id in ref_ids:
        path = find_mp4_for_id(sentence_dir, ref_id)
        if path is None:
            print(
                f"[WARN] No mp4 found for ref_id={ref_id} "
                f"in sentence_id={sentence_id}"
            )
        video_paths.append(path)

    n = len(ref_ids)
    for i, ref_id in enumerate(ref_ids):
        video_path = video_paths[i]
        if video_path is None:
            continue

        # First element: only end frame
        if i == 0 and n == 1:
            # Only one element: both start and end could be useful;
            # but follow the described pattern: only end for first,
            # only start for last. For a single element, we save both.
            frame_start = extract_frame(video_path, "start")
            frame_end = extract_frame(video_path, "end")

            if frame_start is not None:
                out_path = os.path.join(out_dir, f"{ref_id}_start.jpg")
                cv2.imwrite(out_path, frame_start)
            if frame_end is not None:
                out_path = os.path.join(out_dir, f"{ref_id}_end.jpg")
                cv2.imwrite(out_path, frame_end)
        elif i == 0:
            frame_end = extract_frame(video_path, "end")
            if frame_end is not None:
                out_path = os.path.join(out_dir, f"{ref_id}_end.jpg")
                cv2.imwrite(out_path, frame_end)
        # Last element: only start frame
        elif i == n - 1:
            frame_start = extract_frame(video_path, "start")
            if frame_start is not None:
                out_path = os.path.join(out_dir, f"{ref_id}_start.jpg")
                cv2.imwrite(out_path, frame_start)
        # Middle elements: both start and end
        else:
            frame_start = extract_frame(video_path, "start")
            frame_end = extract_frame(video_path, "end")

            if frame_start is not None:
                out_path = os.path.join(out_dir, f"{ref_id}_start.jpg")
                cv2.imwrite(out_path, frame_start)
            if frame_end is not None:
                out_path = os.path.join(out_dir, f"{ref_id}_end.jpg")
                cv2.imwrite(out_path, frame_end)


def main() -> None:
    args = parse_args()

    csv_path = os.path.abspath(args.csv)
    mp4_root = os.path.abspath(args.mp4_root)
    out_root = os.path.abspath(args.out_root)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isdir(mp4_root):
        raise NotADirectoryError(f"mp4-root not found: {mp4_root}")
    ensure_dir(out_root)

    processed_count = 0
    # Use utf-8-sig to handle CSV files that may have a BOM.
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            if args.num is not None and processed_count >= args.num:
                break

            sentence_id = row.get("sentence_id")
            ref_ids_raw = row.get("ref_ids", "")
            if not sentence_id:
                continue

            ref_ids = parse_ref_ids(ref_ids_raw)
            if not ref_ids:
                continue

            process_sentence(sentence_id, ref_ids, mp4_root, out_root)
            processed_count += 1

    print(f"Done. Processed {processed_count} sentences into '{out_root}'.")


if __name__ == "__main__":
    main()
