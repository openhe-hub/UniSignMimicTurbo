import argparse
import csv
import os
from typing import List, Optional

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each sentence folder, extract ALL frames from its mp4s "
            "in the order specified by CSV ref_ids, and save as "
            "frame_{frame_id}.jpg."
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
            "Expected layout: <mp4-root>/<sentence_id>/{ref_id}_{timestamp}.mp4"
        ),
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help=(
            "Root directory to save extracted frames. "
            "Frames will be saved under <out-root>/<sentence_id>/ "
            "as frame_{frame_id}.jpg."
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


def extract_all_frames_for_sentence(
    sentence_id: str,
    ref_ids: List[str],
    mp4_root: str,
    out_root: str,
) -> None:
    """
    For a given sentence and its ordered ref_ids list, extract all frames
    from each mp4 in that order and save as:

        <out-root>/<sentence_id>/frame_{frame_id}.jpg

    where frame_id starts from 0 and increases by 1 for each frame,
    and the file name is {frame_id}_{ref_id}.jpg.
    """
    if not ref_ids:
        return

    sentence_dir = os.path.join(mp4_root, sentence_id)
    out_dir = os.path.join(out_root, sentence_id)
    ensure_dir(out_dir)

    frame_id = 0

    for ref_id in ref_ids:
        video_path = find_mp4_for_id(sentence_dir, ref_id)
        if video_path is None:
            print(
                f"[WARN] No mp4 found for ref_id={ref_id} "
                f"in sentence_id={sentence_id}"
            )
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Failed to open video: {video_path}")
            continue

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            out_path = os.path.join(out_dir, f"{frame_id}_{ref_id}.jpg")
            cv2.imwrite(out_path, frame)
            frame_id += 1

        cap.release()


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
        for row in reader:
            if args.num is not None and processed_count >= args.num:
                break

            sentence_id = row.get("sentence_id")
            ref_ids_raw = row.get("ref_ids", "")
            if not sentence_id:
                continue

            ref_ids = parse_ref_ids(ref_ids_raw)
            if not ref_ids:
                continue

            extract_all_frames_for_sentence(sentence_id, ref_ids, mp4_root, out_root)
            processed_count += 1

    print(f"Done. Processed {processed_count} sentences into '{out_root}'.")


if __name__ == "__main__":
    main()

# python scripts/sentence/extract_all_frames_seq.py --csv assets/sentence/sentences.csv --mp4-root output/sentence_level/sentences --out-root output/sentence_level/frames --num 5
