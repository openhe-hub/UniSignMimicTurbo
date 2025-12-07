import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    ref_id: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MP4 videos for each word (ref_id) from sentence frame sequences. "
            "Each word gets its own video file."
        )
    )
    parser.add_argument(
        "--frames-root",
        type=str,
        required=True,
        help=(
            "Root directory of sentence frame sequences. "
            "Expected layout: <frames-root>/<sentence_id>/{frame_id}_{ref_id}.jpg"
        ),
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help=(
            "Root directory to save word videos. "
            "Videos will be saved under <out-root>/<sentence_id>/{ref_id}.mp4"
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Frames per second for output videos (default: 25).",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Optional: specific subfolder (sentence_id) to process.",
    )
    return parser.parse_args()


def parse_filename(filename: str) -> Tuple[int, str]:
    """
    Extract frame_id and ref_id from filename like '0_vkhwescgz9.jpg'
    Returns: (frame_id, ref_id)
    """
    match = re.match(r"(\d+)_([^.]+)\.jpg", filename)
    if match:
        return int(match.group(1)), match.group(2)
    return -1, ""


def get_frames_by_ref(folder: Path) -> Dict[str, List[FrameRecord]]:
    """
    Group frames by ref_id and sort by frame_id within each group.
    Returns: {ref_id: [(frame_id, path), ...]}
    """
    ref_groups: Dict[str, List[FrameRecord]] = defaultdict(list)

    for frame_file in folder.glob("*.jpg"):
        frame_id, ref_id = parse_filename(frame_file.name)
        if frame_id >= 0 and ref_id:
            ref_groups[ref_id].append(FrameRecord(frame_id, ref_id, frame_file))

    for ref_id in ref_groups:
        ref_groups[ref_id].sort(key=lambda rec: rec.frame_id)

    return ref_groups


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_sentence_folders(frames_root: Path, subfolder: Optional[str]) -> List[Path]:
    if subfolder:
        target = frames_root / subfolder
        if not target.exists():
            raise FileNotFoundError(f"Subfolder not found: {target}")
        return [target]

    return [d for d in frames_root.iterdir() if d.is_dir()]


def generate_word_video(
    frames: List[FrameRecord], output_path: Path, fps: int = 25
) -> bool:
    """
    Generate a video from a list of frames.

    Args:
        frames: List of (frame_id, frame_path) tuples, sorted by frame_id
        output_path: Output video file path
        fps: Frames per second

    Returns:
        True if successful, False otherwise
    """
    if len(frames) < 1:
        print(f"    [WARN] No frames to generate video")
        return False

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frames[0].path))
    if first_frame is None:
        print(f"    [ERROR] Could not read first frame: {frames[0][1]}")
        return False

    height, width = first_frame.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"    [ERROR] Could not create video writer")
        return False

    # Write all frames
    frame_count = 0
    for frame_record in frames:
        frame = cv2.imread(str(frame_record.path))
        if frame is not None:
            out.write(frame)
            frame_count += 1
        else:
            print(f"    [WARN] Could not read frame: {frame_record.path}")

    out.release()

    if frame_count > 0:
        print(f"    Generated: {output_path.name} ({frame_count} frames)")
        return True
    else:
        print(f"    [ERROR] No frames written to video")
        return False


def process_sentence_folder(
    sentence_folder: Path, out_root: Path, fps: int
) -> Tuple[int, int]:
    """
    Process a single sentence folder and generate videos for each word.

    Returns: (total_words, successful_videos)
    """
    sentence_id = sentence_folder.name
    print(f"Processing sentence: {sentence_id}")

    # Create output directory for this sentence
    sentence_out_dir = out_root / sentence_id
    ensure_dir(sentence_out_dir)

    # Group frames by ref_id (each ref_id is a word)
    ref_groups = get_frames_by_ref(sentence_folder)

    if not ref_groups:
        print(f"  [WARN] No valid frames found, skipping.")
        return 0, 0

    print(f"  Found {len(ref_groups)} word(s)")

    success_count = 0
    for ref_id, frames in sorted(ref_groups.items()):
        if not frames:
            continue

        print(f"  Word [{ref_id}]: {len(frames)} frames")

        # Generate video for this word
        output_path = sentence_out_dir / f"{ref_id}.mp4"
        if generate_word_video(frames, output_path, fps):
            success_count += 1

    return len(ref_groups), success_count


def main() -> None:
    args = parse_args()

    frames_root = Path(args.frames_root)
    out_root = Path(args.out_root)

    if not frames_root.exists():
        raise FileNotFoundError(f"frames-root not found: {frames_root}")

    ensure_dir(out_root)

    # Determine which folders to process
    folders = get_sentence_folders(frames_root, args.subfolder)

    if not folders:
        print("No folders found to process.")
        return

    print(f"Found {len(folders)} sentence folder(s) to process.")
    print(f"Output FPS: {args.fps}\n")

    # Process each sentence folder
    total_sentences = 0
    total_words = 0
    total_success = 0

    for folder in sorted(folders):
        word_count, success_count = process_sentence_folder(folder, out_root, args.fps)
        if word_count > 0:
            total_sentences += 1
            total_words += word_count
            total_success += success_count
        print()

    print(f"Done. Processed {total_sentences} sentence(s), {total_words} word(s).")
    print(f"Successfully generated {total_success} video(s) in '{out_root}'.")


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/word/generate_word_videos.py --frames-root output/sentence_level/frames_512x320_filtered2 --out-root output/word_level/word_videos --fps 25
