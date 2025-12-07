import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine interpolation GIFs and word videos into complete sequences. "
            "For each word: ref->word transition (GIF) + word video + word->ref transition (GIF)."
        )
    )
    parser.add_argument(
        "--word-videos-root",
        type=str,
        required=True,
        help=(
            "Root directory of word videos. "
            "Expected layout: <word-videos-root>/<sentence_id>/{ref_id}.mp4"
        ),
    )
    parser.add_argument(
        "--interp-root",
        type=str,
        required=True,
        help=(
            "Root directory of interpolation GIFs. "
            "Expected layout: <interp-root>/<sentence_id>/{ref_id}_0.gif, {ref_id}_1.gif"
        ),
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help=(
            "Root directory to save combined videos. "
            "Videos will be saved under <out-root>/<sentence_id>/{ref_id}_complete.mp4"
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Add stage labels ([PRE], [WORD], [POST]) to video frames for debugging.",
    )
    return parser.parse_args()


def add_text_label(frame, label: str) -> None:
    """
    Add a text label to the frame (in-place modification).

    Args:
        frame: OpenCV image (numpy array)
        label: Text to display (e.g., "[PRE]", "[WORD]", "[POST]")
    """
    height, width = frame.shape[:2]

    # Text configuration
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 3  # Thicker for bold effect
    color = (0, 255, 0)  # Green

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Position: top-left corner with padding
    x = 10
    y = text_height + 15

    # Draw background rectangle for better visibility
    cv2.rectangle(
        frame,
        (x - 5, y - text_height - 5),
        (x + text_width + 5, y + baseline + 5),
        (0, 0, 0),
        -1
    )

    # Draw text
    cv2.putText(frame, label, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_frames_from_gif(gif_path: Path, temp_dir: Path) -> List[Path]:
    """
    Extract all frames from a GIF file using OpenCV.
    Returns list of frame paths.
    """
    frames = []
    cap = cv2.VideoCapture(str(gif_path))

    if not cap.isOpened():
        print(f"    [ERROR] Could not open GIF: {gif_path}")
        return []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = temp_dir / f"gif_{gif_path.stem}_{frame_idx:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frames.append(frame_path)
        frame_idx += 1

    cap.release()
    return frames


def extract_frames_from_video(video_path: Path, temp_dir: Path) -> List[Path]:
    """
    Extract all frames from a video file using OpenCV.
    Returns list of frame paths.
    """
    frames = []
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"    [ERROR] Could not open video: {video_path}")
        return []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = temp_dir / f"video_{video_path.stem}_{frame_idx:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frames.append(frame_path)
        frame_idx += 1

    cap.release()
    return frames


def create_video_from_frames(
    frame_paths: List[Path],
    output_path: Path,
    fps: int = 25,
    labels: Optional[List[str]] = None,
    verbose: bool = False
) -> bool:
    """
    Create a video from a list of frame paths.

    Args:
        frame_paths: List of paths to frame images
        output_path: Output video path
        fps: Frame rate
        labels: Optional list of labels for each frame (e.g., "[PRE]", "[WORD]", "[POST]")
        verbose: If True and labels provided, draw labels on frames
    """
    if not frame_paths:
        print(f"    [ERROR] No frames to create video")
        return False

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        print(f"    [ERROR] Could not read first frame: {frame_paths[0]}")
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
    for idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            # Add label if verbose mode and labels provided
            if verbose and labels and idx < len(labels):
                add_text_label(frame, labels[idx])

            out.write(frame)
            frame_count += 1
        else:
            print(f"    [WARN] Could not read frame: {frame_path}")

    out.release()

    if frame_count > 0:
        return True
    else:
        print(f"    [ERROR] No frames written to video")
        return False


def combine_word_sequence(
    ref_id: str,
    word_video_path: Path,
    gif_0_path: Path,
    gif_1_path: Path,
    output_path: Path,
    fps: int,
    verbose: bool = False,
) -> bool:
    """
    Combine: ref->word GIF + word video + word->ref GIF into a complete sequence.

    Args:
        ref_id: Reference ID (word identifier)
        word_video_path: Path to word MP4 video
        gif_0_path: Path to ref->word transition GIF
        gif_1_path: Path to word->ref transition GIF
        output_path: Output video path
        fps: Frame rate
        verbose: If True, add stage labels ([PRE], [WORD], [POST]) to frames

    Returns:
        True if successful, False otherwise
    """
    # Check if all input files exist
    if not word_video_path.exists():
        print(f"    [ERROR] Word video not found: {word_video_path}")
        return False

    if not gif_0_path.exists():
        print(f"    [ERROR] Transition GIF (0) not found: {gif_0_path}")
        return False

    if not gif_1_path.exists():
        print(f"    [ERROR] Transition GIF (1) not found: {gif_1_path}")
        return False

    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        # Extract frames from all sources
        print(f"    Extracting frames from GIF (ref->word)...")
        frames_0 = extract_frames_from_gif(gif_0_path, temp_dir)

        print(f"    Extracting frames from word video...")
        frames_word = extract_frames_from_video(word_video_path, temp_dir)

        print(f"    Extracting frames from GIF (word->ref)...")
        frames_1 = extract_frames_from_gif(gif_1_path, temp_dir)

        if not frames_0 or not frames_word or not frames_1:
            print(f"    [ERROR] Failed to extract frames")
            return False

        # Combine all frames in order
        all_frames = frames_0 + frames_word + frames_1

        # Generate labels for each frame if verbose mode
        labels = None
        if verbose:
            labels = (
                ["[PRE]"] * len(frames_0) +
                ["[WORD]"] * len(frames_word) +
                ["[POST]"] * len(frames_1)
            )

        print(
            f"    Total frames: {len(all_frames)} "
            f"(GIF0: {len(frames_0)}, Word: {len(frames_word)}, GIF1: {len(frames_1)})"
        )

        # Create output video
        print(f"    Creating combined video...")
        success = create_video_from_frames(all_frames, output_path, fps, labels, verbose)

        if success:
            print(f"    Generated: {output_path.name}")
            return True
        else:
            return False


def process_sentence_folder(
    sentence_id: str,
    word_videos_dir: Path,
    interp_dir: Path,
    out_root: Path,
    fps: int,
    verbose: bool = False,
) -> Tuple[int, int]:
    """
    Process all words in a sentence folder.

    Returns: (total_words, successful_videos)
    """
    print(f"Processing sentence: {sentence_id}")

    # Check if directories exist
    if not word_videos_dir.exists():
        print(f"  [WARN] Word videos directory not found: {word_videos_dir}")
        return 0, 0

    if not interp_dir.exists():
        print(f"  [WARN] Interpolation directory not found: {interp_dir}")
        return 0, 0

    # Create output directory
    sentence_out_dir = out_root / sentence_id
    ensure_dir(sentence_out_dir)

    # Find all word videos
    word_videos = list(word_videos_dir.glob("*.mp4"))

    if not word_videos:
        print(f"  [WARN] No word videos found")
        return 0, 0

    print(f"  Found {len(word_videos)} word(s)")

    success_count = 0
    for word_video_path in sorted(word_videos):
        ref_id = word_video_path.stem  # Remove .mp4 extension

        print(f"  Word [{ref_id}]:")

        # Find corresponding GIFs
        gif_0_path = interp_dir / f"{ref_id}_0.gif"
        gif_1_path = interp_dir / f"{ref_id}_1.gif"

        # Output path
        output_path = sentence_out_dir / f"{ref_id}_complete.mp4"

        # Combine sequence
        if combine_word_sequence(
            ref_id, word_video_path, gif_0_path, gif_1_path, output_path, fps, verbose
        ):
            success_count += 1
        print()

    return len(word_videos), success_count


def get_sentence_folders(root: Path, subfolder: Optional[str]) -> List[Path]:
    """Get list of sentence folders to process."""
    if subfolder:
        target = root / subfolder
        if not target.exists():
            raise FileNotFoundError(f"Subfolder not found: {target}")
        return [target]

    return sorted([d for d in root.iterdir() if d.is_dir()])


def main() -> None:
    args = parse_args()

    word_videos_root = Path(args.word_videos_root)
    interp_root = Path(args.interp_root)
    out_root = Path(args.out_root)

    if not word_videos_root.exists():
        raise FileNotFoundError(f"word-videos-root not found: {word_videos_root}")

    if not interp_root.exists():
        raise FileNotFoundError(f"interp-root not found: {interp_root}")

    ensure_dir(out_root)

    # Determine which folders to process
    folders = get_sentence_folders(word_videos_root, args.subfolder)

    if not folders:
        print("No folders found to process.")
        return

    print(f"Found {len(folders)} sentence folder(s) to process.")
    print(f"Output FPS: {args.fps}")
    if args.verbose:
        print(f"Verbose mode: ON (frames will be labeled with [PRE], [WORD], [POST])")
    print()

    # Process each sentence folder
    total_sentences = 0
    total_words = 0
    total_success = 0

    for sentence_folder in folders:
        sentence_id = sentence_folder.name

        word_videos_dir = word_videos_root / sentence_id
        interp_dir = interp_root / sentence_id

        word_count, success_count = process_sentence_folder(
            sentence_id, word_videos_dir, interp_dir, out_root, args.fps, args.verbose
        )

        if word_count > 0:
            total_sentences += 1
            total_words += word_count
            total_success += success_count

    print(f"\nDone. Processed {total_sentences} sentence(s), {total_words} word(s).")
    print(f"Successfully generated {total_success} complete video(s) in '{out_root}'.")


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/word/combine_word_with_interp.py --word-videos-root output/word_level/word_videos --interp-root output/word_level/interp_512x320 --out-root output/word_level/word_videos_complete --fps 25
# With verbose mode (adds [PRE], [WORD], [POST] labels):
# python scripts/word/combine_word_with_interp.py --word-videos-root output/word_level/word_videos --interp-root output/word_level/interp_512x320 --out-root output/word_level/word_videos_complete --fps 5 --verbose
