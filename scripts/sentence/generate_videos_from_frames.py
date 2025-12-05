import argparse
import re
from pathlib import Path
from typing import List, Tuple

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MP4 videos from frame sequences."
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Directory containing subdirectories with frame sequences.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/videos_from_cleaned_frames",
        help="Directory to save generated videos.",
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
        help="Optional: specific subfolder to process.",
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


def get_ordered_frames(folder: Path) -> List[Tuple[int, str, Path]]:
    """
    Get all frames sorted by frame_id.
    Returns: [(frame_id, ref_id, path), ...]
    """
    frames = []
    for f in folder.glob("*.jpg"):
        frame_id, ref_id = parse_filename(f.name)
        if frame_id >= 0 and ref_id:
            frames.append((frame_id, ref_id, f))

    frames.sort(key=lambda x: x[0])
    return frames


def generate_video(folder: Path, output_dir: Path, fps: int) -> None:
    """
    Generate MP4 video from frames in a folder.
    """
    folder_name = folder.name
    print(f"\nProcessing: {folder_name}")
    print("=" * 60)

    ordered_frames = get_ordered_frames(folder)

    if len(ordered_frames) == 0:
        print(f"  [WARN] No frames found, skipping.")
        return

    print(f"  Total frames: {len(ordered_frames)}")

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(ordered_frames[0][2]))
    if first_frame is None:
        print(f"  [ERROR] Failed to read first frame.")
        return

    height, width = first_frame.shape[:2]
    print(f"  Frame size: {width}x{height}")
    print(f"  FPS: {fps}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup video writer
    output_path = output_dir / f"{folder_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not video_writer.isOpened():
        print(f"  [ERROR] Failed to open video writer.")
        return

    # Write frames
    for i, (frame_id, ref_id, frame_path) in enumerate(ordered_frames):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"  [WARN] Failed to read frame: {frame_path.name}")
            continue

        video_writer.write(frame)

        if (i + 1) % 100 == 0:
            print(f"  Written {i + 1}/{len(ordered_frames)} frames...")

    video_writer.release()

    print(f"  Video saved to: {output_path}")
    print(f"  Duration: {len(ordered_frames) / fps:.2f} seconds")


def main():
    args = parse_args()

    frames_root = Path(args.frames_dir)
    output_dir = Path(args.output_dir)

    if not frames_root.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_root}")

    # Determine which folders to process
    if args.subfolder:
        folders = [frames_root / args.subfolder]
        if not folders[0].exists():
            raise FileNotFoundError(f"Subfolder not found: {folders[0]}")
    else:
        folders = [d for d in frames_root.iterdir() if d.is_dir()]

    if not folders:
        print("No folders found to process.")
        return

    print(f"Found {len(folders)} folder(s) to process")
    print(f"FPS: {args.fps}")
    print(f"Output directory: {output_dir}\n")

    # Process each folder
    for folder in sorted(folders):
        generate_video(folder, output_dir, args.fps)

    print("\n" + "=" * 60)
    print("Done. All videos generated.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
