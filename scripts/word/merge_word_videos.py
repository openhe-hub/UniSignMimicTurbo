import argparse
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge all MP4 videos in each subfolder into a single MP4 file. "
            "One merged video per folder."
        )
    )
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help=(
            "Root directory containing subfolders with MP4 files. "
            "Expected layout: <input-root>/<folder_id>/*.mp4"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help=(
            "Root directory to save merged MP4 files. "
            "Each merged video will be named <folder_id>.mp4"
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Output video FPS (default: 25).",
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=512,
        help="Output video width (default: 512).",
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=512,
        help="Output video height (default: 512).",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Optional: specific subfolder to process.",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort MP4 files alphabetically before merging (default: file system order).",
    )
    parser.add_argument(
        "--fill-mode",
        type=str,
        default="stretch",
        choices=["pad", "cover", "stretch"],
        help=(
            "Fill mode for aspect ratio handling: "
            "'pad' adds black borders to maintain aspect ratio, "
            "'cover' scales and crops to fill the frame completely, "
            "'stretch' directly resizes ignoring aspect ratio (default, may distort)."
        ),
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resize_with_padding(frame, target_width: int, target_height: int):
    """
    Resize frame to target resolution with padding (black borders).
    Maintains aspect ratio by adding padding on top/bottom or left/right.

    Args:
        frame: Input frame
        target_width: Target width
        target_height: Target height

    Returns:
        Resized frame with padding
    """
    h, w = frame.shape[:2]

    # Calculate scaling factor to fit within target size
    scale = min(target_width / w, target_height / h)

    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize frame
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Create black canvas
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # Calculate padding
    top = (target_height - new_h) // 2
    left = (target_width - new_w) // 2

    # Place resized frame on canvas
    canvas[top:top+new_h, left:left+new_w] = resized

    return canvas


def resize_with_cover(frame, target_width: int, target_height: int):
    """
    Resize frame to cover the target resolution completely (may crop).
    Maintains aspect ratio by scaling to cover and cropping excess.

    Args:
        frame: Input frame
        target_width: Target width
        target_height: Target height

    Returns:
        Resized and cropped frame
    """
    h, w = frame.shape[:2]

    # Calculate scaling factor to cover the target size
    scale = max(target_width / w, target_height / h)

    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize frame
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Calculate crop offsets to center the image
    top = (new_h - target_height) // 2
    left = (new_w - target_width) // 2

    # Crop to target size
    cropped = resized[top:top+target_height, left:left+target_width]

    return cropped


def resize_with_stretch(frame, target_width: int, target_height: int):
    """
    Resize frame to target resolution by stretching (ignores aspect ratio).
    May cause distortion if aspect ratios don't match.

    Args:
        frame: Input frame
        target_width: Target width
        target_height: Target height

    Returns:
        Resized frame (may be distorted)
    """
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)


def get_video_properties(video_path: str) -> tuple:
    """
    Get video properties (width, height, fps).
    Returns: (width, height, fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()
    return width, height, fps


def merge_videos(video_paths: List[str], output_path: str, target_fps: int,
                 output_width: int, output_height: int, fill_mode: str = "pad") -> None:
    """
    Merge multiple MP4 videos into a single output video with specified resolution.

    Args:
        video_paths: List of input video paths
        output_path: Path to save merged video
        target_fps: Target FPS for output video
        output_width: Target width for output video
        output_height: Target height for output video
        fill_mode: Fill mode ('pad' or 'cover') for aspect ratio handling
    """
    if not video_paths:
        print(f"  [WARN] No videos to merge")
        return

    print(f"  Output resolution: {output_width}x{output_height} @ {target_fps} fps")
    print(f"  Fill mode: {fill_mode}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (output_width, output_height))

    if not out.isOpened():
        print(f"  [ERROR] Cannot create output video: {output_path}")
        return

    total_frames = 0

    # Process each video
    for i, video_path in enumerate(video_paths, 1):
        print(f"    [{i}/{len(video_paths)}] Processing: {os.path.basename(video_path)}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"      [WARN] Cannot open video, skipping: {video_path}")
            continue

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame based on fill mode
            if fill_mode == "cover":
                resized_frame = resize_with_cover(frame, output_width, output_height)
            elif fill_mode == "stretch":
                resized_frame = resize_with_stretch(frame, output_width, output_height)
            else:
                resized_frame = resize_with_padding(frame, output_width, output_height)

            out.write(resized_frame)
            frame_count += 1
            total_frames += 1

        cap.release()
        print(f"      Wrote {frame_count} frames")

    out.release()

    print(f"  [OK] Merged {len(video_paths)} videos -> {output_path}")
    print(f"  Total frames: {total_frames}")


def process_folder(folder_path: Path, output_root: Path, target_fps: int,
                   output_width: int, output_height: int, sort_files: bool, fill_mode: str) -> bool:
    """
    Process a single folder: merge all MP4 files into one.

    Args:
        folder_path: Path to the folder containing MP4 files
        output_root: Root directory for output
        target_fps: Target FPS for merged video
        output_width: Target width for output video
        output_height: Target height for output video
        sort_files: Whether to sort files alphabetically
        fill_mode: Fill mode ('pad' or 'cover') for aspect ratio handling

    Returns: True if successful, False otherwise
    """
    folder_name = folder_path.name
    print(f"\nProcessing folder: {folder_name}")
    print("=" * 60)

    # Find all MP4 files
    mp4_files = list(folder_path.glob("*.mp4"))

    if not mp4_files:
        print(f"  [WARN] No MP4 files found, skipping")
        return False

    # Sort if requested
    if sort_files:
        mp4_files.sort()

    print(f"  Found {len(mp4_files)} MP4 file(s)")

    # Output path
    output_path = output_root / f"{folder_name}.mp4"

    # Merge videos
    merge_videos([str(p) for p in mp4_files], str(output_path), target_fps,
                 output_width, output_height, fill_mode)

    return True


def main() -> None:
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    ensure_dir(str(output_root))

    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Output resolution: {args.output_width}x{args.output_height}")
    print(f"Target FPS: {args.fps}")
    print(f"Fill mode: {args.fill_mode}")
    print(f"Sort files: {args.sort}")

    # Determine which folders to process
    if args.subfolder:
        folders = [input_root / args.subfolder]
        if not folders[0].exists():
            raise FileNotFoundError(f"Subfolder not found: {folders[0]}")
    else:
        folders = [d for d in input_root.iterdir() if d.is_dir()]

    if not folders:
        print("\nNo folders found to process.")
        return

    print(f"\nFound {len(folders)} folder(s) to process\n")

    # Process each folder
    successful = 0
    for folder in sorted(folders):
        if process_folder(folder, output_root, args.fps, args.output_width,
                         args.output_height, args.sort, args.fill_mode):
            successful += 1

    print("\n" + "=" * 60)
    print(f"DONE. Processed {successful}/{len(folders)} folder(s) successfully.")
    print(f"Merged videos saved to: {output_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()

# Example usage:
# 1. Merge all folders to 512x512 (default):
#    python scripts/word/merge_word_videos.py \
#        --input-root output/word_level/word_videos_complete \
#        --output-root output/word_level/merged_videos_512x512 \
#        --fps 25
#
# 2. Merge all folders with custom resolution:
#    python scripts/word/merge_word_videos.py \
#        --input-root output/word_level/word_videos_complete \
#        --output-root output/word_level/merged_videos \
#        --fps 25 \
#        --output-width 640 \
#        --output-height 640
#
# 3. Merge a specific folder:
#    python scripts/word/merge_word_videos.py \
#        --input-root output/word_level/word_videos_complete \
#        --output-root output/word_level/merged_videos_512x512 \
#        --subfolder test01 \
#        --fps 25
