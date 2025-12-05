import argparse
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Add project root to Python path for rtmlib import
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from rtmlib import Wholebody


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter frames based on hand pose confidence using RTMLib. "
            "Remove frames where both hands have low confidence."
        )
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Directory containing subdirectories with frame sequences.",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Optional: specific subfolder to analyze.",
    )
    parser.add_argument(
        "--hand-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for hand keypoints (default: 0.3).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/frames_hand_filtered",
        help="Directory to save filtered frames.",
    )
    parser.add_argument(
        "--save-filtered",
        action="store_true",
        help="Save filtered frames to output directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on (default: cpu).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="onnxruntime",
        choices=["opencv", "onnxruntime", "openvino"],
        help="Backend for inference (default: onnxruntime).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="balanced",
        choices=["performance", "lightweight", "balanced"],
        help="RTMLib mode (default: balanced).",
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


def check_hand_confidence(keypoints: np.ndarray, scores: np.ndarray, threshold: float) -> Tuple[bool, bool, float, float]:
    """
    Check if hands have sufficient confidence.

    RTMLib Wholebody keypoints format (133 keypoints total):
    - 0-16: Body keypoints (17 points)
    - 17-22: Face box (6 points)
    - 23-90: Face keypoints (68 points)
    - 91-111: Left hand keypoints (21 points)
    - 112-132: Right hand keypoints (21 points)

    Returns: (left_hand_valid, right_hand_valid, left_hand_avg_score, right_hand_avg_score)
    """
    if keypoints is None or scores is None or len(scores) == 0:
        return False, False, 0.0, 0.0

    # Hand keypoint indices
    left_hand_start = 91
    left_hand_end = 112
    right_hand_start = 112
    right_hand_end = 133

    # Extract hand scores
    if len(scores.shape) == 1:
        # Single person detected
        left_hand_scores = scores[left_hand_start:left_hand_end]
        right_hand_scores = scores[right_hand_start:right_hand_end]
    else:
        # Multiple people detected, use first person
        left_hand_scores = scores[0, left_hand_start:left_hand_end]
        right_hand_scores = scores[0, right_hand_start:right_hand_end]

    # Calculate average confidence for each hand
    left_hand_avg = np.mean(left_hand_scores) if len(left_hand_scores) > 0 else 0.0
    right_hand_avg = np.mean(right_hand_scores) if len(right_hand_scores) > 0 else 0.0

    # Check if hands meet threshold
    left_hand_valid = left_hand_avg >= threshold
    right_hand_valid = right_hand_avg >= threshold

    return left_hand_valid, right_hand_valid, float(left_hand_avg), float(right_hand_avg)


def analyze_folder(
    folder: Path,
    wholebody: Wholebody,
    hand_threshold: float,
    output_dir: Path,
    save_filtered: bool,
) -> dict:
    """
    Analyze a folder and filter frames based on hand confidence.
    """
    folder_name = folder.name
    print(f"\nAnalyzing: {folder_name}")
    print("=" * 60)

    ordered_frames = get_ordered_frames(folder)

    if len(ordered_frames) == 0:
        print(f"  Skipping (no frames found)")
        return None

    print(f"  Total frames: {len(ordered_frames)}")

    # Analyze each frame
    frames_to_keep = []
    frames_to_remove = []

    left_hand_scores = []
    right_hand_scores = []
    both_hands_low = 0
    left_only_low = 0
    right_only_low = 0
    both_hands_high = 0
    no_detection = 0

    for i, (frame_id, ref_id, frame_path) in enumerate(ordered_frames):
        # Read frame
        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"  [WARN] Failed to read frame: {frame_path}")
            frames_to_remove.append((frame_id, ref_id, frame_path))
            no_detection += 1
            continue

        # Run pose detection
        try:
            keypoints, scores = wholebody(img)

            # Check hand confidence
            left_valid, right_valid, left_avg, right_avg = check_hand_confidence(
                keypoints, scores, hand_threshold
            )

            left_hand_scores.append(left_avg)
            right_hand_scores.append(right_avg)

            # Categorize frame
            if not left_valid and not right_valid:
                # Both hands low confidence - remove
                frames_to_remove.append((frame_id, ref_id, frame_path))
                both_hands_low += 1
            else:
                # At least one hand has good confidence - keep
                frames_to_keep.append((frame_id, ref_id, frame_path))

                if left_valid and right_valid:
                    both_hands_high += 1
                elif left_valid:
                    right_only_low += 1
                else:
                    left_only_low += 1

        except Exception as e:
            print(f"  [WARN] Pose detection failed for {frame_path.name}: {e}")
            frames_to_remove.append((frame_id, ref_id, frame_path))
            no_detection += 1

        # Progress update
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(ordered_frames)} frames...")

    # Statistics
    print(f"\n  Analysis Results:")
    print(f"    Total frames: {len(ordered_frames)}")
    print(f"    Frames to keep: {len(frames_to_keep)} ({100*len(frames_to_keep)/len(ordered_frames):.1f}%)")
    print(f"    Frames to remove: {len(frames_to_remove)} ({100*len(frames_to_remove)/len(ordered_frames):.1f}%)")
    print(f"\n  Hand Confidence Breakdown:")
    print(f"    Both hands high confidence: {both_hands_high}")
    print(f"    Left hand only high: {left_only_low} (right hand low)")
    print(f"    Right hand only high: {right_only_low} (left hand low)")
    print(f"    Both hands low confidence: {both_hands_low} [REMOVED]")
    print(f"    No detection / errors: {no_detection} [REMOVED]")

    if left_hand_scores and right_hand_scores:
        print(f"\n  Average Hand Scores:")
        print(f"    Left hand: {np.mean(left_hand_scores):.3f} ± {np.std(left_hand_scores):.3f}")
        print(f"    Right hand: {np.mean(right_hand_scores):.3f} ± {np.std(right_hand_scores):.3f}")

    # Save filtered frames if requested
    if save_filtered and len(frames_to_keep) > 0:
        out_folder = output_dir / folder_name
        out_folder.mkdir(parents=True, exist_ok=True)

        # Copy frames preserving original frame_id to maintain ref_id continuity
        for orig_frame_id, ref_id, frame_path in frames_to_keep:
            new_filename = f"{orig_frame_id}_{ref_id}.jpg"
            dst_path = out_folder / new_filename
            shutil.copy2(frame_path, dst_path)

        print(f"\n  Filtered frames saved to: {out_folder}")

    return {
        'folder': folder_name,
        'total_frames': len(ordered_frames),
        'kept_frames': len(frames_to_keep),
        'removed_frames': len(frames_to_remove),
        'both_hands_high': both_hands_high,
        'left_only_high': left_only_low,
        'right_only_high': right_only_low,
        'both_hands_low': both_hands_low,
        'no_detection': no_detection,
        'left_hand_avg': np.mean(left_hand_scores) if left_hand_scores else 0.0,
        'right_hand_avg': np.mean(right_hand_scores) if right_hand_scores else 0.0,
    }


def main():
    args = parse_args()

    frames_root = Path(args.frames_dir)
    output_dir = Path(args.output_dir)

    if not frames_root.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_root}")

    # Initialize RTMLib Wholebody model
    print("Initializing RTMLib Wholebody model...")
    print(f"  Device: {args.device}")
    print(f"  Backend: {args.backend}")
    print(f"  Mode: {args.mode}")

    wholebody = Wholebody(
        to_openpose=False,
        mode=args.mode,
        backend=args.backend,
        device=args.device
    )

    print("Model initialized successfully.\n")

    # Determine which folders to process
    if args.subfolder:
        folders = [frames_root / args.subfolder]
        if not folders[0].exists():
            raise FileNotFoundError(f"Subfolder not found: {folders[0]}")
    else:
        folders = [d for d in frames_root.iterdir() if d.is_dir()]

    if not folders:
        print("No folders found to analyze.")
        return

    print(f"Found {len(folders)} folder(s) to analyze")
    print(f"Hand confidence threshold: {args.hand_threshold}")
    print(f"Save filtered frames: {args.save_filtered}")

    # Analyze each folder
    results = []
    for folder in sorted(folders):
        result = analyze_folder(
            folder, wholebody, args.hand_threshold, output_dir, args.save_filtered
        )
        if result:
            results.append(result)

    # Overall summary
    if results:
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)

        total_frames = sum(r['total_frames'] for r in results)
        total_kept = sum(r['kept_frames'] for r in results)
        total_removed = sum(r['removed_frames'] for r in results)
        total_both_high = sum(r['both_hands_high'] for r in results)
        total_left_only = sum(r['left_only_high'] for r in results)
        total_right_only = sum(r['right_only_high'] for r in results)
        total_both_low = sum(r['both_hands_low'] for r in results)
        total_no_detection = sum(r['no_detection'] for r in results)

        print(f"\nTotal frames analyzed: {total_frames}")
        print(f"Total frames kept: {total_kept} ({100*total_kept/total_frames:.1f}%)")
        print(f"Total frames removed: {total_removed} ({100*total_removed/total_frames:.1f}%)")
        print(f"\nBreakdown:")
        print(f"  Both hands high: {total_both_high}")
        print(f"  Left only high: {total_left_only}")
        print(f"  Right only high: {total_right_only}")
        print(f"  Both hands low: {total_both_low} [REMOVED]")
        print(f"  No detection: {total_no_detection} [REMOVED]")

        if args.save_filtered:
            print(f"\nFiltered frames saved to: {output_dir}")


if __name__ == "__main__":
    main()

# python .\scripts\sentence\filter_frames_by_hand_pose.py --frames-dir .\output\frames_512x320\ --save-filtered --output-dir .\output\frames_hand_filtered_0.7_v2
