import argparse
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

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
        default=0.8,
        help="Confidence threshold for hand keypoints (default: 0.8).",
    )
    parser.add_argument(
        "--head-threshold",
        type=float,
        default=0.9,
        help="Confidence threshold for head/face keypoints (default: 0.9).",
    )
    parser.add_argument(
        "--hand-height-threshold",
        type=float,
        default=0.1,
        help="Height threshold ratio for filtering low hands (default: 0.1 = bottom 10%%).",
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
        "--verbose",
        action="store_true",
        help="Annotate saved frames with confidence scores.",
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


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    ref_id: str
    path: Path


@dataclass
class FrameDecision:
    record: FrameRecord
    keep: bool
    left_valid: bool
    right_valid: bool
    head_valid: bool
    hands_too_low: bool
    detection_failed: bool
    left_avg: float = 0.0
    right_avg: float = 0.0
    head_avg: float = 0.0


@dataclass
class PoseStats:
    frames_to_keep: List[FrameDecision] = field(default_factory=list)
    frames_to_remove: List[FrameRecord] = field(default_factory=list)
    left_hand_scores: List[float] = field(default_factory=list)
    right_hand_scores: List[float] = field(default_factory=list)
    head_scores: List[float] = field(default_factory=list)
    both_hands_low: int = 0
    left_only_low: int = 0
    right_only_low: int = 0
    both_hands_high: int = 0
    head_low: int = 0
    hands_too_low: int = 0
    no_detection: int = 0

    def add_decision(self, decision: FrameDecision) -> None:
        if decision.detection_failed:
            self.frames_to_remove.append(decision.record)
            self.no_detection += 1
            return

        self.left_hand_scores.append(decision.left_avg)
        self.right_hand_scores.append(decision.right_avg)
        self.head_scores.append(decision.head_avg)

        if not decision.head_valid:
            self.frames_to_remove.append(decision.record)
            self.head_low += 1
            return

        if decision.hands_too_low:
            self.frames_to_remove.append(decision.record)
            self.hands_too_low += 1
            return

        if not decision.left_valid and not decision.right_valid:
            self.frames_to_remove.append(decision.record)
            self.both_hands_low += 1
            return

        self.frames_to_keep.append(decision)
        if decision.left_valid and decision.right_valid:
            self.both_hands_high += 1
        elif decision.left_valid:
            self.right_only_low += 1
        else:
            self.left_only_low += 1


def parse_filename(filename: str) -> Tuple[int, str]:
    """
    Extract frame_id and ref_id from filename like '0_vkhwescgz9.jpg'
    Returns: (frame_id, ref_id)
    """
    match = re.match(r"(\d+)_([^.]+)\.jpg", filename)
    if match:
        return int(match.group(1)), match.group(2)
    return -1, ""


def get_ordered_frames(folder: Path) -> List[FrameRecord]:
    """
    Get all frames sorted by frame_id.
    Returns: [FrameRecord, ...]
    """
    frames: List[FrameRecord] = []
    for f in folder.glob("*.jpg"):
        frame_id, ref_id = parse_filename(f.name)
        if frame_id >= 0 and ref_id:
            frames.append(FrameRecord(frame_id, ref_id, f))

    frames.sort(key=lambda x: x.frame_id)
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


def check_head_confidence(keypoints: np.ndarray, scores: np.ndarray, threshold: float) -> Tuple[bool, float]:
    """
    Check if head/face has sufficient confidence.

    RTMLib Wholebody keypoints format (133 keypoints total):
    - 0-16: Body keypoints (17 points) - includes nose, eyes, ears
    - 17-22: Face box (6 points)
    - 23-90: Face keypoints (68 points)

    Returns: (head_valid, head_avg_score)
    """
    if keypoints is None or scores is None or len(scores) == 0:
        return False, 0.0

    # Use body keypoints for head: nose(0), left_eye(1), right_eye(2), left_ear(3), right_ear(4)
    head_keypoint_indices = [0, 1, 2, 3, 4]

    # Extract head scores
    if len(scores.shape) == 1:
        # Single person detected
        head_scores = scores[head_keypoint_indices]
    else:
        # Multiple people detected, use first person
        head_scores = scores[0, head_keypoint_indices]

    # Calculate average confidence for head
    head_avg = np.mean(head_scores) if len(head_scores) > 0 else 0.0

    # Check if head meets threshold
    head_valid = head_avg >= threshold

    return head_valid, float(head_avg)


def check_hand_height(keypoints: np.ndarray, scores: np.ndarray, img_height: int,
                      threshold_ratio: float = 0.1) -> Tuple[bool, float, float]:
    """
    Check if hands are too low in the frame (below threshold_ratio from bottom).

    RTMLib Wholebody keypoints format (133 keypoints total):
    - 91-111: Left hand keypoints (21 points)
    - 112-132: Right hand keypoints (21 points)

    Args:
        keypoints: Keypoint coordinates (x, y)
        scores: Confidence scores for keypoints
        img_height: Height of the image
        threshold_ratio: Ratio from top (default 0.1 means bottom 10%)

    Returns: (hands_too_low, left_hand_avg_y_ratio, right_hand_avg_y_ratio)
    """
    if keypoints is None or scores is None or len(keypoints) == 0:
        return False, 0.0, 0.0

    # Hand keypoint indices
    left_hand_start = 91
    left_hand_end = 112
    right_hand_start = 112
    right_hand_end = 133

    # Extract hand keypoints and scores
    if len(keypoints.shape) == 2:
        # Single person detected
        left_hand_kpts = keypoints[left_hand_start:left_hand_end]
        right_hand_kpts = keypoints[right_hand_start:right_hand_end]
        left_hand_scores = scores[left_hand_start:left_hand_end]
        right_hand_scores = scores[right_hand_start:right_hand_end]
    else:
        # Multiple people detected, use first person
        left_hand_kpts = keypoints[0, left_hand_start:left_hand_end]
        right_hand_kpts = keypoints[0, right_hand_start:right_hand_end]
        left_hand_scores = scores[0, left_hand_start:left_hand_end]
        right_hand_scores = scores[0, right_hand_start:right_hand_end]

    # Calculate average Y position for each hand (only use confident keypoints)
    conf_threshold = 0.3  # Minimum confidence to consider a keypoint

    # Left hand average Y
    left_valid_y = [kpt[1] for kpt, score in zip(left_hand_kpts, left_hand_scores) if score > conf_threshold]
    left_avg_y = np.mean(left_valid_y) if len(left_valid_y) > 0 else img_height / 2

    # Right hand average Y
    right_valid_y = [kpt[1] for kpt, score in zip(right_hand_kpts, right_hand_scores) if score > conf_threshold]
    right_avg_y = np.mean(right_valid_y) if len(right_valid_y) > 0 else img_height / 2

    # Calculate ratio (0.0 = top, 1.0 = bottom)
    left_y_ratio = left_avg_y / img_height
    right_y_ratio = right_avg_y / img_height

    # Check if BOTH hands are in the bottom threshold_ratio area
    # If ratio > (1 - threshold_ratio), hand is too low
    threshold_from_top = 1.0 - threshold_ratio
    both_hands_too_low = (left_y_ratio > threshold_from_top) and (right_y_ratio > threshold_from_top)

    return both_hands_too_low, float(left_y_ratio), float(right_y_ratio)


def evaluate_frame(record: FrameRecord, wholebody: Wholebody, hand_threshold: float,
                   head_threshold: float, hand_height_threshold: float) -> Tuple[FrameDecision, Optional[str]]:
    """
    Run pose estimation on a frame and decide whether to keep it.
    Returns the decision and an optional warning message.
    """
    img = cv2.imread(str(record.path))
    if img is None:
        decision = FrameDecision(
            record=record,
            keep=False,
            left_valid=False,
            right_valid=False,
            head_valid=False,
            hands_too_low=False,
            detection_failed=True,
        )
        return decision, f"[WARN] Failed to read frame: {record.path}"

    try:
        keypoints, scores = wholebody(img)
    except Exception as exc:  # noqa: BLE001 - mirror previous broad catch
        decision = FrameDecision(
            record=record,
            keep=False,
            left_valid=False,
            right_valid=False,
            head_valid=False,
            hands_too_low=False,
            detection_failed=True,
        )
        return decision, f"[WARN] Pose detection failed for {record.path.name}: {exc}"

    img_height = img.shape[0]

    left_valid, right_valid, left_avg, right_avg = check_hand_confidence(
        keypoints, scores, hand_threshold
    )
    head_valid, head_avg = check_head_confidence(
        keypoints, scores, head_threshold
    )
    hands_low_position, _, _ = check_hand_height(
        keypoints, scores, img_height, threshold_ratio=hand_height_threshold
    )

    keep = head_valid and not hands_low_position and (left_valid or right_valid)
    decision = FrameDecision(
        record=record,
        keep=keep,
        left_valid=left_valid,
        right_valid=right_valid,
        head_valid=head_valid,
        hands_too_low=hands_low_position,
        detection_failed=False,
        left_avg=float(left_avg),
        right_avg=float(right_avg),
        head_avg=float(head_avg),
    )
    return decision, None


def annotate_frame(img: np.ndarray, left_conf: float, right_conf: float, head_conf: float) -> np.ndarray:
    """Annotate an image with confidence scores."""
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (280, 85), (0, 0, 0), -1)
    annotated = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    cv2.putText(annotated, f"Left Hand:  {left_conf:.3f}", (10, 25),
               font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(annotated, f"Right Hand: {right_conf:.3f}", (10, 50),
               font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(annotated, f"Head:       {head_conf:.3f}", (10, 75),
               font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    return annotated


def save_filtered_frames(frames_to_keep: List[FrameDecision], output_dir: Path,
                         folder_name: str, verbose: bool) -> None:
    """Persist kept frames, optionally annotated with confidence scores."""
    out_folder = output_dir / folder_name
    out_folder.mkdir(parents=True, exist_ok=True)

    for decision in frames_to_keep:
        record = decision.record
        new_filename = f"{record.frame_id}_{record.ref_id}.jpg"
        dst_path = out_folder / new_filename

        if verbose:
            img = cv2.imread(str(record.path))
            if img is not None:
                annotated = annotate_frame(img, decision.left_avg, decision.right_avg, decision.head_avg)
                cv2.imwrite(str(dst_path), annotated)
                continue

        shutil.copy2(record.path, dst_path)

    print(f"\n  Filtered frames saved to: {out_folder}")
    if verbose:
        print(f"  Frames annotated with confidence scores")


def analyze_folder(
    folder: Path,
    wholebody: Wholebody,
    hand_threshold: float,
    head_threshold: float,
    hand_height_threshold: float,
    output_dir: Path,
    save_filtered: bool,
    verbose: bool = False,
) -> dict:
    """
    Analyze a folder and filter frames based on hand and head confidence.
    """
    folder_name = folder.name
    print(f"\nAnalyzing: {folder_name}")
    print("=" * 60)

    ordered_frames = get_ordered_frames(folder)

    if len(ordered_frames) == 0:
        print(f"  Skipping (no frames found)")
        return None

    print(f"  Total frames: {len(ordered_frames)}")

    stats = PoseStats()

    for i, record in enumerate(ordered_frames):
        decision, warning = evaluate_frame(
            record, wholebody, hand_threshold, head_threshold, hand_height_threshold
        )
        if warning:
            print(f"  {warning}")

        stats.add_decision(decision)

        # Progress update
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(ordered_frames)} frames...")

    # Statistics
    kept_count = len(stats.frames_to_keep)
    removed_count = len(stats.frames_to_remove)
    print(f"\n  Analysis Results:")
    print(f"    Total frames: {len(ordered_frames)}")
    print(f"    Frames to keep: {kept_count} ({100*kept_count/len(ordered_frames):.1f}%)")
    print(f"    Frames to remove: {removed_count} ({100*removed_count/len(ordered_frames):.1f}%)")
    print(f"\n  Hand Confidence Breakdown:")
    print(f"    Both hands high confidence: {stats.both_hands_high}")
    print(f"    Left hand only high: {stats.left_only_low} (right hand low)")
    print(f"    Right hand only high: {stats.right_only_low} (left hand low)")
    print(f"    Both hands low confidence: {stats.both_hands_low} [REMOVED]")
    print(f"    Head low confidence: {stats.head_low} [REMOVED]")
    print(f"    Both hands too low in frame: {stats.hands_too_low} [REMOVED]")
    print(f"    No detection / errors: {stats.no_detection} [REMOVED]")

    if stats.left_hand_scores and stats.right_hand_scores and stats.head_scores:
        print(f"\n  Average Confidence Scores:")
        print(f"    Left hand: {np.mean(stats.left_hand_scores):.3f} +/- {np.std(stats.left_hand_scores):.3f}")
        print(f"    Right hand: {np.mean(stats.right_hand_scores):.3f} +/- {np.std(stats.right_hand_scores):.3f}")
        print(f"    Head: {np.mean(stats.head_scores):.3f} +/- {np.std(stats.head_scores):.3f}")

    # Save filtered frames if requested
    if save_filtered and kept_count > 0:
        save_filtered_frames(stats.frames_to_keep, output_dir, folder_name, verbose)

    return {
        'folder': folder_name,
        'total_frames': len(ordered_frames),
        'kept_frames': kept_count,
        'removed_frames': removed_count,
        'both_hands_high': stats.both_hands_high,
        'left_only_high': stats.left_only_low,
        'right_only_high': stats.right_only_low,
        'both_hands_low': stats.both_hands_low,
        'head_low': stats.head_low,
        'hands_too_low': stats.hands_too_low,
        'no_detection': stats.no_detection,
        'left_hand_avg': np.mean(stats.left_hand_scores) if stats.left_hand_scores else 0.0,
        'right_hand_avg': np.mean(stats.right_hand_scores) if stats.right_hand_scores else 0.0,
        'head_avg': np.mean(stats.head_scores) if stats.head_scores else 0.0,
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
    print(f"Head confidence threshold: {args.head_threshold}")
    print(f"Hand height threshold: {args.hand_height_threshold} (bottom {args.hand_height_threshold*100:.0f}%)")
    print(f"Save filtered frames: {args.save_filtered}")
    print(f"Verbose mode (annotate frames): {args.verbose}")

    # Analyze each folder
    results = []
    for folder in sorted(folders):
        result = analyze_folder(
            folder, wholebody, args.hand_threshold, args.head_threshold,
            args.hand_height_threshold, output_dir, args.save_filtered, args.verbose
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
        total_head_low = sum(r['head_low'] for r in results)
        total_hands_too_low = sum(r['hands_too_low'] for r in results)
        total_no_detection = sum(r['no_detection'] for r in results)

        print(f"\nTotal frames analyzed: {total_frames}")
        print(f"Total frames kept: {total_kept} ({100*total_kept/total_frames:.1f}%)")
        print(f"Total frames removed: {total_removed} ({100*total_removed/total_frames:.1f}%)")
        print(f"\nBreakdown:")
        print(f"  Both hands high: {total_both_high}")
        print(f"  Left only high: {total_left_only}")
        print(f"  Right only high: {total_right_only}")
        print(f"  Both hands low: {total_both_low} [REMOVED]")
        print(f"  Head low: {total_head_low} [REMOVED]")
        print(f"  Hands too low in frame: {total_hands_too_low} [REMOVED]")
        print(f"  No detection: {total_no_detection} [REMOVED]")

        if args.save_filtered:
            print(f"\nFiltered frames saved to: {output_dir}")


if __name__ == "__main__":
    main()

# python scripts/sentence/filter_frames_by_pose.py --frames-dir output/sentence_level/frames_512x320_filtered1 --save-filtered --output-dir output/sentence_level/frames_512x320_filtered2 --hand-threshold 0.8 --head-threshold 0.9 --hand-height-threshold 0.1
