import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect duplicate/redundant frames in sign language videos using frame difference."
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
        "--duplicate-threshold",
        type=float,
        default=0.5,
        help="Pixel difference threshold (percentage) below which frames are considered duplicates (default: 0.5%%).",
    )
    parser.add_argument(
        "--min-duplicate-length",
        type=int,
        default=2,
        help="Minimum number of consecutive duplicate frames to report (default: 3).",
    )
    parser.add_argument(
        "--boundary-frames",
        type=int,
        default=5,
        help="Number of frames at start/end of each ref_id segment to analyze closely (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/duplicate_analysis",
        help="Directory to save analysis results and plots.",
    )
    parser.add_argument(
        "--generate-video",
        action="store_true",
        help="Generate cleaned MP4 videos with duplicate frames removed.",
    )
    parser.add_argument(
        "--save-cleaned-frames",
        action="store_true",
        help="Save cleaned frames (without duplicates) as JPG files.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Output video FPS (default: 25).",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class FrameData:
    frame_id: int
    ref_id: str
    path: Path


@dataclass
class SegmentAnalysis:
    ref_id: str
    num_frames: int
    frame_ids: List[int]
    differences: List[float]
    duplicate_sequences: List[Tuple[int, int]]
    total_duplicates: int
    mean_diff: float
    min_diff: float
    max_diff: float
    start_duplicates: int
    end_duplicates: int
    start_region_size: int
    end_region_size: int

    def to_dict(self) -> Dict[str, object]:
        return {
            'ref_id': self.ref_id,
            'num_frames': self.num_frames,
            'frame_ids': self.frame_ids,
            'differences': self.differences,
            'duplicate_sequences': self.duplicate_sequences,
            'total_duplicates': self.total_duplicates,
            'mean_diff': self.mean_diff,
            'min_diff': self.min_diff,
            'max_diff': self.max_diff,
            'start_duplicates': self.start_duplicates,
            'end_duplicates': self.end_duplicates,
            'start_region_size': self.start_region_size,
            'end_region_size': self.end_region_size,
        }


def parse_filename(filename: str) -> Tuple[int, str]:
    """
    Extract frame_id and ref_id from filename like '0_vkhwescgz9.jpg'
    Returns: (frame_id, ref_id)
    """
    match = re.match(r"(\d+)_([^.]+)\.jpg", filename)
    if match:
        return int(match.group(1)), match.group(2)
    return -1, ""


def get_frames_by_ref(folder: Path) -> Dict[str, List[FrameData]]:
    """
    Group frames by ref_id and sort by frame_id within each group.
    Returns: {ref_id: [FrameData, ...]}
    """
    ref_groups: Dict[str, List[FrameData]] = defaultdict(list)

    for f in folder.glob("*.jpg"):
        frame_id, ref_id = parse_filename(f.name)
        if frame_id >= 0 and ref_id:
            ref_groups[ref_id].append(FrameData(frame_id, ref_id, f))

    for ref_id in ref_groups:
        ref_groups[ref_id].sort(key=lambda frame: frame.frame_id)

    return ref_groups


def get_ordered_frames(folder: Path) -> List[FrameData]:
    """
    Get all frames sorted by frame_id.
    Returns: [FrameData, ...]
    """
    frames: List[FrameData] = []
    for f in folder.glob("*.jpg"):
        frame_id, ref_id = parse_filename(f.name)
        if frame_id >= 0 and ref_id:
            frames.append(FrameData(frame_id, ref_id, f))

    frames.sort(key=lambda frame: frame.frame_id)
    return frames


def calculate_frame_difference_percent(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculate the percentage of pixels that differ between two frames.
    Returns the percentage of changed pixels.
    """
    # Calculate absolute difference
    diff = cv2.absdiff(frame1, frame2)

    # Convert to grayscale if needed
    if len(diff.shape) == 3:
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    else:
        diff_gray = diff

    # Threshold to detect changed pixels (anything > 5 in pixel value)
    _, thresh = cv2.threshold(diff_gray, 5, 255, cv2.THRESH_BINARY)

    # Calculate percentage of changed pixels
    total_pixels = thresh.shape[0] * thresh.shape[1]
    changed_pixels = np.count_nonzero(thresh)

    return (changed_pixels / total_pixels) * 100


def compute_frame_differences(frames: List[FrameData]) -> List[float]:
    """
    Compute pixel difference percentage between consecutive frames.
    Returns a list with length len(frames) - 1.
    """
    differences: List[float] = []

    prev_img = cv2.imread(str(frames[0].path))
    for frame in frames[1:]:
        curr_img = cv2.imread(str(frame.path))
        differences.append(calculate_frame_difference_percent(prev_img, curr_img))
        prev_img = curr_img

    return differences


def select_frames_to_keep(ordered_frames: List[FrameData], differences: List[float], threshold: float) -> List[FrameData]:
    """
    Determine which frames to keep based on the difference threshold.
    Always keeps the first frame.
    """
    frames_to_keep: List[FrameData] = [ordered_frames[0]]

    for diff, frame in zip(differences, ordered_frames[1:]):
        if diff >= threshold:
            frames_to_keep.append(frame)

    return frames_to_keep


def build_ordered_frame_data(ordered_frames: List[FrameData], progress_interval: int = 100) -> List[dict]:
    """
    Build per-frame difference data for the full ordered sequence.
    """
    ordered_frame_data: List[dict] = []
    prev_frame = ordered_frames[0]
    prev_img = cv2.imread(str(prev_frame.path))
    prev_ref_id = prev_frame.ref_id

    for idx, curr_frame in enumerate(ordered_frames[1:], start=1):
        curr_img = cv2.imread(str(curr_frame.path))
        diff = calculate_frame_difference_percent(prev_img, curr_img)
        ordered_frame_data.append({
            'frame_id': curr_frame.frame_id,
            'ref_id': curr_frame.ref_id,
            'difference': diff,
            'is_boundary': prev_ref_id != curr_frame.ref_id
        })

        prev_img = curr_img
        prev_ref_id = curr_frame.ref_id

        if progress_interval and (idx + 1) % progress_interval == 0:
            print(f"  Processed {idx + 1}/{len(ordered_frames)} frames...")

    return ordered_frame_data


def generate_cleaned_video(ordered_frames: List[FrameData], differences: List[float],
                          threshold: float, folder_name: str, output_dir: Path,
                          fps: int = 25) -> None:
    """
    Generate a video with duplicate frames removed.

    Args:
        ordered_frames: Ordered frames with ids/ref ids/paths
        differences: List of frame difference percentages
        threshold: Threshold below which frames are considered duplicates
        folder_name: Name of the folder being processed
        output_dir: Output directory for the video
        fps: Frames per second for output video
    """
    frames_to_keep = select_frames_to_keep(ordered_frames, differences, threshold)

    if len(frames_to_keep) < 2:
        print(f"  Warning: Only {len(frames_to_keep)} frames would remain after filtering. Skipping video generation.")
        return

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frames_to_keep[0].path))
    if first_frame is None:
        print(f"  Error: Could not read first frame. Skipping video generation.")
        return

    height, width = first_frame.shape[:2]

    # Create output video
    output_path = output_dir / f"{folder_name}_cleaned.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"  Error: Could not create video writer. Skipping video generation.")
        return

    # Write frames
    for frame_data in frames_to_keep:
        frame = cv2.imread(str(frame_data.path))
        if frame is not None:
            out.write(frame)

    out.release()

    original_count = len(ordered_frames)
    cleaned_count = len(frames_to_keep)
    removed_count = original_count - cleaned_count

    print(f"  Video saved to: {output_path}")
    print(f"    Original frames: {original_count}")
    print(f"    Cleaned frames: {cleaned_count}")
    print(f"    Removed frames: {removed_count} ({100*removed_count/original_count:.1f}%)")


def save_frames_to_disk(ordered_frames: List[FrameData], differences: List[float],
                       threshold: float, folder_name: str, output_dir: Path) -> None:
    """
    Save cleaned frames (without duplicates) to a new directory.

    Args:
        ordered_frames: Ordered frames with ids/ref ids/paths
        differences: List of frame difference percentages
        threshold: Threshold below which frames are considered duplicates
        folder_name: Name of the folder being processed
        output_dir: Output directory for the cleaned frames
    """
    import shutil

    frames_to_keep = select_frames_to_keep(ordered_frames, differences, threshold)

    if len(frames_to_keep) < 2:
        print(f"  Warning: Only {len(frames_to_keep)} frames would remain after filtering. Skipping frame saving.")
        return

    # Create output directory for this folder
    frames_output_dir = output_dir / folder_name
    frames_output_dir.mkdir(parents=True, exist_ok=True)

    # Copy frames preserving original frame_id to maintain ref_id continuity
    for frame_data in frames_to_keep:
        new_filename = f"{frame_data.frame_id}_{frame_data.ref_id}.jpg"
        dst_path = frames_output_dir / new_filename

        shutil.copy2(frame_data.path, dst_path)

    original_count = len(ordered_frames)
    cleaned_count = len(frames_to_keep)
    removed_count = original_count - cleaned_count

    print(f"  Cleaned frames saved to: {frames_output_dir}")
    print(f"    Original frames: {original_count}")
    print(f"    Cleaned frames: {cleaned_count}")
    print(f"    Removed frames: {removed_count} ({100*removed_count/original_count:.1f}%)")


def find_duplicate_sequences(differences: List[float], threshold: float, min_length: int) -> List[Tuple[int, int]]:
    """
    Find sequences of consecutive frames with very low difference (duplicates).
    Returns: [(start_idx, end_idx), ...] where indices are in the differences array
    """
    sequences = []
    start = None

    for i, diff in enumerate(differences):
        if diff < threshold:
            if start is None:
                start = i
        else:
            if start is not None:
                length = i - start
                if length >= min_length:
                    sequences.append((start, i - 1))
                start = None

    # Check if sequence extends to the end
    if start is not None:
        length = len(differences) - start
        if length >= min_length:
            sequences.append((start, len(differences) - 1))

    return sequences


def analyze_ref_segment(ref_id: str, frames: List[FrameData],
                       threshold: float, min_dup_length: int, boundary_frames: int) -> Optional[SegmentAnalysis]:
    """
    Analyze a single ref_id segment for duplicate frames.
    """
    if len(frames) < 2:
        return None

    frame_ids = [frame.frame_id for frame in frames]
    differences = compute_frame_differences(frames)

    dup_sequences = find_duplicate_sequences(differences, threshold, min_dup_length)

    start_region_diffs = differences[:min(boundary_frames, len(differences))]
    end_region_diffs = differences[-min(boundary_frames, len(differences)):]

    start_duplicates = sum(1 for d in start_region_diffs if d < threshold)
    end_duplicates = sum(1 for d in end_region_diffs if d < threshold)

    return SegmentAnalysis(
        ref_id=ref_id,
        num_frames=len(frames),
        frame_ids=frame_ids[1:],  # Align with differences array
        differences=differences,
        duplicate_sequences=dup_sequences,
        total_duplicates=sum(1 for d in differences if d < threshold),
        mean_diff=float(np.mean(differences)),
        min_diff=float(np.min(differences)),
        max_diff=float(np.max(differences)),
        start_duplicates=start_duplicates,
        end_duplicates=end_duplicates,
        start_region_size=len(start_region_diffs),
        end_region_size=len(end_region_diffs),
    )


def analyze_folder(folder: Path, threshold: float, min_dup_length: int,
                   boundary_frames: int, output_dir: Path, generate_video: bool = False,
                   save_cleaned_frames: bool = False, fps: int = 25) -> dict:
    """
    Analyze a folder of frames for duplicate detection with ref_id awareness.
    """
    print(f"\nAnalyzing: {folder.name}")
    print("="*60)

    ref_groups = get_frames_by_ref(folder)
    ordered_frames = get_ordered_frames(folder)

    if len(ordered_frames) < 2:
        print(f"  Skipping (less than 2 frames)")
        return None

    print(f"  Total frames: {len(ordered_frames)}")
    print(f"  Number of ref_id segments: {len(ref_groups)}")

    # Analyze each ref_id segment
    segment_analyses: List[SegmentAnalysis] = []
    for ref_id, frames in sorted(ref_groups.items(), key=lambda x: x[1][0].frame_id):
        result = analyze_ref_segment(ref_id, frames, threshold, min_dup_length, boundary_frames)
        if result:
            segment_analyses.append(result)

    ordered_frame_data = build_ordered_frame_data(ordered_frames)

    # Summary statistics
    print(f"\n  Per-segment analysis (duplicate threshold: {threshold}%):")
    print(f"  {'ref_id':<15} {'frames':<8} {'duplicates':<12} {'start_dup':<11} {'end_dup':<9} {'mean_diff':<10}")
    print(f"  {'-'*70}")

    total_frames = 0
    total_duplicates = 0
    total_dup_sequences = 0
    boundary_duplicates = 0

    for result in segment_analyses:
        print(f"  {result.ref_id:<15} {result.num_frames:<8} "
              f"{result.total_duplicates:<12} "
              f"{result.start_duplicates}/{result.start_region_size:<8} "
              f"{result.end_duplicates}/{result.end_region_size:<6} "
              f"{result.mean_diff:<10.2f}%")

        if result.duplicate_sequences:
            for seq_start, seq_end in result.duplicate_sequences:
                seq_len = seq_end - seq_start + 1
                frame_range = f"{result.frame_ids[seq_start]}-{result.frame_ids[seq_end]}"
                print(f"    -> Duplicate sequence: frames {frame_range} (length: {seq_len})")

        total_frames += len(result.differences)
        total_duplicates += result.total_duplicates
        total_dup_sequences += len(result.duplicate_sequences)

    # Count boundary duplicates
    for data in ordered_frame_data:
        if data['is_boundary'] and data['difference'] < threshold:
            boundary_duplicates += 1

    print(f"\n  Overall:")
    print(f"    Total transitions: {total_frames}")
    print(f"    Duplicate transitions: {total_duplicates} ({100*total_duplicates/total_frames:.1f}%)")
    print(f"    Duplicate sequences found: {total_dup_sequences}")
    print(f"    Duplicates at ref_id boundaries: {boundary_duplicates}")

    # Create visualization
    create_visualization(folder.name, segment_analyses, ordered_frame_data,
                        threshold, boundary_frames, output_dir)

    # Generate cleaned video if requested
    if generate_video:
        differences = [d['difference'] for d in ordered_frame_data]
        generate_cleaned_video(ordered_frames, differences, threshold,
                             folder.name, output_dir, fps)

    # Save cleaned frames if requested
    if save_cleaned_frames:
        differences = [d['difference'] for d in ordered_frame_data]
        save_frames_to_disk(ordered_frames, differences, threshold,
                           folder.name, output_dir)

    return {
        'folder': folder.name,
        'total_frames': len(ordered_frames),
        'num_segments': len(ref_groups),
        'segment_results': [analysis.to_dict() for analysis in segment_analyses],
        'ordered_data': ordered_frame_data,
        'total_duplicates': total_duplicates,
        'total_dup_sequences': total_dup_sequences,
        'boundary_duplicates': boundary_duplicates
    }


def create_visualization(folder_name: str, segment_results: List[SegmentAnalysis],
                        ordered_data: List[dict], threshold: float,
                        boundary_frames: int, output_dir: Path):
    """
    Create comprehensive visualization of duplicate frame analysis.
    """
    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Overall frame differences over time (colored by ref_id)
    ax1 = plt.subplot(3, 3, 1)

    frame_ids = [d['frame_id'] for d in ordered_data]
    differences = [d['difference'] for d in ordered_data]
    ref_ids = [d['ref_id'] for d in ordered_data]

    unique_refs = sorted(set(ref_ids), key=lambda x: ref_ids.index(x))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_refs)))
    ref_to_color = {ref: colors[i] for i, ref in enumerate(unique_refs)}

    for i in range(len(frame_ids)):
        ax1.plot([frame_ids[i], frame_ids[i]], [0, differences[i]],
                color=ref_to_color[ref_ids[i]], alpha=0.6, linewidth=0.8)

    ax1.axhline(y=threshold, color='r', linestyle='--', linewidth=2,
               label=f'Duplicate threshold ({threshold}%)')
    ax1.set_xlabel('Frame ID')
    ax1.set_ylabel('Pixel Difference %')
    ax1.set_title('Frame Differences (colored by ref_id)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Mark ref_id boundaries
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(frame_ids, differences, 'b-', linewidth=0.5, alpha=0.7)

    boundary_frames_list = [frame_ids[i] for i in range(len(ordered_data))
                           if ordered_data[i]['is_boundary']]
    boundary_diffs = [differences[i] for i in range(len(ordered_data))
                     if ordered_data[i]['is_boundary']]

    ax2.scatter(boundary_frames_list, boundary_diffs, c='red', s=80,
               alpha=0.8, label='ref_id boundaries', marker='x', linewidths=2)
    ax2.axhline(y=threshold, color='r', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold}%)')
    ax2.set_xlabel('Frame ID')
    ax2.set_ylabel('Pixel Difference %')
    ax2.set_title('Differences at ref_id Boundaries')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Plot 3: Distribution of differences
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(differences, bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(x=threshold, color='r', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold}%)')
    ax3.set_xlabel('Pixel Difference %')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Frame Differences')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Per-segment mean differences
    ax4 = plt.subplot(3, 3, 4)
    if segment_results:
        seg_names = [s.ref_id[:10] for s in segment_results]
        seg_means = [s.mean_diff for s in segment_results]
        seg_colors = ['red' if m < threshold else 'green' for m in seg_means]

        ax4.barh(range(len(segment_results)), seg_means, color=seg_colors,
                alpha=0.7, edgecolor='black')
        ax4.axvline(x=threshold, color='r', linestyle='--', linewidth=2,
                   label=f'Threshold ({threshold}%)')
        ax4.set_yticks(range(len(segment_results)))
        ax4.set_yticklabels(seg_names)
        ax4.set_xlabel('Mean Difference %')
        ax4.set_title('Mean Difference per ref_id Segment')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='x')

    # Plot 5: Duplicate ratio per segment
    ax5 = plt.subplot(3, 3, 5)
    if segment_results:
        seg_names = [s.ref_id[:10] for s in segment_results]
        dup_ratios = [100 * s.total_duplicates / len(s.differences)
                     if len(s.differences) > 0 else 0
                     for s in segment_results]

        ax5.barh(range(len(segment_results)), dup_ratios,
                color='orange', alpha=0.7, edgecolor='black')
        ax5.set_yticks(range(len(segment_results)))
        ax5.set_yticklabels(seg_names)
        ax5.set_xlabel('Duplicate Transitions %')
        ax5.set_title('Duplicate Ratio per Segment')
        ax5.grid(True, alpha=0.3, axis='x')

    # Plot 6: Start vs End boundary duplicates
    ax6 = plt.subplot(3, 3, 6)
    if segment_results:
        seg_names = [s.ref_id[:10] for s in segment_results]
        start_ratios = [100 * s.start_duplicates / s.start_region_size
                       if s.start_region_size > 0 else 0
                       for s in segment_results]
        end_ratios = [100 * s.end_duplicates / s.end_region_size
                     if s.end_region_size > 0 else 0
                     for s in segment_results]

        x = np.arange(len(segment_results))
        width = 0.35

        ax6.barh(x - width/2, start_ratios, width, label='Start',
                color='blue', alpha=0.7, edgecolor='black')
        ax6.barh(x + width/2, end_ratios, width, label='End',
                color='red', alpha=0.7, edgecolor='black')

        ax6.set_yticks(x)
        ax6.set_yticklabels(seg_names)
        ax6.set_xlabel(f'Duplicate % (first/last {boundary_frames} frames)')
        ax6.set_title('Duplicates at Segment Start vs End')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='x')

    # Plot 7: Cumulative duplicate sequences
    ax7 = plt.subplot(3, 3, 7)
    if segment_results:
        # Show which segments have duplicate sequences
        seg_names = [s.ref_id[:10] for s in segment_results]
        num_sequences = [len(s.duplicate_sequences) for s in segment_results]

        colors_seq = ['red' if n > 0 else 'gray' for n in num_sequences]
        ax7.barh(range(len(segment_results)), num_sequences,
                color=colors_seq, alpha=0.7, edgecolor='black')
        ax7.set_yticks(range(len(segment_results)))
        ax7.set_yticklabels(seg_names)
        ax7.set_xlabel('Number of Duplicate Sequences')
        ax7.set_title(f'Duplicate Sequences (>={3} consecutive frames)')
        ax7.grid(True, alpha=0.3, axis='x')

    # Plot 8: Overall statistics
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')

    total_transitions = sum(len(s.differences) for s in segment_results)
    total_dups = sum(s.total_duplicates for s in segment_results)
    total_sequences = sum(len(s.duplicate_sequences) for s in segment_results)

    stats_text = f"""
    Overall Statistics:

    Total transitions: {total_transitions}
    Duplicate transitions: {total_dups} ({100*total_dups/total_transitions:.1f}%)
    Duplicate sequences: {total_sequences}

    Threshold: {threshold}% pixel difference
    Min sequence length: {3} frames
    """

    ax8.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 9: Highlight problem segments
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    # Find segments with high duplicate ratios
    problem_segments = [(s.ref_id,
                        100 * s.total_duplicates / len(s.differences) if len(s.differences) > 0 else 0,
                        len(s.duplicate_sequences))
                       for s in segment_results]
    problem_segments.sort(key=lambda x: x[1], reverse=True)

    problem_text = "Problem Segments (highest duplicate %):\n\n"
    for i, (ref_id, dup_pct, num_seq) in enumerate(problem_segments[:10]):
        problem_text += f"{i+1}. {ref_id[:12]}: {dup_pct:.1f}% (seqs: {num_seq})\n"

    ax9.text(0.1, 0.9, problem_text, fontsize=10, verticalalignment='top',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    plt.tight_layout()

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{folder_name}_duplicate_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Plot saved to: {plot_path}")


def main():
    args = parse_args()

    frames_root = Path(args.frames_dir)
    output_dir = Path(args.output_dir)

    if not frames_root.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_root}")

    # Determine which folders to analyze
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
    print(f"Duplicate threshold: {args.duplicate_threshold}% pixel difference")
    print(f"Minimum duplicate sequence length: {args.min_duplicate_length} frames")
    print(f"Boundary analysis: first/last {args.boundary_frames} frames of each segment")

    # Analyze each folder
    results = []
    for folder in sorted(folders):
        result = analyze_folder(folder, args.duplicate_threshold,
                               args.min_duplicate_length, args.boundary_frames,
                               output_dir, args.generate_video,
                               args.save_cleaned_frames, args.fps)
        if result:
            results.append(result)

    # Summary statistics
    if results:
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)

        total_frames = sum(r['total_frames'] for r in results) - len(results)
        total_duplicates = sum(r['total_duplicates'] for r in results)
        total_sequences = sum(r['total_dup_sequences'] for r in results)
        total_boundary_dups = sum(r['boundary_duplicates'] for r in results)

        print(f"\nTotal transitions analyzed: {total_frames}")
        print(f"Total duplicate transitions: {total_duplicates} ({100*total_duplicates/total_frames:.1f}%)")
        print(f"Total duplicate sequences: {total_sequences}")
        print(f"Duplicates at ref_id boundaries: {total_boundary_dups}")


if __name__ == "__main__":
    main()

# python scripts/sentence/filter_duplicate_frames.py --frames-dir output/sentence_level/frames --duplicate-threshold 3.0 --min-duplicate-length 2 --boundary-frames 15 --save-cleaned-frames --output-dir output/sentence_level/frames_512x320_filtered1
