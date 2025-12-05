import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect static frames using optical flow analysis with ref_id awareness."
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
        help="Optional: specific subfolder to analyze. If not specified, analyzes all subfolders.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Motion magnitude threshold below which frames are considered static (default: 2.0).",
    )
    parser.add_argument(
        "--boundary-window",
        type=int,
        default=3,
        help="Number of frames to check at ref_id boundaries (default: 3).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/motion_analysis",
        help="Directory to save analysis results and plots.",
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


def get_frames_by_ref(folder: Path) -> Dict[str, List[Tuple[int, Path]]]:
    """
    Group frames by ref_id and sort by frame_id within each group.
    Returns: {ref_id: [(frame_id, path), ...]}
    """
    ref_groups = defaultdict(list)

    for f in folder.glob("*.jpg"):
        frame_id, ref_id = parse_filename(f.name)
        if frame_id >= 0 and ref_id:
            ref_groups[ref_id].append((frame_id, f))

    # Sort each group by frame_id
    for ref_id in ref_groups:
        ref_groups[ref_id].sort(key=lambda x: x[0])

    return ref_groups


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


def calculate_optical_flow_magnitude(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculate the magnitude of optical flow between two frames.
    Returns the mean magnitude across all pixels.
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # Calculate magnitude of flow vectors
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    # Return mean magnitude
    return np.mean(magnitude)


def analyze_folder(folder: Path, threshold: float, boundary_window: int, output_dir: Path) -> dict:
    """
    Analyze a folder of frames for motion with ref_id awareness.
    """
    print(f"\nAnalyzing: {folder.name}")
    print("="*60)

    # Get frames grouped by ref_id
    ref_groups = get_frames_by_ref(folder)
    ordered_frames = get_ordered_frames(folder)

    if len(ordered_frames) < 2:
        print(f"  Skipping (less than 2 frames)")
        return None

    print(f"  Total frames: {len(ordered_frames)}")
    print(f"  Number of ref_id segments: {len(ref_groups)}")

    # Calculate motion for each transition
    frame_ids = []
    ref_ids = []
    magnitudes = []
    is_boundary = []  # Whether this transition crosses ref_id boundary
    is_near_boundary = []  # Whether this transition is near a boundary

    prev_frame_id, prev_ref_id, prev_path = ordered_frames[0]
    prev_img = cv2.imread(str(prev_path))

    for i in range(1, len(ordered_frames)):
        curr_frame_id, curr_ref_id, curr_path = ordered_frames[i]
        curr_img = cv2.imread(str(curr_path))

        magnitude = calculate_optical_flow_magnitude(prev_img, curr_img)

        frame_ids.append(curr_frame_id)
        ref_ids.append(curr_ref_id)
        magnitudes.append(magnitude)

        # Check if this is a boundary transition
        is_boundary.append(prev_ref_id != curr_ref_id)

        # Check if near boundary (within boundary_window frames of ref_id change)
        near_bound = False
        for j in range(max(0, i - boundary_window), min(len(ordered_frames), i + boundary_window + 1)):
            if j > 0 and ordered_frames[j][1] != ordered_frames[j-1][1]:
                near_bound = True
                break
        is_near_boundary.append(near_bound)

        prev_img = curr_img
        prev_ref_id = curr_ref_id

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(ordered_frames)} frames...")

    # Analyze results
    magnitudes_arr = np.array(magnitudes)
    is_boundary_arr = np.array(is_boundary)
    is_near_boundary_arr = np.array(is_near_boundary)

    static_frames = magnitudes_arr < threshold

    boundary_static = np.sum(is_boundary_arr & static_frames)
    near_boundary_static = np.sum(is_near_boundary_arr & static_frames)
    interior_static = np.sum(~is_near_boundary_arr & static_frames)

    print(f"\n  Static frames (< {threshold}): {np.sum(static_frames)}/{len(magnitudes)} ({100*np.sum(static_frames)/len(magnitudes):.1f}%)")
    print(f"    - At ref_id boundaries: {boundary_static}")
    print(f"    - Near boundaries (±{boundary_window} frames): {near_boundary_static}")
    print(f"    - In segment interiors: {interior_static}")

    # Analyze each ref_id segment
    print(f"\n  Per-segment analysis:")
    segment_stats = []
    for ref_id, frames in sorted(ref_groups.items(), key=lambda x: x[1][0][0]):
        if len(frames) < 2:
            continue

        # Find indices in magnitudes array for this ref_id
        segment_magnitudes = []
        for idx, rid in enumerate(ref_ids):
            if rid == ref_id:
                segment_magnitudes.append(magnitudes[idx])

        if segment_magnitudes:
            seg_mean = np.mean(segment_magnitudes)
            seg_static = np.sum(np.array(segment_magnitudes) < threshold)
            seg_total = len(segment_magnitudes)

            segment_stats.append({
                'ref_id': ref_id,
                'frames': len(frames),
                'mean_motion': seg_mean,
                'static_ratio': seg_static / seg_total if seg_total > 0 else 0
            })

            print(f"    {ref_id}: {len(frames)} frames, mean motion={seg_mean:.2f}, static={seg_static}/{seg_total} ({100*seg_static/seg_total:.1f}%)")

    # Create visualization
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Motion magnitude over time with ref_id coloring
    ax1 = plt.subplot(3, 2, 1)

    # Color by ref_id
    unique_refs = sorted(set(ref_ids), key=lambda x: ref_ids.index(x))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_refs)))
    ref_to_color = {ref: colors[i] for i, ref in enumerate(unique_refs)}

    for i in range(len(frame_ids)):
        ax1.plot([frame_ids[i], frame_ids[i]], [0, magnitudes[i]],
                color=ref_to_color[ref_ids[i]], alpha=0.6, linewidth=0.8)

    ax1.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax1.set_xlabel('Frame ID')
    ax1.set_ylabel('Motion Magnitude')
    ax1.set_title('Motion Analysis (colored by ref_id)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Highlight boundary frames
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(frame_ids, magnitudes, 'b-', linewidth=0.5, alpha=0.5, label='Motion')

    # Mark boundaries
    boundary_frames = [frame_ids[i] for i in range(len(is_boundary)) if is_boundary[i]]
    boundary_mags = [magnitudes[i] for i in range(len(is_boundary)) if is_boundary[i]]
    ax2.scatter(boundary_frames, boundary_mags, c='red', s=50, alpha=0.8,
               label=f'ref_id boundaries', marker='x', linewidths=2)

    ax2.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax2.set_xlabel('Frame ID')
    ax2.set_ylabel('Motion Magnitude')
    ax2.set_title('Motion at ref_id Boundaries')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Distribution of motion magnitudes
    ax3 = plt.subplot(3, 2, 3)
    ax3.hist(magnitudes, bins=50, alpha=0.7, edgecolor='black')
    ax3.axvline(x=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax3.set_xlabel('Motion Magnitude')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Motion Magnitudes')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Compare boundary vs interior
    ax4 = plt.subplot(3, 2, 4)
    boundary_mags = [magnitudes[i] for i in range(len(is_near_boundary)) if is_near_boundary[i]]
    interior_mags = [magnitudes[i] for i in range(len(is_near_boundary)) if not is_near_boundary[i]]

    ax4.hist([interior_mags, boundary_mags], bins=30, alpha=0.7,
            label=['Interior', f'Near boundary (±{boundary_window})'],
            color=['blue', 'orange'], edgecolor='black')
    ax4.axvline(x=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax4.set_xlabel('Motion Magnitude')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Boundary vs Interior Motion')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Per-segment statistics
    ax5 = plt.subplot(3, 2, 5)
    if segment_stats:
        seg_names = [s['ref_id'][:8] for s in segment_stats]  # Truncate long names
        seg_means = [s['mean_motion'] for s in segment_stats]
        seg_colors = ['red' if m < threshold else 'green' for m in seg_means]

        ax5.bar(range(len(segment_stats)), seg_means, color=seg_colors, alpha=0.7, edgecolor='black')
        ax5.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        ax5.set_xlabel('ref_id Segment')
        ax5.set_ylabel('Mean Motion')
        ax5.set_title('Mean Motion per ref_id Segment')
        ax5.set_xticks(range(len(segment_stats)))
        ax5.set_xticklabels(seg_names, rotation=45, ha='right')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')

    # Plot 6: Static ratio comparison
    ax6 = plt.subplot(3, 2, 6)
    categories = ['At Boundary', f'Near Boundary\n(±{boundary_window})', 'Interior', 'Overall']
    static_ratios = [
        boundary_static / np.sum(is_boundary_arr) if np.sum(is_boundary_arr) > 0 else 0,
        near_boundary_static / np.sum(is_near_boundary_arr) if np.sum(is_near_boundary_arr) > 0 else 0,
        interior_static / np.sum(~is_near_boundary_arr) if np.sum(~is_near_boundary_arr) > 0 else 0,
        np.sum(static_frames) / len(magnitudes)
    ]

    colors_bar = ['red', 'orange', 'blue', 'purple']
    ax6.bar(categories, [r * 100 for r in static_ratios], color=colors_bar, alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Static Frame %')
    ax6.set_title('Static Frame Ratio by Location')
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{folder.name}_motion_analysis.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Plot saved to: {plot_path}")

    return {
        'folder': folder.name,
        'total_frames': len(ordered_frames),
        'num_segments': len(ref_groups),
        'frame_ids': frame_ids,
        'magnitudes': magnitudes,
        'static_frames': np.sum(static_frames),
        'static_ratio': np.sum(static_frames) / len(magnitudes),
        'boundary_static': boundary_static,
        'near_boundary_static': near_boundary_static,
        'interior_static': interior_static,
        'segment_stats': segment_stats
    }


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
    print(f"Motion threshold: {args.threshold}")
    print(f"Boundary window: ±{args.boundary_window} frames")

    # Analyze each folder
    results = []
    for folder in sorted(folders):
        result = analyze_folder(folder, args.threshold, args.boundary_window, output_dir)
        if result:
            results.append(result)

    # Summary statistics
    if results:
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)

        total_frames = sum(r['total_frames'] for r in results) - len(results)
        total_static = sum(r['static_frames'] for r in results)
        total_boundary_static = sum(r['boundary_static'] for r in results)
        total_near_boundary_static = sum(r['near_boundary_static'] for r in results)
        total_interior_static = sum(r['interior_static'] for r in results)

        print(f"\nTotal transitions analyzed: {total_frames}")
        print(f"Total static frames: {total_static} ({100*total_static/total_frames:.1f}%)")
        print(f"  - At ref_id boundaries: {total_boundary_static}")
        print(f"  - Near boundaries: {total_near_boundary_static}")
        print(f"  - In interiors: {total_interior_static}")


if __name__ == "__main__":
    main()
