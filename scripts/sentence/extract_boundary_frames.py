import argparse
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each sentence folder of frame sequences, "
            "extract boundary frames between ref_ids. "
            "Input: {frame_id}_{ref_id}.jpg format."
        )
    )
    parser.add_argument(
        "--frames-root",
        type=str,
        required=True,
        help=(
            "Root directory of frame sequences. "
            "Expected layout: <frames-root>/<sentence_id>/{frame_id}_{ref_id}.jpg"
        ),
    )
    parser.add_argument(
        "--out-root",
        type=str,
        required=True,
        help=(
            "Root directory to save extracted boundary frames. "
            "Frames will be saved under <out-root>/<sentence_id>/."
        ),
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default=None,
        help="Optional: specific subfolder to process.",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


def get_ordered_ref_ids(ref_groups: Dict[str, List[Tuple[int, Path]]]) -> List[str]:
    """
    Get ref_ids ordered by the first frame_id appearance.
    Returns: [ref_id1, ref_id2, ...]
    """
    ref_with_first_frame = []
    for ref_id, frames in ref_groups.items():
        if frames:
            first_frame_id = frames[0][0]
            ref_with_first_frame.append((first_frame_id, ref_id))

    ref_with_first_frame.sort(key=lambda x: x[0])
    return [ref_id for _, ref_id in ref_with_first_frame]


def process_sentence_folder(sentence_folder: Path, out_root: Path) -> None:
    """
    Process a single sentence folder and extract boundary frames.

    For ref_ids = [a, b, c], extracts:
      - Boundary 0: a→b (0_start.jpg from a's last frame, 0_end.jpg from b's first frame)
      - Boundary 1: b→c (1_start.jpg from b's last frame, 1_end.jpg from c's first frame)
    """
    sentence_id = sentence_folder.name
    print(f"Processing: {sentence_id}")

    # Get frames grouped by ref_id
    ref_groups = get_frames_by_ref(sentence_folder)

    if len(ref_groups) < 2:
        print(f"  [WARN] Only {len(ref_groups)} ref_id(s) found, skipping (need at least 2).")
        return

    # Get ordered ref_ids
    ordered_ref_ids = get_ordered_ref_ids(ref_groups)
    print(f"  Found {len(ordered_ref_ids)} ref_ids: {ordered_ref_ids}")

    # Create output directory
    out_dir = out_root / sentence_id
    ensure_dir(str(out_dir))

    # Extract boundaries
    num_boundaries = len(ordered_ref_ids) - 1
    for boundary_idx in range(num_boundaries):
        prev_ref_id = ordered_ref_ids[boundary_idx]
        next_ref_id = ordered_ref_ids[boundary_idx + 1]

        prev_frames = ref_groups[prev_ref_id]
        next_frames = ref_groups[next_ref_id]

        # Get last frame of previous ref_id
        if prev_frames:
            last_frame_path = prev_frames[-1][1]
            out_path = out_dir / f"{boundary_idx}_start.jpg"
            shutil.copy2(last_frame_path, out_path)
            print(f"    Boundary {boundary_idx}: {prev_ref_id}→{next_ref_id}")
            print(f"      {boundary_idx}_start.jpg ← {last_frame_path.name}")

        # Get first frame of next ref_id
        if next_frames:
            first_frame_path = next_frames[0][1]
            out_path = out_dir / f"{boundary_idx}_end.jpg"
            shutil.copy2(first_frame_path, out_path)
            print(f"      {boundary_idx}_end.jpg ← {first_frame_path.name}")

    print(f"  Extracted {num_boundaries} boundaries.")


def main() -> None:
    args = parse_args()

    frames_root = Path(args.frames_root)
    out_root = Path(args.out_root)

    if not frames_root.exists():
        raise FileNotFoundError(f"frames-root not found: {frames_root}")

    ensure_dir(str(out_root))

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

    print(f"Found {len(folders)} folder(s) to process.\n")

    # Process each folder
    processed_count = 0
    for folder in sorted(folders):
        process_sentence_folder(folder, out_root)
        processed_count += 1
        print()

    print(f"Done. Processed {processed_count} sentence(s) into '{out_root}'.")


if __name__ == "__main__":
    main()
