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
            "For each sentence folder containing word frames, "
            "extract boundary frames for ref->word and word->ref transitions. "
            "Generates 2 pairs per word: ref->first_frame and last_frame->ref."
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
        "--ref-image",
        type=str,
        required=True,
        help="Path to the reference image to use for all words.",
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
        help="Optional: specific subfolder (sentence_id) to process.",
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


def process_sentence_folder(sentence_folder: Path, ref_image_path: Path, out_root: Path) -> int:
    """
    Process a single sentence folder and extract boundary frames for each word.

    For each word (ref_id), generates 2 boundary pairs:
      - Pair 0: ref->word ({ref_id}_0_start.jpg=ref, {ref_id}_0_end.jpg=first_frame)
      - Pair 1: word->ref ({ref_id}_1_start.jpg=last_frame, {ref_id}_1_end.jpg=ref)

    Returns: number of words processed
    """
    sentence_id = sentence_folder.name
    print(f"Processing sentence: {sentence_id}")

    # Create output directory for this sentence
    sentence_out_dir = out_root / sentence_id
    ensure_dir(str(sentence_out_dir))

    # Group frames by ref_id (each ref_id is a word)
    ref_groups = get_frames_by_ref(sentence_folder)

    if not ref_groups:
        print(f"  [WARN] No valid frames found, skipping.")
        return 0

    print(f"  Found {len(ref_groups)} word(s)")

    word_count = 0
    for ref_id, frames in sorted(ref_groups.items()):
        if not frames:
            continue

        # Get first and last frame for this word
        first_frame = frames[0][1]
        last_frame = frames[-1][1]

        print(f"  Word [{ref_id}]: {len(frames)} frames")
        print(f"    First: {first_frame.name}, Last: {last_frame.name}")

        # Pair 0: ref -> word (入场)
        # {ref_id}_0_start.jpg = ref image
        # {ref_id}_0_end.jpg = first frame
        shutil.copy2(ref_image_path, sentence_out_dir / f"{ref_id}_0_start.jpg")
        shutil.copy2(first_frame, sentence_out_dir / f"{ref_id}_0_end.jpg")

        # Pair 1: word -> ref (退场)
        # {ref_id}_1_start.jpg = last frame
        # {ref_id}_1_end.jpg = ref image
        shutil.copy2(last_frame, sentence_out_dir / f"{ref_id}_1_start.jpg")
        shutil.copy2(ref_image_path, sentence_out_dir / f"{ref_id}_1_end.jpg")

        word_count += 1

    print(f"  Generated {word_count * 2} boundary pairs for {word_count} words")
    return word_count


def main() -> None:
    args = parse_args()

    frames_root = Path(args.frames_root)
    ref_image_path = Path(args.ref_image)
    out_root = Path(args.out_root)

    if not frames_root.exists():
        raise FileNotFoundError(f"frames-root not found: {frames_root}")

    if not ref_image_path.exists():
        raise FileNotFoundError(f"ref-image not found: {ref_image_path}")

    print(f"Using reference image: {ref_image_path}\n")

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

    print(f"Found {len(folders)} sentence folder(s) to process.\n")

    # Process each sentence folder
    total_sentences = 0
    total_words = 0
    for folder in sorted(folders):
        word_count = process_sentence_folder(folder, ref_image_path, out_root)
        if word_count > 0:
            total_sentences += 1
            total_words += word_count
        print()

    print(f"Done. Processed {total_sentences} sentence(s), {total_words} word(s) into '{out_root}'.")
    print(f"Total boundary pairs generated: {total_words * 2}")


if __name__ == "__main__":
    main()

# Example usage:
# python scripts/word/extract_boundary_frames.py --frames-root output/sentence_level/frames_512x320_filtered2 --ref-image assets/example_data/images/test3_512x320.jpg --out-root output/word_level/word_boundary_frames
