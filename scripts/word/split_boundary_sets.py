import argparse
import math
import re
import shutil
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a boundary frame folder into N parts without breaking start/end pairs. "
            "Files are expected as <ref_id>_<pairIdx>_(start|end).jpg."
        )
    )
    parser.add_argument(
        "--boundary-dir",
        required=True,
        type=str,
        help="Directory containing boundary frames for a single word set (flat JPGs).",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        type=str,
        help="Output root; parts will be created as <out-root>/part_<idx>/.",
    )
    parser.add_argument(
        "--splits",
        required=True,
        type=int,
        help="Number of parts to split into.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (default: copy).",
    )
    return parser.parse_args()


def collect_groups(boundary_dir: Path) -> Dict[str, List[Path]]:
    pattern = re.compile(r"(.+)_([01])_(start|end)\.(jpg|png)$", re.IGNORECASE)
    groups: Dict[str, List[Path]] = {}
    for file_path in boundary_dir.iterdir():
        if not file_path.is_file():
            continue
        match = pattern.match(file_path.name)
        if not match:
            continue
        ref_id = match.group(1)
        groups.setdefault(ref_id, []).append(file_path)
    return groups


def validate_groups(groups: Dict[str, List[Path]]) -> None:
    for ref_id, files in groups.items():
        if len(files) % 2 != 0:
            raise ValueError(f"{ref_id}: odd number of boundary files ({len(files)})")
        starts = sum(1 for f in files if "_start." in f.name.lower())
        ends = sum(1 for f in files if "_end." in f.name.lower())
        if starts != ends:
            raise ValueError(f"{ref_id}: start/end count mismatch ({starts} start, {ends} end)")


def chunk_ids(ref_ids: List[str], splits: int) -> List[List[str]]:
    chunk_size = math.ceil(len(ref_ids) / splits)
    chunks = []
    for i in range(splits):
        start = i * chunk_size
        end = start + chunk_size
        if start >= len(ref_ids):
            chunks.append([])
        else:
            chunks.append(ref_ids[start:end])
    return chunks


def transfer_files(ref_ids: List[str], groups: Dict[str, List[Path]],
                   out_dir: Path, move: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    transfer = shutil.move if move else shutil.copy2
    count = 0
    for ref_id in ref_ids:
        for src in groups[ref_id]:
            dst = out_dir / src.name
            transfer(src, dst)
            count += 1
    return count


def main() -> None:
    args = parse_args()

    boundary_dir = Path(args.boundary_dir)
    out_root = Path(args.out_root)

    if args.splits < 1:
        raise ValueError("splits must be >= 1")
    if not boundary_dir.exists():
        raise FileNotFoundError(f"boundary-dir not found: {boundary_dir}")

    groups = collect_groups(boundary_dir)
    if not groups:
        raise ValueError(f"No boundary files found in {boundary_dir}")

    validate_groups(groups)

    ref_ids = sorted(groups.keys())
    chunks = chunk_ids(ref_ids, args.splits)

    total = 0
    for idx, chunk in enumerate(chunks, start=1):
        if not chunk:
            print(f"[SKIP] part_{idx}: no refs assigned")
            continue
        part_dir = out_root / f"part_{idx}"
        moved = transfer_files(chunk, groups, part_dir, args.move)
        total += moved
        print(f"[DONE] part_{idx}: {len(chunk)} ref_ids, {moved} files")

    print(f"Finished. {len(ref_ids)} ref_ids, {total} files processed.")


if __name__ == "__main__":
    main()
