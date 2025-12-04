import argparse
import csv
import os
import shutil
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For the first N sentences in a CSV, "
            "copy the corresponding mp4 files (by ref_ids) "
            "into per-sentence subfolders."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to sentences.csv.",
    )
    parser.add_argument(
        "--mp4-dir",
        type=str,
        required=True,
        help="Directory containing source mp4 files (named <ref_id>.mp4).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output root directory; subfolders per sentence will be created here.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="Number of data rows (sentences) to process from the CSV.",
    )
    return parser.parse_args()


def parse_ref_ids(raw: str) -> List[str]:
    """
    Parse the ref_ids field into a list of ids.

    The CSV stores ref_ids as a comma-separated string, e.g.:
    "id1, id2, id3"

    We also skip any placeholder tokens like 'NO_REF_xxx'.
    """
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p and not p.startswith("NO_REF")]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def copy_mp4s_for_sentence(
    sentence_id: str,
    ref_ids: List[str],
    mp4_dir: str,
    out_root: str,
) -> None:
    sentence_out_dir = os.path.join(out_root, sentence_id)
    ensure_dir(sentence_out_dir)

    for ref_id in ref_ids:
        src = os.path.join(mp4_dir, f"{ref_id}.mp4")
        if not os.path.isfile(src):
            print(f"[WARN] mp4 not found for ref_id={ref_id} (sentence_id={sentence_id})")
            continue

        dst = os.path.join(sentence_out_dir, f"{ref_id}.mp4")
        if os.path.isfile(dst):
            # Already copied; skip to avoid extra I/O.
            continue
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()

    csv_path = os.path.abspath(args.csv)
    mp4_dir = os.path.abspath(args.mp4_dir)
    out_dir = os.path.abspath(args.out_dir)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isdir(mp4_dir):
        raise NotADirectoryError(f"mp4-dir not found: {mp4_dir}")
    ensure_dir(out_dir)

    # Use utf-8-sig to be robust to possible BOM in the CSV.
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if count >= args.num:
                break

            sentence_id = row.get("sentence_id")
            ref_ids_raw = row.get("ref_ids", "")

            if not sentence_id:
                continue

            ref_ids = parse_ref_ids(ref_ids_raw)
            copy_mp4s_for_sentence(sentence_id, ref_ids, mp4_dir, out_dir)

            count += 1

    print(f"Done. Processed {count} sentences into '{out_dir}'.")


if __name__ == "__main__":
    main()

# python scripts/sentence/split_sentence.py --csv assets/sentence/sentence.csv --mp4-dir assets/Asl --out-dir assets/sentence/ --num 5
