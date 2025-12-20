import argparse
import random
import subprocess
import tempfile
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create N sample videos by randomly picking K MP4s and concatenating them."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=str,
        help="Directory containing source MP4 files.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=str,
        help="Output directory to save concatenated samples.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of sample videos to generate (default: 10).",
    )
    parser.add_argument(
        "--per",
        type=int,
        default=5,
        help="Number of clips to concatenate per sample (default: 5).",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help="Optional: directory to save extracted frames per source MP4. Default: <out-dir>/edited_frames",
    )
    return parser.parse_args()


def pick_clips(mp4_files: List[Path], per: int) -> List[Path]:
    if len(mp4_files) < per:
        raise ValueError(f"Not enough MP4 files: need {per}, found {len(mp4_files)}")
    return random.sample(mp4_files, per)


def concat_with_ffmpeg(clips: List[Path], output_path: Path) -> bool:
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        for clip in clips:
            abs_path = clip.resolve()
            f.write(f"file '{abs_path.as_posix()}'\n")
        list_path = f.name

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c",
        "copy",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found; please install ffmpeg.")
        return False

    if result.returncode != 0:
        print(f"[ERROR] ffmpeg failed for {output_path.name}")
        print(result.stderr.decode("utf-8", errors="ignore"))
        return False

    return True


def extract_frames(video_path: Path, target_dir: Path) -> None:
    """
    Extract all frames from a video into target_dir/frame_*.jpg.
    Skips extraction if target folder already has jpgs.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    if any(target_dir.glob("*.jpg")):
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path.resolve()),
        str((target_dir / "frame_%05d.jpg").resolve()),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError:
        print("[ERROR] ffmpeg not found; please install ffmpeg.")
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] ffmpeg extract failed for {video_path.name}")
        print(exc.stderr.decode("utf-8", errors="ignore"))


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_root = Path(args.frames_dir) if args.frames_dir else out_dir / "edited_frames"
    frames_root.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(input_dir.glob("*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No MP4 files found in {input_dir}")

    for idx in range(1, args.count + 1):
        clips = pick_clips(mp4_files, args.per)
        output_path = out_dir / f"sample_{idx:02d}.mp4"
        print(f"[INFO] Generating {output_path.name} from {[c.name for c in clips]}")
        if concat_with_ffmpeg(clips, output_path):
            extract_frames(output_path, frames_root / output_path.stem)

    print(f"Done. Generated {args.count} samples in {out_dir}")


if __name__ == "__main__":
    main()
