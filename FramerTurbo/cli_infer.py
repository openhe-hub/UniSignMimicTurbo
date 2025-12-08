import os
from typing import Dict, List, Tuple

from app import Drag, get_args
from gradio_demo.utils_drag import ensure_dirname


def collect_pairs(input_dir: str) -> List[Tuple[str, str, str]]:
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".jpg")]
    pairs: Dict[str, Dict[str, str]] = {}

    for name in files:
        if name.endswith("_start.jpg"):
            sample_id = name[: -len("_start.jpg")]
            pairs.setdefault(sample_id, {})["start"] = os.path.join(input_dir, name)
        elif name.endswith("_end.jpg"):
            sample_id = name[: -len("_end.jpg")]
            pairs.setdefault(sample_id, {})["end"] = os.path.join(input_dir, name)

    result = []
    for sample_id, paths in pairs.items():
        if "start" in paths and "end" in paths:
            result.append((sample_id, paths["start"], paths["end"]))

    result.sort(key=lambda x: x[0])
    return result


def main():
    args = get_args()
    if args.input_dir is None:
        raise ValueError("--input_dir must be specified for CLI inference.")

    ensure_dirname(args.output_dir)

    drag = Drag("cuda", args, args.height, args.width, args.num_frames, use_sift=bool(args.use_sift))

    pairs = collect_pairs(args.input_dir)
    if not pairs:
        raise ValueError(
            f"No valid pairs found in {args.input_dir}. "
            f"Expected files named {{id}}_start.jpg and {{id}}_end.jpg."
        )

    print(f"Found {len(pairs)} pairs in {args.input_dir}")

    for sample_id, start_path, end_path in pairs:
        print(f"Processing id={sample_id}")
        output_path = os.path.join(args.output_dir, f"{sample_id}.gif")

        # Empty list means: if use_sift=True when constructing Drag,
        # the model will automatically estimate trajectories.
        tracking_points = []

        gif_path = drag.run(
            start_path,
            end_path,
            tracking_points,
            args.controlnet_cond_scale,
            args.motion_bucket_id,
            output_path=output_path,
        )

        print(f"Saved result for id={sample_id} to {gif_path}")


if __name__ == "__main__":
    main()

# /home/zl6890/.conda/envs/framer_py38/bin/python cli_infer.py --input_dir assets/pairs --model checkpoints/framer_512x320/ --output_dir outputs
