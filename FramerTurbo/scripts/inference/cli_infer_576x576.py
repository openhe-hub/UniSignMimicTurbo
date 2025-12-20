"""
FramerTurbo CLI Inference for 576x576 Sign Language Videos

使用示例：
    # 基础使用（DPM++ 调度器，15步）
    python cli_infer_576x576.py --input_dir data/sign_pairs --model checkpoints/framer_512x320/ --output_dir outputs_576

    # LCM 快速模式（6步）
    python cli_infer_576x576.py --input_dir data/sign_pairs --model checkpoints/framer_512x320/ \\
           --output_dir outputs_576 --scheduler lcm --num_inference_steps 6

    # 使用 SIFT 自动轨迹估计
    python cli_infer_576x576.py --input_dir data/sign_pairs --model checkpoints/framer_512x320/ \\
           --output_dir outputs_576 --use_sift 1

输入格式：
    目录下需要包含成对的图像文件：
    - {id}_start.jpg  （起始帧）
    - {id}_end.jpg    （结束帧）
    - 例如：001_start.jpg, 001_end.jpg
"""

import os
import sys
import time
from typing import List, Tuple, Dict

# Add FramerTurbo to path
sys.path.insert(0, os.path.dirname(__file__))

from apps.app_turbo_v2 import Drag, get_args, configure_scheduler
from gradio_demo.utils_drag import ensure_dirname


def collect_pairs(input_dir: str) -> List[Tuple[str, str, str]]:
    """收集输入目录中的图像对"""
    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png"))]
    pairs: Dict[str, Dict[str, str]] = {}

    for name in files:
        if "_start." in name:
            sample_id = name.split("_start.")[0]
            pairs.setdefault(sample_id, {})["start"] = os.path.join(input_dir, name)
        elif "_end." in name:
            sample_id = name.split("_end.")[0]
            pairs.setdefault(sample_id, {})["end"] = os.path.join(input_dir, name)

    result = []
    for sample_id, paths in pairs.items():
        if "start" in paths and "end" in paths:
            result.append((sample_id, paths["start"], paths["end"]))

    result.sort(key=lambda x: x[0])
    return result


def main():
    args = get_args()

    # 强制使用 576x576 分辨率
    args.width = 576
    args.height = 576

    if args.input_dir is None:
        raise ValueError("--input_dir must be specified for CLI inference.")

    ensure_dirname(args.output_dir)

    print("\n" + "=" * 70)
    print("FramerTurbo - 576x576 Sign Language Video Interpolation")
    print("=" * 70)
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Scheduler: {args.scheduler.upper()}")
    print(f"Inference Steps: {args.num_inference_steps}")
    print(f"Use SIFT: {bool(args.use_sift)}")

    if args.scheduler == "dpm++":
        print(f"DPM Solver Order: {args.dpm_solver_order}")
        print(f"Karras Sigmas: {args.dpm_use_karras_sigmas}")
    elif args.scheduler == "lcm":
        print(f"LCM Train Timesteps: {args.lcm_num_train_timesteps}")

    print("=" * 70 + "\n")

    print("Loading model...")
    drag = Drag("cuda", args, args.height, args.width, args.num_frames, use_sift=bool(args.use_sift))
    print("✓ Model loaded successfully\n")

    pairs = collect_pairs(args.input_dir)
    if not pairs:
        raise ValueError(
            f"No valid pairs found in {args.input_dir}. "
            f"Expected files named {{id}}_start.jpg and {{id}}_end.jpg."
        )

    print(f"Found {len(pairs)} pairs in {args.input_dir}\n")

    inference_times = []

    print("=" * 70)
    print("Starting inference...")
    print("=" * 70 + "\n")

    for idx, (sample_id, start_path, end_path) in enumerate(pairs, 1):
        print(f"[{idx}/{len(pairs)}] Processing {sample_id}")
        output_path = os.path.join(args.output_dir, f"{sample_id}.gif")

        # 如果 use_sift=True，空列表会触发自动轨迹估计
        tracking_points = []

        try:
            gif_path, inference_time = drag.run(
                start_path,
                end_path,
                tracking_points,
                args.controlnet_cond_scale,
                args.motion_bucket_id,
                output_path=output_path,
            )

            inference_times.append(inference_time)
            print(f"  ✓ Saved to {gif_path}")
            print(f"  ⏱  Inference time: {inference_time:.2f}s\n")

        except Exception as e:
            print(f"  ✗ Failed: {e}\n")
            import traceback
            traceback.print_exc()
            continue

    # 统计信息
    if inference_times:
        total_time = sum(inference_times)
        avg_time = total_time / len(inference_times)

        print("\n" + "=" * 70)
        print(f"Successfully processed {len(inference_times)}/{len(pairs)} pairs")
        print("=" * 70)
        print(f"Statistics:")
        print(f"  - Total inference time: {total_time:.2f}s")
        print(f"  - Average per pair: {avg_time:.2f}s")
        print(f"  - Output directory: {args.output_dir}")
        print("=" * 70 + "\n")

        # 保存统计信息
        time_file = os.path.join(args.output_dir, ".inference_stats.txt")
        with open(time_file, "w") as f:
            f.write(f"Total inference time: {total_time:.2f}s\n")
            f.write(f"Average per pair: {avg_time:.2f}s\n")
            f.write(f"Pairs processed: {len(inference_times)}/{len(pairs)}\n")
            f.write(f"Resolution: {args.width}x{args.height}\n")
            f.write(f"Scheduler: {args.scheduler}\n")
            f.write(f"Steps: {args.num_inference_steps}\n")


if __name__ == "__main__":
    main()
