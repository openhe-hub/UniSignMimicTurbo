import os
import time
from typing import Dict, List, Tuple

from app_turbo_v2 import Drag, get_args, configure_scheduler
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

    print(f"\n{'='*70}")
    print(f"Framer Turbo V2 CLI - Multi-Scheduler Support")
    print(f"{'='*70}")
    print(f"Scheduler: {args.scheduler.upper()}")
    print(f"Inference Steps: {args.num_inference_steps}")

    if args.scheduler == "dpm++":
        print(f"DPM Solver Order: {args.dpm_solver_order}")
        print(f"Karras Sigmas: {args.dpm_use_karras_sigmas}")
    elif args.scheduler == "lcm":
        print(f"LCM Train Timesteps: {args.lcm_num_train_timesteps}")
        print(f"Sigma Range: [{args.scheduler_sigma_min}, {args.scheduler_sigma_max}]")

    print(f"{'='*70}\n")

    print("Loading model...")
    drag = Drag("cuda", args, args.height, args.width, args.num_frames, use_sift=bool(args.use_sift))
    print("✓ Model loaded\n")

    pairs = collect_pairs(args.input_dir)
    if not pairs:
        raise ValueError(
            f"No valid pairs found in {args.input_dir}. "
            f"Expected files named {{id}}_start.jpg and {{id}}_end.jpg."
        )

    print(f"Found {len(pairs)} pairs in {args.input_dir}\n")

    # 开始计时 - 只计算推理时间
    inference_times = []

    print(f"{'='*70}")
    print("Starting inference (timing begins now)...")
    print(f"{'='*70}\n")

    for idx, (sample_id, start_path, end_path) in enumerate(pairs, 1):
        print(f"[{idx}/{len(pairs)}] Processing id={sample_id}")
        output_path = os.path.join(args.output_dir, f"{sample_id}.gif")

        # Empty list means: if use_sift=True when constructing Drag,
        # the model will automatically estimate trajectories.
        tracking_points = []

        # Run inference and get pure inference time from inside drag.run()
        gif_path, inference_time = drag.run(
            start_path,
            end_path,
            tracking_points,
            args.controlnet_cond_scale,
            args.motion_bucket_id,
            output_path=output_path,
        )

        inference_times.append(inference_time)

        print(f"✓ Saved result for id={sample_id} to {gif_path}")
        print(f"  Pure inference time: {inference_time:.2f}s\n")

    # 计算统计信息
    total_inference_time = sum(inference_times)
    avg_inference_time = total_inference_time / len(inference_times) if inference_times else 0

    print(f"\n{'='*70}")
    print(f"All {len(pairs)} pairs processed successfully!")
    print(f"{'='*70}")
    print(f"Inference Statistics:")
    print(f"  - Total PURE inference time: {total_inference_time:.2f}s")
    print(f"  - Average per image: {avg_inference_time:.2f}s")
    print(f"  - Images processed: {len(pairs)}")
    print(f"  - Note: Excludes model loading, image loading, SIFT, visualization, and GIF saving")
    print(f"{'='*70}\n")

    # 保存时间到文件，供测试脚本使用
    time_file = os.path.join(args.output_dir, ".inference_time.txt")
    with open(time_file, "w") as f:
        f.write(f"{total_inference_time:.2f}\n")
        f.write(f"{avg_inference_time:.2f}\n")
        f.write(f"{len(pairs)}\n")


if __name__ == "__main__":
    main()

# ============================================================================
# Usage Examples
# ============================================================================
#
# NOTE: All timing measurements now report PURE inference time only, excluding:
#   - Model loading
#   - Image loading and preprocessing
#   - SIFT matching (if enabled)
#   - Trajectory preprocessing
#   - Visualization generation
#   - GIF/MP4 saving
#
# 1. DPM++ mode (default, recommended - balanced quality/speed):
#    python cli_infer_turbo_v2.py --input_dir assets/pairs --model checkpoints/framer_512x320/ --output_dir outputs
#
# 2. DPM++ mode with Karras sigmas (better quality):
#    python cli_infer_turbo_v2.py --input_dir assets/pairs --model checkpoints/framer_512x320/ \
#           --output_dir outputs --dpm_use_karras_sigmas
#
# 3. DPM++ mode with custom steps:
#    python cli_infer_turbo_v2.py --input_dir assets/pairs --model checkpoints/framer_512x320/ \
#           --output_dir outputs --num_inference_steps 15
#
# 4. LCM mode (fastest):
#    python cli_infer_turbo_v2.py --input_dir assets/pairs --model checkpoints/framer_512x320/ \
#           --output_dir outputs --scheduler lcm
#
# 5. LCM mode (ultra-fast, 4 steps):
#    python cli_infer_turbo_v2.py --input_dir assets/pairs --model checkpoints/framer_512x320/ \
#           --output_dir outputs --scheduler lcm --num_inference_steps 4
#
# 6. Original Euler mode (best quality):
#    python cli_infer_turbo_v2.py --input_dir assets/pairs --model checkpoints/framer_512x320/ \
#           --output_dir outputs --scheduler euler
#
# ============================================================================
# Performance Comparison
# ============================================================================
#
# IMPORTANT: These timings measure ONLY the self.pipeline() inference call,
# excluding all preprocessing (image loading, SIFT matching, trajectory
# interpolation) and postprocessing (visualization, GIF saving).
#
# Scheduler   Steps    Speed       Quality    Use Case
# ----------  -------  ----------  ---------  --------------------------
# Euler       30       Baseline    Best       Production, final output
# DPM++       15-20    ~1.7x       Excellent  Recommended for most uses
# LCM         6        ~5x         Good       Fast previews, iteration
# LCM (4)     4        ~7.5x       Fair       Ultra-fast previews
#
