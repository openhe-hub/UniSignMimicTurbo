"""
Validate LoRA training by inferring on training samples
Extracts first/last frames from training videos and generates interpolation
"""

import os
import sys
import argparse
import torch
import cv2
import random
from PIL import Image
from peft import PeftModel
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.getcwd())

from models_diffusers.controlnet_svd import ControlNetSVDModel
from models_diffusers.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from pipelines.pipeline_stable_video_diffusion_interp_control import StableVideoDiffusionInterpControlPipeline
from diffusers.utils import export_to_gif


def parse_args():
    parser = argparse.ArgumentParser(description="Validate LoRA training on training samples")

    parser.add_argument(
        "--base_model",
        type=str,
        default="checkpoints/framer_512x320",
        help="Path to base FramerTurbo model"
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        required=True,
        help="Path to LoRA weights to validate"
    )
    parser.add_argument(
        "--svd_model",
        type=str,
        default="checkpoints/stable-video-diffusion-img2vid-xt",
        help="Path to Stable Video Diffusion model"
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Training data directory (contains video files)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/validation",
        help="Output directory for validation results"
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=10,
        help="Number of random videos to test"
    )
    parser.add_argument(
        "--windows_per_video",
        type=int,
        default=3,
        help="Number of frame windows to extract per video"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=3,
        help="Number of frames to generate (must match training)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=576,
        help="Frame height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=576,
        help="Frame width"
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="LoRA scaling factor (0-1)"
    )
    parser.add_argument(
        "--save_comparison",
        action="store_true",
        help="Save side-by-side comparison (original vs generated)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )

    return parser.parse_args()


def extract_frame_windows_from_video(video_path, height, width, num_frames=3, num_windows=3):
    """
    Extract consecutive frame windows from video (matching training)

    Args:
        video_path: Path to video file
        height, width: Target resolution
        num_frames: Window size (e.g., 3 means extract [0,1,2], [5,6,7], etc.)
        num_windows: How many windows to extract from this video

    Returns:
        List of (first_frame, middle_frames, last_frame) tuples
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        cap.release()
        raise ValueError(f"Video has only {total_frames} frames, need at least {num_frames}")

    # Calculate possible starting positions
    max_start = total_frames - num_frames

    # Sample window starting positions
    if num_windows == 1:
        # Take from the beginning
        start_positions = [0]
    elif max_start <= num_windows:
        # If video is short, take all possible positions
        start_positions = list(range(0, max_start + 1))
    else:
        # Uniformly sample positions
        step = max_start // (num_windows - 1)
        start_positions = [i * step for i in range(num_windows)]
        if start_positions[-1] != max_start:
            start_positions[-1] = max_start

    windows = []

    for start_idx in start_positions:
        frames = []

        # Read consecutive frames
        for offset in range(num_frames):
            frame_idx = start_idx + offset
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height))
            frames.append(Image.fromarray(frame))

        if len(frames) == num_frames:
            # Split into first, middle, last
            first = frames[0]
            last = frames[-1]
            middle = frames[1:-1] if len(frames) > 2 else []
            windows.append((first, middle, last, start_idx))

    cap.release()

    return windows


def collect_video_files(data_dir):
    """Collect all video files from training directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    video_files = []

    data_path = Path(data_dir)
    for ext in video_extensions:
        video_files.extend(list(data_path.glob(f"*{ext}")))

    return [str(f) for f in video_files]


def load_lora_model(base_model_path, lora_weights_path, lora_scale=1.0, device="cuda"):
    """Load UNet with LoRA weights"""
    print(f"[INFO] Loading base UNet from {base_model_path}...")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        os.path.join(base_model_path, "unet"),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        custom_resume=True,
    )

    print(f"[INFO] Loading LoRA weights from {lora_weights_path}...")
    unet = PeftModel.from_pretrained(
        unet,
        lora_weights_path,
        torch_dtype=torch.float16,
    )

    if lora_scale != 1.0:
        print(f"[INFO] Setting LoRA scale to {lora_scale}")
        for name, module in unet.named_modules():
            if hasattr(module, 'scaling'):
                module.scaling = {k: lora_scale for k in module.scaling.keys()} if isinstance(module.scaling, dict) else lora_scale

    unet = unet.to(device, dtype=torch.float16)
    return unet


def create_comparison_image(original_frames, generated_frames):
    """Create side-by-side comparison image"""
    import numpy as np

    # Ensure both have same number of frames
    num_frames = min(len(original_frames), len(generated_frames))

    # Convert to numpy arrays
    orig_arrays = [np.array(f) for f in original_frames[:num_frames]]
    gen_arrays = [np.array(f) for f in generated_frames[:num_frames]]

    # Stack horizontally
    comparisons = [np.hstack([orig, gen]) for orig, gen in zip(orig_arrays, gen_arrays)]

    # Convert back to PIL
    return [Image.fromarray(comp) for comp in comparisons]


def main():
    args = parse_args()

    print("=" * 70)
    print("LoRA Training Validation")
    print("=" * 70)
    print(f"Training data: {args.train_data_dir}")
    print(f"LoRA weights: {args.lora_weights}")
    print(f"Videos to test: {args.num_videos}")
    print(f"Windows per video: {args.windows_per_video}")
    print(f"Frames per window: {args.num_frames}")
    print(f"LoRA scale: {args.lora_scale}")
    print("=" * 70 + "\n")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frames_dir = output_path / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Collect video files
    print(f"[INFO] Scanning training data directory...")
    video_files = collect_video_files(args.train_data_dir)

    if not video_files:
        print(f"[ERROR] No video files found in {args.train_data_dir}")
        return

    print(f"[INFO] Found {len(video_files)} training videos")

    # Sample random videos
    num_videos = min(args.num_videos, len(video_files))
    sampled_videos = random.sample(video_files, num_videos)
    print(f"[INFO] Selected {num_videos} random videos")
    print(f"[INFO] Will extract {args.windows_per_video} windows per video\n")

    # Load models
    print("=" * 70)
    print("Loading Models")
    print("=" * 70)

    unet = load_lora_model(
        args.base_model,
        args.lora_weights,
        lora_scale=args.lora_scale,
        device=args.device,
    )

    print(f"[INFO] Loading ControlNet from {args.base_model}/controlnet...")
    controlnet = ControlNetSVDModel.from_pretrained(
        os.path.join(args.base_model, "controlnet"),
        torch_dtype=torch.float16,
    )
    controlnet = controlnet.to(args.device, dtype=torch.float16)

    print(f"[INFO] Creating pipeline with SVD model {args.svd_model}...")
    pipe = StableVideoDiffusionInterpControlPipeline.from_pretrained(
        args.svd_model,
        unet=unet,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
    )
    pipe = pipe.to(args.device)

    # Enable xformers
    try:
        from diffusers.utils.import_utils import is_xformers_available
        if is_xformers_available():
            pipe.unet.enable_xformers_memory_efficient_attention()
            print("[INFO] xformers enabled")
    except:
        print("[WARNING] xformers not available")

    # Process samples
    print("\n" + "=" * 70)
    print("Processing Samples")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  Frames per window: {args.num_frames}")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  Size: {args.width}x{args.height}")
    print("=" * 70 + "\n")

    success_count = 0
    total_windows = 0
    failed_windows = []

    for idx, video_path in enumerate(tqdm(sampled_videos, desc="Processing videos"), 1):
        video_name = Path(video_path).stem

        try:
            # Extract consecutive frame windows (matching training)
            windows = extract_frame_windows_from_video(
                video_path,
                args.height,
                args.width,
                num_frames=args.num_frames,
                num_windows=args.windows_per_video
            )

            if not windows:
                print(f"\n[WARNING] No valid windows extracted from {video_name}")
                continue

            # Process each window
            for window_idx, (first_frame, middle_frames, last_frame, start_frame_idx) in enumerate(windows):
                total_windows += 1
                window_name = f"{video_name}_window{window_idx}_f{start_frame_idx}"

                try:
                    # Save extracted frames
                    first_frame.save(frames_dir / f"{window_name}_first.jpg")
                    last_frame.save(frames_dir / f"{window_name}_last.jpg")

                    # Save original middle frames
                    for i, mf in enumerate(middle_frames):
                        mf.save(frames_dir / f"{window_name}_original_mid{i}.jpg")

                    # Run inference
                    with torch.no_grad():
                        output = pipe(
                            first_frame,
                            last_frame,
                            with_control=False,
                            num_frames=args.num_frames,
                            width=args.width,
                            height=args.height,
                            num_inference_steps=args.num_inference_steps,
                            motion_bucket_id=100,
                            fps=7,
                        )

                    generated_frames = output.frames[0]

                    # Save generated GIF
                    output_gif = output_path / f"{window_name}_generated.gif"
                    export_to_gif(generated_frames, str(output_gif))

                    # Save comparison if requested
                    if args.save_comparison and middle_frames:
                        # Create original GIF
                        original_frames = [first_frame] + middle_frames + [last_frame]
                        original_gif = output_path / f"{window_name}_original.gif"
                        export_to_gif(original_frames, str(original_gif))

                        # Create side-by-side comparison
                        comparison_frames = create_comparison_image(original_frames, generated_frames)
                        comparison_gif = output_path / f"{window_name}_comparison.gif"
                        export_to_gif(comparison_frames, str(comparison_gif))

                    success_count += 1

                except Exception as e:
                    print(f"\n[ERROR] Failed window {window_name}: {e}")
                    failed_windows.append((window_name, str(e)))
                    continue

        except Exception as e:
            print(f"\n[ERROR] Failed to process video {video_name}: {e}")
            continue

    # Summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)
    print(f"Videos processed: {num_videos}")
    print(f"Total windows: {total_windows}")
    print(f"Successful windows: {success_count}")
    print(f"Failed windows: {len(failed_windows)}")

    if failed_windows:
        print("\nFailed windows:")
        for name, error in failed_windows[:10]:  # Show first 10
            print(f"  - {name}: {error}")
        if len(failed_windows) > 10:
            print(f"  ... and {len(failed_windows) - 10} more")

    print(f"\nOutput directory: {output_path}")
    print(f"Extracted frames: {frames_dir}")
    print("=" * 70)

    # Save summary
    summary_file = output_path / "validation_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"LoRA Validation Results\n")
        f.write(f"=" * 70 + "\n")
        f.write(f"LoRA weights: {args.lora_weights}\n")
        f.write(f"Training data: {args.train_data_dir}\n")
        f.write(f"LoRA scale: {args.lora_scale}\n")
        f.write(f"Frames per window: {args.num_frames}\n")
        f.write(f"Windows per video: {args.windows_per_video}\n")
        f.write(f"Inference steps: {args.num_inference_steps}\n")
        f.write(f"\nResults:\n")
        f.write(f"  Videos processed: {num_videos}\n")
        f.write(f"  Total windows: {total_windows}\n")
        f.write(f"  Successful: {success_count}\n")
        f.write(f"  Failed: {len(failed_windows)}\n")

        if failed_windows:
            f.write(f"\nFailed windows:\n")
            for name, error in failed_windows:
                f.write(f"  - {name}: {error}\n")


if __name__ == "__main__":
    main()


# ============================================================================
# Usage Example
# ============================================================================
"""
# Basic validation (10 videos, 3 windows each = 30 samples)
python training/validate_on_trainset.py \
    --lora_weights outputs/lora_576x576/final/unet_lora \
    --train_data_dir assets/AslToHiya-01 \
    --output_dir outputs/validation_0.3 \
    --num_videos 10 \
    --windows_per_video 3 \
    --lora_scale 0.3

# Validate with comparison GIFs (see original vs generated side-by-side)
python training/validate_on_trainset.py \
    --lora_weights outputs/lora_576x576/final/unet_lora \
    --train_data_dir assets/AslToHiya-01 \
    --output_dir outputs/validation \
    --num_videos 20 \
    --windows_per_video 3 \
    --save_comparison

# Test different LoRA scales
python training/validate_on_trainset.py \
    --lora_weights outputs/lora_576x576/final/unet_lora \
    --train_data_dir assets/AslToHiya-01 \
    --output_dir outputs/validation_lora05 \
    --num_videos 10 \
    --windows_per_video 3 \
    --lora_scale 0.5 \
    --save_comparison
"""
