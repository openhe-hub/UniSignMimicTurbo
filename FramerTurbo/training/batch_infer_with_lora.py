"""
Batch inference script for FramerTurbo with LoRA fine-tuned weights
Processes all image pairs in a directory: {xxx}_start.jpg, {xxx}_end.jpg -> {xxx}.gif
"""

import os
import sys
import argparse
import torch
from PIL import Image
from peft import PeftModel
from pathlib import Path
from tqdm import tqdm
import glob

sys.path.insert(0, os.getcwd())

from models_diffusers.controlnet_svd import ControlNetSVDModel
from models_diffusers.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from pipelines.pipeline_stable_video_diffusion_interp_control import StableVideoDiffusionInterpControlPipeline
from diffusers.utils import export_to_gif


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference with LoRA fine-tuned FramerTurbo")

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
        help="Path to LoRA weights (e.g., outputs/lora_576x576/final/unet_lora)"
    )
    parser.add_argument(
        "--svd_model",
        type=str,
        default="checkpoints/stable-video-diffusion-img2vid-xt",
        help="Path to Stable Video Diffusion model"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing {xxx}_start.jpg and {xxx}_end.jpg pairs"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated GIFs"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=3,
        help="Number of frames to generate"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=15,
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
        "--motion_bucket_id",
        type=int,
        default=100,
        help="Motion bucket ID"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=7,
        help="Output FPS"
    )
    parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="Merge LoRA weights into base model (faster inference)"
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="LoRA scaling factor (0-1, lower means closer to base model)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )

    return parser.parse_args()


def find_image_pairs(input_dir):
    """
    Find all image pairs matching {xxx}_start.jpg and {xxx}_end.jpg pattern

    Returns:
        List of tuples: (prefix, start_path, end_path)
    """
    input_path = Path(input_dir)
    start_images = sorted(input_path.glob("*_start.jpg"))

    pairs = []
    for start_img in start_images:
        # Extract prefix from {xxx}_start.jpg
        prefix = start_img.stem.replace("_start", "")

        # Check if corresponding end image exists
        end_img = input_path / f"{prefix}_end.jpg"

        if end_img.exists():
            pairs.append((prefix, str(start_img), str(end_img)))
        else:
            print(f"[WARNING] Missing end image for: {start_img.name}")

    return pairs


def load_lora_model(base_model_path, lora_weights_path, lora_scale=1.0, merge=False, device="cuda"):
    """Load UNet with LoRA weights

    Args:
        lora_scale: LoRA scaling factor (0-1).
                   1.0 = full LoRA effect
                   0.5 = 50% LoRA + 50% base model
                   0.0 = pure base model (disable LoRA)
    """
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

    # Set LoRA scaling factor
    if lora_scale != 1.0:
        print(f"[INFO] Setting LoRA scale to {lora_scale}")
        for name, module in unet.named_modules():
            if hasattr(module, 'scaling'):
                # Adjust the alpha/r ratio which controls LoRA strength
                module.scaling = {k: lora_scale for k in module.scaling.keys()} if isinstance(module.scaling, dict) else lora_scale

    if merge:
        print("[INFO] Merging LoRA weights into base model...")
        unet = unet.merge_and_unload()

    unet = unet.to(device, dtype=torch.float16)
    return unet


def main():
    args = parse_args()

    print("=" * 70)
    print("FramerTurbo Batch Inference with LoRA")
    print("=" * 70)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_path}")

    # Find image pairs
    print(f"[INFO] Scanning input directory: {args.input_dir}")
    image_pairs = find_image_pairs(args.input_dir)

    if not image_pairs:
        print("[ERROR] No image pairs found!")
        print(f"        Expected pattern: {{xxx}}_start.jpg and {{xxx}}_end.jpg")
        return

    print(f"[INFO] Found {len(image_pairs)} image pairs")

    # Load models
    print("\n" + "=" * 70)
    print("Loading Models")
    print("=" * 70)

    unet = load_lora_model(
        args.base_model,
        args.lora_weights,
        lora_scale=args.lora_scale,
        merge=args.merge_lora,
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

    # Enable xformers if available
    try:
        from diffusers.utils.import_utils import is_xformers_available
        if is_xformers_available():
            pipe.unet.enable_xformers_memory_efficient_attention()
            print("[INFO] xformers enabled")
    except:
        print("[WARNING] xformers not available")

    # Process each image pair
    print("\n" + "=" * 70)
    print("Processing Image Pairs")
    print("=" * 70)
    print(f"Parameters:")
    print(f"  Frames: {args.num_frames}")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  Size: {args.width}x{args.height}")
    print(f"  FPS: {args.fps}")
    print("=" * 70 + "\n")

    success_count = 0
    failed_pairs = []

    for prefix, start_path, end_path in tqdm(image_pairs, desc="Generating"):
        try:
            # Load and resize images
            start_image = Image.open(start_path).convert("RGB")
            start_image = start_image.resize((args.width, args.height))

            end_image = Image.open(end_path).convert("RGB")
            end_image = end_image.resize((args.width, args.height))

            # Run inference
            with torch.no_grad():
                output = pipe(
                    start_image,
                    end_image,
                    with_control=False,
                    num_frames=args.num_frames,
                    width=args.width,
                    height=args.height,
                    num_inference_steps=args.num_inference_steps,
                    motion_bucket_id=args.motion_bucket_id,
                    fps=args.fps,
                )

            frames = output.frames[0]

            # Save output
            output_file = output_path / f"{prefix}.gif"
            export_to_gif(frames, str(output_file), fps=args.fps)

            success_count += 1

        except Exception as e:
            print(f"\n[ERROR] Failed to process {prefix}: {e}")
            failed_pairs.append((prefix, str(e)))
            continue

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total pairs: {len(image_pairs)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_pairs)}")

    if failed_pairs:
        print("\nFailed pairs:")
        for prefix, error in failed_pairs:
            print(f"  - {prefix}: {error}")

    print(f"\nOutput directory: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()


# ============================================================================
# Usage Example
# ============================================================================
"""
# Basic usage (run from FramerTurbo project root)
python training/batch_infer_with_lora.py \
    --lora_weights outputs/lora_576x576/final/unet_lora \
    --input_dir assets/test01 \
    --output_dir outputs/test01 \
    --height 576 \
    --width 576 \
    --num_frames 3 \
    --num_inference_steps 15

# With merged LoRA for faster inference
python training/batch_infer_with_lora.py \
    --lora_weights outputs/lora_576x576/final/unet_lora \
    --input_dir assets/test01 \
    --output_dir outputs/test01 \
    --height 576 \
    --width 576 \
    --merge_lora

# High quality (more frames, more steps)
python training/batch_infer_with_lora.py \
    --lora_weights outputs/lora_576x576/final/unet_lora \
    --input_dir assets/test01 \
    --output_dir outputs/test01_hq \
    --height 576 \
    --width 576 \
    --num_frames 7 \
    --num_inference_steps 25 \
    --merge_lora
"""
