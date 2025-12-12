"""
Inference script for FramerTurbo with LoRA fine-tuned weights
This script shows how to load and use LoRA fine-tuned models
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
    parser = argparse.ArgumentParser(description="Inference with LoRA fine-tuned FramerTurbo")

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
        help="Path to LoRA weights (e.g., outputs/lora_finetune/final/unet_lora)"
    )
    parser.add_argument(
        "--svd_model",
        type=str,
        default="checkpoints/stable-video-diffusion-img2vid-xt",
        help="Path to Stable Video Diffusion model"
    )
    parser.add_argument(
        "--start_image",
        type=str,
        required=True,
        help="Path to start frame image"
    )
    parser.add_argument(
        "--end_image",
        type=str,
        required=True,
        help="Path to end frame image"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.gif",
        help="Output GIF path"
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
        default=320,
        help="Frame height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Frame width"
    )
    parser.add_argument(
        "--motion_bucket_id",
        type=int,
        default=100,
        help="Motion bucket ID"
    )
    parser.add_argument(
        "--controlnet_scale",
        type=float,
        default=1.0,
        help="ControlNet conditioning scale"
    )
    parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="Merge LoRA weights into base model (faster inference)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )

    return parser.parse_args()


def load_lora_model(base_model_path, lora_weights_path, merge=False, device="cuda"):
    """
    Load UNet with LoRA weights

    Args:
        base_model_path: Path to base UNet model
        lora_weights_path: Path to LoRA adapter weights
        merge: Whether to merge LoRA into base model
        device: Device to load model on

    Returns:
        UNet model with LoRA weights loaded
    """
    print(f"Loading base UNet from {base_model_path}...")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        os.path.join(base_model_path, "unet"),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        custom_resume=True,
    )

    print(f"Loading LoRA weights from {lora_weights_path}...")
    unet = PeftModel.from_pretrained(
        unet,
        lora_weights_path,
        torch_dtype=torch.float16,
    )

    if merge:
        print("Merging LoRA weights into base model...")
        unet = unet.merge_and_unload()

    unet = unet.to(device, dtype=torch.float16)
    return unet


def main():
    args = parse_args()

    print("=" * 70)
    print("FramerTurbo Inference with LoRA")
    print("=" * 70)

    # Load UNet with LoRA
    unet = load_lora_model(
        args.base_model,
        args.lora_weights,
        merge=args.merge_lora,
        device=args.device,
    )

    # Load ControlNet
    print(f"Loading ControlNet from {args.base_model}/controlnet...")
    controlnet = ControlNetSVDModel.from_pretrained(
        os.path.join(args.base_model, "controlnet"),
        torch_dtype=torch.float16,
    )
    controlnet = controlnet.to(args.device, dtype=torch.float16)

    # Create pipeline
    print(f"Creating pipeline with SVD model {args.svd_model}...")
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
            print("✓ xformers enabled")
    except:
        print("⚠ xformers not available")

    # Load images
    print(f"Loading start image: {args.start_image}")
    start_image = Image.open(args.start_image).convert("RGB")
    start_image = start_image.resize((args.width, args.height))

    print(f"Loading end image: {args.end_image}")
    end_image = Image.open(args.end_image).convert("RGB")
    end_image = end_image.resize((args.width, args.height))

    # Run inference
    print("\nGenerating frames...")
    print(f"  Frames: {args.num_frames}")
    print(f"  Steps: {args.num_inference_steps}")
    print(f"  Size: {args.width}x{args.height}")

    with torch.no_grad():
        output = pipe(
            start_image,
            end_image,
            with_control=False,  # Set to True if using trajectory control
            num_frames=args.num_frames,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps,
            motion_bucket_id=args.motion_bucket_id,
            fps=7,
            controlnet_cond_scale=args.controlnet_scale,
        )

    frames = output.frames[0]

    # Save output
    print(f"\nSaving output to {args.output_path}...")
    export_to_gif(frames, args.output_path, fps=7)

    print("=" * 70)
    print(f"✓ Done! Output saved to: {args.output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()


# ============================================================================
# Usage Examples
# ============================================================================
"""
# Basic usage (run from FramerTurbo project root)
python training/infer_with_lora.py \
    --lora_weights outputs/lora_576x576/final/unet_lora \
    --start_image assets/test01/0f2hdqx1ud_0_start.jpg \
    --end_image assets/test01/0f2hdqx1ud_0_end.jpg \
    --output_path outputs/0f2hdqx1ud_0.gif \
    --height 576 \
    --width 576 \
    --num_frames 3 \
    --num_inference_steps 50

# With merged LoRA (faster)
python training/infer_with_lora.py \\
    --lora_weights outputs/lora_finetune/final/unet_lora \\
    --start_image examples/start.jpg \\
    --end_image examples/end.jpg \\
    --output_path output.gif \\
    --merge_lora

# Custom parameters
python training/infer_with_lora.py \\
    --lora_weights outputs/lora_finetune/final/unet_lora \\
    --start_image examples/start.jpg \\
    --end_image examples/end.jpg \\
    --output_path output.gif \\
    --num_frames 5 \\
    --num_inference_steps 20 \\
    --motion_bucket_id 120
"""
