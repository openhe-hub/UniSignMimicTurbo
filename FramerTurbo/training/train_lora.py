"""
LoRA Fine-tuning script for FramerTurbo
Supports efficient training with LoRA adapters
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model
import wandb

# Add parent directory to path
sys.path.insert(0, os.getcwd())

from models_diffusers.controlnet_svd import ControlNetSVDModel
from models_diffusers.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from pipelines.pipeline_stable_video_diffusion_interp_control import StableVideoDiffusionInterpControlPipeline
from train_dataset import VideoFrameDataset, ImagePairDataset, collate_fn

# Check diffusers version
check_min_version("0.25.0")

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for FramerTurbo")

    # Model arguments
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        required=True,
        help="Path to pretrained FramerTurbo model"
    )
    parser.add_argument(
        "--svd_model_path",
        type=str,
        default="checkpoints/stable-video-diffusion-img2vid-xt",
        help="Path to Stable Video Diffusion base model"
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing training videos or image pairs"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="video",
        choices=["video", "image_pair"],
        help="Type of dataset: 'video' for video files, 'image_pair' for start-end frame pairs"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=3,
        help="Number of frames to sample per video"
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

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help="Learning rate scheduler type",
        choices=["linear", "cosine", "constant", "constant_with_warmup"]
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1"
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2"
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Adam weight decay"
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Adam epsilon"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping"
    )

    # LoRA arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=64,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--train_unet",
        action="store_true",
        help="Whether to train UNet with LoRA"
    )
    parser.add_argument(
        "--train_controlnet",
        action="store_true",
        help="Whether to train ControlNet"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    # Logging
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Logging directory"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="framer-turbo-lora",
        help="W&B project name"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb", "all"],
        help="Reporting tool"
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training"
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=0.05,
        help="Noise offset for training stability"
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        choices=["epsilon", "v_prediction"],
        help="Prediction type for diffusion model"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.logging_dir,
    )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)#, main_process_only=False)

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    logger.info("Loading models...")

    # Load UNet
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        os.path.join(args.pretrained_model_path, "unet"),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        custom_resume=True,
    )

    # Load ControlNet
    controlnet = ControlNetSVDModel.from_pretrained(
        os.path.join(args.pretrained_model_path, "controlnet"),
        torch_dtype=torch.float16,
    )

    # Load VAE
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.svd_model_path,
        subfolder="vae",
        torch_dtype=torch.float16,
    )
    vae.requires_grad_(False)
    vae.eval()

    # Load image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.svd_model_path,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    )
    image_encoder.requires_grad_(False)
    image_encoder.eval()

    # Load feature extractor
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.svd_model_path,
        subfolder="feature_extractor",
    )

    # Load scheduler - Use DDPM for training (Euler is for inference only)
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.svd_model_path,
        subfolder="scheduler",
    )

    # Setup LoRA for UNet
    trainable_params = []
    if args.train_unet:
        logger.info(f"Setting up LoRA for UNet (rank={args.lora_rank}, alpha={args.lora_alpha})")

        # Configure LoRA
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=[
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "ff.net.0.proj", "ff.net.2",
            ],
            lora_dropout=args.lora_dropout,
        )

        # Apply LoRA to UNet
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()

        # Enable gradient checkpointing to save memory
        unet.enable_gradient_checkpointing()
        logger.info("✓ Gradient checkpointing enabled for UNet")

        trainable_params.extend(unet.parameters())
    else:
        unet.requires_grad_(False)

    # Setup ControlNet training
    if args.train_controlnet:
        logger.info("Training ControlNet")
        controlnet.requires_grad_(True)
        trainable_params.extend(controlnet.parameters())
    else:
        controlnet.requires_grad_(False)

    if not trainable_params:
        raise ValueError("No parameters to train! Enable --train_unet or --train_controlnet")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Create dataset
    logger.info(f"Loading {args.dataset_type} dataset from {args.data_dir}")

    if args.dataset_type == "video":
        train_dataset = VideoFrameDataset(
            video_dir=args.data_dir,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
        )
    else:
        train_dataset = ImagePairDataset(
            data_dir=args.data_dir,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare with accelerator
    if args.train_unet and args.train_controlnet:
        unet, controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, controlnet, optimizer, train_dataloader, lr_scheduler
        )
    elif args.train_unet:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
        controlnet = controlnet.to(accelerator.device)
    else:  # train_controlnet
        controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            controlnet, optimizer, train_dataloader, lr_scheduler
        )
        unet = unet.to(accelerator.device)

    # Move frozen models to device
    vae = vae.to(accelerator.device)
    image_encoder = image_encoder.to(accelerator.device)

    # Initialize W&B
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(project=args.wandb_project, config=vars(args))

    # Training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    # Training loop
    global_step = 0
    progress_bar = tqdm(
        range(max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    for epoch in range(args.num_train_epochs):
        unet.train() if args.train_unet else None
        controlnet.train() if args.train_controlnet else None

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet if args.train_unet else controlnet):
                # Get batch data
                pixel_values = batch["pixel_values"]  # (B, num_frames, 3, H, W)
                first_frames = batch["first_frames"]  # (B, 3, H, W)
                last_frames = batch["last_frames"]    # (B, 3, H, W)

                batch_size = pixel_values.shape[0]

                # Encode images to latents
                with torch.no_grad():
                    # Encode all frames
                    b, f, c, h, w = pixel_values.shape
                    pixel_values_flat = pixel_values.reshape(b * f, c, h, w)

                    # Convert to same dtype as VAE (fix dtype mismatch)
                    pixel_values_flat = pixel_values_flat.to(dtype=vae.dtype)

                    # VAE encoding
                    latents = vae.encode(pixel_values_flat).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.reshape(b, f, *latents.shape[1:])  # (B, F, C, H, W)

                    # Prepare frames for CLIP (requires 224x224)
                    # Resize first/last frames to 224x224 for CLIP encoder
                    first_frames_clip = F.interpolate(
                        first_frames,
                        size=(224, 224),
                        mode='bicubic',
                        align_corners=False
                    ).to(dtype=image_encoder.dtype)

                    last_frames_clip = F.interpolate(
                        last_frames,
                        size=(224, 224),
                        mode='bicubic',
                        align_corners=False
                    ).to(dtype=image_encoder.dtype)

                    # Encode image conditions with CLIP
                    # Shape: (B, hidden_dim) -> (B, 1, hidden_dim)
                    first_frame_embeds = image_encoder(first_frames_clip).image_embeds.unsqueeze(1)
                    last_frame_embeds = image_encoder(last_frames_clip).image_embeds.unsqueeze(1)

                    # Clean up temporary tensors to save memory
                    del first_frames_clip, last_frames_clip, pixel_values_flat

                # Sample noise
                noise = torch.randn_like(latents)

                # Add noise offset for better training stability
                if args.noise_offset > 0:
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1, 1),
                        device=latents.device
                    )

                # Sample random timesteps
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=latents.device,
                ).long()

                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Prepare conditional latents for FramerTurbo
                # The model expects: [noisy_latents (4ch), conditional_latents (4ch), mask (1ch)] = 9 channels
                with torch.no_grad():
                    # Get first and last frame latents (clean, without noise)
                    first_frame_latent = latents[:, 0:1, :, :, :]  # (B, 1, 4, H, W)
                    last_frame_latent = latents[:, -1:, :, :, :]   # (B, 1, 4, H, W)

                    # Get mask token from UNet and create middle frames
                    num_frames = latents.shape[1]
                    _, _, latent_c, latent_h, latent_w = latents.shape

                    # Access mask_token from the base model (unwrap if using PEFT)
                    if hasattr(unet, 'get_base_model'):
                        mask_token = unet.get_base_model().mask_token
                    else:
                        mask_token = unet.mask_token

                    # Create mask tokens for middle frames
                    if num_frames > 2:
                        conditional_latents_mask = mask_token.repeat(batch_size, num_frames - 2, 1, latent_h, latent_w)
                        # Concatenate: first + middle(mask) + last
                        conditional_latents = torch.cat([first_frame_latent, conditional_latents_mask, last_frame_latent], dim=1)
                    else:
                        # If only 2 frames, just use first and last
                        conditional_latents = torch.cat([first_frame_latent, last_frame_latent], dim=1)

                    # Create mask channel: 0 for known frames (first and last), 1 for unknown
                    mask_channel = torch.ones((batch_size, num_frames, 1, latent_h, latent_w),
                                             dtype=conditional_latents.dtype,
                                             device=conditional_latents.device)
                    mask_channel[:, 0:1, :, :, :] = 0  # First frame is known
                    mask_channel[:, -1:, :, :, :] = 0  # Last frame is known

                    # Concatenate conditional latents with mask channel: (B, F, 5, H, W)
                    conditional_latents = torch.cat([conditional_latents, mask_channel], dim=2)

                # Concatenate noisy latents with conditional latents: (B, F, 9, H, W)
                latent_model_input = torch.cat([noisy_latents, conditional_latents], dim=2)

                # Prepare added_time_ids for SVD (fps, motion_bucket_id, noise_aug_strength)
                # Using default values for training
                fps = 7  # SVD default
                motion_bucket_id = 127  # Medium motion
                noise_aug_strength = 0.0  # No noise augmentation during training

                add_time_ids = torch.tensor(
                    [[fps, motion_bucket_id, noise_aug_strength]],
                    dtype=latent_model_input.dtype,
                    device=latent_model_input.device,
                )
                add_time_ids = add_time_ids.repeat(batch_size, 1)

                # Predict noise with UNet
                # For now, we skip ControlNet conditioning to simplify
                # You can add trajectory point conditioning here if needed

                model_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=torch.cat([first_frame_embeds, last_frame_embeds], dim=1),
                    added_time_ids=add_time_ids,
                ).sample

                # Calculate loss
                if args.prediction_type == "epsilon":
                    target = noise
                elif args.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {args.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = trainable_params
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Clear CUDA cache periodically to prevent fragmentation
                if global_step % 10 == 0:
                    torch.cuda.empty_cache()

                # Logging
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }
                progress_bar.set_postfix(**logs)

                if args.use_wandb and accelerator.is_main_process:
                    wandb.log(logs, step=global_step)

                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)

                        # Save LoRA weights
                        if args.train_unet:
                            unwrapped_unet = accelerator.unwrap_model(unet)
                            unwrapped_unet.save_pretrained(os.path.join(save_path, "unet_lora"))

                        if args.train_controlnet:
                            unwrapped_controlnet = accelerator.unwrap_model(controlnet)
                            torch.save(
                                unwrapped_controlnet.state_dict(),
                                os.path.join(save_path, "controlnet.pth")
                            )

                        logger.info(f"Saved checkpoint to {save_path}")

            if global_step >= max_train_steps:
                break

    # Save final model
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "final")
        os.makedirs(save_path, exist_ok=True)

        if args.train_unet:
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_unet.save_pretrained(os.path.join(save_path, "unet_lora"))

        if args.train_controlnet:
            unwrapped_controlnet = accelerator.unwrap_model(controlnet)
            torch.save(
                unwrapped_controlnet.state_dict(),
                os.path.join(save_path, "controlnet.pth")
            )

        logger.info(f"Saved final model to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
