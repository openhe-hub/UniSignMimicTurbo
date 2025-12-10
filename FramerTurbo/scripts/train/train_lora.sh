#!/bin/bash

# FramerTurbo LoRA Fine-tuning Script
# Optimized for A100 GPU with 576x576 resolution

# ============================================================================
# Configuration
# ============================================================================

# Paths
PRETRAINED_MODEL="checkpoints/framer_512x320"
SVD_MODEL="checkpoints/stable-video-diffusion-img2vid-xt"
DATA_DIR="assets/AslToHiya-01"  # Change this to your data directory
OUTPUT_DIR="outputs/lora_576x576"

# Dataset settings - 576x576 for sign language videos
DATASET_TYPE="video"  # Options: "video" or "image_pair"
NUM_FRAMES=5
HEIGHT=576    # Optimized for your 576x576 videos
WIDTH=576     # Optimized for your 576x576 videos

# Training hyperparameters - A100 optimized for 576x576 with gradient checkpointing
BATCH_SIZE=1              # Reduced for 576x576 resolution
GRADIENT_ACCUM=8          # Effective batch = 1 x 8 = 8
EPOCHS=30                 # Train for 30 epochs first
LEARNING_RATE=1e-4
LR_SCHEDULER="constant_with_warmup"
WARMUP_STEPS=500

# LoRA settings - Reduced rank to save memory
LORA_RANK=64
LORA_ALPHA=64             # Match rank
LORA_DROPOUT=0.0

# Mixed precision - BF16 for A100
MIXED_PRECISION="bf16"    # A100 supports BF16 better than FP16

# Checkpointing - More frequent saves
CHECKPOINT_STEPS=200      # Save every 200 steps (was 500)

# Logging
USE_WANDB=""  # Add "--use_wandb" to enable W&B logging
WANDB_PROJECT="framer-turbo-lora-576"

# GPU settings
NUM_WORKERS=8             # A100 can handle more workers
SEED=42

# ============================================================================
# Launch Training
# ============================================================================

echo "========================================="
echo "FramerTurbo LoRA Fine-tuning (A100)"
echo "========================================="
echo "Model: $PRETRAINED_MODEL"
echo "Resolution: ${WIDTH}x${HEIGHT}"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE x $GRADIENT_ACCUM (effective: $((BATCH_SIZE * GRADIENT_ACCUM)))"
echo "LoRA rank: $LORA_RANK"
echo "Mixed precision: $MIXED_PRECISION"
echo "========================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

accelerate launch training/train_lora.py \
  --pretrained_model_path "$PRETRAINED_MODEL" \
  --svd_model_path "$SVD_MODEL" \
  --data_dir "$DATA_DIR" \
  --dataset_type "$DATASET_TYPE" \
  --output_dir "$OUTPUT_DIR" \
  --num_frames $NUM_FRAMES \
  --height $HEIGHT \
  --width $WIDTH \
  --train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACCUM \
  --num_train_epochs $EPOCHS \
  --learning_rate $LEARNING_RATE \
  --lr_scheduler "$LR_SCHEDULER" \
  --lr_warmup_steps $WARMUP_STEPS \
  --lora_rank $LORA_RANK \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --train_unet \
  --checkpointing_steps $CHECKPOINT_STEPS \
  --mixed_precision "$MIXED_PRECISION" \
  --num_workers $NUM_WORKERS \
  --seed $SEED \
  $USE_WANDB

echo ""
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
