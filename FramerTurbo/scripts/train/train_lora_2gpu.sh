#!/bin/bash

# FramerTurbo LoRA Fine-tuning Script - 2-GPU with DeepSpeed
# Optimized for 2x A100/V100 GPUs with 576x576 resolution @ 10 frames

# ============================================================================
# Configuration
# ============================================================================

# Paths
PRETRAINED_MODEL="checkpoints/framer_512x320"
SVD_MODEL="checkpoints/stable-video-diffusion-img2vid-xt"
DATA_DIR="assets/AslToHiya-01"
OUTPUT_DIR="outputs/lora_576x576_10f_2gpu"

# Dataset settings - 576x576 @ 10 frames
DATASET_TYPE="video"
NUM_FRAMES=10
HEIGHT=576
WIDTH=576

# Training hyperparameters - 2-GPU optimized
NUM_GPUS=2
BATCH_SIZE=1              # Per GPU batch size
GRADIENT_ACCUM=4          # Effective batch = 2 GPUs x 1 x 4 = 8 (same as 4-GPU setup)
EPOCHS=30
LEARNING_RATE=1e-4
LR_SCHEDULER="constant_with_warmup"
WARMUP_STEPS=500

# LoRA settings - Keep full rank for quality
LORA_RANK=64
LORA_ALPHA=64
LORA_DROPOUT=0.0

# Mixed precision - BF16 for A100 / FP16 for V100
MIXED_PRECISION="fp16"

# DeepSpeed config
DEEPSPEED_CONFIG="configs/deepspeed_zero2.json"  # Use zero3 for V100-32GB
ACCELERATE_CONFIG="configs/accelerate_config_2gpu.yaml"

# Checkpointing
CHECKPOINT_STEPS=200

# Logging
USE_WANDB=""  # Add "--use_wandb" to enable W&B logging
WANDB_PROJECT="framer-turbo-lora-10f-2gpu"

# GPU settings
NUM_WORKERS=8
SEED=42

# ============================================================================
# Launch Training
# ============================================================================

echo "========================================="
echo "FramerTurbo LoRA Multi-GPU Training"
echo "========================================="
echo "GPUs: $NUM_GPUS"
echo "Model: $PRETRAINED_MODEL"
echo "Resolution: ${WIDTH}x${HEIGHT} @ ${NUM_FRAMES} frames"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $NUM_GPUS x $BATCH_SIZE x $GRADIENT_ACCUM (effective: $((NUM_GPUS * BATCH_SIZE * GRADIENT_ACCUM)))"
echo "LoRA rank: $LORA_RANK"
echo "DeepSpeed: $DEEPSPEED_CONFIG"
echo "Mixed precision: $MIXED_PRECISION"
echo "========================================="

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

# Set CUDA_HOME for DeepSpeed (modify if your CUDA is in a different location)
module load cuda/12.2.0
export CUDA_HOME=${CUDA_HOME:/share/apps/NYUAD5/cuda/12.2.0/}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA_HOME: $CUDA_HOME"
echo ""

# Launch with accelerate + DeepSpeed
accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes $NUM_GPUS \
  training/train_lora_deepspeed.py \
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
