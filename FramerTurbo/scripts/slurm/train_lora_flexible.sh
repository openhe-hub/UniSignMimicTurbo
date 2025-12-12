#!/bin/bash
#SBATCH --job-name=Framer-Flex
#SBATCH --output=logs/train_flex_%j.out
#SBATCH --error=logs/train_flex_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --partition=nvidia

# ============================================================================
# FramerTurbo Flexible Multi-GPU Training
# Automatically adapts to available GPUs (A100, V100, or mixed)
# ============================================================================

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "========================================="
echo "Job Info"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo ""

echo "========================================="
echo "GPU Info"
echo "========================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# Detect GPU types and memory
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
echo ""
echo "Detected GPUs:"
echo "$GPU_INFO"
echo ""

# Check if we have V100 or mixed GPUs with less memory
if echo "$GPU_INFO" | grep -q "V100"; then
    echo "⚠️  V100 detected - Using ZeRO-3 for memory optimization"
    DEEPSPEED_CONFIG="zero3"
elif echo "$GPU_INFO" | grep -q "32768 MiB"; then
    echo "⚠️  32GB GPU detected - Using ZeRO-3 for memory optimization"
    DEEPSPEED_CONFIG="zero3"
else
    echo "✅ Using ZeRO-2 (faster, sufficient for A100-40GB+)"
    DEEPSPEED_CONFIG="zero2"
fi

# Check for mixed GPU types
GPU_MODELS=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | wc -l)
if [ $GPU_MODELS -gt 1 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚠️  WARNING: Mixed GPU types detected!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    nvidia-smi --query-gpu=index,name,memory.total --format=table
    echo ""
    echo "Mixed GPUs can cause issues:"
    echo "  - Training speed limited by slowest GPU"
    echo "  - Potential communication problems"
    echo "  - Memory imbalance between GPUs"
    echo ""
    echo "Recommended: Request homogeneous GPUs"
    echo "Example: --gres=gpu:a100:4 or --gres=gpu:v100:4"
    echo ""
    echo "Continuing anyway in 10 seconds..."
    echo "Press Ctrl+C to cancel"
    sleep 10
fi

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

echo ""
echo "Master address: $MASTER_ADDR:$MASTER_PORT"
echo "Using DeepSpeed config: $DEEPSPEED_CONFIG"
echo ""

echo "========================================="
echo "Starting Training"
echo "========================================="

# Dynamically select training script based on GPU count
NUM_GPUS=$SLURM_GPUS_ON_NODE

if [ -f "scripts/train/train_lora_${NUM_GPUS}gpu.sh" ]; then
    # Temporarily update the DeepSpeed config if needed
    if [ "$DEEPSPEED_CONFIG" = "zero3" ]; then
        sed -i 's/deepspeed_zero2.json/deepspeed_zero3.json/' "scripts/train/train_lora_${NUM_GPUS}gpu.sh"
    fi

    ./scripts/train/train_lora_${NUM_GPUS}gpu.sh

    # Restore original config
    if [ "$DEEPSPEED_CONFIG" = "zero3" ]; then
        sed -i 's/deepspeed_zero3.json/deepspeed_zero2.json/' "scripts/train/train_lora_${NUM_GPUS}gpu.sh"
    fi
else
    echo "Error: No training script found for $NUM_GPUS GPUs"
    echo "Available scripts: train_lora_{2,3,4}gpu.sh"
    exit 1
fi

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
