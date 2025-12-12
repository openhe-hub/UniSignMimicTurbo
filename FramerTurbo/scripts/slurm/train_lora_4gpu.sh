#!/bin/bash
#SBATCH --job-name=Framer-4GPU
#SBATCH --output=logs/train_4gpu_%j.out
#SBATCH --error=logs/train_4gpu_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=nvidia

# ============================================================================
# FramerTurbo Multi-GPU Training on Slurm
# Requires: 4x A100-40GB or 4x A100-80GB
# For V100-32GB: Change deepspeed config to zero3 in train_lora_multigpu.sh
# ============================================================================

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "========================================="
echo "Job Info"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "Number of tasks: $SLURM_NTASKS"
echo ""

echo "========================================="
echo "GPU Info"
echo "========================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Set master address for distributed training
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Master address: $MASTER_ADDR:$MASTER_PORT"
echo ""

echo "========================================="
echo "Starting Training"
echo "========================================="

# Launch training
./scripts/train/train_lora_multigpu.sh

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
