#!/bin/bash
#SBATCH --job-name=Framer-2GPU
#SBATCH --output=logs/train_2gpu_%j.out
#SBATCH --error=logs/train_2gpu_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=nvidia

# ============================================================================
# FramerTurbo 2-GPU Training on Slurm
# Requires: 2x A100-40GB or 2x A100-80GB
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
echo ""

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Master address: $MASTER_ADDR:$MASTER_PORT"
echo ""

echo "========================================="
echo "Starting Training"
echo "========================================="

# Launch 2-GPU training
./scripts/train/train_lora_2gpu.sh

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
