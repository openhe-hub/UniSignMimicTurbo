#!/bin/bash
#SBATCH --job-name=Framer
#SBATCH --output=logs/train_%j.out        
#SBATCH --error=logs/train_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=nvidia

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export CUDA_VISIBLE_DEVICES=0        # let Slurm handle GPU binding

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "Using input_dir=${INPUT_DIR}"
echo "Output to ${OUTPUT_DIR}"

# Batch inference; default use_sift=1. Disable with --use_sift 0 if needed.
# High quality settings for better output quality
./scripts/train_lora.sh

echo "Job completed at: $(date)"
