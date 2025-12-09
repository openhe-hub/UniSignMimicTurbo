#!/bin/bash
#SBATCH --job-name=Framer
#SBATCH --output=logs/train_%j.out        
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=nvidia

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export CUDA_VISIBLE_DEVICES=0        # let Slurm handle GPU binding

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv


INPUT_DIR="assets/test01"
OUTPUT_DIR="outputs/test01"
MODEL_DIR="checkpoints/framer_512x320"

echo "Using input_dir=${INPUT_DIR}"
echo "Output to ${OUTPUT_DIR}"

# Batch inference; default use_sift=1. Disable with --use_sift 0 if needed.
# High quality settings for better output quality
/home/zl6890/.conda/envs/framer_py38/bin/python cli_infer.py \
  --input_dir "${INPUT_DIR}" \
  --model "${MODEL_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_sift 0 \
  --num_inference_steps 50 \
  --min_guidance_scale 2.0 \
  --max_guidance_scale 4.0 \
  --controlnet_cond_scale 1.5 \
  --noise_aug 0.01 \
  --num_frames 14

echo "Job completed at: $(date)"
