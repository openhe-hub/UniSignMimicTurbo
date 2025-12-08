#!/bin/bash
#SBATCH --job-name=Framer
#SBATCH --output=train_%A_%a.out        # each array task gets its own log
#SBATCH --error=train_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=nvidia
#SBATCH --array=1-1                    # 5 GPUs -> 5 subfolders

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export CUDA_VISIBLE_DEVICES=0        # let Slurm handle GPU binding

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID  ArrayID: $SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

IDX=$(printf "%d" "$SLURM_ARRAY_TASK_ID")   # 1..5

# Assign subfolder by array index
FOLDERS=(W10_000001 W10_000002 W10_000003 W10_000004 W10_000005)
ARR_IDX=$((IDX-1))
INPUT_SUBDIR=${FOLDERS[$ARR_IDX]}

INPUT_DIR="assets/${INPUT_SUBDIR}"
OUTPUT_DIR="outputs/${INPUT_SUBDIR}"
MODEL_DIR="checkpoints/framer_512x320"

echo "Using input_dir=${INPUT_DIR}"
echo "Output to ${OUTPUT_DIR}"

# Batch inference; default use_sift=1. Disable with --use_sift 0 if needed.
/home/zl6890/.conda/envs/framer_py38/bin/python cli_infer.py \
  --input_dir "${INPUT_DIR}" \
  --model "${MODEL_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --use_sift 0

echo "Job completed at: $(date)"
