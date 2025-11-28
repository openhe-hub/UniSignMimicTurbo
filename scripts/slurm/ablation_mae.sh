#!/bin/bash
#SBATCH --job-name=extract_pose
#SBATCH --output=train_%A_%a.out        # ← 每个 array 任务单独日志
#SBATCH --error=train_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=nvidia
#SBATCH --array=1-10                    # ← 一次提交 10 个任务

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export CUDA_VISIBLE_DEVICES=0          # 建议让 Slurm 分配，不手动绑

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID  ArrayID: $SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv

# 若用 conda，非交互脚本里通常需要：
# source ~/.bashrc
# conda activate mst
# 或者：source /path/to/miniconda3/etc/profile.d/conda.sh && conda activate mst

IDX=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")   # ← 01..10
python ablation_mae.py \
  --inference_config configs/mimicmotion/test.yaml \
  --batch_folder assets/part${IDX}

echo "Job completed at: $(date)"
