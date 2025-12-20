#!/bin/bash
#SBATCH --job-name=Framer576
#SBATCH --output=logs/framer_576_%A_%a.out
#SBATCH --error=logs/framer_576_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=nvidia
#SBATCH --array=1-8                    # 8 parts -> 8 subfolders

# ============================================================================
# Framer 576x576 Inference - Euler Scheduler (30 steps)
# ============================================================================

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=============================================================================="
echo "Job started at: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "=============================================================================="
echo ""

IDX=$(printf "%d" "$SLURM_ARRAY_TASK_ID")   # 1..8

# Input/output roots (override via env if needed)
BOUNDARY_ROOT="assets/boundary_frames"
OUTPUT_ROOT="outputs"

# Assign subfolder by array index (part_1 ... part_8)
FOLDERS=(part_1 part_2 part_3 part_4 part_5 part_6 part_7 part_8)
ARR_IDX=$((IDX-1))
INPUT_SUBDIR=${FOLDERS[$ARR_IDX]}

INPUT_DIR="${BOUNDARY_ROOT}/${INPUT_SUBDIR}"
OUTPUT_DIR="${OUTPUT_ROOT}/${INPUT_SUBDIR}"
MODEL_DIR="checkpoints/framer_512x320"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input dir not found: ${INPUT_DIR}"
  exit 1
fi
mkdir -p "${OUTPUT_DIR}"

echo "Using input_dir=${INPUT_DIR}"
echo "Output to ${OUTPUT_DIR}"

# ============================================================================
# Run Inference (High Quality Settings)
# ============================================================================

python cli_infer_576x576.py \
    --input_dir "$INPUT_DIR" \
    --model "$MODEL_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --scheduler euler \
    --num_inference_steps 50 \
    --noise_aug 0.01 \
    --min_guidance_scale 2.0 \
    --max_guidance_scale 4.0 \
    --motion_bucket_id 127

EXIT_CODE=$?

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=============================================================================="
echo "Job completed at: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS"
    echo "Results: $OUTPUT_DIR"
else
    echo "Status: FAILED (exit code: $EXIT_CODE)"
fi
echo "=============================================================================="

exit $EXIT_CODE
