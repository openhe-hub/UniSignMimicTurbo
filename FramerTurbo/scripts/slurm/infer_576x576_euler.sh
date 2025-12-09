#!/bin/bash
#SBATCH --job-name=Framer576
#SBATCH --output=framer_576_%j.out
#SBATCH --error=framer_576_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=nvidia

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

# ============================================================================
# Configuration
# ============================================================================

INPUT_DIR="assets/test02"
OUTPUT_DIR="outputs/test02"
MODEL_DIR="checkpoints/framer_512x320"

echo "Configuration:"
echo "  Input:  $INPUT_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Model:  $MODEL_DIR"
echo ""

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
