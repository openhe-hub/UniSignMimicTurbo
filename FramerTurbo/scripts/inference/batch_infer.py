"""
Quick launcher for batch inference
"""
import subprocess
import sys

# Configuration
LORA_WEIGHTS = "outputs/lora_576x576/final/unet_lora"
BASE_MODEL = "checkpoints/framer_512x320"
SVD_MODEL = "checkpoints/stable-video-diffusion-img2vid-xt"

# Parse command line
if len(sys.argv) < 2:
    print("Usage: python scripts/inference/batch_infer.py <input_dir> [output_dir]")
    print("")
    print("Example:")
    print("  python scripts/inference/batch_infer.py assets/test01")
    print("  python scripts/inference/batch_infer.py assets/test01 outputs/test01_custom")
    sys.exit(1)

input_dir = sys.argv[1]
output_dir = sys.argv[2] if len(sys.argv) > 2 else f"outputs/{input_dir.split('/')[-1]}"

print(f"[INFO] Input: {input_dir}")
print(f"[INFO] Output: {output_dir}")
print(f"[INFO] LoRA: {LORA_WEIGHTS}")
print("")

# Run batch inference
cmd = [
    "python", "training/batch_infer_with_lora.py",
    "--lora_weights", LORA_WEIGHTS,
    "--base_model", BASE_MODEL,
    "--svd_model", SVD_MODEL,
    "--input_dir", input_dir,
    "--output_dir", output_dir,
    "--height", "576",
    "--width", "576",
    "--num_frames", "3",  # Must match training config (NUM_FRAMES=3)
    "--num_inference_steps", "15",
    "--merge_lora",
]

subprocess.run(cmd)
