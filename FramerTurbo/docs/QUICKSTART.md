# FramerTurbo - Quick Start Guide

Welcome to FramerTurbo! This guide helps you quickly understand the project structure and usage.

## 📁 Project Structure

```
FramerTurbo/
├── 📖 README.md                 # Main project documentation
├── 📖 STRUCTURE.md              # Detailed directory structure
├── 📖 QUICKSTART.md             # This file
│
├── 🎨 apps/                     # Gradio interactive apps
│   └── app_turbo_v2.py         # Recommended (supports multiple schedulers)
│
├── 🔧 scripts/                  # Script tools
│   ├── inference/              # Inference scripts
│   │   └── cli_infer_turbo_v2.py  # Recommended
│   ├── slurm/                  # Cluster jobs
│   └── train/
│       └── train_lora.sh       # Training launch script
│
├── 🎓 training/                 # LoRA fine-tuning
│   ├── train_lora.py           # Training script
│   ├── train_dataset.py        # Dataset implementation
│   ├── train_config.py         # Configuration example
│   ├── infer_with_lora.py      # LoRA inference
│   ├── batch_infer_with_lora.py # Batch LoRA inference
│   └── validate_on_trainset.py  # Validation on training data
│
├── 🏗️ models_diffusers/         # Model definitions
├── 🔄 pipelines/                # Pipelines
├── 🎯 gradio_demo/              # Gradio utilities
└── 📦 assets/                   # Example assets
```

## 🚀 Quick Start

### 1️⃣ Inference (Generate Videos)

**Command-line Inference** (Recommended):
```bash
# Using DPM++ scheduler (balanced speed and quality)
python scripts/inference/cli_infer_turbo_v2.py \
    --input_dir assets/test_single \
    --model checkpoints/framer_512x320 \
    --output_dir outputs
```

**Graphical Interface**:
```bash
python apps/app_turbo_v2.py
```

### 2️⃣ Training (Fine-tune Model)

See complete training documentation:
```bash
cat docs/TRAINING.md
```

Quick start training:
```bash
# 1. Prepare data (place videos in data/training_videos/)
# 2. Edit configuration
nano scripts/train/train_lora.sh

# 3. Start training
bash scripts/train/train_lora.sh
```

### 3️⃣ Use Fine-tuned Model

```bash
python training/infer_with_lora.py \
    --lora_weights outputs/lora_finetune/final/unet_lora \
    --start_image examples/start.jpg \
    --end_image examples/end.jpg \
    --output_path output.gif
```

## 📝 Common Commands

### Inference

```bash
# Basic inference (Euler, 30 steps, best quality)
python scripts/inference/cli_infer_turbo_v2.py \
    --input_dir INPUT_DIR \
    --model checkpoints/framer_512x320 \
    --scheduler euler \
    --num_inference_steps 30

# Fast inference (DPM++, 15 steps, recommended)
python scripts/inference/cli_infer_turbo_v2.py \
    --input_dir INPUT_DIR \
    --model checkpoints/framer_512x320 \
    --scheduler dpm++ \
    --num_inference_steps 15

# Ultra-fast inference (LCM, 4 steps)
python scripts/inference/cli_infer_turbo_v2.py \
    --input_dir INPUT_DIR \
    --model checkpoints/framer_512x320 \
    --scheduler lcm \
    --num_inference_steps 4

# High-resolution inference (576x576)
python scripts/inference/cli_infer_576x576.py \
    --input_dir INPUT_DIR \
    --model checkpoints/framer_576x576 \
    --output_dir outputs_hd
```

### Training

```bash
# View training configuration
cat training/train_config.py

# Edit training script
nano scripts/train/train_lora.sh

# Start training
bash scripts/train/train_lora.sh

# Train with custom parameters
python training/train_lora.py \
    --pretrained_model_path checkpoints/framer_512x320 \
    --data_dir data/my_videos \
    --output_dir outputs/my_lora \
    --train_batch_size 2 \
    --lora_rank 128 \
    --train_unet
```

## 📚 Detailed Documentation

- **Project Overview**: [README.md](../README.md)
- **Directory Structure**: [STRUCTURE.md](STRUCTURE.md)
- **Training Guide**: [TRAINING.md](TRAINING.md)

## ⚙️ Scheduler Comparison

| Scheduler | Steps | Speed      | Quality | Use Case |
|-----------|-------|------------|---------|----------|
| Euler     | 30    | Slow       | Best    | Final production |
| DPM++     | 15    | Fast       | Excellent | Daily use (recommended) |
| LCM       | 4-6   | Ultra-fast | Good    | Quick preview |

## 💡 Tips

1. **First-time use**: Start with the GUI (`python apps/app_turbo_v2.py`)
2. **Batch processing**: Use command-line scripts (`scripts/inference/cli_infer_turbo_v2.py`)
3. **Fine-tuning**: Refer to `docs/TRAINING.md` for detailed steps
4. **Out of memory**: Reduce batch_size, use FP16, or use smaller lora_rank

## ❓ FAQ

**Q: How to switch between different app versions?**
```bash
# Original version (Euler)
python apps/app.py

# Turbo v2 (recommended, supports multiple schedulers)
python apps/app_turbo_v2.py
```

**Q: Where should training data be placed?**

Place video files in any directory, then specify the `DATA_DIR` path in the training script.

**Q: How to view all available parameters?**
```bash
python scripts/inference/cli_infer_turbo_v2.py --help
python training/train_lora.py --help
```

## 📞 Get Help

- View detailed documentation: `STRUCTURE.md` and `TRAINING.md`
- Check examples: `assets/` directory
- Review configuration: `training/train_config.py`

---

**Happy Framing! 🎬**
