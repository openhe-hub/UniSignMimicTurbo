# FramerTurbo Directory Structure

```
FramerTurbo/
│
├── README.md                    # Main project documentation
├── requirements.txt             # Python dependencies
├── .gitignore
├── docs/
│   ├── STRUCTURE.md             # This file - Directory structure
│   ├── QUICKSTART.md            # Quick start guide
│   ├── TRAINING.md              # Training documentation
│   └── DATA_PREPARATION.md      # Data preparation guide
│
├── models_diffusers/            # Model definitions
│   ├── unet_spatio_temporal_condition.py
│   ├── controlnet_svd.py
│   ├── attention.py
│   ├── attention_processor.py
│   ├── transformer_temporal.py
│   ├── unet_3d_blocks.py
│   ├── lcm_scheduler.py
│   ├── sift_match.py
│   └── utils.py
│
├── pipelines/                   # Inference pipelines
│   └── pipeline_stable_video_diffusion_interp_control.py
│
├── gradio_demo/                 # Gradio demo utilities
│   └── utils_drag.py
│
├── apps/                        # Gradio applications
│   ├── app.py                  # Original version
│   ├── app_turbo.py            # Turbo version
│   └── app_turbo_v2.py         # Turbo v2 version (latest)
│
├── scripts/                     # Scripts directory
│   ├── inference/              # Inference scripts
│   │   ├── cli_infer.py               # Basic inference
│   │   ├── cli_infer_turbo.py         # Turbo inference
│   │   ├── cli_infer_turbo_v2.py      # Turbo v2 inference (recommended)
│   │   └── cli_infer_576x576.py       # High-resolution inference
│   ├── slurm/                  # SLURM cluster scripts
│   │   └── infer_576x576_euler.sh
│   ├── train/                  # Training scripts
│   │   └── train_lora.sh       # LoRA training launch script
│   └── eval/                   # Evaluation scripts
│       └── parse_training_log.py
│
├── training/                    # Training code (LoRA fine-tuning)
│   ├── train_lora.py           # LoRA training main script
│   ├── train_dataset.py        # Dataset definition
│   ├── train_config.py         # Training configuration example
│   ├── infer_with_lora.py      # LoRA model inference script
│   ├── batch_infer_with_lora.py # Batch inference with LoRA
│   └── validate_on_trainset.py  # Validation on training data
│
└── assets/                      # Asset files
    └── logo/
        └── framer.png
```

## 📁 Directory Descriptions

### Core Modules

- **models_diffusers/** - Custom Diffusers model components
  - UNet, ControlNet implementations
  - Custom attention mechanisms and schedulers

- **pipelines/** - Inference pipelines
  - SVD interpolation pipeline integrated with ControlNet

### Applications and Scripts

- **apps/** - Gradio interactive applications
  - `app_turbo_v2.py` is the latest version, supporting multiple schedulers

- **scripts/** - Various scripts
  - `inference/` - Command-line inference scripts
    - Recommended: `cli_infer_turbo_v2.py`
  - `slurm/` - Cluster job scripts
  - `train/` - Training scripts
    - `train_lora.sh` - Training launch script
  - `eval/` - Evaluation and analysis scripts

### Training Module

- **training/** - LoRA fine-tuning
  - Complete training code and documentation
  - Supports video files and image pair datasets
  - See `docs/TRAINING.md` for details

### Documentation

- **docs/** - Project documentation
  - `QUICKSTART.md` - Quick start guide
  - `STRUCTURE.md` - This file
  - `TRAINING.md` - Complete training tutorial
  - `DATA_PREPARATION.md` - Data preparation guide

## 🚀 Quick Usage

### Inference
```bash
# Recommended: Use Turbo v2 version
python scripts/inference/cli_infer_turbo_v2.py \
    --input_dir assets/pairs \
    --model checkpoints/framer_512x320/ \
    --output_dir outputs
```

### Training
```bash
# View training documentation
cat docs/TRAINING.md

# Start LoRA training
bash scripts/train/train_lora.sh
```

### Gradio Application
```bash
# Launch Turbo v2 app
python apps/app_turbo_v2.py
```

## 📝 Version Notes

- **Basic Version** (`app.py`, `cli_infer.py`): Original implementation using Euler scheduler
- **Turbo Version** (`app_turbo.py`, `cli_infer_turbo.py`): Added LCM scheduler support
- **Turbo v2 Version** (`app_turbo_v2.py`, `cli_infer_turbo_v2.py`): Supports Euler/DPM++/LCM multiple schedulers (recommended)

## 🔄 Migration Guide

If you previously used scripts in the root directory, please update paths:

- `cli_infer_turbo_v2.py` → `scripts/inference/cli_infer_turbo_v2.py`
- `app_turbo_v2.py` → `apps/app_turbo_v2.py`
- `train_lora.py` → `training/train_lora.py`
