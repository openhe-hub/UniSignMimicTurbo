# UniSignMimicTurbo

Sign language video generation and interpolation system combining MimicMotion and FramerTurbo with LoRA fine-tuning.

## 📁 Project Structure

```
UniSignMimicTurbo/
│
├── FramerTurbo/              # Frame interpolation with LoRA fine-tuning
│   ├── training/             # LoRA training code
│   ├── models_diffusers/     # Custom diffusion models
│   ├── pipelines/            # Inference pipelines
│   ├── apps/                 # Gradio applications
│   ├── scripts/              # Training and inference scripts
│   └── docs/                 # Complete documentation
│
├── mimicmotion/              # MimicMotion core library
│   └── ...                   # Video generation based on pose
│
├── rtmlib/                   # RTM pose estimation library
│   └── ...                   # Real-time pose detection
│
├── scripts/                  # Utility scripts
│   ├── inference/            # Inference scripts (moved from root)
│   │   ├── inference.py
│   │   ├── inference_batch.py
│   │   ├── inference_raw_batch.py
│   │   ├── inference_raw_batch_cache.py
│   │   ├── inference_raw_batch_turbo.py
│   │   ├── predict.py
│   │   └── ...
│   ├── word/                 # Word-level processing
│   ├── sentence/             # Sentence-level processing
│   ├── rtm-demo/             # RTM demonstration scripts
│   └── slurm/                # Cluster job scripts
│
├── configs/                  # Configuration files
│   ├── constants.py          # Global constants (ASPECT_RATIO, etc.)
│   └── test.yaml             # Test configurations
│
├── assets/                   # Resource files
│   └── ...                   # Images, videos, test data
│
├── output/                   # Output directory
│   └── ...                   # Generated results
│
└── doc/                      # Project documentation
    └── ...
```

## 🚀 Quick Start

### FramerTurbo Training

See [FramerTurbo/README.md](FramerTurbo/README.md) for complete training documentation.

```bash
cd FramerTurbo
bash scripts/train/train_lora.sh
```

### Inference

```bash
# Single video inference
python scripts/inference/inference.py

# Batch inference
python scripts/inference/inference_batch.py

# With caching (faster)
python scripts/inference/inference_raw_batch_cache.py
```

## 📚 Documentation

- **FramerTurbo Training Guide**: [FramerTurbo/docs/TRAINING.md](FramerTurbo/docs/TRAINING.md)
- **LoRA Principles**: [FramerTurbo/docs/LORA_PRINCIPLES.md](FramerTurbo/docs/LORA_PRINCIPLES.md)
- **Project Structure**: [FramerTurbo/docs/STRUCTURE.md](FramerTurbo/docs/STRUCTURE.md)
- **Scripts Guide**: [scripts/README.md](scripts/README.md)

## 🔧 Key Components

### FramerTurbo
- **Purpose**: Frame interpolation with LoRA fine-tuning
- **Tech**: Diffusion models, PEFT, PyTorch
- **GPU**: A100 40GB (BF16 mixed precision)
- **Dataset**: 351 sign language videos at 576×576

### MimicMotion
- **Purpose**: Pose-driven video generation
- **Input**: Pose sequences
- **Output**: Animated sign language videos

### RTMLib
- **Purpose**: Real-time pose estimation
- **Models**: RTMPose, RTMDet
- **Usage**: Extract pose from videos

## 📝 Recent Changes

- Organized inference scripts into `scripts/inference/`
- Moved `constants.py` to `configs/` directory
- Updated all imports: `from configs.constants import ASPECT_RATIO`
- Added comprehensive LoRA training documentation

## 🛠️ Development

All inference scripts should be executed from project root:
```bash
python scripts/inference/<script_name>.py
```

Configuration constants are in `configs/constants.py`.

---

**Project**: Sign Language Video Generation System
**Last Updated**: 2025-12-10
