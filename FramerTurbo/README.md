# FramerTurbo

LoRA Fine-tuning for Frame Interpolation

## 🎯 Overview

This project implements LoRA fine-tuning pipeline for FramerTurbo model, specialized for video frame interpolation tasks.

### Tech Stack

- **Base Model**: FramerTurbo (based on Stable Video Diffusion)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) via PEFT library
- **Deep Learning Framework**: PyTorch 2.x + CUDA
- **Diffusion Library**: Diffusers 0.25.0
- **Mixed Precision**: BF16 (A100 GPU)
- **Model Components**:
  - UNet3D (Spatio-temporal conditional UNet with LoRA)
  - ControlNet (Control network)
  - VAE (Variational autoencoder)
  - CLIP (Image encoder)

### Key Features

- **Parameter Efficient**: LoRA rank=64, trains only ~0.5% of model parameters
- **Memory Optimized**: Gradient checkpointing + mixed precision, supports 576x576 video on 40GB A100
- **Flexible Inference**: Adjustable LoRA weight coefficient (lora_scale)
- **Complete Toolkit**: Training, inference, validation, and log analysis

## 📚 Documentation

### Getting Started
- **[Quickstart Guide](docs/QUICKSTART.md)** - 5-minute quick start
  - Inference usage
  - Training basics
  - Common commands

### Project Docs
- **[Project Structure](docs/STRUCTURE.md)** - Directory organization
  - Directory functions
  - File descriptions
  - Version comparison

### Training Docs
- **[LoRA Training Guide](docs/TRAINING.md)** - Complete training tutorial
  - Installation & setup
  - Data preparation
  - Training parameters
  - Troubleshooting
  - Advanced usage

- **[Data Preparation Guide](docs/DATA_PREPARATION.md)** - Dataset preparation
  - Video format requirements
  - Data organization
  - Quality recommendations
  - Sign language specific tips

### Technical Docs
- **[LoRA Principles](docs/LORA_PRINCIPLES.md)** - Deep dive into LoRA fine-tuning
  - Mathematical foundation
  - Diffusion model integration
  - Architecture details
  - Training implementation
  - Experimental analysis

## 🚀 Quick Navigation

### I want to...

**...start using FramerTurbo**
→ See [docs/QUICKSTART.md](docs/QUICKSTART.md)

**...understand project structure**
→ See [docs/STRUCTURE.md](docs/STRUCTURE.md)

**...train my own model**
→ See [docs/TRAINING.md](docs/TRAINING.md)

**...understand LoRA principles**
→ See [docs/LORA_PRINCIPLES.md](docs/LORA_PRINCIPLES.md)

**...learn about scheduler differences**
→ See [docs/QUICKSTART.md - Scheduler Comparison](docs/QUICKSTART.md#⚙️-调度器对比)

**...troubleshoot training issues**
→ See [docs/TRAINING.md - FAQ](docs/TRAINING.md#❓-常见问题)

## 📖 Additional Resources

### Code Documentation
- `models_diffusers/` - Model implementations
- `pipelines/` - Pipeline implementations
- `training/` - Training code

### Configuration Examples
- `training/train_config.py` - Training configuration example
- `scripts/train_lora.sh` - Training launch script

## 🔗 External Links

- [Paper](https://arxiv.org/abs/2410.18978)
- [Project Page](https://aim-uofa.github.io/Framer)
- [Hugging Face Demo](https://huggingface.co/spaces/wwen1997/Framer)

---

**Note**: All documentation is accessible from project root, e.g., `cat docs/QUICKSTART.md`
