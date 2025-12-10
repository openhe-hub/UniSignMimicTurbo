# FramerTurbo LoRA Fine-tuning Guide

This is the LoRA fine-tuning training code for FramerTurbo. Supports efficient fine-tuning on custom datasets.

## 📋 Features

- ✅ **Efficient LoRA Fine-tuning**: Uses PEFT library, memory-friendly (~16-24GB)
- ✅ **Multiple Data Formats**: Supports video files or image pairs
- ✅ **Mixed Precision Training**: Supports FP16/BF16
- ✅ **Gradient Accumulation**: Enables training on limited memory
- ✅ **Accelerate Integration**: Supports single/multi-GPU training
- ✅ **Flexible Configuration**: Optional training for UNet and/or ControlNet

## 🚀 Quick Start

> **Important**: Run all commands from the FramerTurbo project root directory!

### 1. Install Dependencies

```bash
# From project root directory
pip install -r requirements.txt
pip install accelerate peft wandb
```

### 2. Prepare Dataset

#### Option A: Video Files (Recommended)

Place video files in a directory:

```
data/training_videos/
    video_001.mp4
    video_002.mp4
    video_003.mp4
    ...
```

#### Option B: Image Pairs

Pair start and end frames:

```
data/image_pairs/
    sample_001_start.jpg
    sample_001_end.jpg
    sample_002_start.jpg
    sample_002_end.jpg
    ...
```

### 3. Configure Training Script

Edit `scripts/train/train_lora.sh`:

```bash
# Modify data path
DATA_DIR="data/training_videos"  # Your data directory

# Choose dataset type
DATASET_TYPE="video"  # or "image_pair"

# Adjust training parameters
BATCH_SIZE=1          # Adjust based on GPU memory
GRADIENT_ACCUM=4      # Effective batch size = BATCH_SIZE × GRADIENT_ACCUM
EPOCHS=10             # Number of epochs
LEARNING_RATE=1e-4    # Learning rate
```

### 4. Start Training

```bash
cd FramerTurbo
bash scripts/train/train_lora.sh
```

## ⚙️ Training Parameters

### Core Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--pretrained_model_path` | Pre-trained model path | `checkpoints/framer_512x320` |
| `--data_dir` | Training data directory | - |
| `--output_dir` | Output directory | - |
| `--dataset_type` | Dataset type: `video` or `image_pair` | `video` |

### LoRA Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--lora_rank` | LoRA rank (higher = better quality but more memory) | 64 |
| `--lora_alpha` | LoRA alpha (usually equals rank) | 64 |
| `--lora_dropout` | LoRA dropout | 0.0 |
| `--train_unet` | Train UNet (required) | ✓ |
| `--train_controlnet` | Train ControlNet (optional) | - |

### Training Hyperparameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `--train_batch_size` | Batch size per GPU | 1 |
| `--gradient_accumulation_steps` | Gradient accumulation steps | 4-8 |
| `--num_train_epochs` | Number of epochs | 10-20 |
| `--learning_rate` | Learning rate | 1e-4 |
| `--mixed_precision` | Mixed precision: `fp16` or `bf16` | `fp16` |

## 💾 Memory Requirements

| Configuration | Memory Required | Notes |
|--------------|-----------------|-------|
| LoRA (rank=64) + UNet | ~16-20 GB | Recommended |
| LoRA (rank=128) + UNet | ~20-24 GB | Higher quality |
| LoRA + UNet + ControlNet | ~24-32 GB | Full training |
| Full fine-tuning | ~40+ GB | Best quality |

**Memory Saving Tips**:
- Reduce `--lora_rank` (e.g., 32)
- Reduce `--train_batch_size` and increase `--gradient_accumulation_steps`
- Use `--mixed_precision fp16`
- Reduce `--num_frames` (e.g., 3 → 2)

## 📊 Monitor Training

### TensorBoard

```bash
tensorboard --logdir logs
```

### Weights & Biases

```bash
# Add to training script
USE_WANDB="--use_wandb"
```

## 🔄 Use Trained Model

### Load LoRA Weights

```python
from peft import PeftModel
from models_diffusers.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

# Load base model
unet = UNetSpatioTemporalConditionModel.from_pretrained(
    "checkpoints/framer_512x320/unet",
    torch_dtype=torch.float16,
)

# Load LoRA weights
unet = PeftModel.from_pretrained(
    unet,
    "outputs/lora_finetune/final/unet_lora",
)

# Merge LoRA (optional, for inference speedup)
unet = unet.merge_and_unload()
```

### Inference Example

```python
# Integrate fine-tuned UNet into inference pipeline
pipe = StableVideoDiffusionInterpControlPipeline.from_pretrained(
    "checkpoints/stable-video-diffusion-img2vid-xt",
    unet=unet,  # Use fine-tuned UNet
    controlnet=controlnet,
    torch_dtype=torch.float16,
)

# Normal inference
frames = pipe(
    first_image,
    last_image,
    num_frames=3,
    ...
).frames
```

## 🛠️ Advanced Usage

### Multi-GPU Training

Use Accelerate configuration:

```bash
accelerate config  # Configure multi-GPU settings
accelerate launch training/train_lora.py ...  # Automatic multi-GPU training
```

### Resume from Checkpoint

```bash
python training/train_lora.py \
  --resume_from_checkpoint outputs/lora_finetune/checkpoint-1000 \
  ...
```

### Train ControlNet Only

```bash
python training/train_lora.py \
  --train_controlnet \  # Train only ControlNet
  --learning_rate 5e-5 \  # Smaller learning rate recommended for ControlNet
  ...
```

### Mixed Training (UNet LoRA + Full ControlNet)

```bash
python training/train_lora.py \
  --train_unet \
  --train_controlnet \
  --lora_rank 64 \
  ...
```

## 📁 Output Structure

```
outputs/lora_finetune/
├── checkpoint-500/
│   └── unet_lora/          # LoRA weights checkpoint
├── checkpoint-1000/
│   └── unet_lora/
└── final/
    └── unet_lora/          # Final LoRA weights
        ├── adapter_config.json
        └── adapter_model.safetensors
```

## ❓ FAQ

### Q: Out of memory during training?

A: Try these solutions:
1. Reduce `--train_batch_size` to 1
2. Increase `--gradient_accumulation_steps` to 8 or higher
3. Reduce `--lora_rank` to 32 or 16
4. Reduce `--num_frames` to 2

### Q: How many training steps are appropriate?

A: Depends on dataset size:
- Small dataset (< 100 videos): 10-20 epochs
- Medium dataset (100-1000 videos): 5-10 epochs
- Large dataset (> 1000 videos): 2-5 epochs

Recommend saving checkpoints every 500 steps, choose the best model based on validation performance.

### Q: How to adjust learning rate?

A: Recommended values:
- LoRA training: `1e-4` to `5e-5`
- ControlNet training: `5e-5` to `1e-5`
- If loss doesn't decrease, try increasing learning rate
- If loss oscillates, try decreasing learning rate

### Q: Support for custom trajectory annotations?

A: The current training code doesn't integrate trajectory point annotations. To train trajectory control:
1. Prepare dataset with trajectory annotations
2. Modify `train_dataset.py` to load trajectory data
3. Add ControlNet conditioning in `train_lora.py`

We plan to add full trajectory annotation training support in future versions.

## 📚 References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate)
- [FramerTurbo Paper](https://arxiv.org/abs/2410.18978)

## 📝 License

This training code follows the FramerTurbo license.
