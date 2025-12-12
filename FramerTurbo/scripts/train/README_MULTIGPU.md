# Multi-GPU Training with DeepSpeed

This directory contains scripts for efficient multi-GPU training of FramerTurbo using DeepSpeed ZeRO optimization.

## 📋 Features

- **DeepSpeed ZeRO-2/3**: Reduce memory usage by 50-70%
- **4-GPU Support**: Train 10-frame 576x576 videos on 4x A100-40GB
- **Slurm Integration**: Ready for HPC cluster deployment
- **Flexible Configuration**: Switch between ZeRO-2 and ZeRO-3 easily

## 🎯 Memory Requirements

### Configuration Comparison

| Setup | Resolution | Frames | GPUs | Memory/GPU | Config |
|-------|-----------|--------|------|------------|--------|
| Single GPU | 576x576 | 5 | 1 | ~20GB | Original |
| Single GPU | 576x576 | 10 | 1 | **~40GB (OOM on A100-40GB)** | Original |
| Multi-GPU ZeRO-2 | 576x576 | 10 | 4 | **~25GB** ✅ | This guide |
| Multi-GPU ZeRO-3 | 576x576 | 10 | 4 | **~18GB** ✅ | This guide |

### Hardware Support

- ✅ **4x A100-80GB**: Use ZeRO-2 (recommended)
- ✅ **4x A100-40GB**: Use ZeRO-2 (fits comfortably)
- ✅ **4x V100-32GB**: Use ZeRO-3 (required)
- ❌ **1x A100-40GB**: Not enough for 10 frames @ 576x576

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install deepspeed
```

### 2. Choose Your Configuration

**For A100-40GB/80GB (Recommended):**
```bash
# Edit scripts/train/train_lora_multigpu.sh
DEEPSPEED_CONFIG="configs/deepspeed_zero2.json"
```

**For V100-32GB:**
```bash
# Edit scripts/train/train_lora_multigpu.sh
DEEPSPEED_CONFIG="configs/deepspeed_zero3.json"
```

### 3. Launch Training

**Local Machine (4 GPUs):**
```bash
cd FramerTurbo
bash scripts/train/train_lora_multigpu.sh
```

**HPC with Slurm:**
```bash
cd FramerTurbo
sbatch scripts/slurm/train_lora_4gpu.sh
```

## 🏗️ Architecture Overview

### How Accelerate and DeepSpeed Work Together

```
┌─────────────────────────────────────────────┐
│              Your Training Code              │
│         (train_lora.py)                      │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           Accelerate (调度层)                │
│  - 检测硬件环境                               │
│  - 统一API                                   │
│  - 自动分布式                                 │
│  - 集成各种后端                               │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐    ┌────────────────┐
│  PyTorch DDP  │    │   DeepSpeed    │
│  (原生多卡)    │    │  (显存优化)     │
└───────────────┘    └────────────────┘
```

### What is Accelerate?

**Accelerate** is a library that simplifies distributed training by providing a unified API.

**What it does:**
- **Auto-detects environment**: Single GPU, multi-GPU, multi-node, TPU
- **Unified API**: Same code works everywhere
- **Gradient accumulation**: Automatic management
- **Mixed precision**: Automatic FP32/FP16/BF16 conversion
- **Checkpoint handling**: Automatic multi-process synchronization

**Without Accelerate (manual):**
```python
# Traditional PyTorch multi-GPU - lots of boilerplate
dist.init_process_group(backend='nccl')
model = DistributedDataParallel(model, device_ids=[local_rank])
sampler = DistributedSampler(dataset)
# ... manual gradient accumulation logic
# ... manual checkpoint saving for multi-process
```

**With Accelerate (automatic):**
```python
accelerator = Accelerator(gradient_accumulation_steps=8)
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
# Everything else is automatic!
```

### What is DeepSpeed?

**DeepSpeed** is a memory optimization framework that makes training large models feasible.

**Key innovation - ZeRO (Zero Redundancy Optimizer):**

Traditional multi-GPU training stores full copies on each GPU (wasteful):
```
┌────────────────────┐
│ GPU 0: 40GB        │  Full model + optimizer on each GPU
│ GPU 1: 40GB        │  ← Redundant!
│ GPU 2: 40GB        │
│ GPU 3: 40GB        │
└────────────────────┘
```

DeepSpeed ZeRO partitions memory across GPUs:
```
┌────────────────────┐
│ GPU 0: 25GB        │  Optimizer & gradients partitioned
│ GPU 1: 25GB        │  ← Saves 15GB per GPU!
│ GPU 2: 25GB        │
│ GPU 3: 25GB        │
└────────────────────┘
```

**ZeRO Stages:**
- **ZeRO-1**: Partition optimizer states → Save ~7% memory
- **ZeRO-2**: Partition optimizer + gradients → Save ~40% memory
- **ZeRO-3**: Partition optimizer + gradients + model → Save ~60% memory (slower)

### Why Use Both?

| Tool | Role | What it solves |
|------|------|----------------|
| **Accelerate** | Orchestrator | "How do I run on multiple GPUs?" |
| **DeepSpeed** | Optimizer | "What if I run out of memory?" |

**In our setup:**
- Accelerate handles distributed coordination
- DeepSpeed reduces memory from 40GB → 25GB per GPU
- Together they enable 10-frame training on 4x A100-40GB

## ⚙️ Configuration Files

### Accelerate Config
`configs/accelerate_config_4gpu.yaml`
- Sets up 4-GPU distributed training
- Enables DeepSpeed integration
- Configures BF16 mixed precision

### DeepSpeed Configs

**ZeRO-2** (`configs/deepspeed_zero2.json`)
- Partitions optimizer states and gradients across GPUs
- Offloads optimizer to CPU
- Best for A100-40GB/80GB
- Memory savings: ~40%

**ZeRO-3** (`configs/deepspeed_zero3.json`)
- Partitions model parameters, gradients, AND optimizer states
- Offloads both optimizer and parameters to CPU
- Best for V100-32GB
- Memory savings: ~60%
- Slightly slower than ZeRO-2

## 📊 Expected Performance

### Training Speed

| GPUs | Config | Steps/sec | Speedup |
|------|--------|-----------|---------|
| 1 GPU | Baseline (5 frames) | 1.0 | 1x |
| 4 GPUs | ZeRO-2 (10 frames) | 3.2 | 3.2x |
| 4 GPUs | ZeRO-3 (10 frames) | 2.8 | 2.8x |

### Memory Usage (10 frames @ 576x576)

| Component | Single GPU | ZeRO-2 (4 GPU) | ZeRO-3 (4 GPU) |
|-----------|-----------|----------------|----------------|
| Model | 4GB | 4GB | 1GB (sharded) |
| Optimizer | 3GB | 0.75GB (sharded) | 0.2GB (sharded + offload) |
| Activations | 25GB | 25GB | 25GB |
| Gradients | 1GB | 0.25GB (sharded) | 0.25GB (sharded) |
| **Total** | **~40GB** | **~25GB** | **~18GB** |

## 🔧 Customization

### Adjust Number of GPUs

Edit `scripts/train/train_lora_multigpu.sh`:
```bash
NUM_GPUS=4  # Change to 2, 4, or 8
```

And `configs/accelerate_config_4gpu.yaml`:
```yaml
num_processes: 4  # Match NUM_GPUS
```

### Adjust Batch Size

```bash
# In train_lora_multigpu.sh
BATCH_SIZE=1          # Per-GPU batch size
GRADIENT_ACCUM=2      # Gradient accumulation steps

# Effective batch size = NUM_GPUS × BATCH_SIZE × GRADIENT_ACCUM
# Example: 4 × 1 × 2 = 8
```

### Switch Resolution

```bash
# For lower memory usage
HEIGHT=512
WIDTH=512

# Original resolution
HEIGHT=576
WIDTH=576
```

## 🐛 Troubleshooting

### OOM on V100-32GB with ZeRO-2

**Solution**: Switch to ZeRO-3
```bash
DEEPSPEED_CONFIG="configs/deepspeed_zero3.json"
```

### Training is slower than expected

**Check**:
1. Network bandwidth between GPUs (use `nvidia-smi topo -m`)
2. CPU offloading overhead (ZeRO-3 is slower than ZeRO-2)
3. Disk I/O for data loading

**Solutions**:
- Use ZeRO-2 instead of ZeRO-3 if you have enough memory
- Increase `num_workers` in training script
- Use faster storage (NVMe SSD)

### "No module named 'deepspeed'"

```bash
pip install deepspeed
```

### Multi-node training not working

Set up hostfile in `configs/hostfile`:
```
worker0 slots=4
worker1 slots=4
```

Then uncomment in `train_lora_multigpu.sh`:
```bash
export DEEPSPEED_HOSTFILE="configs/hostfile"
```

## 📈 Monitoring

### Check GPU Memory

```bash
watch -n 1 nvidia-smi
```

### W&B Logging

Enable in `train_lora_multigpu.sh`:
```bash
USE_WANDB="--use_wandb"
```

### Parse Training Logs

```bash
python scripts/eval/parse_training_log.py logs/train_4gpu_JOBID.err
```

## 🎓 Advanced: Multi-Node Training

For 8+ GPUs across multiple nodes:

1. Create `configs/accelerate_config_multinode.yaml`
2. Set `num_machines: 2` (or more)
3. Configure `machine_rank` on each node
4. Launch with same command on all nodes simultaneously

Example Slurm script provided in `scripts/slurm/train_lora_8gpu_2node.sh` (coming soon).

## 📚 References

- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
