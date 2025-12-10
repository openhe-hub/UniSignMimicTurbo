# FramerTurbo LoRA Fine-tuning Guide

这是 FramerTurbo 的 LoRA 微调训练代码。支持在自定义数据集上进行高效微调。

## 📋 特性

- ✅ **LoRA 高效微调**: 使用 PEFT 库，显存友好（~16-24GB）
- ✅ **多种数据格式**: 支持视频文件或图像对
- ✅ **混合精度训练**: 支持 FP16/BF16
- ✅ **梯度累积**: 支持小显存训练
- ✅ **Accelerate 集成**: 支持单卡/多卡训练
- ✅ **灵活配置**: 可选训练 UNet 和/或 ControlNet

## 🚀 快速开始

> **重要**: 请从 FramerTurbo 项目根目录运行所有命令！

### 1. 安装依赖

```bash
# 在项目根目录下
pip install -r requirements.txt
pip install accelerate peft wandb
```

### 2. 准备数据集

#### 方式 A: 视频文件（推荐）

将视频文件放在一个目录下：

```
data/training_videos/
    video_001.mp4
    video_002.mp4
    video_003.mp4
    ...
```

#### 方式 B: 图像对

将起始帧和结束帧配对：

```
data/image_pairs/
    sample_001_start.jpg
    sample_001_end.jpg
    sample_002_start.jpg
    sample_002_end.jpg
    ...
```

### 3. 配置训练脚本

编辑 `scripts/train_lora.sh`:

```bash
# 修改数据路径
DATA_DIR="data/training_videos"  # 你的数据目录

# 选择数据集类型
DATASET_TYPE="video"  # 或 "image_pair"

# 调整训练参数
BATCH_SIZE=1          # 根据显存调整
GRADIENT_ACCUM=4      # 有效 batch size = BATCH_SIZE × GRADIENT_ACCUM
EPOCHS=10             # 训练轮数
LEARNING_RATE=1e-4    # 学习率
```

### 4. 启动训练

```bash
cd FramerTurbo
bash scripts/train_lora.sh
```

## ⚙️ 训练参数说明

### 核心参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--pretrained_model_path` | 预训练模型路径 | `checkpoints/framer_512x320` |
| `--data_dir` | 训练数据目录 | - |
| `--output_dir` | 输出目录 | - |
| `--dataset_type` | 数据集类型: `video` 或 `image_pair` | `video` |

### LoRA 参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--lora_rank` | LoRA 秩（越大效果越好但显存越多） | 64 |
| `--lora_alpha` | LoRA alpha（通常等于 rank） | 64 |
| `--lora_dropout` | LoRA dropout | 0.0 |
| `--train_unet` | 训练 UNet（必选） | ✓ |
| `--train_controlnet` | 训练 ControlNet（可选） | - |

### 训练超参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--train_batch_size` | 每卡 batch size | 1 |
| `--gradient_accumulation_steps` | 梯度累积步数 | 4-8 |
| `--num_train_epochs` | 训练轮数 | 10-20 |
| `--learning_rate` | 学习率 | 1e-4 |
| `--mixed_precision` | 混合精度: `fp16` 或 `bf16` | `fp16` |

## 💾 显存需求

| 配置 | 显存需求 | 说明 |
|------|----------|------|
| LoRA (rank=64) + UNet | ~16-20 GB | 推荐配置 |
| LoRA (rank=128) + UNet | ~20-24 GB | 更高质量 |
| LoRA + UNet + ControlNet | ~24-32 GB | 完整训练 |
| 全量微调 | ~40+ GB | 最佳效果 |

**节省显存技巧**:
- 减小 `--lora_rank` (如 32)
- 减小 `--train_batch_size` 并增加 `--gradient_accumulation_steps`
- 使用 `--mixed_precision fp16`
- 减小 `--num_frames` (如 3 → 2)

## 📊 监控训练

### TensorBoard

```bash
tensorboard --logdir logs
```

### Weights & Biases

```bash
# 在训练脚本中添加
USE_WANDB="--use_wandb"
```

## 🔄 使用训练好的模型

### 加载 LoRA 权重

```python
from peft import PeftModel
from models_diffusers.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

# 加载基础模型
unet = UNetSpatioTemporalConditionModel.from_pretrained(
    "checkpoints/framer_512x320/unet",
    torch_dtype=torch.float16,
)

# 加载 LoRA 权重
unet = PeftModel.from_pretrained(
    unet,
    "outputs/lora_finetune/final/unet_lora",
)

# 合并 LoRA（可选，用于推理加速）
unet = unet.merge_and_unload()
```

### 推理示例

```python
# 将微调后的 UNet 集成到推理 pipeline
pipe = StableVideoDiffusionInterpControlPipeline.from_pretrained(
    "checkpoints/stable-video-diffusion-img2vid-xt",
    unet=unet,  # 使用微调后的 UNet
    controlnet=controlnet,
    torch_dtype=torch.float16,
)

# 正常推理
frames = pipe(
    first_image,
    last_image,
    num_frames=3,
    ...
).frames
```

## 🛠️ 高级用法

### 多卡训练

使用 Accelerate 配置文件:

```bash
accelerate config  # 配置多卡设置
accelerate launch training/train_lora.py ...  # 自动多卡训练
```

### 从检查点恢复

```bash
python training/train_lora.py \
  --resume_from_checkpoint outputs/lora_finetune/checkpoint-1000 \
  ...
```

### 仅训练 ControlNet

```bash
python training/train_lora.py \
  --train_controlnet \  # 只训练 ControlNet
  --learning_rate 5e-5 \  # ControlNet 建议更小的学习率
  ...
```

### 混合训练（UNet LoRA + ControlNet 全量）

```bash
python training/train_lora.py \
  --train_unet \
  --train_controlnet \
  --lora_rank 64 \
  ...
```

## 📁 输出结构

```
outputs/lora_finetune/
├── checkpoint-500/
│   └── unet_lora/          # LoRA 权重检查点
├── checkpoint-1000/
│   └── unet_lora/
└── final/
    └── unet_lora/          # 最终 LoRA 权重
        ├── adapter_config.json
        └── adapter_model.safetensors
```

## ❓ 常见问题

### Q: 训练过程中显存溢出怎么办？

A: 尝试以下方法：
1. 减小 `--train_batch_size` 至 1
2. 增加 `--gradient_accumulation_steps` 至 8 或更高
3. 减小 `--lora_rank` 至 32 或 16
4. 减小 `--num_frames` 至 2

### Q: 训练多少步合适？

A: 取决于数据集大小：
- 小数据集 (< 100 视频): 10-20 epochs
- 中数据集 (100-1000 视频): 5-10 epochs
- 大数据集 (> 1000 视频): 2-5 epochs

建议每 500 步保存检查点，根据验证效果选择最佳模型。

### Q: 如何调整学习率？

A: 参考建议：
- LoRA 训练: `1e-4` 到 `5e-5`
- ControlNet 训练: `5e-5` 到 `1e-5`
- 如果 loss 不下降，尝试提高学习率
- 如果 loss 震荡，尝试降低学习率

### Q: 支持自定义轨迹标注吗？

A: 当前版本的训练代码暂未集成轨迹点标注。如需训练轨迹控制能力，需要：
1. 准备带轨迹标注的数据集
2. 修改 `train_dataset.py` 加载轨迹数据
3. 在 `train_lora.py` 中添加 ControlNet 条件

我们计划在后续版本中添加完整的轨迹标注训练支持。

## 📚 参考资料

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [Accelerate 文档](https://huggingface.co/docs/accelerate)
- [FramerTurbo 论文](https://arxiv.org/abs/2410.18978)

## 📝 许可证

本训练代码遵循 FramerTurbo 的许可证。
