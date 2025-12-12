# FramerTurbo 训练模块

此目录包含 FramerTurbo 的 LoRA 微调训练代码。

## 📖 完整文档

训练的详细文档已移至: **[../docs/TRAINING.md](../docs/TRAINING.md)**

## 📁 文件说明

- `train_lora.py` - 主训练脚本
- `train_dataset.py` - 数据集定义（支持视频和图像对）
- `train_config.py` - 训练配置示例
- `infer_with_lora.py` - LoRA 模型推理脚本

## 🚀 快速开始

### 启动训练

```bash
# 从项目根目录运行
bash scripts/train_lora.sh
```

### 自定义训练

```bash
python training/train_lora.py \
    --pretrained_model_path checkpoints/framer_512x320 \
    --data_dir data/my_videos \
    --output_dir outputs/my_lora \
    --train_unet \
    --lora_rank 64
```

### 使用训练好的模型

```bash
python training/infer_with_lora.py \
    --lora_weights outputs/lora_finetune/final/unet_lora \
    --start_image examples/start.jpg \
    --end_image examples/end.jpg \
    --output_path output.gif
```

## 📚 更多信息

- 完整训练指南: [../docs/TRAINING.md](../docs/TRAINING.md)
- 快速开始: [../docs/QUICKSTART.md](../docs/QUICKSTART.md)
- 项目结构: [../docs/STRUCTURE.md](../docs/STRUCTURE.md)
