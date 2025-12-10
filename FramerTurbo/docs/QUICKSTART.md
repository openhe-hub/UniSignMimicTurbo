# FramerTurbo - 快速导航

欢迎使用 FramerTurbo！本文档帮助你快速了解项目结构和使用方法。

## 📁 项目结构

```
FramerTurbo/
├── 📖 README.md                 # 项目主文档
├── 📖 STRUCTURE.md              # 详细的目录结构说明
├── 📖 QUICKSTART.md             # 本文件
│
├── 🎨 apps/                     # Gradio 交互应用
│   └── app_turbo_v2.py         # 推荐使用（支持多种调度器）
│
├── 🔧 scripts/                  # 脚本工具
│   ├── inference/              # 推理脚本
│   │   └── cli_infer_turbo_v2.py  # 推荐使用
│   ├── slurm/                  # 集群任务
│   └── train_lora.sh           # 训练启动脚本
│
├── 🎓 training/                 # LoRA 微调训练
│   ├── README.md               # 训练详细文档
│   ├── train_lora.py           # 训练脚本
│   ├── train_dataset.py        # 数据集
│   ├── train_config.py         # 配置示例
│   └── infer_with_lora.py      # LoRA 推理
│
├── 🏗️ models_diffusers/         # 模型定义
├── 🔄 pipelines/                # Pipeline
├── 🎯 gradio_demo/              # Gradio 工具
└── 📦 assets/                   # 示例资源
```

## 🚀 快速开始

### 1️⃣ 推理（生成视频）

**命令行推理**（推荐）:
```bash
# 使用 DPM++ 调度器（平衡速度和质量）
python scripts/inference/cli_infer_turbo_v2.py \
    --input_dir assets/test_single \
    --model checkpoints/framer_512x320 \
    --output_dir outputs
```

**图形界面**:
```bash
python apps/app_turbo_v2.py
```

### 2️⃣ 训练（微调模型）

查看完整训练文档:
```bash
cat training/README.md
```

快速开始训练:
```bash
# 1. 准备数据（将视频放在 data/training_videos/）
# 2. 编辑配置
nano scripts/train_lora.sh

# 3. 启动训练
bash scripts/train_lora.sh
```

### 3️⃣ 使用微调后的模型

```bash
python training/infer_with_lora.py \
    --lora_weights outputs/lora_finetune/final/unet_lora \
    --start_image examples/start.jpg \
    --end_image examples/end.jpg \
    --output_path output.gif
```

## 📝 常用命令

### 推理相关

```bash
# 基础推理（Euler，30步，最佳质量）
python scripts/inference/cli_infer_turbo_v2.py \
    --input_dir INPUT_DIR \
    --model checkpoints/framer_512x320 \
    --scheduler euler \
    --num_inference_steps 30

# 快速推理（DPM++，15步，推荐）
python scripts/inference/cli_infer_turbo_v2.py \
    --input_dir INPUT_DIR \
    --model checkpoints/framer_512x320 \
    --scheduler dpm++ \
    --num_inference_steps 15

# 超快推理（LCM，4步）
python scripts/inference/cli_infer_turbo_v2.py \
    --input_dir INPUT_DIR \
    --model checkpoints/framer_512x320 \
    --scheduler lcm \
    --num_inference_steps 4

# 高分辨率推理（576x576）
python scripts/inference/cli_infer_576x576.py \
    --input_dir INPUT_DIR \
    --model checkpoints/framer_576x576 \
    --output_dir outputs_hd
```

### 训练相关

```bash
# 查看训练配置
cat training/train_config.py

# 编辑训练脚本
nano scripts/train_lora.sh

# 启动训练
bash scripts/train_lora.sh

# 使用自定义参数训练
python training/train_lora.py \
    --pretrained_model_path checkpoints/framer_512x320 \
    --data_dir data/my_videos \
    --output_dir outputs/my_lora \
    --train_batch_size 2 \
    --lora_rank 128 \
    --train_unet
```

## 📚 详细文档

- **项目介绍**: [README.md](README.md)
- **目录结构**: [STRUCTURE.md](STRUCTURE.md)
- **训练指南**: [training/README.md](training/README.md)

## ⚙️ 调度器对比

| 调度器 | 步数 | 速度 | 质量 | 使用场景 |
|--------|------|------|------|----------|
| Euler  | 30   | 慢   | 最佳 | 最终产出 |
| DPM++  | 15   | 快   | 优秀 | 日常使用（推荐）|
| LCM    | 4-6  | 极快 | 良好 | 快速预览 |

## 💡 提示

1. **首次使用**: 从图形界面开始（`python apps/app_turbo_v2.py`）
2. **批量处理**: 使用命令行脚本（`scripts/inference/cli_infer_turbo_v2.py`）
3. **微调训练**: 参考 `training/README.md` 了解详细步骤
4. **显存不足**: 减小 batch_size，使用 FP16，或使用更小的 lora_rank

## ❓ 常见问题

**Q: 如何切换不同版本的应用？**
```bash
# 原始版本（Euler）
python apps/app.py

# Turbo v2（推荐，支持多调度器）
python apps/app_turbo_v2.py
```

**Q: 训练数据应该放在哪里？**

视频文件放在任意目录，然后在训练脚本中指定 `DATA_DIR` 路径。

**Q: 如何查看所有可用参数？**
```bash
python scripts/inference/cli_infer_turbo_v2.py --help
python training/train_lora.py --help
```

## 📞 获取帮助

- 查看详细文档: `STRUCTURE.md` 和 `training/README.md`
- 查看示例: `assets/` 目录
- 检查配置: `training/train_config.py`

---

**Happy Framing! 🎬**
