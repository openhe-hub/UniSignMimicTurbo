# FramerTurbo 目录结构说明

```
FramerTurbo/
│
├── README.md                    # 项目主文档
├── requirements.txt             # Python 依赖
├── .gitignore
├── STRUCTURE.md                 # 本文件 - 目录结构说明
│
├── models_diffusers/            # 模型定义
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
├── pipelines/                   # 推理 Pipeline
│   └── pipeline_stable_video_diffusion_interp_control.py
│
├── gradio_demo/                 # Gradio 演示相关工具
│   └── utils_drag.py
│
├── apps/                        # Gradio 应用程序
│   ├── app.py                  # 原始版本
│   ├── app_turbo.py            # Turbo 版本
│   └── app_turbo_v2.py         # Turbo v2 版本（最新）
│
├── scripts/                     # 脚本目录
│   ├── inference/              # 推理脚本
│   │   ├── cli_infer.py               # 基础推理
│   │   ├── cli_infer_turbo.py         # Turbo 推理
│   │   ├── cli_infer_turbo_v2.py      # Turbo v2 推理（推荐）
│   │   └── cli_infer_576x576.py       # 高分辨率推理
│   ├── slurm/                  # SLURM 集群脚本
│   │   └── infer_576x576_euler.sh
│   └── train_lora.sh           # LoRA 训练启动脚本
│
├── training/                    # 训练相关（LoRA 微调）
│   ├── README.md               # 训练文档
│   ├── train_lora.py           # LoRA 训练主脚本
│   ├── train_dataset.py        # 数据集定义
│   ├── train_config.py         # 训练配置示例
│   └── infer_with_lora.py      # LoRA 模型推理脚本
│
└── assets/                      # 资源文件
    └── logo/
        └── framer.png
```

## 📁 目录说明

### 核心模块

- **models_diffusers/** - 自定义的 Diffusers 模型组件
  - UNet、ControlNet 实现
  - 自定义的注意力机制和调度器

- **pipelines/** - 推理管道
  - 集成了 ControlNet 的 SVD 插帧 pipeline

### 应用和脚本

- **apps/** - Gradio 交互应用
  - `app_turbo_v2.py` 是最新版本，支持多种调度器

- **scripts/** - 各类脚本
  - `inference/` - 命令行推理脚本
    - 推荐使用 `cli_infer_turbo_v2.py`
  - `slurm/` - 集群任务脚本
  - `train_lora.sh` - 训练启动脚本

### 训练模块

- **training/** - LoRA 微调训练
  - 完整的训练代码和文档
  - 支持视频文件和图像对数据集
  - 详见 `training/README.md`

## 🚀 快速使用

### 推理
```bash
# 推荐使用 Turbo v2 版本
python scripts/inference/cli_infer_turbo_v2.py \
    --input_dir assets/pairs \
    --model checkpoints/framer_512x320/ \
    --output_dir outputs
```

### 训练
```bash
# 查看训练文档
cat training/README.md

# 启动 LoRA 训练
bash scripts/train_lora.sh
```

### Gradio 应用
```bash
# 启动 Turbo v2 应用
python apps/app_turbo_v2.py
```

## 📝 版本说明

- **基础版本** (`app.py`, `cli_infer.py`): 原始实现，使用 Euler 调度器
- **Turbo 版本** (`app_turbo.py`, `cli_infer_turbo.py`): 增加 LCM 调度器支持
- **Turbo v2 版本** (`app_turbo_v2.py`, `cli_infer_turbo_v2.py`): 支持 Euler/DPM++/LCM 多种调度器（推荐）

## 🔄 迁移指南

如果你之前使用根目录下的脚本，请更新路径：

- `cli_infer_turbo_v2.py` → `scripts/inference/cli_infer_turbo_v2.py`
- `app_turbo_v2.py` → `apps/app_turbo_v2.py`
- `train_lora.py` → `training/train_lora.py`
