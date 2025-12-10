# FramerTurbo 训练数据准备指南

## 📋 数据集要求总结

### ✅ 视频格式要求

| 项目 | 要求 | 说明 |
|------|------|------|
| **格式** | `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` | 常见视频格式都支持 |
| **最小帧数** | ≥ 16 帧 | 太短的视频会被自动过滤 |
| **分辨率** | **任意** | 会自动 resize 到目标分辨率 |
| **帧率** | 任意 | 训练时会自动采样 |
| **时长** | 建议 1-10 秒 | 短视频训练效果更好 |

### 🎯 推荐的训练分辨率

```python
# 标准分辨率（默认）
height=320, width=512   # 推荐，显存友好

# 高分辨率（需要更多显存）
height=576, width=576   # 如果你有更大的显存

# 自定义分辨率
height=H, width=W       # 必须是 8 的倍数
```

**重要**:
- ✅ 宽高必须是 **8 的倍数**（VAE 编码要求）
- ✅ 训练分辨率可以和视频原始分辨率不同
- ✅ 推荐使用 512x320（和预训练模型一致）

## 📁 数据集组织方式

### 方式 1: 视频文件（推荐）

```
data/
└── training_videos/
    ├── video_001.mp4
    ├── video_002.mp4
    ├── video_003.mp4
    ├── subfolder/
    │   ├── video_004.mp4
    │   └── video_005.mp4
    └── ...
```

**优点**:
- 简单，直接扔视频就行
- 自动递归搜索子目录
- 自动采样帧序列

### 方式 2: 图像对（适合已处理的数据）

```
data/
└── image_pairs/
    ├── sample_001_start.jpg
    ├── sample_001_end.jpg
    ├── sample_002_start.jpg
    ├── sample_002_end.jpg
    └── ...
```

**注意**: 命名必须遵循 `{id}_start.{ext}` 和 `{id}_end.{ext}` 格式

## 🎬 数据质量建议

### ✅ 好的训练数据

1. **清晰的运动**
   - 明显的物体移动
   - 平滑的运动轨迹
   - 避免剧烈抖动

2. **良好的画质**
   - 分辨率 ≥ 480p
   - 无明显压缩伪影
   - 光照均匀

3. **合适的内容**
   - 与你的应用场景相关
   - 例如：手语数据集 → 收集手语视频

4. **多样性**
   - 不同的场景
   - 不同的运动速度
   - 不同的背景

### ❌ 避免的数据

- 静态画面（没有运动）
- 过度模糊或压缩
- 快速闪烁或场景切换
- 极暗或过曝的画面

## 📊 数据集规模建议

| 数据量 | 训练轮数 | 效果 | 适用场景 |
|--------|---------|------|----------|
| **50-100 视频** | 10-20 epochs | 初步适应 | 概念验证 |
| **100-500 视频** | 5-10 epochs | 良好适应 | 小型项目 |
| **500-1000 视频** | 3-5 epochs | 很好效果 | 推荐规模 |
| **1000+ 视频** | 2-3 epochs | 最佳效果 | 生产环境 |

**重要**:
- 质量 > 数量
- 50 个高质量视频 > 500 个低质量视频

## 🛠️ 数据准备实战

### 步骤 1: 创建数据目录

```bash
cd FramerTurbo
mkdir -p data/training_videos
```

### 步骤 2: 放置视频文件

```bash
# 直接复制视频到目录
cp /path/to/your/videos/*.mp4 data/training_videos/

# 或者创建软链接（节省空间）
ln -s /path/to/your/videos/*.mp4 data/training_videos/
```

### 步骤 3: 验证数据集

创建一个测试脚本：

```python
# test_dataset.py
from training.train_dataset import VideoFrameDataset

# 测试数据集
dataset = VideoFrameDataset(
    video_dir="data/training_videos",
    num_frames=3,
    height=320,
    width=512,
    min_video_frames=16,
)

print(f"总共找到 {len(dataset)} 个有效视频")

# 查看第一个样本
sample = dataset[0]
print(f"样本形状: {sample['pixel_values'].shape}")  # 应该是 (3, 3, 320, 512)
print(f"视频路径: {sample['video_path']}")
```

运行测试：
```bash
python test_dataset.py
```

### 步骤 4: 检查视频质量

```bash
# 安装 ffprobe（如果没有）
# sudo apt-get install ffmpeg

# 检查单个视频信息
ffprobe -v quiet -show_entries format=duration -show_entries stream=width,height,nb_frames,r_frame_rate -of json data/training_videos/video_001.mp4
```

## 🎯 针对手语数据集的特殊建议

既然你在做手语相关的项目（从目录名 `UniSignMimicTurbo` 推测），这里有一些专门建议：

### 手语视频数据特点

1. **关注手部区域**
   - 确保手部清晰可见
   - 避免严重的手部遮挡
   - 手部运动轨迹完整

2. **背景处理**
   - 纯色背景最佳
   - 避免复杂背景干扰
   - 考虑使用 `scripts/word/replace_video_background.py` 预处理

3. **视频剪辑**
   - 每个视频对应一个完整的手语动作
   - 起始帧和结束帧应该是动作的关键姿态
   - 避免动作开始前/结束后的静止帧

4. **数据增强建议**
   - 同一动作的不同执行者
   - 不同的拍摄角度
   - 不同的速度（如果有）

### 手语数据的采样策略

修改 `scripts/train_lora.sh`:

```bash
# 针对手语，可能需要更多帧
NUM_FRAMES=5  # 增加到 5 帧，更好地捕捉手部运动

# 如果视频较短，减小采样步长
# 在 train_dataset.py 中 sample_stride=2
```

## 📝 数据准备检查清单

在开始训练前，确认：

- [ ] 视频格式正确（.mp4, .avi, .mov, .mkv, .webm）
- [ ] 每个视频 ≥ 16 帧
- [ ] 视频质量良好（清晰、无过度压缩）
- [ ] 数据集规模合理（建议 ≥ 100 个视频）
- [ ] 运行 `test_dataset.py` 成功
- [ ] 检查了几个样本的质量
- [ ] 确定了训练分辨率（推荐 512x320）

## 🚀 快速开始训练

数据准备好后：

```bash
# 1. 编辑训练脚本
nano scripts/train_lora.sh

# 修改这一行：
DATA_DIR="data/training_videos"  # 改成你的数据目录

# 2. 启动训练
bash scripts/train_lora.sh
```

## ❓ 常见问题

**Q: 视频分辨率必须一致吗？**
A: 不需要！代码会自动 resize 到目标分辨率。

**Q: 可以混合不同帧率的视频吗？**
A: 可以！训练时会自动采样。

**Q: 最少需要多少视频？**
A: 理论上 50 个就能开始，但建议 100+ 以获得更好效果。

**Q: 视频太长怎么办？**
A: 代码会自动随机采样片段，或者你可以预先切分视频。

**Q: 如何切分长视频？**
```bash
# 使用 ffmpeg 切分成 2 秒片段
ffmpeg -i input.mp4 -c copy -map 0 -segment_time 2 -f segment output_%03d.mp4
```

**Q: 我的视频是竖屏的，需要转吗？**
A: 不需要！代码会 resize，但建议保持训练数据的方向一致。

## 📚 相关文档

- [完整训练教程](TRAINING.md)
- [快速开始](QUICKSTART.md)
- [训练配置](../training/train_config.py)

---

**准备好数据后，查看**: [docs/TRAINING.md](TRAINING.md) 开始训练！
