# FramerTurbo Training Data Preparation Guide

## 📋 Dataset Requirements Summary

### ✅ Video Format Requirements

| Item | Requirement | Notes |
|------|-------------|-------|
| **Format** | `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` | Common video formats are supported |
| **Minimum Frames** | ≥ 16 frames | Too short videos will be automatically filtered |
| **Resolution** | **Any** | Will be automatically resized to target resolution |
| **Frame Rate** | Any | Will be automatically sampled during training |
| **Duration** | Recommended 1-10 seconds | Shorter videos train better |

### 🎯 Recommended Training Resolutions

```python
# Standard resolution (default)
height=320, width=512   # Recommended, memory-friendly

# High resolution (requires more memory)
height=576, width=576   # If you have larger GPU memory

# Custom resolution
height=H, width=W       # Must be multiples of 8
```

**Important**:
- ✅ Width and height must be **multiples of 8** (VAE encoding requirement)
- ✅ Training resolution can differ from video's original resolution
- ✅ Recommended to use 512x320 (consistent with pre-trained model)

## 📁 Dataset Organization

### Method 1: Video Files (Recommended)

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

**Advantages**:
- Simple, just drop videos
- Automatically searches subdirectories recursively
- Automatically samples frame sequences

### Method 2: Image Pairs (For Preprocessed Data)

```
data/
└── image_pairs/
    ├── sample_001_start.jpg
    ├── sample_001_end.jpg
    ├── sample_002_start.jpg
    ├── sample_002_end.jpg
    └── ...
```

**Note**: Naming must follow `{id}_start.{ext}` and `{id}_end.{ext}` format

## 🎬 Data Quality Recommendations

### ✅ Good Training Data

1. **Clear Motion**
   - Obvious object movement
   - Smooth motion trajectories
   - Avoid severe shaking

2. **Good Image Quality**
   - Resolution ≥ 480p
   - No obvious compression artifacts
   - Even lighting

3. **Appropriate Content**
   - Relevant to your application scenario
   - Example: Sign language dataset → collect sign language videos

4. **Diversity**
   - Different scenes
   - Different motion speeds
   - Different backgrounds

### ❌ Data to Avoid

- Static frames (no motion)
- Overly blurry or compressed
- Rapid flickering or scene changes
- Extremely dark or overexposed frames

## 📊 Dataset Scale Recommendations

| Data Volume | Training Epochs | Effect | Use Case |
|------------|----------------|--------|----------|
| **50-100 videos** | 10-20 epochs | Initial adaptation | Proof of concept |
| **100-500 videos** | 5-10 epochs | Good adaptation | Small projects |
| **500-1000 videos** | 3-5 epochs | Great results | Recommended scale |
| **1000+ videos** | 2-3 epochs | Best results | Production |

**Important**:
- Quality > Quantity
- 50 high-quality videos > 500 low-quality videos

## 🛠️ Data Preparation Walkthrough

### Step 1: Create Data Directory

```bash
cd FramerTurbo
mkdir -p data/training_videos
```

### Step 2: Place Video Files

```bash
# Directly copy videos to directory
cp /path/to/your/videos/*.mp4 data/training_videos/

# Or create symbolic links (saves space)
ln -s /path/to/your/videos/*.mp4 data/training_videos/
```

### Step 3: Validate Dataset

Create a test script:

```python
# test_dataset.py
from training.train_dataset import VideoFrameDataset

# Test dataset
dataset = VideoFrameDataset(
    video_dir="data/training_videos",
    num_frames=3,
    height=320,
    width=512,
    min_video_frames=16,
)

print(f"Found {len(dataset)} valid videos")

# View first sample
sample = dataset[0]
print(f"Sample shape: {sample['pixel_values'].shape}")  # Should be (3, 3, 320, 512)
print(f"Video path: {sample['video_path']}")
```

Run test:
```bash
python test_dataset.py
```

### Step 4: Check Video Quality

```bash
# Install ffprobe (if not available)
# sudo apt-get install ffmpeg

# Check single video info
ffprobe -v quiet -show_entries format=duration -show_entries stream=width,height,nb_frames,r_frame_rate -of json data/training_videos/video_001.mp4
```

## 🎯 Special Recommendations for Sign Language Dataset

Since you're working on sign language-related projects (inferred from `UniSignMimicTurbo` directory name), here are specific recommendations:

### Sign Language Video Data Characteristics

1. **Focus on Hand Regions**
   - Ensure hands are clearly visible
   - Avoid severe hand occlusion
   - Complete hand motion trajectories

2. **Background Processing**
   - Solid color background is best
   - Avoid complex background interference
   - Consider using `scripts/word/replace_video_background.py` for preprocessing

3. **Video Clipping**
   - Each video corresponds to a complete sign language gesture
   - Start and end frames should be key poses of the gesture
   - Avoid static frames before/after gesture

4. **Data Augmentation Recommendations**
   - Different performers for the same gesture
   - Different shooting angles
   - Different speeds (if available)

### Sampling Strategy for Sign Language Data

Modify `scripts/train/train_lora.sh`:

```bash
# For sign language, may need more frames
NUM_FRAMES=5  # Increase to 5 frames for better hand motion capture

# If videos are short, reduce sampling stride
# In train_dataset.py, set sample_stride=2
```

## 📝 Data Preparation Checklist

Before starting training, confirm:

- [ ] Video format is correct (.mp4, .avi, .mov, .mkv, .webm)
- [ ] Each video ≥ 16 frames
- [ ] Good video quality (clear, not over-compressed)
- [ ] Reasonable dataset size (recommended ≥ 100 videos)
- [ ] Successfully ran `test_dataset.py`
- [ ] Checked quality of several samples
- [ ] Determined training resolution (recommended 512x320)

## 🚀 Quick Start Training

After data is prepared:

```bash
# 1. Edit training script
nano scripts/train/train_lora.sh

# Modify this line:
DATA_DIR="data/training_videos"  # Change to your data directory

# 2. Start training
bash scripts/train/train_lora.sh
```

## ❓ FAQ

**Q: Must video resolutions be consistent?**
A: No! Code will automatically resize to target resolution.

**Q: Can I mix videos with different frame rates?**
A: Yes! Training will automatically sample frames.

**Q: What's the minimum number of videos needed?**
A: Theoretically 50 can start, but 100+ is recommended for better results.

**Q: What if videos are too long?**
A: Code will automatically sample random segments, or you can pre-split videos.

**Q: How to split long videos?**
```bash
# Use ffmpeg to split into 2-second segments
ffmpeg -i input.mp4 -c copy -map 0 -segment_time 2 -f segment output_%03d.mp4
```

**Q: My videos are portrait orientation, need to convert?**
A: No! Code will resize, but it's recommended to keep training data orientation consistent.

## 📚 Related Documentation

- [Complete Training Tutorial](TRAINING.md)
- [Quick Start](QUICKSTART.md)
- [Training Configuration](../training/train_config.py)

---

**After data is prepared, see**: [docs/TRAINING.md](TRAINING.md) to start training!
