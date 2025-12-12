## Workflow

### 1. Get word videos from UniSignMimic
Obtain word-level videos from the UniSignMimic dataset.

### 2. Extract frames from word videos
```bash
python scripts/sentence/extract_all_frames_seq.py \
  --csv assets/sentence/sentences.csv \
  --mp4-root data/word_videos \
  --out-root output/sentence_level/frames_512x320 \
  --num 5
```
- Input: Raw word videos (MP4 files)
- Output: `output/sentence_level/frames_512x320/` - Extracted frame sequences

### 3. Filter1: Frame difference detection (remove duplicates)
Detect and remove duplicate/static frames first (fast, reduces data):
```bash
python scripts/sentence/filter_duplicate_frames.py \
  --frames-dir output/sentence_level/frames_512x320 \
  --save-cleaned-frames \
  --output-dir output/sentence_level/frames_512x320_filtered1 \
  --duplicate-threshold 3.0 \
  --min-duplicate-length 2
```
- **duplicate-threshold**: Frame difference threshold in percentage (recommended: 2.0 = 2.0%)
  - Frames with < 2% pixel change are considered duplicates
- **min-duplicate-length**: Minimum consecutive duplicates to detect
- Output: `output/sentence_level/frames_512x320_filtered1/` - Deduplicated frames

### 4. Filter2: RTMPose detection (hand + head + position)
Use RTMLib Wholebody to filter frames based on pose criteria (slower, but on fewer frames):
```bash
python scripts/sentence/filter_frames_by_pose.py \
  --frames-dir output/sentence_level/frames_512x320_filtered1 \
  --save-filtered \
  --output-dir output/sentence_level/frames_512x320_filtered2 \
  --hand-threshold 0.8 \
  --head-threshold 0.9 \
  --hand-height-threshold 0.1 \
  --device cuda \
  --verbose
```
- **hand-threshold**: Hand keypoint confidence threshold (recommended: 0.7-0.8)
  - At least one hand must meet this threshold
- **head-threshold**: Head/face keypoint confidence threshold (recommended: 0.9)
  - Ensures face is clearly visible
- **hand-height-threshold**: Bottom region ratio for hand position filtering (recommended: 0.1)
  - Removes frames where both hands are in the bottom X% of the frame
  - Larger value = stricter filtering
- **verbose**: Annotate saved frames with confidence scores
- Output: `output/sentence_level/frames_512x320_filtered2/` - Final filtered frames

### 5. Extract boundary frames
Extract start and end frames for each word:
```bash
python scripts/sentence/extract_boundary_frames.py \
  --csv assets/sentence/sentences.csv \
  --frames-root output/sentence_level/frames_512x320_filtered2 \
  --out-root output/sentence_level/boundary_frames \
  --num 5
```
- Output: `output/sentence_level/boundary_frames/` - Boundary frame pairs (for interpolation)

### 6. Run FramerTurbo: Generate interpolation frames
Use FramerTurbo with LoRA fine-tuning for frame interpolation:

```bash
cd FramerTurbo

# Using LoRA fine-tuned model
python training/batch_infer_with_lora.py \
  --lora_weights outputs/lora_finetune/checkpoint-XXXX/unet_lora \
  --input_dir ../output/sentence_level/boundary_frames \
  --output_dir ../output/sentence_level/interp_frames \
  --height 576 \
  --width 576 \
  --lora_scale 1.0

cd ..
```

See [FramerTurbo/docs/TRAINING.md](../FramerTurbo/docs/TRAINING.md) for training details.

### 7. Combine frames and interpolation GIFs
Merge cleaned word frames with interpolation GIFs into final videos:
```bash
python scripts/sentence/combine_frames_and_interp.py \
  --csv assets/sentence/sentences.csv \
  --frames-root output/sentence_level/frames_512x320_filtered2 \
  --interp-root output/sentence_level/interp_frames \
  --out-root output/sentence_level/videos_combined \
  --fps 25 \
  --num 5
```
- **fps**: Output video frame rate (recommended: 25)
- Output: `output/sentence_level/videos_combined/` - Final MP4 videos

## Directory Structure
```
output/
├── sentence_level/                      # Sentence-level processing outputs
│   ├── frames_512x320/                  # Step 2: Raw frames
│   ├── frames_512x320_filtered1/        # Step 3: Deduplicated frames
│   ├── frames_512x320_filtered2/        # Step 4: Pose-filtered frames
│   ├── boundary_frames/                 # Step 5: Boundary frames
│   ├── interp_frames/                   # Step 6: Interpolation results
│   └── videos_combined/                 # Step 7: Final videos
├── word_level/                          # Word-level processing outputs
└── metrics/                             # Evaluation metrics
```

## Filter Order Rationale

**Why deduplication before pose detection?**

1. **Performance**: Frame difference is cheap (pixel comparison), RTMPose is expensive (deep learning model)
2. **Data reduction**: Deduplication removes ~30-50% of frames, reducing pose detection workload
3. **Quality**: Static duplicate frames usually fail pose criteria anyway

**Filter1 (Deduplication)**: Fast → Reduces data volume
**Filter2 (Pose)**: Slow but thorough → Applied to smaller dataset

## Filter Criteria (Step 4)

A frame is **removed** if ANY of the following conditions are true:
1. **Head confidence < 0.9** - Face not clearly visible
2. **Both hands too low** - Both hands in bottom 10% of frame
3. **Both hands low confidence** - Both hands below threshold

A frame is **kept** if ALL of the following are true:
- Head confidence ≥ 0.9
- At least one hand above bottom 10% region
- At least one hand confidence ≥ threshold

## Additional Utilities

### Resize frames
Resize frames to target resolution:
```bash
python scripts/sentence/resize_frames.py \
  --frames-dir output/sentence_level/frames_512x320 \
  --output-dir output/sentence_level/frames_576x576 \
  --size 576 576
```

### Generate videos from frames
Alternative way to generate videos:
```bash
python scripts/sentence/generate_videos_from_frames.py \
  --frames-root output/sentence_level/frames_512x320 \
  --out-root output/sentence_level/videos \
  --fps 25
```

## Notes
- All scripts preserve `frame_id` and `ref_id` continuity
- Use `--num` parameter to control number of sentences to process
- Recommend using version suffixes for output management (e.g., `_v2`)
- Enable `--verbose` in Step 4 to visualize confidence scores on frames
- Execute all scripts from project root: `python scripts/sentence/<script>.py`
- All outputs are organized under `output/sentence_level/` directory
