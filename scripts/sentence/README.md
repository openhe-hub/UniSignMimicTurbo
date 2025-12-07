## Workflow

### 1. Get word videos from UniSignMimic
Obtain word-level videos from the UniSignMimic dataset.

### 2. Extract frames from word videos
```bash
python scripts/sentence/extract_frames_from_videos.py \
  --csv assets/sentence/sentences.csv \
  --videos-root data/word_videos \
  --out-root output/frames_512x320 \
  --size 512 320 \
  --num 5
```
- Input: Raw word videos
- Output: `output/frames_512x320/` - Extracted frame sequences

### 3. Filter1: RTMPose detection (hand + head + position)
Use RTMLib Wholebody to detect and filter frames based on multiple criteria:
```bash
python scripts/sentence/filter_frames_by_pose.py \
  --frames-dir output/frames_512x320 \
  --save-filtered \
  --output-dir output/frames_512x320_filtered1 \
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
- Output: `output/frames_512x320_filtered1.8/` - Filtered frames

### 4. Filter2: Frame difference detection
Detect and remove duplicate/static frames:
```bash
python scripts/sentence/filter_duplicate_frames.py \
  --frames-dir output/frames_512x320_filtered1 \
  --save-cleaned-frames \
  --output-dir output/frames_512x320_filtered2 \
  --duplicate-threshold 3.0 \
  --min-duplicate-length 2
```
- **duplicate-threshold**: Frame difference threshold in percentage (recommended: 2.0 = 2.0%)
  - Frames with < 2% pixel change are considered duplicates
- **min-duplicate-length**: Minimum consecutive duplicates to detect
- Output: `output/frames_512x320_filtered2/` - Final cleaned frames

### 5. Extract boundary frames
Extract start and end frames for each word:
```bash
python scripts/sentence/extract_boundary_frames.py \
  --csv assets/sentence/sentences.csv \
  --frames-root output/frames_512x320_filtered2 \
  --out-root output/boundary_frames \
  --num 5
```
- Output: `output/boundary_frames/` - Boundary frame pairs (for interpolation)

### 6. Run Framer: Generate interpolation frames
https://github.com/aim-uofa/Framer

- Input: Boundary frame pairs
- Output: `output/interp_right_512x320_cleaned/` - Interpolation GIFs

### 7. Combine frames and interpolation GIFs
Merge cleaned word frames with interpolation GIFs into final videos:
```bash
python scripts/sentence/combine_frames_and_interp.py \
  --csv assets/sentence/sentences.csv \
  --frames-root output/frames_512x320_cleaned \
  --interp-root output/interp_right_512x320_cleaned \
  --out-root output/videos_combined \
  --fps 25 \
  --num 5
```
- **fps**: Output video frame rate (recommended: 25)
- Output: `output/videos_combined/` - Final MP4 videos

## Directory Structure
```
output/
├── frames_512x320/              # Step 2: Raw frames
├── frames_512x320_filtered1/    # Step 3: Pose-filtered frames
├── frames_512x320_filtered2/      # Step 4: Deduplicated frames
├── boundary_frames/             # Step 5: Boundary frames
├── interp_right_512x320_cleaned/ # Step 6: Interpolation GIFs
└── videos_combined/             # Step 7: Final videos
```

## Filter Criteria (Step 3)

A frame is **removed** if ANY of the following conditions are true:
1. **Head confidence < 0.9** - Face not clearly visible
2. **Both hands too low** - Both hands in bottom 10% of frame
3. **Both hands low confidence** - Both hands below threshold

A frame is **kept** if ALL of the following are true:
- Head confidence ≥ 0.9
- At least one hand above bottom 10% region
- At least one hand confidence ≥ threshold

## Notes
- All scripts preserve `frame_id` and `ref_id` continuity
- Use `--num` parameter to control number of sentences to process
- Recommend using `_v2` suffix for version management
- Enable `--verbose` in Step 3 to visualize confidence scores on frames