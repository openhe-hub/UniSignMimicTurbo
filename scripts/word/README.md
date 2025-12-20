# Word 级流水线（AslToHiya-01 示例）

## 依赖
- 已安装并可用的 `conda` 环境 `diff_turbo`（包含 `cv2`、`torch`、`rtmlib` 等）。
- 默认参考图：`assets/example_data/images/test5_576x576.jpg`。

快速自检：
```bash
conda run -n diff_turbo python - <<'PY'
import cv2, torch
print("cv2", cv2.__version__)
print("torch", torch.__version__)
PY
```

## 一键跑通
```bash
conda run -n diff_turbo scripts/word/run_word_pipeline.sh
```

流水线步骤（脚本自动执行）：
- 链接 MP4 根目录：`output/word_level/mp4_asltohiya/<WORD_ID>`（指向 `output/word_level/word_videos/<WORD_ID>`）
- 抽帧：`scripts/sentence/extract_all_frames_seq.py` → `output/word_level/frames_asltohiya/<WORD_ID>`
- 去重过滤：`scripts/sentence/filter_duplicate_frames.py` → `..._filtered1/<WORD_ID>`
- 姿态过滤：`scripts/sentence/filter_frames_by_pose.py` → `..._filtered2/<WORD_ID>`
- 词级 boundary：`scripts/word/extract_boundary_frames.py` → `output/word_level/word_boundary_frames/<WORD_ID>`

## 拆分 boundary 成 N 份（保持成对 start/end）
```bash
conda run -n diff_turbo python scripts/word/split_boundary_sets.py \
  --boundary-dir output/word_level/word_boundary_frames/AslToHiya-01 \
  --out-root output/word_level/word_boundary_frames_split \
  --splits 4
```
- 会按 `ref_id` 整组划分，确保每个 part 里 start/end 成对（文件数为偶数）；空 part 会跳过。
- 默认复制，如需移动加 `--move`。

## 可配置参数（环境变量覆盖）
- `WORD_ID`（默认 `AslToHiya-01`）
- `REF_IMAGE`（默认 `assets/example_data/images/test5_576x576.jpg`）
- `DEVICE`（默认 `cuda`）
- `DUP_THRESHOLD`、`MIN_DUP_LEN`、`HAND_THRESHOLD`、`HEAD_THRESHOLD`、`HAND_HEIGHT_THRESHOLD`

示例：指定参考图或设备
```bash
conda run -n diff_turbo REF_IMAGE=assets/example_data/images/test5_576x576.jpg DEVICE=cuda scripts/word/run_asltohiya_pipeline.sh
```
