# Scripts Directory

This directory contains utility scripts for processing videos, frames, and managing sign language datasets.

## Main Scripts

### split_files.py
Splits MP4 files into equal parts for parallel processing.

**Usage:**
```bash
python split_files.py --src-dir <source_directory> --parts <number_of_parts> --mode <move|copy>
```

**Options:**
- `--src-dir`: Directory containing MP4 files (default: `assets/bad_videos/bad_videos`)
- `--parts`: Number of splits to create (default: 10)
- `--mode`: Either `move` or `copy` files into splits (default: `move`)

**Note:** Despite the function name `split_pkls`, this script actually processes `.mp4` files.

### stat_mp4.py
Extracts and displays metadata from video files.

**Usage:**
```bash
python stat_mp4.py --video_path <path_to_video>
```

**Output:**
- Video path
- FPS (frames per second)
- Resolution (width x height)
- Total frame count

## Sentence Processing (`sentence/`)

Scripts for processing sign language sentence datasets with reference IDs.

### split_sentence_mp4.py
Organizes MP4 files into per-sentence folders based on CSV metadata.

**Usage:**
```bash
python sentence/split_sentence_mp4.py --csv <sentences.csv> --mp4-dir <source_mp4_directory> --out-dir <output_directory> --num <number_of_sentences>
```

**Example:**
```bash
python scripts/sentence/split_sentence.py --csv assets/sentence/sentence.csv --mp4-dir assets/Asl --out-dir assets/sentence/ --num 5
```

**Options:**
- `--csv`: Path to sentences CSV file containing `sentence_id` and `ref_ids` columns
- `--mp4-dir`: Directory with source MP4 files named `<ref_id>.mp4`
- `--out-dir`: Output root directory where per-sentence subfolders will be created
- `--num`: Number of sentences to process (default: 10)

**Note:** Automatically filters out placeholder tokens starting with `NO_REF`.

### extract_all_frames_seq.py
Extracts all frames from sentence videos in the order specified by CSV ref_ids.

**Usage:**
```bash
python sentence/extract_all_frames_seq.py --csv <sentences.csv> --mp4-root <mp4_directory> --out-root <output_directory> [--num <count>]
```

**Options:**
- `--csv`: Path to sentences CSV file
- `--mp4-root`: Root directory containing processed MP4s organized as `<sentence_id>/{ref_id}_{timestamp}.mp4`
- `--out-root`: Output directory for extracted frames
- `--num`: Optional limit on number of sentences to process

**Output Format:**
- Frames saved as `<out-root>/<sentence_id>/{frame_id}_{ref_id}.jpg`
- Frame IDs start from 0 and increment across all videos in a sentence

### Other Sentence Scripts
- `combine_frames_and_interp.py`: Combines and interpolates frames
- `crop_resize_frames.py`: Crops and resizes extracted frames
- `crop_resize_gifs_right.py`: Processes GIF files with crop and resize
- `extract_boundary_frames.py`: Extracts first and last frames from videos

## SLURM Job Scripts (`slurm/`)

Scripts for running inference jobs on HPC clusters with SLURM workload manager.

### infer_raw_array.sh
SLURM batch script for running parallel inference tasks using job arrays.

**Configuration:**
- Job array: 1-5 tasks
- Resources per task:
  - 1 GPU
  - 32 CPUs
  - 32GB RAM
  - 24 hour time limit
- Partition: nvidia

**Usage:**
```bash
sbatch slurm/infer_raw_array.sh
```

**Runs:**
```bash
python inference_raw_batch.py \
  --inference_config configs/test.yaml \
  --batch_folder assets/sentence/W10_00000${ARRAY_ID}
```

## Dependencies

Required Python packages:
- `opencv-python` (cv2)
- Standard library: `argparse`, `csv`, `os`, `math`, `shutil`, `pathlib`

## Directory Structure

```
scripts/
  |-- README.md                           # This file
  |-- split_files.py                      # Split MP4 files into parts
  |-- stat_mp4.py                         # Get video metadata
  |
  |-- sentence/                           # Sentence dataset processing
  |     |-- split_sentence_mp4.py         # Organize videos by sentence
  |     |-- extract_all_frames_seq.py     # Extract frames in sequence
  |     |-- combine_frames_and_interp.py
  |     |-- crop_resize_frames.py
  |     |-- crop_resize_gifs_right.py
  |     `-- extract_boundary_frames.py
  |
  `-- slurm/                              # HPC cluster job scripts
        `-- infer_raw_array.sh            # Parallel inference array job
```

## Notes

- Most scripts expect specific directory structures and file naming conventions
- CSV files should be UTF-8 encoded (scripts handle UTF-8 with BOM)
- MP4 file naming convention for sentence processing: `{ref_id}_{timestamp}.mp4`
- All scripts include error handling for missing files and directories
