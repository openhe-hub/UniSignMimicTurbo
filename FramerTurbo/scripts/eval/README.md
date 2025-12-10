# Training Log Analysis Tool

Pure Python tool for parsing and analyzing FramerTurbo training logs. Cross-platform compatible.

## Usage

```bash
# Basic usage - print summary only
python scripts/eval/parse_training_log.py logs/train_1210.err --no-plot

# Generate plots (auto-saves next to log file)
python scripts/eval/parse_training_log.py logs/train_1210.err

# Save plot to specific location
python scripts/eval/parse_training_log.py logs/train_1210.err --plot results/my_plot.png
```

## Requirements

```bash
# Core parsing (text summary only)
# No dependencies needed!

# For visualization
pip install matplotlib numpy
```

## Features

- **Cross-platform**: Pure Python, works on Windows/Linux/Mac
- **Zero dependencies for text output**: Only matplotlib needed for plots
- Extract loss curves, learning rates, and training statistics
- Generate training curve visualizations
- Print comprehensive training summaries
- Per-epoch statistics

## Output

### Console Summary

```
======================================================================
                         TRAINING SUMMARY
======================================================================

[Configuration]
  Dataset: 351 examples
  Epochs: 10
  Batch size: 1
  Gradient accumulation: 8
  Effective batch size: 8
  Total steps: 430

[LoRA Configuration]
  Rank: 64
  Alpha: 64

[Model Parameters]
  Trainable: 52,117,504
  Total: 1,628,860,974
  Trainable %: 3.1997%

[Training Statistics]
  Initial loss: 2.8400
  Final loss: 1.0100
  Loss reduction: 1.8300 (64.44%)
  Min loss: 0.9650
  Mean loss: 2.0234 +/- 0.5123

[Per-Epoch Statistics]
  Epoch    Mean Loss    Min Loss     Final Loss
  -------- ------------ ------------ ------------
  0        2.8765       2.4700       2.7200
  1        2.8234       2.5200       2.7500
  ...
  9        1.1456       0.9650       1.0100

[Output]
  Model saved to: outputs/lora_576x576/final
======================================================================
```

### Training Curves Plot

Generates a 2-subplot figure:
1. **Loss Curve**: Raw and smoothed training loss over steps
   - Epoch boundaries marked with vertical lines
   - Smoothed curve for trend visualization
2. **Learning Rate Schedule**: Learning rate over training steps

## Quick Start

```bash
# Find your log file
ls logs/

# Run analysis (text only)
python scripts/eval/parse_training_log.py logs/train_1210.err --no-plot

# Generate visualization
python scripts/eval/parse_training_log.py logs/train_1210.err
# Plot saved as: logs/train_1210_curves.png
```

## Troubleshooting

**No training data found:**
- Ensure the log file contains training progress lines
- Check log format matches expected pattern

**matplotlib not installed:**
```bash
pip install matplotlib numpy
# Or use --no-plot flag for text-only output
```

**Plot generation fails:**
- Use `--no-plot` to check if parsing works
- Verify write permissions to output directory
