"""
Parse and analyze FramerTurbo training logs
Extracts loss curves, learning rate, and training statistics
"""

import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path: str) -> Dict:
    """
    Parse training log file and extract metrics

    Args:
        log_path: Path to the log file (.err or .log)

    Returns:
        Dictionary containing parsed metrics
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    metrics = {
        'steps': [],
        'epochs': [],
        'losses': [],
        'learning_rates': [],
        'timestamps': [],
    }

    # Parse training progress lines
    # Format: Training:  XX%|...| step/total [time<time, speed, epoch=X, loss=X.XX, lr=X.XXe-X]
    pattern = r'Training:\s+\d+%.*?\|\s*(\d+)/\d+\s+\[.*?epoch=(\d+),\s*loss=([\d.]+),\s*lr=([\d.e+-]+)\]'

    matches = re.finditer(pattern, content)
    for match in matches:
        step = int(match.group(1))
        epoch = int(match.group(2))
        loss = float(match.group(3))
        lr = float(match.group(4))

        metrics['steps'].append(step)
        metrics['epochs'].append(epoch)
        metrics['losses'].append(loss)
        metrics['learning_rates'].append(lr)

    # Parse training configuration
    config_pattern = r'Num examples = (\d+).*?Num epochs = (\d+).*?Batch size per device = (\d+).*?Gradient accumulation steps = (\d+).*?Total optimization steps = (\d+)'
    config_match = re.search(config_pattern, content, re.DOTALL)

    if config_match:
        metrics['config'] = {
            'num_examples': int(config_match.group(1)),
            'num_epochs': int(config_match.group(2)),
            'batch_size': int(config_match.group(3)),
            'gradient_accum': int(config_match.group(4)),
            'total_steps': int(config_match.group(5)),
        }

    # Parse LoRA config
    lora_pattern = r'Setting up LoRA for UNet \(rank=(\d+), alpha=(\d+)\)'
    lora_match = re.search(lora_pattern, content)
    if lora_match:
        metrics['lora_config'] = {
            'rank': int(lora_match.group(1)),
            'alpha': int(lora_match.group(2)),
        }

    # Parse trainable parameters
    params_pattern = r'trainable params: ([\d,]+) \|\| all params: ([\d,]+) \|\| trainable%: ([\d.]+)'
    params_match = re.search(params_pattern, content)
    if params_match:
        metrics['parameters'] = {
            'trainable': int(params_match.group(1).replace(',', '')),
            'total': int(params_match.group(2).replace(',', '')),
            'trainable_percent': float(params_match.group(3)),
        }

    # Parse final save message
    final_pattern = r'Saved final model to (.+)'
    final_match = re.search(final_pattern, content)
    if final_match:
        metrics['output_dir'] = final_match.group(1).strip()

    return metrics


def compute_statistics(metrics: Dict) -> Dict:
    """Compute training statistics"""
    losses = np.array(metrics['losses'])

    stats = {}

    if len(losses) > 0:
        stats['initial_loss'] = losses[0]
        stats['final_loss'] = losses[-1]
        stats['loss_reduction'] = losses[0] - losses[-1]
        stats['loss_reduction_percent'] = (losses[0] - losses[-1]) / losses[0] * 100
        stats['min_loss'] = np.min(losses)
        stats['max_loss'] = np.max(losses)
        stats['mean_loss'] = np.mean(losses)
        stats['std_loss'] = np.std(losses)

    # Compute per-epoch statistics
    epochs = np.array(metrics['epochs'])
    unique_epochs = sorted(set(epochs))

    stats['epoch_stats'] = []
    for epoch in unique_epochs:
        epoch_mask = epochs == epoch
        epoch_losses = losses[epoch_mask]

        if len(epoch_losses) > 0:
            stats['epoch_stats'].append({
                'epoch': epoch,
                'mean_loss': np.mean(epoch_losses),
                'min_loss': np.min(epoch_losses),
                'max_loss': np.max(epoch_losses),
                'final_loss': epoch_losses[-1],
            })

    return stats


def plot_training_curves(metrics: Dict, output_path: str = None):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    steps = metrics['steps']
    losses = metrics['losses']
    lrs = metrics['learning_rates']
    epochs = metrics['epochs']

    # Plot 1: Loss curve
    ax1 = axes[0]
    ax1.plot(steps, losses, linewidth=1.5, alpha=0.7, label='Training Loss')

    # Add smoothed loss curve
    if len(losses) > 20:
        window = min(50, len(losses) // 10)
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        smoothed_steps = steps[window-1:]
        ax1.plot(smoothed_steps, smoothed, linewidth=2.5, color='red',
                label=f'Smoothed (window={window})')

    # Mark epoch boundaries
    epoch_changes = [0]
    for i in range(1, len(epochs)):
        if epochs[i] != epochs[i-1]:
            epoch_changes.append(i)

    for idx in epoch_changes[1:]:
        ax1.axvline(x=steps[idx], color='gray', linestyle='--', alpha=0.3)

    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Learning rate
    ax2 = axes[1]
    ax2.plot(steps, lrs, linewidth=1.5, color='green')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved plot to: {output_path}")
    else:
        plt.show()

    plt.close()


def print_summary(metrics: Dict, stats: Dict):
    """Print training summary"""
    print("\n" + "="*70)
    print("TRAINING SUMMARY".center(70))
    print("="*70)

    # Configuration
    if 'config' in metrics:
        config = metrics['config']
        print(f"\n[Configuration]")
        print(f"  Dataset: {config['num_examples']} examples")
        print(f"  Epochs: {config['num_epochs']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Gradient accumulation: {config['gradient_accum']}")
        print(f"  Effective batch size: {config['batch_size'] * config['gradient_accum']}")
        print(f"  Total steps: {config['total_steps']}")

    # LoRA config
    if 'lora_config' in metrics:
        lora = metrics['lora_config']
        print(f"\n[LoRA Configuration]")
        print(f"  Rank: {lora['rank']}")
        print(f"  Alpha: {lora['alpha']}")

    # Parameters
    if 'parameters' in metrics:
        params = metrics['parameters']
        print(f"\n[Model Parameters]")
        print(f"  Trainable: {params['trainable']:,}")
        print(f"  Total: {params['total']:,}")
        print(f"  Trainable %: {params['trainable_percent']:.4f}%")

    # Training statistics
    if stats:
        print(f"\n[Training Statistics]")
        print(f"  Initial loss: {stats['initial_loss']:.4f}")
        print(f"  Final loss: {stats['final_loss']:.4f}")
        print(f"  Loss reduction: {stats['loss_reduction']:.4f} ({stats['loss_reduction_percent']:.2f}%)")
        print(f"  Min loss: {stats['min_loss']:.4f}")
        print(f"  Mean loss: {stats['mean_loss']:.4f} +/- {stats['std_loss']:.4f}")

    # Per-epoch statistics
    if 'epoch_stats' in stats and len(stats['epoch_stats']) > 0:
        print(f"\n[Per-Epoch Statistics]")
        print(f"  {'Epoch':<8} {'Mean Loss':<12} {'Min Loss':<12} {'Final Loss':<12}")
        print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12}")

        for epoch_stat in stats['epoch_stats']:
            print(f"  {epoch_stat['epoch']:<8} "
                  f"{epoch_stat['mean_loss']:<12.4f} "
                  f"{epoch_stat['min_loss']:<12.4f} "
                  f"{epoch_stat['final_loss']:<12.4f}")

    # Output location
    if 'output_dir' in metrics:
        print(f"\n[Output]")
        print(f"  Model saved to: {metrics['output_dir']}")

    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Parse FramerTurbo training logs')
    parser.add_argument('log_file', type=str, help='Path to training log file')
    parser.add_argument('--plot', type=str, default=None,
                       help='Save plot to this path (e.g., training_curves.png)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Do not generate plots')
    args = parser.parse_args()

    log_path = Path(args.log_file)

    if not log_path.exists():
        print(f"[ERROR] Log file not found: {log_path}")
        return

    print(f"[INFO] Parsing log file: {log_path}")

    # Parse log file
    metrics = parse_log_file(str(log_path))

    if not metrics['steps']:
        print("[ERROR] No training data found in log file")
        return

    print(f"[INFO] Found {len(metrics['steps'])} training steps")

    # Compute statistics
    stats = compute_statistics(metrics)

    # Print summary
    print_summary(metrics, stats)

    # Generate plots
    if not args.no_plot:
        try:
            import matplotlib
            if args.plot:
                plot_path = args.plot
            else:
                plot_path = log_path.parent / f"{log_path.stem}_curves.png"

            plot_training_curves(metrics, str(plot_path))
        except ImportError:
            print("[WARNING] matplotlib not installed, skipping plots")
            print("          Install with: pip install matplotlib")
        except Exception as e:
            print(f"[WARNING] Failed to generate plots: {e}")


if __name__ == '__main__':
    main()
