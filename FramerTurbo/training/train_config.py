"""
Example training configuration for FramerTurbo LoRA fine-tuning
You can use this as a starting point and modify based on your needs
"""

# ==============================================================================
# Model Configuration
# ==============================================================================
MODEL_CONFIG = {
    # Path to pretrained FramerTurbo model
    "pretrained_model_path": "checkpoints/framer_512x320",

    # Path to Stable Video Diffusion base model
    "svd_model_path": "checkpoints/stable-video-diffusion-img2vid-xt",
}

# ==============================================================================
# Data Configuration
# ==============================================================================
DATA_CONFIG = {
    # Directory containing training data
    "data_dir": "data/training_videos",

    # Dataset type: "video" or "image_pair"
    "dataset_type": "video",

    # Number of frames to sample per video
    "num_frames": 3,

    # Frame dimensions
    "height": 320,
    "width": 512,

    # Dataloader workers
    "num_workers": 4,
}

# ==============================================================================
# Training Configuration
# ==============================================================================
TRAINING_CONFIG = {
    # Output directory
    "output_dir": "outputs/lora_finetune",

    # Batch size per device
    "train_batch_size": 1,

    # Gradient accumulation steps
    # Effective batch size = train_batch_size × gradient_accumulation_steps
    "gradient_accumulation_steps": 4,

    # Number of training epochs
    "num_train_epochs": 10,

    # Learning rate
    "learning_rate": 1e-4,

    # Learning rate scheduler
    "lr_scheduler": "constant_with_warmup",  # Options: linear, cosine, constant, constant_with_warmup

    # Warmup steps
    "lr_warmup_steps": 500,

    # Optimizer parameters
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_weight_decay": 1e-2,
    "adam_epsilon": 1e-8,

    # Gradient clipping
    "max_grad_norm": 1.0,

    # Mixed precision training
    "mixed_precision": "fp16",  # Options: "no", "fp16", "bf16"

    # Noise offset for better training stability
    "noise_offset": 0.05,

    # Prediction type
    "prediction_type": "epsilon",  # Options: "epsilon", "v_prediction"

    # Random seed
    "seed": 42,
}

# ==============================================================================
# LoRA Configuration
# ==============================================================================
LORA_CONFIG = {
    # LoRA rank (higher = more capacity but more memory)
    "lora_rank": 64,

    # LoRA alpha (usually same as rank)
    "lora_alpha": 64,

    # LoRA dropout
    "lora_dropout": 0.0,

    # Which modules to train
    "train_unet": True,
    "train_controlnet": False,  # Set to True if you want to train ControlNet as well
}

# ==============================================================================
# Checkpointing Configuration
# ==============================================================================
CHECKPOINT_CONFIG = {
    # Save checkpoint every N steps
    "checkpointing_steps": 500,

    # Resume from checkpoint (None or path to checkpoint)
    "resume_from_checkpoint": None,
}

# ==============================================================================
# Logging Configuration
# ==============================================================================
LOGGING_CONFIG = {
    # Logging directory
    "logging_dir": "logs",

    # Report to (tensorboard, wandb, or all)
    "report_to": "tensorboard",

    # Use Weights & Biases
    "use_wandb": False,

    # W&B project name
    "wandb_project": "framer-turbo-lora",
}

# ==============================================================================
# Hardware-Specific Presets
# ==============================================================================

# Preset for RTX 3090 / RTX 4090 (24GB)
PRESET_24GB = {
    **TRAINING_CONFIG,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "mixed_precision": "fp16",
    "lora_rank": 64,
}

# Preset for A100 40GB
PRESET_40GB = {
    **TRAINING_CONFIG,
    "train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "mixed_precision": "fp16",
    "lora_rank": 128,
}

# Preset for A100 80GB
PRESET_80GB = {
    **TRAINING_CONFIG,
    "train_batch_size": 4,
    "gradient_accumulation_steps": 2,
    "mixed_precision": "bf16",
    "lora_rank": 128,
}

# Preset for limited memory (16GB)
PRESET_16GB = {
    **TRAINING_CONFIG,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "mixed_precision": "fp16",
    "lora_rank": 32,
    "num_frames": 2,  # Reduce frames to save memory
}

# ==============================================================================
# Usage Example
# ==============================================================================
"""
To use a preset, you can pass it as command-line arguments:

# For 40GB GPU
python train_lora.py \\
    --pretrained_model_path checkpoints/framer_512x320 \\
    --data_dir data/training_videos \\
    --output_dir outputs/lora_finetune \\
    --train_batch_size 2 \\
    --gradient_accumulation_steps 4 \\
    --lora_rank 128 \\
    --train_unet

Or create a wrapper script that uses these configs programmatically.
"""
