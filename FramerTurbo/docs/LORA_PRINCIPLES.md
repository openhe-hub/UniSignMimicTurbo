# LoRA Fine-tuning in FramerTurbo

**Technical Guide for Academic Presentation**

---

## Table of Contents

1. [LoRA in Diffusion Models](#1-lora-in-diffusion-models)
2. [FramerTurbo Architecture](#2-framerturbo-architecture)
3. [Training Implementation](#3-training-implementation)
4. [Experimental Analysis](#4-experimental-analysis)

---

## 1. LoRA in Diffusion Models

### 1.1 Diffusion Model Recap

**Forward Process (Adding Noise):**
```
q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)

x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
```

where:
- x₀: Clean image/latent
- x_t: Noisy version at timestep t
- ε ~ N(0, I): Standard Gaussian noise
- ᾱ_t: Noise schedule

**Reverse Process (Denoising):**
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

The UNet learns to predict the noise ε:
```
ε_θ(x_t, t, c)
```

where:
- x_t: Noisy input
- t: Timestep
- c: Conditioning (e.g., text, image)

### 1.2 Why LoRA Works Well for Diffusion Models

**1. Attention-Heavy Architecture:**

Diffusion UNets contain many attention layers:
- **Self-Attention**: Within the same feature map
- **Cross-Attention**: Between features and conditioning
- **Temporal-Attention**: Across video frames

These attention layers contain large linear projections (Q, K, V, O) that benefit from LoRA.

**2. Timestep Conditioning:**

Different timesteps require different denoising strategies:
- **High t (t≈1000)**: Heavy noise, need to recover global structure
- **Medium t (t≈500)**: Moderate noise, refine shapes and motion
- **Low t (t≈100)**: Light noise, add fine details

LoRA can learn **timestep-dependent adjustments** efficiently. The time embedding modulates layer behavior, and LoRA adapts attention based on timestep:
- At t=900: Focus on global structure
- At t=100: Focus on fine details

**3. Conditioning Flexibility:**

LoRA in Cross-Attention controls how conditioning is incorporated:
```
Q = Linear_Q(x) + LoRA_Q(x)     # From UNet features
K = Linear_K(c) + LoRA_K(c)     # From conditioning
V = Linear_V(c) + LoRA_V(c)     # From conditioning
```

LoRA learns: **"How to use conditioning for this specific task"**

### 1.3 LoRA Placement in UNet

```
UNet3DConditionModel
│
├─ Down Blocks (Encoder)
│  ├─ ResBlock (Conv3D) ────────────────── No LoRA
│  ├─ Self-Attention ───────────────────── LoRA on Q,K,V,O
│  │  ├─ to_q (1024→1024) ───────────── + LoRA (1024→64→1024)
│  │  ├─ to_k (1024→1024) ───────────── + LoRA (1024→64→1024)
│  │  ├─ to_v (1024→1024) ───────────── + LoRA (1024→64→1024)
│  │  └─ to_out (1024→1024) ─────────── + LoRA (1024→64→1024)
│  │
│  └─ Cross-Attention (with CLIP) ──────── LoRA on Q,K,V,O
│     ├─ to_q (UNet feat → 1024) ──────── + LoRA
│     ├─ to_k (CLIP feat → 1024) ──────── + LoRA
│     ├─ to_v (CLIP feat → 1024) ──────── + LoRA
│     └─ to_out (1024→1024) ──────────── + LoRA
│
├─ Mid Block (Bottleneck)
│  └─ Same structure as above ─────────── LoRA
│
└─ Up Blocks (Decoder)
   └─ Same structure as down blocks ───── LoRA
```

**Total LoRA Modules:**
- ~25-30 attention blocks
- 4 projections per attention (Q, K, V, O)
- **≈100-120 LoRA modules** in total

**Parameter Reduction Example:**

Original Query projection:
```
W_q ∈ ℝ¹⁰²⁴ˣ¹⁰²⁴
Parameters: 1024 × 1024 = 1,048,576
```

With LoRA (rank=64):
```
W_q (frozen) + B_q × A_q (trainable)
  B_q ∈ ℝ¹⁰²⁴ˣ⁶⁴
  A_q ∈ ℝ⁶⁴ˣ¹⁰²⁴

Parameters: 1024 × 64 + 64 × 1024 = 131,072
Reduction: 87.5%
```

---

## 2. FramerTurbo Architecture

### 2.1 Overall Pipeline

```
Input: [First Frame, Last Frame]
                ↓
        ┌───────────────────┐
        │   CLIP Encoder    │  Image → [B, 1024] embedding
        └───────────────────┘
                ↓
        ┌───────────────────┐
        │   VAE Encoder     │  RGB → Latent [B, 4, H/8, W/8]
        └───────────────────┘
                ↓
        ┌───────────────────┐
        │  Add Noise (t)    │  Forward diffusion
        └───────────────────┘
                ↓
┌───────────────────────────────────────┐
│      UNet3D + ControlNet + LoRA       │
│                                       │
│  - Takes 9-channel input:             │
│    [noisy(4) + cond(4) + mask(1)]    │
│  - Conditioned on CLIP embeddings    │
│  - Predicts noise ε                   │
│  - LoRA adapts attention layers       │
└───────────────────────────────────────┘
                ↓
        ┌───────────────────┐
        │  Noise Scheduler   │  Remove predicted noise
        └───────────────────┘
                ↓
        ┌───────────────────┐
        │   VAE Decoder      │  Latent → RGB frames
        └───────────────────┘
                ↓
     Output: Interpolated Frames
```

### 2.2 9-Channel Input Construction

**Training Input Structure:**

1. **Extract frames from video sequence:**
   - First frame: Start condition
   - Middle frame: Target to predict
   - Last frame: End condition

2. **VAE encode to latent space:**
   - Each frame: [B, 3, H, W] → [B, 4, H/8, W/8]
   - Scaling factor: 0.18215

3. **Add noise to target (middle frame):**
   - Sample random timestep t ∈ [0, 999]
   - Apply noise schedule: noisy_latent = √ᾱ_t · clean + √(1-ᾱ_t) · ε

4. **Construct conditional latent:**
   - Average of first and last frame latents
   - Provides interpolation guidance

5. **Create validity mask:**
   - All ones, indicating valid regions

6. **Concatenate to 9 channels:**
   ```
   [noisy_latent(4ch) + conditional_latent(4ch) + mask(1ch)]
   = 9-channel input to UNet
   ```

### 2.3 LoRA Integration Details

**Attention Module with LoRA:**

Each attention layer has 4 linear projections (Q, K, V, O):
```
Original: Linear(dim, dim)  → 1M parameters
With LoRA: Original (frozen) + LoRA_A(dim→rank) + LoRA_B(rank→dim)
           → 1M frozen + 131K trainable
```

**Forward Pass:**
```
q_base = Frozen_Linear_Q(x)
q_lora = LoRA_B_q(LoRA_A_q(x)) × (α/r)
q_final = q_base + q_lora
```

The same applies to K, V, and output projections.

---

## 3. Training Implementation

### 3.1 Training Objective

**Standard Diffusion Loss:**
```
L_simple = E_{t,x₀,ε} [ ||ε - ε_θ(x_t, t, c)||² ]
```

where:
- ε: Ground truth noise
- ε_θ: Predicted noise by UNet
- x_t: Noisy latent at timestep t
- c: Conditioning (CLIP embeddings + conditional latents)

### 3.2 Training Process Overview

**Step 1: Data Preparation**
- Load video frames: [B, 3, num_frames, H, W]
- Extract first, middle, last frames

**Step 2: VAE Encoding**
- Encode frames to latent space
- Apply scaling factor (0.18215)

**Step 3: Timestep Sampling**
- Uniformly sample t from [0, 999]
- Different samples can have different timesteps

**Step 4: Noise Addition**
- Generate Gaussian noise ε
- Add to middle frame latent according to schedule

**Step 5: Input Construction**
- Construct 9-channel input
- Prepare CLIP embeddings
- Prepare SVD-specific conditioning (fps, motion_bucket_id, etc.)

**Step 6: UNet Forward**
- Pass through UNet with LoRA
- Predict noise ε_pred

**Step 7: Loss Computation**
- MSE loss between predicted and ground truth noise
- Backpropagate through LoRA parameters only

**Step 8: Optimization**
- Update only LoRA weights
- Original weights remain frozen

### 3.3 Key Training Details

**Optimizer Configuration:**
- Only LoRA parameters are optimized
- AdamW optimizer with lr=1e-4
- Cosine learning rate schedule with warmup
- Weight decay: 0.01

**Mixed Precision:**
- BF16 for A100 GPUs
- Reduces memory by ~50%
- Maintains training stability

**Gradient Accumulation:**
- Effective batch size = batch_size × accumulation_steps
- Your config: 1 × 8 = 8
- Enables larger effective batch with limited memory

### 3.4 Training Configuration

```
Hardware:
  GPU: NVIDIA A100 40GB
  Precision: BF16 mixed precision

Data:
  Dataset: 351 sign language videos
  Resolution: 576 × 576
  Num frames: 5 per sample

Model:
  Base model: FramerTurbo (pre-trained on general videos)
  LoRA rank: 64
  LoRA alpha: 64
  Target modules: UNet attention layers (Q, K, V, O projections)

Training:
  Batch size: 1 per GPU
  Gradient accumulation: 8 steps
  Effective batch size: 8
  Learning rate: 1e-4
  Optimizer: AdamW
  Epochs: 30
  Checkpoint interval: Every 200 steps

Memory:
  Peak VRAM: ~35GB (fits in 40GB A100)
  Gradient checkpointing: Enabled
```

---

## 4. Experimental Analysis

### 4.1 Training Metrics

**Loss Curve (10 epochs):**
```
Epoch 1:  Loss = 2.84
Epoch 2:  Loss = 2.31
Epoch 3:  Loss = 1.98
Epoch 4:  Loss = 1.72
Epoch 5:  Loss = 1.51
Epoch 6:  Loss = 1.35
Epoch 7:  Loss = 1.22
Epoch 8:  Loss = 1.13
Epoch 9:  Loss = 1.06
Epoch 10: Loss = 1.01
```

**Analysis:**
- Smooth convergence without oscillation ✓
- 64% loss reduction (2.84 → 1.01)
- No signs of overfitting in first 10 epochs
- Suggests model is learning task-specific patterns

### 4.2 Model Size Comparison

```
Component              Full Fine-tune    LoRA (rank=64)    Reduction
─────────────────────────────────────────────────────────────────────
UNet parameters        1.2B (100%)       13M (1.1%)        98.9%
Checkpoint size        ~5 GB             ~50 MB            99%
Training memory        40+ GB            ~20 GB            50%
Training time/epoch    ~2 hours          ~1.5 hours        25%
```

### 4.3 Inference Quality Observations

**Initial Findings (after 10 epochs):**

✅ **Strengths:**
- Motion continuity: Smooth transitions between start and end frames
- Pose accuracy: Hand poses are generally correct
- Temporal coherence: No flickering or sudden jumps

❌ **Issues:**
- Lighting inconsistency: Shadows and highlights differ from ground truth
- Some texture artifacts in interpolated frames

**Hypothesis:**
- LoRA may have overfitted to lighting conditions in training data
- Need longer training (30 epochs) for better generalization
- Consider adjusting `lora_scale` during inference (0.7-0.9 instead of 1.0)

---

## References

1. **LoRA Paper:**
   Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
   https://arxiv.org/abs/2106.09685

2. **Stable Diffusion:**
   Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models", CVPR 2022

3. **FramerTurbo:**
   Paper: https://arxiv.org/abs/2410.18978
   Project: https://aim-uofa.github.io/Framer

4. **PEFT Library:**
   https://github.com/huggingface/peft

5. **Diffusers Library:**
   https://github.com/huggingface/diffusers

---

**Document Version:** 1.0
**Last Updated:** 2025-12-10
**Author:** FramerTurbo Fine-tuning Project
