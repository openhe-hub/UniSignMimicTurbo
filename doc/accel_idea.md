# Overview of Diffusion Speedup Directions

> Scope: diffusion-based generative models only (no flow matching / rectified flow here).

Modern diffusion speedups can roughly be grouped into four directions:

1. **Training-free efficient samplers (ODE/SDE solvers)**
2. **Distillation: multi-step → few-step / one-step**
3. **Consistency / Latent Consistency Models (LCM / LCM-LoRA)**
4. **Engineering-level pipeline tricks**

Below is a brief description of each category plus representative papers / repos.

---

## 1. Training-Free Samplers: Smarter ODE/SDE Solvers

**Idea:** View diffusion sampling as solving an ODE (probability flow ODE) and design dedicated high-order solvers that approximate the full trajectory in very few steps (5–20), **without retraining** the model.

Representative work:

- **DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps**  
  NeurIPS 2022  
  Paper: <https://arxiv.org/abs/2206.00927>  
  OpenReview: <https://openreview.net/forum?id=2uAaGwlP_V>  
  Official code (also includes DPM-Solver++): <https://github.com/LuChengTHU/dpm-solver>

Key points:

- Analytically solve the “linear part” of the diffusion ODE, and only numerically integrate the nonlinear residual.
- Supports various parameterizations (ε / v / x₀ prediction) and guided sampling (classifier-free guidance, etc.).
- Drop-in replacement for DDIM / Euler schedulers in many frameworks (e.g., Stable Diffusion with `diffusers`).

> **Good for:** quick, low-risk speedups on existing SD / MimicMotion pipelines with no retraining.

---

## 2. Distillation: Compressing Multi-Step Diffusion into Few-Step / One-Step

**Idea:** Treat a high-quality multi-step diffusion sampler as a **teacher** and train a **student** that requires far fewer steps (or just one step), trading extra training for faster inference.

### 2.1 Multi-Step → Few-Step: Progressive Distillation

- **Progressive Distillation for Fast Sampling of Diffusion Models**  
  ICLR 2022  
  Paper: <https://arxiv.org/abs/2202.00512>  
  OpenReview: <https://openreview.net/forum?id=TIdIXIpzhoI>  

Concept:

- Start from a teacher requiring `N` sampling steps.
- Distill it into a student using `N/2` steps by training the student to mimic **two teacher steps at once**.
- Repeat this process: `N → N/2 → N/4 → …` until you reach 4–8 steps, while retaining most of the perceptual quality.

> **Good for:** compressing a very slow but strong diffusion teacher into a moderately fast few-step sampler (e.g., 50–100 steps → 4–8 steps).

### 2.2 Multi-Step → One-Step: One-Step Diffusion Distillation

**Idea:** Train a generator `G(z)` to directly map Gaussian noise to images, so that the output distribution matches the teacher diffusion’s multi-step outputs.

Representative lines of work (not exhaustive):

- **Distribution Matching Distillation (DMD)** — one-step generator trained by matching distributions between teacher outputs and student outputs (e.g., via KL / perceptual losses).
- **Score Implicit Matching (SIM)** — uses score-based divergences and implicit matching; relies on teacher’s score function.
- **Unified frameworks (e.g., “Uni-Instruct” style)** — unify different one-step distillation methods under a generalized divergence / instruction perspective.

> **Good for:** research on **one-step T2I / T2V**, where the teacher is SD / AnimateDiff / MimicMotion and the goal is real-time generation.

---

## 3. Consistency Models & Latent Consistency Models (LCM / LCM-LoRA)

**Idea:** Instead of iteratively denoising along a fine-grained time grid, train a “consistency mapping” that takes a noisy sample at any time `t` directly back to the clean data, and enforces that outputs along the same trajectory are consistent. This naturally supports one- or few-step sampling.

### 3.1 Consistency Models (CM)

- **Consistency Models**  
  Paper: <https://arxiv.org/abs/2303.01469>  
  Official repo: <https://github.com/openai/consistency_models>  

Core properties:

- Direct one-step mapping from noise to data, with optional multi-step refinement.
- Can be trained by distilling pre-trained diffusion models or from scratch as standalone models.
- Supports zero-shot editing (inpainting, super-resolution, colorization) without task-specific retraining.

### 3.2 Latent Consistency Models (LCM) & LCM-LoRA

- **Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference**  
  Paper: <https://arxiv.org/abs/2310.04378>  
  OpenReview: <https://openreview.net/forum?id=duBCwjb68o>  
  Project page: <https://latent-consistency-models.github.io>  
  Official repo (includes LCM-LoRA): <https://github.com/luosiallen/latent-consistency-model>

Key ideas:

- Apply consistency training in **latent space** of latent diffusion models (LDMs), such as Stable Diffusion.
- View guided reverse diffusion as solving an augmented probability flow ODE (PF-ODE) in latent space, and directly predict its solution.
- **LCM-LoRA**: a universal acceleration module implemented as a LoRA adapter on top of pre-trained SD / SDXL models.
  - Typically achieves usable image quality in **2–4 steps**, and near-original quality in **4–8 steps**, instead of 20–30 steps.

> **Good for:** plug-and-play acceleration of existing SD pipelines — simply load an LCM-LoRA and use an LCM-style scheduler.

---

## 4. Engineering-Level Tricks

Not new model families, but important for real-world systems. Examples:

- **Resolution strategies**  
  - Generate at a lower resolution then upsample via super-resolution models.
- **Spatial cropping / tiling**  
  - Generate or refine only relevant regions and stitch results together.
- **Reducing expensive condition computations**  
  - Cache and reuse text / pose / control features instead of recomputing at every step.
- **Parallelism & batching**  
  - Multi-GPU / multi-batch sampling, caching intermediate results across a sequence, etc.

These tricks can—and usually should—be combined with any of the three algorithmic directions above.

---

## TL;DR

- **Fastest “no retrain” path:**  
  - Swap to **DPM-Solver++** (or similar) for training-free speedup.  
  - Add **LCM-LoRA** on SD / SDXL to reduce sampling to 2–8 steps.

- **If you’re okay with extra training:**  
  - Use **Progressive Distillation** to compress a strong teacher into a few-step student.

- **If you’re aiming for a research contribution in one-step generation:**  
  - Build on **one-step diffusion distillation** (DMD / SIM / unified divergence frameworks) or on **Consistency / LCM** architectures.
