# MimicMotion Raw Inference: LCM LoRA 加速说明

## 功能概览
- `inference_raw_turbo.py` 现在支持在创建 MimicMotion pipeline 后加载 LCM LoRA，并可选将调度器替换为 `LCMScheduler`。  
- 目标是把推理步数从原先的 ~25 步降到 4~8 步，显著缩短整条流水线耗时。

## 主要参数
| 参数名 | 说明 |
| --- | --- |
| `--lcm_lora_path` | LCM LoRA 的本地路径或 HF repo id。不填则保持旧流程。 |
| `--lcm_weight_name` | （可选）指定权重文件名，如 `pytorch_lora_weights.safetensors`。 |
| `--lcm_lora_scale` | 融合 LoRA 的缩放系数，默认 `1.0`；可下调到 0.5~0.8 来平衡质量。 |
| `--no_fuse_lcm_lora` | 仅加载 LoRA 而不融合，便于动态切换或调试。 |
| `--skip_lcm_scheduler` | 跳过调度器替换，继续使用 Euler 等原始设置。 |
| `--lcm_beta_schedule` | 替换为 LCM 后的 beta schedule，默认 `linear`。 |

## 使用示例
```bash
python inference_raw_turbo.py \
  --inference_config configs/test.yaml \
  --batch_folder assets/bad_videos \
  --lcm_lora_path models/anim_lcm_lora/AnimateLCM-SVD-xt-1.1.safetensors \
  --lcm_steps 6 \
  --lcm_cfg 1.0
```
> 提示：结合 LCM LoRA 时可将 `configs/mimicmotion/test.yaml` 中的 `num_inference_steps` 调到 6~8，再按质量需求微调。

## 质量与性能建议
- LCM 采样步数越低，推理越快，但细节/动作可能更柔和；建议从 8 步起逐步下探，并观察输出。  
- 如果发现画面漂浮或纹理模糊，可尝试：
  1. 略微提高步数（例如 10 步）。
  2. 减小 `--lcm_lora_scale`（例如 0.7）。
  3. 暂时加上 `--no_fuse_lcm_lora` 只在推理阶段注入，方便比较差异。  
- 随时可通过 `--skip_lcm_scheduler`/不传 `--lcm_lora_path` 退回旧流程，确保对比公平。

## 依赖与兼容
- 推荐使用 `diffusers` ≥ 0.24，以便直接通过 `pipeline.load_lora_weights`/`pipeline.fuse_lora` 管理 LoRA，并保证 `LCMScheduler` 可用。  
- 如果受限于旧版本，脚本会依次尝试：`DiffusionPipeline.load_lora_weights` → `UNet.load_attn_procs` → **手动融合**（直接把 LoRA 权重加到线性层）；最后一种模式只能以融合方式生效，即便指定了 `--no_fuse_lcm_lora` 也会给出提示。  
- 仍建议在条件允许时执行 `pip install --upgrade diffusers`，以获得更完善的 LCM 生态支持。
