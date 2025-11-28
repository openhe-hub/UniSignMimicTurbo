# UniSignMimicTurbo
***
## 推理加速

### LCM LoRA Turbo 工作流
- 入口脚本：`inference_raw_batch_turbo.py`。在创建 MimicMotion pipeline 后，可通过 `--lcm_lora_path` 加载 LCM LoRA，并使用 `--scheduler AnimateLCM_SVD` 将调度器切换为自定义的 `AnimateLCMSVDStochasticIterativeScheduler`。
- 建议把 `--num_inference_steps` 降到 4~8 步；结合 LCM LoRA 能够在保持画质的前提下将采样时间压缩到原来的 1/3~1/5。
- 其他常用参数：`--lcm_weight_name`（指定 LoRA 权重文件）、`--lcm_lora_scale`（默认 1.0，可调低平衡质量）、`--skip_lcm_scheduler`（仅加载 LoRA、不改调度器）。详细用法见 `doc/lcm_lora_integration.md`。

### 特征缓存
- 入口脚本：`inference_raw_batch_cache.py`。它包装了 Turbo 流程，并增加多级缓存机制：
  1. **姿态/图像缓存**：首次运行会把 DWPose + 图像预处理好的 `pose_pixels`、`image_pixels` 落盘（默认目录 `cache/features`），下次命中后无需重复提取。
  2. **CLIP / VAE 缓存**：拦截 pipeline 的 `_encode_image` 与 `_encode_vae_image`，缓存参考图的 CLIP embedding 和噪声扰动后的 VAE latent。只要 `seed`、`noise_aug_strength` 不变，就能直接复用。
- 常用参数：
  - `--feature_cache_dir`: 缓存目录（默认 `cache/features`）。
  - `--disable_feature_cache`: 完全关闭 pose/image 缓存。
  - `--disable_pipeline_feature_cache`: 仅保留 pose/image 缓存，跳过 CLIP/VAE。
  - `--benchmark_cache`: 自动执行冷缓存+热缓存各一次，并在日志里输出 `preprocess/pipeline/total` 耗时对比。
  - 其余调度、步数、输出目录参数与 Turbo 脚本一致，可直接复用。
- 详细说明见 `doc/feature_cache.md`。