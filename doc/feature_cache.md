# MimicMotion 特征缓存加速说明

## 功能概览
- 新增 `inference_raw_batch_cache.py`，在执行 MimicMotion Raw 推理前，先尝试复用同一 `ref_image + ref_video` 组合的姿态/图像特征。
- 缓存命中后可跳过 DWPose 检测与图像预处理，从而减少多次实验或调参时的大量重复开销。
- 在 pipeline 运行阶段会拦截 `_encode_image` / `_encode_vae_image`，将 CLIP image embeddings 以及带噪声的 VAE latents 落盘；第二次运行便可直接复用，避免重复地喂参考图进入 CLIP/VAE。
- 自带计时日志，可通过 `--benchmark_cache` 连续运行两次（冷缓存 + 热缓存）来对比时间收益。
- 缓存键会综合图像/视频绝对路径、文件修改时间、`resolution` 与 `sample_stride`，因此只要任一因素变化就会自动失效并重新构建。

## 主要参数
| 参数名 | 说明 |
| --- | --- |
| `--feature_cache_dir` | 缓存目录，默认 `cache/features`。每个组合会生成一个 `.pt` 文件。 |
| `--disable_feature_cache` | 添加该 flag 将完全跳过缓存逻辑（不读不写）。 |
| `--feature_cache_force_refresh` | 忽略已有缓存，强制重新计算并覆盖。 |
| `--disable_pipeline_feature_cache` | 只缓存 DWPose 预处理结果，跳过 CLIP/VAE 特征缓存。 |
| `--benchmark_cache` | 冷/热缓存各跑一次并记录计时，用于观察缓存带来的速度差异。 |

> 兼容旧脚本需求：如果希望继续使用最初的 `inference_raw_batch_turbo.py`，无需改动；只有当你主动调用新的 `inference_raw_batch_cache.py` 时才会启用缓存机制。

## 使用示例
```bash
python inference_raw_batch_cache.py \
  --inference_config configs/test.yaml \
  --batch_folder assets/bad_videos \
  --feature_cache_dir cache/features \
  --num_inference_steps 4
```

首次运行会输出 “Cache miss…” 并写入缓存；再次针对同一个 `ref`+`video` 执行时，可在日志中看到 “Loaded cached features…”。

## 缓存内容
- `pose_pixels` / `image_pixels`：DWPose + 参考图预处理结果（CPU Tensor）。
- `image_embeddings`：CLIP image encoder 产出的 embedding，命中后可直接跳过 encoder。
- `image_latents`：经过噪声扰动 & VAE 编码后的 latent，附带 `seed`、`noise_aug_strength` 标签；只有当两者与当前任务一致时才会被复用，否则自动回落到重新编码。

## 时间基准
- 默认情况下脚本会在日志中输出单次运行的 `preprocess`、`pipeline`、`total` 三段耗时。
- 加上 `--benchmark_cache` 后会执行两次：第一次强制刷新缓存（冷启动），第二次直接读取缓存（热启动），末尾会给出对比表，便于快速评估加速效果。
- 如需基准模式，需确保未显式禁用缓存；脚本会在 benchmark 期间临时忽略 `--disable_feature_cache`，避免误测。

## 注意事项
- 缓存持久化内容包含 pose tensor 与对齐后图像 tensor，全部保存为 CPU precision，文件体积取决于分辨率与帧数。
- 若你手动替换了参考图像或视频，请确保其修改时间更新（或主动使用 `--feature_cache_force_refresh`），否则缓存会认为文件未变更。
- 更换 `seed` 或 `noise_aug_strength` 时，VAE latent 会自动视为过期，但 pose/image 缓存仍可命中；如需完全避免 pipeline 级缓存，可添加 `--disable_pipeline_feature_cache`。
- 缓存目录位于项目根目录下，可根据磁盘情况定期清理过期条目。
