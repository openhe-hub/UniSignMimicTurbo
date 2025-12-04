# Latent 拼接实验

这个实验脚本（`inference_latent_concat.py`）用来验证“把不同视频的 VAE 之前 latent token 直接拼接再解码”会得到怎样的效果，便于和从头重新采样的结果做对比。它不会改动现有推理代码，只是提供一个外挂式的 encode/decode 流程。

## 工作流

1. **Encode（缓存阶段）**
   - 运行 `python inference_latent_concat.py --mode encode ...`。
   - 对指定的参考视频逐个跑完整条 MimicMotion pipeline，但把 `output_type` 设置为 `latent`，因此不会经过 VAE。
   - 每个视频会生成一个 `<cache_dir>/<video_stem>.pt`，内部字段包含：
     - `latents`：`torch.float16`，形状 `[batch, frames, channels, height, width]`，即 VAE decode 前的原始潜空间序列。
     - `metadata`：记录 `video_name`、`num_frames`、`height`、`width`、`fps`、`seed`、`dtype` 等基本信息。
   - 默认还会把当前视频的 latent 直接 decode 成单独的 MP4（保存到 `outputs/latent_concat/individual`），方便和后续“拼接结果”对比；若不需要可加 `--disable_individual_videos` 关闭。
   - 如果只想更新部分缓存，提供 `--videos a.mp4 b.mp4` 即可；已存在的条目默认跳过，可加 `--overwrite_cache` 强制重算。

2. **Decode（拼接阶段）**
   - 运行 `python inference_latent_concat.py --mode decode ...`。
   - 脚本会按传入视频顺序加载缓存，把 latent 在帧维度上 `torch.cat`，然后只用 VAE decoder 把拼接后的 latent 解码成像素帧，最后输出一个 MP4。
   - 可通过 `--interp_frames N` 启用“过渡帧”功能：在每两个视频之间用 latent 线性插值插入 N 帧，以获得更平滑的衔接（不会涉及重新采样）。默认会像原推理脚本一样丢弃第一帧（reference），若想保留加 `--keep_first_frame`。

> ⚠️ 由于 latents 是直接拼接，没有跨段过渡信息，因此视觉效果会是“硬切换”。这是实验的本意，用来证明拼接不会产生平滑渐变。

## 常用参数

| 参数 | 作用 |
| --- | --- |
| `--mode {encode,decode}` | 选择缓存 or 只解码。 |
| `--inference_config` | 复用原来的推理配置（默认 `configs/test.yaml`）。 |
| `--video_folder` | 视频所在目录。若不指定，沿用 config 里的 `batch.video_folder`。 |
| `--videos a.mp4 b.mp4 c.mp4` | 指定要处理/拼接的文件（顺序即拼接顺序）。若省略，则遍历整个目录。 |
| `--cache_dir` | latent 缓存目录（默认 `latent_cache`）。 |
| `--output_dir` | 解码后 MP4 的输出目录。 |
| `--decode_chunk_size` | VAE 解码 chunk 大小，用来控制显存。 |
| `--max_pose_frames` | 运行 encode 时把 pose/latent 长度裁剪到 `min(num_pose_frames, max_pose_frames)`（默认 179，避免超过 ~180 帧导致爆显存）。 |
| `--interp_frames` / `--interp_mode` | 在 decode 阶段为每两个视频插入指定数量的 latent 插值帧（默认禁用，`--interp_mode` 暂只支持 `linear`）。 |
| `--disable_individual_videos` | encode 模式下不输出每个视频的单独 MP4。 |
| `--keep_first_frame` | 保留第一帧（reference）。默认会丢弃。 |

## 示例

```bash
# 1) 缓存 a/b/c 三个视频的 latent
python inference_latent_concat.py \
  --mode encode \
  --video_folder assets/bad_videos/bad_videos \
  --videos a.mp4 b.mp4 c.mp4 \
  --cache_dir cache/latents

# 2) 直接拼接缓存并解码为新的 MP4
python inference_latent_concat.py \
  --mode decode \
  --video_folder assets/bad_videos/bad_videos \
  --videos a.mp4 b.mp4 c.mp4 \
  --cache_dir cache/latents \
  --output_dir outputs/latents_concat
```

## 限制与注意事项

- Encode 阶段仍需完整跑一遍 MimicMotion pipeline（含 UNet 迭代），因此速度与常规推理一致；Decode 阶段只用 VAE，耗时非常小。
- latent 的空间尺寸、参考图像、seed、噪声增强等条件必须匹配，否则拼接出的结果与真正连续采样存在明显切换，这是实验预期。
- 该脚本不会自动切换 LCM scheduler 或 Feature Cache；它面向“强行拼接 latent”这一独立实验。需要结合 Turbo/Cache 时，可各自运行再对比耗时。
