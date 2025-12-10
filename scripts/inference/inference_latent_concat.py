import argparse
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import (
    center_crop,
    pil_to_tensor,
    resize,
    to_pil_image,
)

from mimicmotion.utils.geglu_patch import patch_geglu_inplace

patch_geglu_inplace()

from configs.constants import ASPECT_RATIO  # noqa: E402
from mimicmotion.dwpose.preprocess0 import get_image_pose, get_video_pose  # noqa: E402
from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline, tensor2vid  # noqa: E402
from mimicmotion.utils.loader import create_pipeline  # noqa: E402
from mimicmotion.utils.utils import save_to_mp4  # noqa: E402


logger = logging.getLogger("latent_concat")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: [%(levelname)s] %(message)s"
)


def clamp_pose_sequence(pose_pixels: torch.Tensor, max_frames: int) -> torch.Tensor:
    if max_frames is None:
        return pose_pixels
    current = pose_pixels.shape[0]
    target = min(current, max_frames)
    target = max(target, 2)
    if target < current:
        logger.info(
            "Clamping pose frames from %d to %d (max_frames=%d)",
            current,
            target,
            max_frames,
        )
        pose_pixels = pose_pixels[:target]
    return pose_pixels


def preprocess(video_path, image_path, resolution=576, sample_stride=2):
    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels)
    h, w = image_pixels.shape[-2:]
    if h > w:
        w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
    else:
        w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution
    h_w_ratio = float(h) / float(w)
    if h_w_ratio < h_target / w_target:
        h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
    else:
        h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
    image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
    image_pixels = center_crop(image_pixels, [h_target, w_target])
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()
    image_pose = get_image_pose(image_pixels)
    video_pose = get_video_pose(video_path, image_pixels, sample_stride=sample_stride)
    pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
    image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
    return torch.from_numpy(pose_pixels.copy()) / 127.5 - 1, torch.from_numpy(
        image_pixels
    ) / 127.5 - 1


def decode_latents_with_pipeline(
    pipeline: MimicMotionPipeline,
    latents: torch.Tensor,
    num_frames: int,
    decode_chunk_size: int,
    device: torch.device,
):
    vae_dtype = next(pipeline.vae.parameters()).dtype
    pipeline.vae.decoder.to(device)
    decoded = pipeline.decode_latents(
        latents.to(device=device, dtype=vae_dtype),
        num_frames,
        decode_chunk_size,
    )
    video_tensor = tensor2vid(decoded, pipeline.image_processor, output_type="pt").cpu()
    pipeline.vae.decoder.cpu()
    return video_tensor


def _linear_interpolate_frames(
    start: torch.Tensor, end: torch.Tensor, steps: int
) -> torch.Tensor:
    if steps <= 0:
        return None
    start_f = start.to(torch.float32)
    end_f = end.to(torch.float32)
    frames = []
    for idx in range(1, steps + 1):
        alpha = idx / float(steps + 1)
        frame = torch.lerp(start_f, end_f, alpha)
        frames.append(frame.to(start.dtype))
    return torch.cat(frames, dim=1)


def inject_latent_interpolations(
    latents_list: List[torch.Tensor], steps: int, mode: str
) -> Tuple[List[torch.Tensor], int]:
    if steps <= 0 or len(latents_list) <= 1:
        return latents_list, 0

    augmented = []
    total_inserted = 0

    for idx, current in enumerate(latents_list):
        augmented.append(current)
        if idx == len(latents_list) - 1:
            continue

        next_latents = latents_list[idx + 1]
        if mode == "linear":
            interp = _linear_interpolate_frames(current[:, -1:], next_latents[:, :1], steps)
        else:
            raise ValueError(f"Unsupported interpolation mode: {mode}")

        if interp is not None and interp.shape[1] > 0:
            augmented.append(interp)
            total_inserted += interp.shape[1]

    return augmented, total_inserted


@torch.no_grad()
def run_pipeline_for_latents(
    pipeline: MimicMotionPipeline,
    image_pixels: torch.Tensor,
    pose_pixels: torch.Tensor,
    device: torch.device,
    task_config,
):
    image_pixels = [
        to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5
    ]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(task_config.seed))

    tile_size = min(int(task_config.num_frames), pose_pixels.size(0))
    tile_size = max(tile_size, 2)

    outputs = pipeline(
        image_pixels,
        image_pose=pose_pixels,
        num_frames=pose_pixels.size(0),
        tile_size=tile_size,
        tile_overlap=int(task_config.frames_overlap),
        height=pose_pixels.shape[-2],
        width=pose_pixels.shape[-1],
        fps=int(task_config.fps),
        noise_aug_strength=float(task_config.noise_aug_strength),
        num_inference_steps=int(task_config.num_inference_steps),
        generator=generator,
        min_guidance_scale=float(task_config.guidance_scale),
        max_guidance_scale=float(task_config.guidance_scale),
        decode_chunk_size=8,
        output_type="latent",
        device=device,
    )
    latents = outputs.frames.detach().cpu()
    return latents


def load_infer_config(args):
    infer_config = OmegaConf.load(args.inference_config)
    if args.video_folder is not None:
        infer_config.batch.video_folder = args.video_folder
    return infer_config


def resolve_video_list(video_folder: str, explicit_list: Sequence[str]) -> List[Path]:
    folder = Path(video_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Video folder {video_folder} not found")

    if explicit_list:
        video_paths = [folder / name for name in explicit_list]
    else:
        video_paths = sorted(folder.glob("*.mp4"))

    if not video_paths:
        raise RuntimeError(f"No mp4 files found under {video_folder}")

    missing = [p for p in video_paths if not p.exists()]
    if missing:
        missing_str = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(f"Could not find the following videos: {missing_str}")

    return video_paths


@torch.no_grad()
def cache_latents(args, infer_config, device, video_paths):
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    task = infer_config.test_case[0]
    pipeline = create_pipeline(infer_config, device)

    for video_path in video_paths:
        cache_path = Path(args.cache_dir) / f"{video_path.stem}.pt"
        if cache_path.exists() and not args.overwrite_cache:
            logger.info(
                "Skipping %s (cache exists). Use --overwrite_cache to refresh.",
                video_path.stem,
            )
            continue

        task.ref_video_path = str(video_path)
        logger.info("Encoding latents for %s", task.ref_video_path)
        pose_pixels, image_pixels = preprocess(
            task.ref_video_path,
            task.ref_image_path,
            resolution=int(task.resolution),
            sample_stride=int(task.sample_stride),
        )
        pose_pixels = clamp_pose_sequence(pose_pixels, args.max_pose_frames)
        logger.info("Using %d pose frames for %s", pose_pixels.shape[0], video_path.name)
        latents = run_pipeline_for_latents(
            pipeline, image_pixels, pose_pixels, device, task
        )

        metadata = {
            "video_name": video_path.name,
            "num_frames": latents.shape[1],
            "height": pose_pixels.shape[-2],
            "width": pose_pixels.shape[-1],
            "fps": int(task.fps),
            "seed": int(task.seed),
            "dtype": str(latents.dtype),
        }
        torch.save(
            {"latents": latents.to(torch.float16), "metadata": metadata}, cache_path
        )
        logger.info("Cached %s -> %s", video_path.name, cache_path)

        if not args.disable_individual_videos:
            video_tensor = decode_latents_with_pipeline(
                pipeline,
                latents,
                latents.shape[1],
                args.decode_chunk_size,
                device,
            )
            clip = video_tensor[0]
            if not args.keep_first_frame and clip.shape[0] > 1:
                clip = clip[1:]
            clip = (clip * 255.0).clamp(0, 255).to(torch.uint8)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            indiv_dir = Path(args.output_dir) / "individual"
            indiv_dir.mkdir(parents=True, exist_ok=True)
            mp4_path = indiv_dir / f"{video_path.stem}_{timestamp}.mp4"
            save_to_mp4(clip, str(mp4_path), fps=int(task.fps))
            logger.info("Saved individual video to %s", mp4_path)


@torch.no_grad()
def decode_concatenated_latents(args, infer_config, device, video_paths):
    cache_dir = Path(args.cache_dir)
    task = infer_config.test_case[0]
    latents_list = []
    fps = int(task.fps)

    for video_path in video_paths:
        cache_path = cache_dir / f"{video_path.stem}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache file {cache_path} is missing. Run encode mode first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        latents_list.append(payload["latents"])
        fps = payload.get("metadata", {}).get("fps", fps)
        logger.info("Loaded %s (frames=%d)", cache_path, payload["latents"].shape[1])

    latents_augmented, inserted = inject_latent_interpolations(
        latents_list, args.interp_frames, args.interp_mode
    )
    if inserted > 0:
        logger.info(
            "Inserted %d interpolated frames (%d per boundary, mode=%s).",
            inserted,
            args.interp_frames,
            args.interp_mode,
        )

    concatenated = torch.cat(latents_augmented, dim=1)
    total_frames = concatenated.shape[1]
    pipeline = create_pipeline(infer_config, device)
    video_tensor = decode_latents_with_pipeline(
        pipeline,
        concatenated,
        total_frames,
        args.decode_chunk_size,
        device,
    )

    video_frames = (video_tensor * 255.0).clamp(0, 255).to(torch.uint8)
    clip = video_frames[0]
    if not args.keep_first_frame and clip.shape[0] > 1:
        clip = clip[1:]

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = (
        Path(args.output_dir)
        / f"{args.experiment_name}_{'_'.join([p.stem for p in video_paths])}_{timestamp}.mp4"
    )
    save_to_mp4(clip, str(output_path), fps=fps)
    logger.info("Saved concatenated decode to %s", output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache and concatenate latent tokens before VAE decode."
    )
    parser.add_argument("--inference_config", type=str, default="configs/test.yaml")
    parser.add_argument(
        "--video_folder",
        type=str,
        default=None,
        help="Override video folder defined in config.",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        default=None,
        help="Video filenames (relative to folder) to include.",
    )
    parser.add_argument("--cache_dir", type=str, default="latent_cache")
    parser.add_argument("--output_dir", type=str, default="outputs/latent_concat")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("encode", "decode"),
        required=True,
        help="encode: run pipeline and cache latents; decode: load cache and run VAE decode only.",
    )
    parser.add_argument(
        "--decode_chunk_size", type=int, default=8, help="Chunk size for VAE decode."
    )
    parser.add_argument(
        "--max_pose_frames",
        type=int,
        default=179,
        help="Clamp pose/latent length to avoid OOM (min(num_pose_frames, max_pose_frames)).",
    )
    parser.add_argument(
        "--interp_frames",
        type=int,
        default=0,
        help="Number of interpolated latent frames to insert between each video.",
    )
    parser.add_argument(
        "--interp_mode",
        type=str,
        choices=("linear",),
        default="linear",
        help="Interpolation method for bridging latents.",
    )
    parser.add_argument("--experiment_name", type=str, default="latent_concat")
    parser.add_argument(
        "--keep_first_frame",
        action="store_true",
        help="Keep the very first frame (default pipeline drops the reference frame).",
    )
    parser.add_argument(
        "--disable_individual_videos",
        action="store_true",
        help="Skip saving per-video MP4 during encode mode.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Recompute cache even if it exists.",
    )
    parser.add_argument(
        "--no_use_float16",
        action="store_true",
        help="Disable float16 (useful for debugging).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.no_use_float16:
        torch.set_default_dtype(torch.float16)

    infer_config = load_infer_config(args)
    video_paths = resolve_video_list(infer_config.batch.video_folder, args.videos)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.mode == "encode":
        cache_latents(args, infer_config, device, video_paths)
    elif args.mode == "decode":
        decode_concatenated_latents(args, infer_config, device, video_paths)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()

# 示例：python inference_latent_concat.py --mode encode --video_folder assets/concat_exp --videos a.mp4 b.mp4 c.mp4 --cache_dir cache/latents
# 示例：python inference_latent_concat.py --mode decode --video_folder assets/concat_exp --videos a.mp4 b.mp4 c.mp4 --cache_dir cache/latents --output_dir outputs/latents_concat --interp_frames 5
