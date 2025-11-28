import argparse
import logging
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import center_crop, pil_to_tensor, resize, to_pil_image

from mimicmotion.utils.geglu_patch import patch_geglu_inplace

patch_geglu_inplace()

from constants import ASPECT_RATIO
from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
from mimicmotion.utils.loader import create_pipeline
from mimicmotion.utils.utils import save_to_mp4
from mimicmotion.dwpose.preprocess0 import get_video_pose, get_image_pose
from mimicmotion.lcm_scheduler import AnimateLCMSVDStochasticIterativeScheduler


logging.basicConfig(level=logging.INFO, format="%(asctime)s: [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCHEDULER_CHOICES = ("EulerDiscreteScheduler", "AnimateLCM_SVD")


def configure_scheduler(pipeline: MimicMotionPipeline, args):
    if args.scheduler == "EulerDiscreteScheduler":
        logger.info("Using EulerDiscreteScheduler for sampling.")
        return pipeline.scheduler

    if args.scheduler == "AnimateLCM_SVD":
        scheduler = AnimateLCMSVDStochasticIterativeScheduler(
            num_train_timesteps=args.lcm_num_train_timesteps,
            sigma_min=args.scheduler_sigma_min,
            sigma_max=args.scheduler_sigma_max,
            sigma_data=args.scheduler_sigma_data,
            s_noise=args.scheduler_s_noise,
            rho=args.scheduler_rho,
            clip_denoised=args.scheduler_clip_denoised,
        )
        pipeline.scheduler = scheduler
        logger.info(
            "Swapped to AnimateLCM_SVD scheduler (train_steps=%d, sigma_min=%.4f, sigma_max=%.1f).",
            args.lcm_num_train_timesteps,
            args.scheduler_sigma_min,
            args.scheduler_sigma_max,
        )
        return scheduler

    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


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
    return torch.from_numpy(pose_pixels.copy()) / 127.5 - 1, torch.from_numpy(image_pixels) / 127.5 - 1


def run_pipeline(pipeline: MimicMotionPipeline, image_pixels, pose_pixels, device, task_config):
    image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
    generator = torch.Generator(device=device)
    generator.manual_seed(task_config.seed)
    frames = pipeline(
        image_pixels,
        image_pose=pose_pixels,
        num_frames=pose_pixels.size(0),
        tile_size=task_config.num_frames,
        tile_overlap=task_config.frames_overlap,
        height=pose_pixels.shape[-2],
        width=pose_pixels.shape[-1],
        fps=7,
        noise_aug_strength=task_config.noise_aug_strength,
        num_inference_steps=task_config.num_inference_steps,
        generator=generator,
        min_guidance_scale=task_config.guidance_scale,
        max_guidance_scale=task_config.guidance_scale,
        decode_chunk_size=8,
        output_type="pt",
        device=device,
    ).frames.cpu()
    video_frames = (frames * 255.0).to(torch.uint8)

    for vid_idx in range(video_frames.shape[0]):
        _video_frames = video_frames[vid_idx, 1:]

    return _video_frames


@torch.no_grad()
def main(args):
    if not args.no_use_float16:
        torch.set_default_dtype(torch.float16)

    infer_config = OmegaConf.load(args.inference_config)
    task = infer_config.test_case[0]
    infer_config.batch.video_folder = args.batch_folder

    if args.num_inference_steps is not None:
        task.num_inference_steps = args.num_inference_steps

    all_videos = sorted([f for f in os.listdir(infer_config.batch.video_folder) if f.endswith("mp4")])
    if not all_videos:
        logger.warning("No mp4 files found in %s", infer_config.batch.video_folder)
        return

    for video in all_videos:
        task.ref_video_path = os.path.join(infer_config.batch.video_folder, video)
        logger.info("Extracting pose from %s", task.ref_video_path)
        pose_pixels, image_pixels = preprocess(
            task.ref_video_path,
            task.ref_image_path,
            resolution=task.resolution,
            sample_stride=task.sample_stride,
        )
        logger.info("Pose extraction finished")

        torch.set_default_dtype(torch.float16)
        pipeline = create_pipeline(infer_config, device)
        configure_scheduler(pipeline, args)

        logger.info("Running MimicMotion pipeline (%s)", args.scheduler)
        # task.num_frames = 16 # min(int(len(pose_pixels) / 16) * 16 - 1, 199)
        logger.info("Found %d pose frames, using %d for inference", len(pose_pixels), task.num_frames)

        _video_frames = run_pipeline(
            pipeline,
            image_pixels,
            pose_pixels,
            device,
            task,
        )

        save_to_mp4(
            _video_frames,
            f"{args.output_dir}/{infer_config.batch.video_folder.split('/')[-1]}/"
            f"{os.path.basename(task.ref_video_path).split('.')[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4",
            fps=task.fps,
        )
        break


def set_logger(log_file=None, log_level=logging.INFO):
    log_handler = logging.FileHandler(log_file, "w")
    log_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    )
    log_handler.setLevel(log_level)
    logger.addHandler(log_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--inference_config", type=str, default="configs/test.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="path to output")
    parser.add_argument("--batch_folder", type=str, default="", help="batch folder with reference videos")
    parser.add_argument(
        "--no_use_float16",
        action="store_true",
        help="Disable float16 to debug numerical differences",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=SCHEDULER_CHOICES,
        default="AnimateLCM_SVD",
        help="Noise scheduler for sampling",
    )
    parser.add_argument("--scheduler_sigma_min", type=float, default=0.002)
    parser.add_argument("--scheduler_sigma_max", type=float, default=700.0)
    parser.add_argument("--scheduler_sigma_data", type=float, default=1.0)
    parser.add_argument("--scheduler_s_noise", type=float, default=1.0)
    parser.add_argument("--scheduler_rho", type=float, default=7.0)
    parser.add_argument(
        "--scheduler_clip_denoised",
        action="store_true",
        help="Enable clipping of LCM outputs",
    )
    parser.add_argument(
        "--lcm_num_train_timesteps",
        type=int,
        default=40,
        help="Number of training timesteps for AnimateLCM_SVD",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=4,
        help="Override step count defined in the config",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.batch_folder == "":
        raise ValueError("--batch_folder must point to a folder with reference videos")

    log_path = (
        args.log_file
        if args.log_file is not None
        else f"{args.output_dir}/{datetime.now().strftime('%Y%m%d%H%M%S')}.log"
    )
    set_logger(log_path)
    main(args)
    logger.info("--- Finished ---")

# python inference_raw_batch_turbo.py --batch_folder assets/bad_videos --num_inference_steps 2