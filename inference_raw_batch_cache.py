import argparse
import hashlib
import json
import logging
import os
import types
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Optional, Tuple

import torch
from omegaconf import OmegaConf

import inference_raw_batch_turbo as turbo


logger = turbo.logger
device = turbo.device

FEATURE_CACHE_VERSION = 2


def _abs_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def _get_mtime(path: str) -> Optional[float]:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _build_cache_metadata(task, args) -> Tuple[str, dict, Path]:
    metadata = {
        "version": FEATURE_CACHE_VERSION,
        "ref_image": _abs_path(task.ref_image_path),
        "ref_video": _abs_path(task.ref_video_path),
        "image_mtime": _get_mtime(task.ref_image_path),
        "video_mtime": _get_mtime(task.ref_video_path),
        "resolution": int(task.resolution),
        "sample_stride": int(task.sample_stride),
    }
    digest = hashlib.sha1(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()
    cache_dir = Path(args.feature_cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{digest}.pt"
    return digest, metadata, cache_path


class FeatureCacheEntry:
    def __init__(self, task, args):
        self.cache_key, self.metadata, self.cache_path = _build_cache_metadata(task, args)
        self._data: dict = {}
        self._dirty = False
        self._loaded = False
        self._force_refresh = args.feature_cache_force_refresh

        if not self._force_refresh:
            self._loaded = self._try_load()

    @property
    def short_key(self) -> str:
        return self.cache_key[:12]

    def _try_load(self) -> bool:
        if not self.cache_path.exists():
            return False
        try:
            payload = torch.load(self.cache_path, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read cached features %s: %s", self.cache_path, exc)
            return False

        if not isinstance(payload, dict):
            return False
        if payload.get("metadata") != self.metadata or payload.get("cache_version") != FEATURE_CACHE_VERSION:
            return False

        self._data = payload.get("data", {})
        return True

    def _to_cpu(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: self._to_cpu(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._to_cpu(v) for v in value]
        return value

    def get(self, key: str):
        return self._data.get(key)

    def set_tensor(self, key: str, tensor: torch.Tensor) -> None:
        if tensor is None:
            return
        self._data[key] = tensor.detach().cpu()
        self._dirty = True

    def set_blob(self, key: str, value) -> None:
        self._data[key] = self._to_cpu(value)
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        payload = {
            "cache_version": FEATURE_CACHE_VERSION,
            "metadata": self.metadata,
            "data": self._data,
        }
        tmp_path = self.cache_path.with_suffix(".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(self.cache_path)
        self._dirty = False


class PipelineFeatureCache:
    def __init__(self, pipeline, cache_entry: Optional[FeatureCacheEntry], task, args):
        self.pipeline = pipeline
        self.cache_entry = cache_entry
        self.disabled = cache_entry is None or args.disable_pipeline_feature_cache
        self.seed = int(getattr(task, "seed", 0))
        self.noise_aug_strength = float(getattr(task, "noise_aug_strength", 0.0))

        self._orig_encode_image = None
        self._orig_encode_vae_image = None
        self._cached_embeddings = None
        self._cached_latents = None
        self._updated = False

    def __enter__(self):
        if self.disabled:
            return self

        self._cached_embeddings = self.cache_entry.get("image_embeddings")
        latents_blob = self.cache_entry.get("image_latents")
        if (
            isinstance(latents_blob, dict)
            and latents_blob.get("seed") == self.seed
            and abs(float(latents_blob.get("noise_aug_strength", 0.0)) - self.noise_aug_strength) < 1e-6
        ):
            self._cached_latents = latents_blob.get("tensor")

        self._orig_encode_image = self.pipeline._encode_image
        self._orig_encode_vae_image = self.pipeline._encode_vae_image
        self.pipeline._encode_image = types.MethodType(self._encode_image_proxy, self.pipeline)
        self.pipeline._encode_vae_image = types.MethodType(self._encode_vae_image_proxy, self.pipeline)
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        if not self.disabled:
            self.pipeline._encode_image = self._orig_encode_image
            self.pipeline._encode_vae_image = self._orig_encode_vae_image
            if self._updated:
                self.cache_entry.save()

    def _encode_image_proxy(self, pipeline_self, *args, **kwargs):
        device = kwargs.get("device")
        if device is None and len(args) >= 2:
            device = args[1]

        if self._cached_embeddings is not None:
            logger.info("Using cached CLIP embeddings (key=%s)", self.cache_entry.short_key)
            return self._cached_embeddings.to(device=device)

        image_embeddings = self._orig_encode_image(*args, **kwargs)
        self.cache_entry.set_tensor("image_embeddings", image_embeddings)
        self._updated = True
        return image_embeddings

    def _encode_vae_image_proxy(self, pipeline_self, *args, **kwargs):
        device = kwargs.get("device")
        if device is None and len(args) >= 2:
            device = args[1]

        if self._cached_latents is not None:
            logger.info(
                "Using cached VAE latents (seed=%d, key=%s)",
                self.seed,
                self.cache_entry.short_key,
            )
            return self._cached_latents.to(device=device, dtype=pipeline_self.vae.dtype)

        image_latents = self._orig_encode_vae_image(*args, **kwargs)
        blob = {
            "seed": self.seed,
            "noise_aug_strength": self.noise_aug_strength,
            "tensor": image_latents,
        }
        self.cache_entry.set_blob("image_latents", blob)
        self._updated = True
        return image_latents


def preprocess_with_cache(task, args):
    if args.disable_feature_cache:
        logger.info("Feature cache disabled, extracting features directly.")
        pose_pixels, image_pixels = turbo.preprocess(
            task.ref_video_path,
            task.ref_image_path,
            resolution=task.resolution,
            sample_stride=task.sample_stride,
        )
        return None, pose_pixels, image_pixels

    cache_entry = FeatureCacheEntry(task, args)
    pose_pixels = cache_entry.get("pose_pixels")
    image_pixels = cache_entry.get("image_pixels")
    if pose_pixels is not None and image_pixels is not None:
        logger.info("Loaded cached pose/image tensors from %s", cache_entry.cache_path)
        return cache_entry, pose_pixels, image_pixels

    logger.info(
        "Cache miss for ref=%s video=%s (key=%s), extracting features...",
        task.ref_image_path,
        task.ref_video_path,
        cache_entry.short_key,
    )
    pose_pixels, image_pixels = turbo.preprocess(
        task.ref_video_path,
        task.ref_image_path,
        resolution=task.resolution,
        sample_stride=task.sample_stride,
    )
    cache_entry.set_tensor("pose_pixels", pose_pixels)
    cache_entry.set_tensor("image_pixels", image_pixels)
    cache_entry.save()
    return cache_entry, pose_pixels, image_pixels


def run_inference_pass(args, label: str, overrides: Optional[dict] = None):
    overrides = overrides or {}
    original = {}
    for key, value in overrides.items():
        if not hasattr(args, key):
            continue
        original[key] = getattr(args, key)
        setattr(args, key, value)
    try:
        return _run_single_video(args, label)
    finally:
        for key, value in original.items():
            setattr(args, key, value)


@torch.no_grad()
def _run_single_video(args, label: str):
    run_start = perf_counter()
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
        return None

    metrics = None
    for video in all_videos:
        task.ref_video_path = os.path.join(infer_config.batch.video_folder, video)
        logger.info("Preparing pose/image features for %s", task.ref_video_path)
        preprocess_start = perf_counter()
        cache_entry, pose_pixels, image_pixels = preprocess_with_cache(task, args)
        preprocess_duration = perf_counter() - preprocess_start
        logger.info("Feature preparation finished (%.2fs)", preprocess_duration)

        torch.set_default_dtype(torch.float16)
        pipeline = turbo.create_pipeline(infer_config, device)
        turbo.configure_scheduler(pipeline, args)

        logger.info("Running MimicMotion pipeline (%s)", args.scheduler)
        logger.info("Found %d pose frames, using %d for inference", len(pose_pixels), task.num_frames)

        pipeline_start = perf_counter()
        with PipelineFeatureCache(pipeline, cache_entry, task, args):
            _video_frames = turbo.run_pipeline(
                pipeline,
                image_pixels,
                pose_pixels,
                device,
                task,
            )
        pipeline_duration = perf_counter() - pipeline_start

        save_path = (
            f"{args.output_dir}/{infer_config.batch.video_folder.split('/')[-1]}/"
            f"{os.path.basename(task.ref_video_path).split('.')[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4"
        )
        turbo.save_to_mp4(_video_frames, save_path, fps=task.fps)

        total_duration = perf_counter() - run_start
        logger.info(
            "[Timing][%s] preprocess=%.2fs pipeline=%.2fs total=%.2fs",
            label,
            preprocess_duration,
            pipeline_duration,
            total_duration,
        )
        metrics = {
            "label": label,
            "preprocess": preprocess_duration,
            "pipeline": pipeline_duration,
            "total": total_duration,
            "output_path": save_path,
        }
        break

    return metrics


def main(args):
    run_specs = [("run", {})]
    if args.benchmark_cache:
        logger.info("Benchmark mode enabled: running cold & warm cache passes.")
        if args.disable_feature_cache:
            logger.warning("Benchmark mode overrides --disable_feature_cache to record cache speedups.")
        run_specs = [
            ("benchmark_cold", {"feature_cache_force_refresh": True, "disable_feature_cache": False}),
            ("benchmark_warm", {"feature_cache_force_refresh": False, "disable_feature_cache": False}),
        ]

    results = []
    for label, overrides in run_specs:
        metrics = run_inference_pass(args, label, overrides)
        if metrics is None:
            return
        results.append(metrics)

    if not results:
        return

    if len(results) > 1:
        logger.info("Timing summary:")
        for item in results:
            logger.info(
                "  %s -> preprocess %.2fs | pipeline %.2fs | total %.2fs",
                item["label"],
                item["preprocess"],
                item["pipeline"],
                item["total"],
            )
    else:
        item = results[0]
        logger.info(
            "Timing summary: preprocess %.2fs | pipeline %.2fs | total %.2fs",
            item["preprocess"],
            item["pipeline"],
            item["total"],
        )


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
        choices=turbo.SCHEDULER_CHOICES,
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
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="cache/features",
        help="Directory used to cache pose/image features",
    )
    parser.add_argument(
        "--disable_feature_cache",
        action="store_true",
        help="Skip cache reads/writes and always recompute pose/image features",
    )
    parser.add_argument(
        "--feature_cache_force_refresh",
        action="store_true",
        help="Ignore existing cache files and rebuild them",
    )
    parser.add_argument(
        "--disable_pipeline_feature_cache",
        action="store_true",
        help="Only cache DWPose features and skip caching CLIP/VAE tensors",
    )
    parser.add_argument(
        "--benchmark_cache",
        action="store_true",
        help="Run twice (cold & warm cache) and record timing improvements",
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
    turbo.set_logger(log_path)
    main(args)
    logger.info("--- Finished ---")

# python inference_raw_batch_cache.py --batch_folder assets/bad_videos/bad_videos
