import os
import argparse
import logging
import math
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path
import pickle
import ipdb

# from mst.dwpose_trans import handle
# from mst.bridge_mst2mm import infer_mae

import numpy as np
import torch.jit
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from torchvision.transforms.functional import to_pil_image


from mimicmotion.utils.geglu_patch import patch_geglu_inplace
patch_geglu_inplace()

from configs.constants import ASPECT_RATIO

from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
from mimicmotion.utils.loader import create_pipeline
from mimicmotion.utils.utils import save_to_mp4
from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose

from loguru import logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s: [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(video_path, image_path, resolution=576, sample_stride=2):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): input video pose path
        image_path (str): reference image path
        resolution (int, optional):  Defaults to 576.
        sample_stride (int, optional): Defaults to 2.
    """
    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels) # (c, h, w)
    h, w = image_pixels.shape[-2:]
    ############################ compute target h/w according to original aspect ratio ###############################
    if h>w:
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
    ##################################### get image&video pose value #################################################
    # image_pose = get_image_pose(image_pixels)
    video_pose_raw = get_video_pose(video_path, image_pixels, sample_stride=sample_stride)
    # pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
    image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
    return video_pose_raw, image_pixels

def get_image_pixels(image_path, resolution=576):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): input video pose path
        image_path (str): reference image path
        resolution (int, optional):  Defaults to 576.
        sample_stride (int, optional): Defaults to 2.
    """
    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels) # (c, h, w)
    h, w = image_pixels.shape[-2:]
    ############################ compute target h/w according to original aspect ratio ###############################
    if h>w:
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
    image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
    return torch.from_numpy(image_pixels) / 127.5 - 1

def run_pipeline(pipeline: MimicMotionPipeline, image_pixels, pose_pixels, device, task_config):
    image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
    generator = torch.Generator(device=device)
    generator.manual_seed(task_config.seed)
    # image_pixels = torch.Tensor(np.transpose(np.array(image_pixels), (0,3,1,2))) 
    # print(image_pixels.shape) # (1,3,1024,576)
    # print(pose_pixels.shape) # (51,3,1024,576)
    pose_pixels = pose_pixels.to(torch.float16)
    # print(pose_pixels.dtype)
    # import ipdb; ipdb.set_trace()
    frames = pipeline(
        image_pixels, image_pose=pose_pixels, num_frames=pose_pixels.size(0),
        tile_size=task_config.num_frames, tile_overlap=task_config.frames_overlap,
        height=pose_pixels.shape[-2], width=pose_pixels.shape[-1], fps=7,
        noise_aug_strength=task_config.noise_aug_strength, num_inference_steps=task_config.num_inference_steps,
        generator=generator, min_guidance_scale=task_config.guidance_scale, 
        max_guidance_scale=task_config.guidance_scale, decode_chunk_size=8, output_type="pt", device=device
    ).frames.cpu()
    video_frames = (frames * 255.0).to(torch.uint8)

    for vid_idx in range(video_frames.shape[0]):
        # deprecated first frame because of ref image
        _video_frames = video_frames[vid_idx, 1:]

    return _video_frames


@torch.no_grad()
def main(args):
    infer_config = OmegaConf.load(args.inference_config)
    if args.batch_video_folder:
        # Allow overriding the configured batch folder from the CLI.
        if "batch" not in infer_config or infer_config.batch is None:
            infer_config.batch = OmegaConf.create()
        infer_config.batch.video_folder = args.batch_video_folder
    task = infer_config.test_case[0]
    all_videos = sorted([f for f in os.listdir(infer_config.batch.video_folder) if f.endswith('pkl')])

    for video in all_videos:
        ############################################## Pre-process data ##############################################
        # torch.set_default_dtype(torch.float32)
        # task.ref_video_path = os.path.join(infer_config.batch.video_folder, video)
        # logger.info(f"begin extracting pose from {task.ref_video_path}")
        # video_pose_raw, image_pixels = preprocess(
        #     task.ref_video_path, task.ref_image_path, 
        #     resolution=task.resolution, sample_stride=task.sample_stride
        # )

        # # os.mkdir(f"outputs/{infer_config.batch.video_folder.split('/')[-1]}")
        # with open(f"outputs/{infer_config.batch.video_folder.split('/')[-1]}/{os.path.basename(video).split('.')[0]}.pkl", 'wb') as f:
        #     pickle.dump(video_pose_raw, f)
        # # handle('video')
        # logger.info("end extracting pose")
        # continue

        ########################################### Run TNet pipeline ###########################################
        # logger.info("begin PoseTNet stage")
        # logger.info("end PoseTNet stage")

        ########################################### Run MAE pipeline ###########################################
        # logger.info(f"begin MAE stage, video = {video}")
        # return_sig = infer_mae(video)
        # if return_sig < 0: continue
        # logger.info("end MAE stage")
        
        ########################################### Run MimicMotion pipeline ###########################################
        logger.info("begin preparing pipeline")
        torch.set_default_dtype(torch.float16)
        pipeline = create_pipeline(infer_config, device)
        logger.info("end preparing pipeline")

        logger.info("begin mimicmotion stage")
        with open(f'{infer_config.batch.video_folder}/{video}', 'rb') as fp:
            bridged_data = pickle.load(fp)
        pose_pixels = bridged_data['pose_pixels']

        image_pixels = get_image_pixels(task.ref_image_path, resolution=task.resolution)
        num_video_frame = len(pose_pixels)
        if num_video_frame <= 16: continue
        task.num_frames = min(int(num_video_frame), 179)
        logger.info(f"using num video frame = {num_video_frame}, num frame = {task.num_frames}")

        _video_frames = run_pipeline(
            pipeline, 
            image_pixels, pose_pixels, 
            device, task
        )
        logger.info("end mimicmotion stage")
            
        # # # ################################### save results to output folder. ###########################################
        save_to_mp4(
            _video_frames, 
            f"{args.output_dir}/{os.path.basename(video).split('.')[0]}" \
            f"_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4",
            fps=task.fps,
        )

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
    parser.add_argument("--inference_config", type=str, default="configs/test.yaml") #ToDo
    parser.add_argument("--output_dir", type=str, default="outputs/", help="path to output")
    parser.add_argument("--bridge_file", type=str, default="outputs/bridge/seg_0.pkl", help="path to output")
    parser.add_argument("--batch_video_folder", type=str, default=None,
                        help="Override infer_config.batch.video_folder")
    parser.add_argument("--no_use_float16",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    set_logger(args.log_file \
               if args.log_file is not None else f"{args.output_dir}/{datetime.now().strftime('%Y%m%d%H%M%S')}.log")
    main(args)
    logger.info(f"--- Finished ---")

# python bridge_mst2mm_batch.py --batch_video_folder assets/bridge_pair_vis/