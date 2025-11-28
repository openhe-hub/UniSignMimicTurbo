import os
import argparse
import logging
import math
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path

import numpy as np
import torch.jit
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
# from torchvision.transforms.functional import to_pil_image
import pickle


from mimicmotion.utils.geglu_patch import patch_geglu_inplace
patch_geglu_inplace()

from constants import ASPECT_RATIO

# from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
# from mimicmotion.utils.loader import create_pipeline
# from mimicmotion.utils.utils import save_to_mp4
from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose

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
    # image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
    return video_pose_raw


# def run_pipeline(pipeline: MimicMotionPipeline, image_pixels, pose_pixels, device, task_config):
#     image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
#     generator = torch.Generator(device=device)
#     generator.manual_seed(task_config.seed)
#     frames = pipeline(
#         image_pixels, image_pose=pose_pixels, num_frames=pose_pixels.size(0),
#         tile_size=task_config.num_frames, tile_overlap=task_config.frames_overlap,
#         height=pose_pixels.shape[-2], width=pose_pixels.shape[-1], fps=7,
#         noise_aug_strength=task_config.noise_aug_strength, num_inference_steps=task_config.num_inference_steps,
#         generator=generator, min_guidance_scale=task_config.guidance_scale, 
#         max_guidance_scale=task_config.guidance_scale, decode_chunk_size=8, output_type="pt", device=device
#     ).frames.cpu()
#     video_frames = (frames * 255.0).to(torch.uint8)

#     for vid_idx in range(video_frames.shape[0]):
#         # deprecated first frame because of ref image
#         _video_frames = video_frames[vid_idx, 1:]

#     return _video_frames


@torch.no_grad()
def main(args):
    if not args.no_use_float16 :
        torch.set_default_dtype(torch.float16)

    infer_config = OmegaConf.load(args.inference_config)
    # pipeline = create_pipeline(infer_config, device)
    task = infer_config.test_case[0]
    cnt = 0

    all_videos = sorted([f for f in os.listdir(infer_config.batch.video_folder) if f.endswith('mp4')])
    videos_to_process = all_videos[args.node_index::args.total_nodes]

    for video in videos_to_process:
        if video.endswith('mp4'):
            task.ref_video_path = os.path.join(infer_config.batch.video_folder, video)
            logger.info(f"handling {task.ref_video_path}")
            ############################################## Pre-process data ##############################################
            video_pose_raw = preprocess(
                task.ref_video_path, task.ref_image_path, 
                resolution=task.resolution, sample_stride=task.sample_stride
            )

            video_name = os.path.basename(task.ref_video_path).split('.')[0]
            with open(f'{args.output_dir}/{video_name}_kps.pkl', 'wb') as f:
                pickle.dump(video_pose_raw, f)
            
            print(f"{cnt} video finished")
            cnt += 1

        # pose_video_frames = ((pose_pixels + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
        # pose_video_frames = pose_video_frames[1:]  
        # pose_video_path = f"{args.output_dir}/{video_name}_video.mp4"
        # save_to_mp4(
        #     pose_video_frames,
        #     pose_video_path,
        #     fps=task.fps if hasattr(task, 'fps') else 30,
        # )

        ########################################### Run MimicMotion pipeline ###########################################
        # _video_frames = run_pipeline(
        #     pipeline, 
        #     image_pixels, pose_pixels, 
        #     device, task
        # )
        # ################################### save results to output folder. ###########################################
        # save_to_mp4(
        #     _video_frames, 
        #     f"{args.output_dir}/{os.path.basename(task.ref_video_path).split('.')[0]}" \
        #     f"_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4",
        #     fps=task.fps,
        # )

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
    parser.add_argument("--no_use_float16",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
    )
    parser.add_argument("--node_index", type=int, default=0, help="curr index id")
    parser.add_argument("--total_nodes", type=int, default=5, help="total indics")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    set_logger(args.log_file \
               if args.log_file is not None else f"{args.output_dir}/{datetime.now().strftime('%Y%m%d%H%M%S')}.log")
    main(args)
    logger.info(f"--- Finished ---")

