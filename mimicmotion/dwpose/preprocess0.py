from tqdm import tqdm
import decord
import numpy as np
import copy
from typing import Dict, Optional, Tuple

from .util import draw_pose
from .dwpose_detector import dwpose_detector as dwprocessor

def align_naive(ref_kpts, result_animation):
    print("Performing naive alignment using bounding box normalization...")
    
    # 1. 计算参考系的几何属性 (我们的"标尺")
    ref_min = ref_kpts.min(axis=0)
    ref_max = ref_kpts.max(axis=0)
    ref_center = (ref_min + ref_max) / 2.0
    ref_size = ref_max - ref_min  # [width, height]
    
    # 防止除以零
    ref_size[ref_size < 1e-6] = 1e-6
    
    frame = result_animation
        
    # 2. 计算当前帧的几何属性
    frame_min = frame.min(axis=0)
    frame_max = frame.max(axis=0)
    frame_center = (frame_min + frame_max) / 2.0
    frame_size = frame_max - frame_min # [width, height]
        
    # 防止除以零
    frame_size[frame_size < 1e-6] = 1e-6

    # 3. 执行归一化
    # 首先，将当前帧中心移到原点
    centered_frame = frame - frame_center
        
    # 其次，根据参考系的尺寸进行缩放
    # 计算缩放比例 (x和y轴分开算)
    scale_factors = ref_size / frame_size # [scale_x, scale_y]
    # 应用缩放
    scaled_frame = centered_frame * scale_factors
        
    # 最后，将缩放后的帧移动到参考系的中心
    aligned_frame = scaled_frame + ref_center
        
    return aligned_frame, ref_kpts

def graft_pose_v5(ref_pose: dict, video_pose: dict) -> dict | None:
    NECK_ID = 1
    R_SHOULDER_ID = 2
    R_ELBOW_ID = 3
    R_WRIST_ID = 4
    L_SHOULDER_ID = 5
    L_ELBOW_ID = 6
    L_WRIST_ID = 7
    FACES = [0, 14, 15, 16, 17]

    grafted_pose = copy.deepcopy(ref_pose)

    # calc shoulder-neck distances
    ref_body = ref_pose['bodies']['candidate']
    video_body = video_pose['bodies']['candidate']
    dist_ref_l_shoulder = np.linalg.norm(ref_body[L_SHOULDER_ID] - ref_body[NECK_ID])
    dist_video_l_shoulder = np.linalg.norm(video_body[L_SHOULDER_ID] - video_body[NECK_ID])
    dist_ref_r_shoulder = np.linalg.norm(ref_body[R_SHOULDER_ID] - ref_body[NECK_ID])
    dist_video_r_shoulder = np.linalg.norm(video_body[R_SHOULDER_ID] - video_body[NECK_ID])
    scale_left = dist_ref_l_shoulder / (dist_video_l_shoulder + 1e-6)
    scale_right = dist_ref_r_shoulder / (dist_video_r_shoulder + 1e-6)

    # scale arm & elbow in body kps
    vec_video_l_shoulder_elbow = video_body[L_ELBOW_ID] - video_body[L_SHOULDER_ID]
    grafted_pose['bodies']['candidate'][L_ELBOW_ID] = ref_body[L_SHOULDER_ID] + vec_video_l_shoulder_elbow * scale_left
    vec_video_l_elbow_wrist = video_body[L_WRIST_ID] - video_body[L_ELBOW_ID]
    grafted_pose['bodies']['candidate'][L_WRIST_ID] = grafted_pose['bodies']['candidate'][L_ELBOW_ID] + vec_video_l_elbow_wrist * scale_left
    vec_video_r_shoulder_elbow = video_body[R_ELBOW_ID] - video_body[R_SHOULDER_ID]
    grafted_pose['bodies']['candidate'][R_ELBOW_ID] = ref_body[R_SHOULDER_ID] + vec_video_r_shoulder_elbow * scale_right
    vec_video_r_elbow_wrist = video_body[R_WRIST_ID] - video_body[R_ELBOW_ID]
    grafted_pose['bodies']['candidate'][R_WRIST_ID] = grafted_pose['bodies']['candidate'][R_ELBOW_ID] + vec_video_r_elbow_wrist * scale_right

    # scale hand kps
    if video_pose['hands'].any():
        video_l_hand_kps = video_pose['hands'][0]
        grafted_l_wrist_coord = grafted_pose['bodies']['candidate'][L_WRIST_ID]
        grafted_pose['hands'][0] = grafted_l_wrist_coord + (video_l_hand_kps - video_body[L_WRIST_ID]) * scale_left

        video_r_hand_kps = video_pose['hands'][1]
        grafted_r_wrist_coord = grafted_pose['bodies']['candidate'][R_WRIST_ID]
        grafted_pose['hands'][1] = grafted_r_wrist_coord + (video_r_hand_kps - video_body[R_WRIST_ID]) * scale_right

    if 'hands_score' in video_pose and video_pose['hands_score'].any():
        grafted_pose['hands_score'] = video_pose['hands_score']

    grafted_pose['faces'][0], _ = align_naive(ref_pose['faces'][0], video_pose['faces'][0])
    grafted_pose['bodies']['candidate'][FACES], _ = align_naive(ref_pose['bodies']['candidate'][FACES], video_pose['bodies']['candidate'][FACES])

    if 'faces_score' in video_pose and 'faces_score' in grafted_pose and video_pose['faces_score'].any():
        grafted_pose['faces_score'] = video_pose['faces_score']

    return grafted_pose


def get_video_pose(
        video_path: str,
        ref_image: np.ndarray,
        sample_stride: int = 1):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): video pose path
        ref_image (np.ndarray): reference image 
        sample_stride (int, optional): Defaults to 1.

    Returns:
        np.ndarray: sequence of video pose
    """
    # select ref-keypoint from reference pose for pose rescale
    ref_pose = dwprocessor(ref_image)
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [i for i in ref_keypoint_id
                       if len(ref_pose['bodies']['subset']) > 0 and ref_pose['bodies']['subset'][0][i] >= .0]
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]

    height, width, _ = ref_image.shape

    # read input video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    sample_stride *= max(1, int(vr.get_avg_fps() / 24))

    frames = vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()
    detected_poses = [dwprocessor(frm) for frm in tqdm(frames, desc="DWPose")]
    dwprocessor.release_memory()

    detected_bodies = np.stack(
        [p['bodies']['candidate'] for p in detected_poses if p['bodies']['candidate'].shape[0] == 18])[:,
                                                                                                       ref_keypoint_id]
    # compute linear-rescale params
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    fh, fw, _tmp = vr[0].shape
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])

    output_pose = []
    frame_idx = 0
    # scale & retarget pose
    for detected_pose in detected_poses:
        detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
        detected_pose['faces'] = detected_pose['faces'] * a + b
        detected_pose['hands'] = detected_pose['hands'] * a + b

        # render
        grafted_pose = graft_pose_v5(ref_pose, detected_pose)

        im = draw_pose(grafted_pose, height, width)
        output_pose.append(np.array(im))

        frame_idx += 1
        
    return np.stack(output_pose)

def get_image_pose(ref_image):
    """process image pose

    Args:
        ref_image (np.ndarray): reference image pixel value

    Returns:
        np.ndarray: pose visual image in RGB-mode
    """
    height, width, _ = ref_image.shape
    ref_pose = dwprocessor(ref_image)
    pose_img = draw_pose(ref_pose, height, width)
    return np.array(pose_img)
