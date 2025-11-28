from tqdm import tqdm
import decord
import numpy as np
import copy
from typing import Dict, Optional, Tuple

from .util import draw_pose
from .dwpose_detector import dwpose_detector as dwprocessor


def graft_pose(ref_pose: dict, video_pose: dict):
    if not ref_pose['bodies']['candidate'].any() or not video_pose['bodies']['candidate'].any():
        return None

    NECK_ID = 1
    HEAD_IDS = [] # [0, 14, 15, 16, 17]  # exclude head part
    ARMS_IDS = [3, 4, 6, 7]  # exclude hand part: l/r elbow & arm
    GRAFT_BODY_IDS = HEAD_IDS + ARMS_IDS

    grafted_pose = copy.deepcopy(ref_pose)

    ref_neck_coord = ref_pose['bodies']['candidate'][NECK_ID]
    video_neck_coord = video_pose['bodies']['candidate'][NECK_ID]
    translation_vector = ref_neck_coord - video_neck_coord

    # add excluded body part
    for part_id in GRAFT_BODY_IDS:
        grafted_pose['bodies']['candidate'][part_id] = video_pose['bodies']['candidate'][part_id] + translation_vector

    # add head part
    # if video_pose['faces'].any():
    #     grafted_pose['faces'] = video_pose['faces'] + translation_vector

    # add hand part
    if video_pose['hands'].any():
        grafted_pose['hands'] = video_pose['hands'] + translation_vector
    
    # cover confidence data from video ref
    # grafted_pose['faces_score'] = video_pose['faces_score']
    grafted_pose['hands_score'] = video_pose['hands_score']


    hand_thres = 0.3
    head_thres = 0.3

    # substitute low confidence
    # if np.mean(video_pose['hands_score'][0]) < hand_thres:
    #     grafted_pose['hands'][0] = grafted_pose['hands'][0]
    #     grafted_pose['hands_score'][0] = grafted_pose['hands_score'][0]
    # if np.mean(video_pose['hands_score'][1]) < hand_thres:
    #     grafted_pose['hands_score'][1] = grafted_pose['hands_score'][1]
    # if np.mean(video_pose['faces_score']) < head_thres:
    #     grafted_pose['faces'] = grafted_pose['faces']
    #     grafted_pose['faces_score'] = grafted_pose['faces_score']

    # filter low confidence

    # for hand_id in range(2):
    #     for kps_id in range(21):
    #         if grafted_pose['hands_score'][hand_id][kps_id] < hand_thres:
    #             grafted_pose['hands_score'][hand_id][kps_id] = 0.0
    # for kps_id in range(68):
    #     if grafted_pose['faces_score'][0][kps_id] < head_thres:
    #         grafted_pose['faces_score'][0][kps_id] = 0.0

    return grafted_pose

def graft_pose_v2(ref_pose: dict, video_pose: dict):
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
    
    # calc nose-neck distances
    dist_ref_nose_neck = np.linalg.norm(ref_body[FACES[0]] - ref_body[NECK_ID])
    dist_video_nose_neck = np.linalg.norm(video_body[FACES[0]] - video_body[NECK_ID])
    scale_up = dist_ref_nose_neck / (dist_video_nose_neck + 1e-6)
    for head_index in FACES:
        vec_video_neck_head = video_body[head_index] - video_body[NECK_ID]
        grafted_pose['bodies']['candidate'][head_index] = ref_body[NECK_ID] + vec_video_neck_head * scale_up

    if 'faces' in video_pose and video_pose['faces'].any():
        video_head_kps = video_pose['faces']
        # The vector is from the video's neck to all of the video's head keypoints
        vec_video_neck_to_heads = video_head_kps - video_body[NECK_ID]
        # The new head keypoints are anchored to the reference neck
        grafted_pose['faces'] = ref_body[NECK_ID] + vec_video_neck_to_heads * scale_up

    if 'faces_score' in video_pose and 'faces_score' in grafted_pose and video_pose['faces_score'].any():
        grafted_pose['faces_score'] = video_pose['faces_score']


    return grafted_pose
class TransformSmoother:
    """Temporal smoother for similarity transform parameters with historical window."""
    
    def __init__(self, window_size: int = 5, alpha: float = 0.3, method: str = 'ema'):
        """
        Initialize the smoother with historical window.
        
        Args:
            window_size: Number of historical frames to keep (default 5)
            alpha: Smoothing factor for EMA method (0-1). Lower values = more smoothing.
            method: Smoothing method - 'weighted_average', 'median', or 'ema'
        """
        self.window_size = max(1, window_size)
        self.alpha = alpha
        self.method = method
        
        # Historical buffers
        self.history_R = []  # List of rotation angles (easier to average than matrices)
        self.history_s = []  # List of scale factors
        self.history_t = []  # List of translation vectors
        
    def smooth(self, R: np.ndarray, s: float, t: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Apply smoothing with historical window to transform parameters.
        
        Args:
            R: Current rotation matrix (2x2)
            s: Current scale factor
            t: Current translation vector (1x2)
            
        Returns:
            Smoothed (R, s, t) tuple
        """
        # Convert rotation matrix to angle for easier averaging
        curr_angle = np.arctan2(R[1, 0], R[0, 0])
        
        # Add current values to history
        if len(self.history_R) > 0:
            # Handle angle wrap-around for consistency
            prev_angle = self.history_R[-1]
            angle_diff = curr_angle - prev_angle
            if angle_diff > np.pi:
                curr_angle -= 2 * np.pi
            elif angle_diff < -np.pi:
                curr_angle += 2 * np.pi
        
        self.history_R.append(curr_angle)
        self.history_s.append(s)
        self.history_t.append(t)
        
        # Maintain window size
        if len(self.history_R) > self.window_size:
            self.history_R.pop(0)
            self.history_s.pop(0)
            self.history_t.pop(0)
        
        # Apply smoothing based on method
        if self.method == 'median':
            smoothed_angle = np.median(self.history_R)
            smoothed_s = np.median(self.history_s)
            smoothed_t = np.median(self.history_t, axis=0)
            
        elif self.method == 'weighted_average':
            # Create weights that give more importance to recent frames
            n = len(self.history_R)
            weights = np.exp(np.linspace(-2, 0, n))  # Exponential decay
            weights = weights / weights.sum()
            
            smoothed_angle = np.average(self.history_R, weights=weights)
            smoothed_s = np.average(self.history_s, weights=weights)
            smoothed_t = np.average(self.history_t, weights=weights, axis=0)
            
        elif self.method == 'ema':
            # Traditional EMA but considering the window
            if len(self.history_R) == 1:
                smoothed_angle = curr_angle
                smoothed_s = s
                smoothed_t = t
            else:
                # Use EMA between current and weighted average of history
                hist_angle = np.mean(self.history_R[:-1])
                hist_s = np.mean(self.history_s[:-1])
                hist_t = np.mean(self.history_t[:-1], axis=0)
                
                smoothed_angle = self.alpha * curr_angle + (1 - self.alpha) * hist_angle
                smoothed_s = self.alpha * s + (1 - self.alpha) * hist_s
                smoothed_t = self.alpha * t + (1 - self.alpha) * hist_t
        else:
            # Simple average (fallback)
            smoothed_angle = np.mean(self.history_R)
            smoothed_s = np.mean(self.history_s)
            smoothed_t = np.mean(self.history_t, axis=0)
        
        # Convert smoothed angle back to rotation matrix
        cos_a = np.cos(smoothed_angle)
        sin_a = np.sin(smoothed_angle)
        smoothed_R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        return smoothed_R, smoothed_s, smoothed_t
    
    def reset(self):
        """Reset the smoother history."""
        self.history_R = []
        self.history_s = []
        self.history_t = []

def graft_pose_v3(ref_pose: Dict, video_pose: Dict, transform_smoother: Optional[TransformSmoother] = None):
    """
    将视频中的姿态（手臂和脸部）移植到参考姿态上。

    V3版本使用普氏分析（Procrustes Analysis）来处理面部移植，
    可以精确地处理旋转、缩放和平移，使得面部移植效果更加稳健和自然。

    Args:
        ref_pose (Dict): 参考姿态数据。
        video_pose (Dict): 单帧视频的姿态数据。
        transform_smoother (Optional[TransformSmoother]): 用于时间平滑的滤波器。

    Returns:
        Optional[Dict]: 移植后的新姿态数据字典，如果输入无效则返回None。
    """
    if 'bodies' not in ref_pose or 'bodies' not in video_pose or \
       not ref_pose['bodies']['candidate'].any() or not video_pose['bodies']['candidate'].any():
        return None

    # 定义关键点ID (基于OpenPose COCO格式)
    NECK_ID = 1
    R_SHOULDER_ID, R_ELBOW_ID, R_WRIST_ID = 2, 3, 4
    L_SHOULDER_ID, L_ELBOW_ID, L_WRIST_ID = 5, 6, 7
    
    # 用于身体姿态中的面部关键点
    BODY_FACE_IDS = [0, 14, 15, 16, 17]
    # 用于普氏分析对齐的稳定面部标志点 (鼻子, 左眼, 右眼)
    FACE_ALIGN_IDS = [0, 15, 16] 

    grafted_pose = copy.deepcopy(ref_pose)
    ref_body = np.array(ref_pose['bodies']['candidate'])
    video_body = np.array(video_pose['bodies']['candidate'])

    # --- 1. 手臂移植 (逻辑与v2版本保持一致) ---
    dist_ref_l_shoulder = np.linalg.norm(ref_body[L_SHOULDER_ID] - ref_body[NECK_ID])
    dist_video_l_shoulder = np.linalg.norm(video_body[L_SHOULDER_ID] - video_body[NECK_ID])
    dist_ref_r_shoulder = np.linalg.norm(ref_body[R_SHOULDER_ID] - ref_body[NECK_ID])
    dist_video_r_shoulder = np.linalg.norm(video_body[R_SHOULDER_ID] - video_body[NECK_ID])
    
    scale_left = dist_ref_l_shoulder / (dist_video_l_shoulder + 1e-6)
    scale_right = dist_ref_r_shoulder / (dist_video_r_shoulder + 1e-6)

    # 左臂
    vec_video_l_shoulder_elbow = video_body[L_ELBOW_ID] - video_body[L_SHOULDER_ID]
    grafted_pose['bodies']['candidate'][L_ELBOW_ID] = ref_body[L_SHOULDER_ID] + vec_video_l_shoulder_elbow * scale_left
    vec_video_l_elbow_wrist = video_body[L_WRIST_ID] - video_body[L_ELBOW_ID]
    grafted_pose['bodies']['candidate'][L_WRIST_ID] = grafted_pose['bodies']['candidate'][L_ELBOW_ID] + vec_video_l_elbow_wrist * scale_left
    
    # 右臂
    vec_video_r_shoulder_elbow = video_body[R_ELBOW_ID] - video_body[R_SHOULDER_ID]
    grafted_pose['bodies']['candidate'][R_ELBOW_ID] = ref_body[R_SHOULDER_ID] + vec_video_r_shoulder_elbow * scale_right
    vec_video_r_elbow_wrist = video_body[R_WRIST_ID] - video_body[R_ELBOW_ID]
    grafted_pose['bodies']['candidate'][R_WRIST_ID] = grafted_pose['bodies']['candidate'][R_ELBOW_ID] + vec_video_r_elbow_wrist * scale_right

    # 手部移植
    if 'hands' in video_pose and video_pose['hands'].any():
        video_l_hand_kps = video_pose['hands'][0]
        grafted_l_wrist_coord = grafted_pose['bodies']['candidate'][L_WRIST_ID]
        grafted_pose['hands'][0] = grafted_l_wrist_coord + (video_l_hand_kps - video_body[L_WRIST_ID]) * scale_left

        video_r_hand_kps = video_pose['hands'][1]
        grafted_r_wrist_coord = grafted_pose['bodies']['candidate'][R_WRIST_ID]
        grafted_pose['hands'][1] = grafted_r_wrist_coord + (video_r_hand_kps - video_body[R_WRIST_ID]) * scale_right

    if 'hands_score' in video_pose and video_pose['hands_score'].any():
        grafted_pose['hands_score'] = video_pose['hands_score']
    
    # --- 2. 面部移植 (使用普氏分析的全新逻辑) ---
    
    # 提取用于对齐的标志点 (只取x,y坐标)
    ref_align_pts = np.array([ref_body[i][:2] for i in FACE_ALIGN_IDS])
    video_align_pts = np.array([video_body[i][:2] for i in FACE_ALIGN_IDS])

    # 鲁棒性检查：如果视频中的标志点无效 (例如，[0,0])，则跳过面部移植
    init_s, init_x = None, video_body[BODY_FACE_IDS[0]][0] - ref_body[BODY_FACE_IDS[0]][0]
    print(f"init x = {init_x}")
    if np.all(video_align_pts > 0):
        # 计算将视频面部对齐到参考面部的变换参数
        R, s, t = solve_similarity_transform(video_align_pts, ref_align_pts)
        if init_s is None: init_s = s
        
        # Apply temporal smoothing if smoother is provided
        if transform_smoother is not None:
            R, s, t = transform_smoother.smooth(R, s, t)

        s = min(s, init_s*0.8)
        s = max(s, init_s*1.2)
        t[0] += init_x

        print(R, s, t)
        
        # 定义一个应用变换的函数
        transform = lambda points: s * (points @ R) + t

        # A. 更新 'bodies' 中的面部关键点
        video_body_face_kps = np.array([video_body[i][:2] for i in BODY_FACE_IDS])
        grafted_body_face_kps = transform(video_body_face_kps)
        
        for i, idx in enumerate(BODY_FACE_IDS):
            # 只更新x,y坐标，保留原始的置信度
            grafted_pose['bodies']['candidate'][idx][:2] = grafted_body_face_kps[i]
            
        # B. 更新 'faces' 中的所有面部关键点 (如果存在)
        if 'faces' in video_pose and video_pose['faces'].any():
            video_face_all_kps = np.array(video_pose['faces'])
            grafted_face_kps = transform(video_face_all_kps)
            grafted_pose['faces'] = grafted_face_kps

    # 继承面部置信度
    if 'faces_score' in video_pose and 'faces_score' in grafted_pose and video_pose['faces_score'].any():
        grafted_pose['faces_score'] = video_pose['faces_score']
        
    return grafted_pose

def solve_similarity_transform(source_points: np.ndarray, target_points: np.ndarray):
    """
    使用普氏分析计算将源点集对齐到目标点集的最佳相似性变换（旋转、缩ăpadă和位移）。
    
    Args:
        source_points (np.ndarray): 源点集，形状为 (N, 2)。
        target_points (np.ndarray): 目标点集，形状为 (N, 2)。

    Returns:
        Tuple[np.ndarray, float, np.ndarray]: 包含旋转矩阵 R (2x2)，
                                             缩放因子 s (float)，
                                             和平移向量 t (1x2) 的元组。
    """
    assert source_points.shape == target_points.shape, "点集必须有相同的形状"
    
    # 1. 中心化点集
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)
    source_centered = source_points - source_center
    target_centered = target_points - target_center

    # 2. 计算协方差矩阵
    H = source_centered.T @ target_centered

    # 3. 使用奇异值分解(SVD)找到最佳旋转
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 4. 处理反射的特殊情况 (可选但推荐)
    # 如果R的行列式为-1，则它是一个反射矩阵，而不是一个纯旋转矩阵。
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 5. 计算最佳缩放因子
    # np.trace(np.dot(R.T, H)) 等价于 SVD 中奇异值的和
    var_source = np.sum(source_centered**2)
    s = np.trace(R.T @ H) / var_source

    # 6. 计算最佳平移向量
    t = target_center - s * (source_center @ R)
    
    return R, s, t

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

def graft_pose_v5(ref_pose: dict, video_pose: dict):
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

    # grafted_pose['faces'][0], _ = align_naive(ref_pose['faces'][0], video_pose['faces'][0])
    # grafted_pose['bodies']['candidate'][FACES], _ = align_naive(ref_pose['bodies']['candidate'][FACES], video_pose['bodies']['candidate'][FACES])

    # if 'faces_score' in video_pose and 'faces_score' in grafted_pose and video_pose['faces_score'].any():
    #     grafted_pose['faces_score'] = video_pose['faces_score']

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

    output_pose, output_pose_raw = [], []
    frame_idx = 0
    # scale & retarget pose
    for detected_pose in detected_poses:
        detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
        detected_pose['faces'] = detected_pose['faces'] * a + b
        detected_pose['hands'] = detected_pose['hands'] * a + b

        # stat_hand_kps_distribution(detected_pose['hands'], frame_idx)
        # stat_hand_score(detected_pose['hands_score'], frame_idx)

        # graft: retarget & filter low-confident points
        grafted_pose = graft_pose_v5(ref_pose, detected_pose)
        
        # stat_hand_kps_distribution(grafted_pose['hands'], frame_idx)
        # stat_hand_score(grafted_pose['hands_score'], frame_idx)

        # render
        # im = draw_pose(grafted_pose, height, width)
        # output_pose.append(np.array(im))
        output_pose_raw.append(grafted_pose)

        frame_idx += 1
        
    # return np.stack(output_pose), output_pose_raw
    return output_pose_raw

def stat_hand_kps_distribution(hands_kps, frame_id):
    right, left = hands_kps[0], hands_kps[1]
    print(f"frame id = {frame_id}")
    print(f"left mean = {np.mean(right)}, {np.mean(right[:, 0])}, {np.mean(right[:, 1])}")
    print(f"right mean = {np.mean(left)}, {np.mean(left[:, 0])}, {np.mean(left[:, 1])}")
    print(f"left std = {np.std(right)}, {np.std(right[:, 0])}, {np.std(right[:, 1])}")
    print(f"right std = {np.std(left)}, {np.std(left[:, 0])}, {np.std(left[:, 1])}")
    print(f"left = {right}")
    print(f"right = {left}")
    # import ipdb; ipdb.set_trace()

def stat_hand_score(hands_score, frame_id):
    print(f"frame id = {frame_id}")
    print(f"left mean = {np.mean(hands_score[0])}")
    print(f"right mean = {np.mean(hands_score[1])}")
    print(f"left = {hands_score[0]}")
    print(f"right = {hands_score[1]}")

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
