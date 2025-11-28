from tqdm import tqdm
import decord
import numpy as np
import copy

from .util import draw_pose
from .dwpose_detector import dwpose_detector as dwprocessor


def crop_scale_normalize(motion, thr=0.3):
    """
    Normalize keypoints to [-1, 1] based on bounding box.
    
    Args:
        motion: Array of shape [(M), T, N, 3] with (x, y, confidence)
        thr: Confidence threshold for valid keypoints
    
    Returns:
        result: Normalized motion
        scale: Scaling factor used
        offset: Translation offset [xs, ys]
    """
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2] > thr][:,:2]
    
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None
    
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    
    ratio = 1
    scale = max(xmax-xmin, ymax-ymin) * ratio
    
    if scale == 0:
        return np.zeros(motion.shape), 0, None
    
    # Center the bounding box
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    
    # Normalize to [0, 1] then to [-1, 1]
    result[...,:2] = (motion[..., :2] - [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    
    # Mask low confidence keypoints
    result[result[...,2] <= thr] = 0
    
    return result, scale, [xs,ys]


def normalize_part_keypoints(detected_pose, ref_pose=None):
    thr = 0.3
    normalized_sequences = detected_pose.copy()
    
    # Prepare data structures
    all_bodies = None
    all_faces = None
    all_hands = None
    
    # Body keypoints [1, 18, 2]
    if detected_pose['bodies']['candidate'].shape[0] == 18:
        body_kps = detected_pose['bodies']['candidate']
        body_conf = detected_pose['bodies']['score'][0]
        all_bodies = np.concatenate([body_kps, body_conf[:, None]], axis=-1)
    else:
        all_bodies = np.zeros((18, 3))
        
    # Face keypoints [1, 68, 2]
    if detected_pose['faces'].shape[1] == 68:
        face_kps = detected_pose['faces'][0]
        face_conf = detected_pose['faces_score'][0]
        all_faces = np.concatenate([face_kps, face_conf[:, None]], axis=-1)
    else:
        all_faces = np.zeros((68, 3))
        
    # Hand keypoints [2, 21, 2]
    if detected_pose['hands'].shape[1] == 21:
        hand_kps = np.vstack([detected_pose['hands'][0], detected_pose['hands'][1]])  # Shape: (42, 2)
        hand_conf = detected_pose['hands_score'].reshape(-1)
        all_hands = np.concatenate([hand_kps, hand_conf[:, None]], axis=-1)
    else:
        all_hands = np.zeros((42, 3))
    
    # Normalize body first to get the scale
    body_normalized, body_scale, body_offset = crop_scale_normalize(all_bodies, thr)
    normalized_sequences['bodies']['candidate'] = body_normalized[:, :2]
    
    # Normalize other parts using body scale
    if body_scale > 0:
        # Face normalization - centered on nose
        face_centered = all_faces.copy()
        if np.any(all_faces[:, 2] > thr):
            face_center = np.mean(all_faces[all_faces[:, 2] > thr, :2], axis=0)
            face_centered[:, :2] = all_faces[:, :2] - face_center
        
        face_normalized = face_centered.copy()
        face_normalized[:, :2] = face_centered[:, :2] / body_scale
        face_normalized = np.clip(face_normalized, -1, 1)
        face_normalized[face_normalized[:, 2] <= thr] = 0
        normalized_sequences['faces'] = face_normalized[:, :2].reshape(1, 68, 2)
        
        # # Left hand normalization - centered on wrist
        left_hand = all_hands[:21, :]
        left_hand_centered = left_hand.copy()
        if left_hand[0, 2] > thr:  # Wrist is first point
            left_hand_centered[:, :2] = left_hand[:, :2] - left_hand[0, :2]
        
        left_normalized = left_hand_centered.copy()
        left_normalized[:, :2] = left_hand_centered[:, :2] / body_scale
        left_normalized = np.clip(left_normalized, -1, 1)
        left_normalized[left_normalized[:, 2] <= thr] = 0
        normalized_sequences['hands'][0] = left_normalized[:, :2]
        
        # Right hand normalization - centered on wrist
        right_hand = all_hands[21:, :]
        right_hand_centered = right_hand.copy()
        if right_hand[0, 2] > thr:  # Wrist is first point
            right_hand_centered[:, :2] = right_hand[:, :2] - right_hand[0, :2]
        
        right_normalized = right_hand_centered.copy()
        right_normalized[:, :2] = right_hand_centered[:, :2] / body_scale
        right_normalized = np.clip(right_normalized, -1, 1)
        right_normalized[right_normalized[:, 2] <= thr] = 0
        normalized_sequences['hands'][1] = right_normalized[:, :2]
    
    # Store scale and offset for potential denormalization
    # normalized_sequences['scale'] = body_scale
    # normalized_sequences['offset'] = body_offset

    # import ipdb; ipdb.set_trace()
    
    return normalized_sequences   


def get_video_pose(
        video_path: str, 
        ref_image: np.ndarray = None, 
        sample_stride: int = 1):
    """
    Extract and normalize video pose using CoSign-style normalization.
    
    Args:
        video_path (str): video path
        ref_image (np.ndarray): optional reference image for alignment
        sample_stride (int, optional): Defaults to 1.
    
    Returns:
        dict: Dictionary containing normalized pose sequences for each part
    """
    # Process reference image if provided
    ref_pose = dwprocessor(ref_image)
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [i for i in ref_keypoint_id \
        if len(ref_pose['bodies']['subset']) > 0 and ref_pose['bodies']['subset'][0][i] >= .0]
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]

    height, width, _ = ref_image.shape
    
    # Read input video
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
    fh, fw, _ = vr[0].shape
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    output_pose = []
    # pose rescale 
    for detected_pose in detected_poses:
        detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
        detected_pose['faces'] = detected_pose['faces'] * a + b
        detected_pose['hands'] = detected_pose['hands'] * a + b

        # check_size(detected_pose)

        detected_pose = normalize_part_keypoints(detected_pose)

        # check_size(detected_pose)

        im = draw_pose(detected_pose, height, width)
        output_pose.append(np.array(im))

    return np.stack(output_pose)


def check_size(dwpose_pack):
    print(dwpose_pack['bodies']['candidate'].shape)
    print(dwpose_pack['bodies']['score'].shape)
    print(dwpose_pack['bodies']['subset'].shape)

    print(dwpose_pack['faces'].shape)
    print(dwpose_pack['faces_score'].shape)

    print(dwpose_pack['hands'].shape)
    print(dwpose_pack['hands_score'].shape)


def get_image_pose(ref_image):
    """
    Process and normalize image pose using CoSign-style normalization.
    
    Args:
        ref_image (np.ndarray): reference image pixel value
    
    Returns:
        dict: Dictionary containing normalized pose for each part
    """
    height, width, _ = ref_image.shape
    ref_pose = dwprocessor(ref_image)
    # check_size(ref_pose)
    ref_pose = normalize_part_keypoints(ref_pose)
    # check_size(ref_pose)
    pose_img = draw_pose(ref_pose, height, width)

    return np.array(pose_img)