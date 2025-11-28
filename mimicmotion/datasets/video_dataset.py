import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import functional as F
from typing import Dict, List, Optional, Tuple
import cv2
from pathlib import Path

from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose


class VideoMotionDataset(Dataset):
    """Dataset for training MimicMotion model.
    
    Each sample contains:
    - Reference image (first frame)
    - Pose sequence from the video
    - Target video frames
    """
    
    def __init__(
        self,
        data_root: str,
        annotation_file: str,
        num_frames: int = 16,
        sample_stride: int = 2,
        resolution: int = 576,
        aspect_ratio: float = 0.5625,  # 9:16
        split: str = 'train',
        transform=None
    ):
        """
        Args:
            data_root: Root directory containing videos
            annotation_file: JSON file with video annotations
            num_frames: Number of frames to sample
            sample_stride: Stride for sampling frames
            resolution: Resolution for preprocessing
            aspect_ratio: Target aspect ratio (h/w)
            split: 'train' or 'val'
            transform: Optional transform to apply
        """
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.sample_stride = sample_stride
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio
        self.split = split
        self.transform = transform
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        self.samples = annotations[split]
        
        # Compute target dimensions
        if aspect_ratio < 1:  # Portrait
            self.w_target = resolution
            self.h_target = int(resolution / aspect_ratio // 64) * 64
        else:  # Landscape
            self.h_target = resolution
            self.w_target = int(resolution * aspect_ratio // 64) * 64
    
    def __len__(self):
        return len(self.samples)
    
    def _load_video_frames(self, video_path: str, start_frame: int = 0) -> np.ndarray:
        """Load video frames from file."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(self.num_frames * self.sample_stride):
            ret, frame = cap.read()
            if not ret:
                break
            
            if len(frames) % self.sample_stride == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            if len(frames) >= self.num_frames:
                break
        
        cap.release()
        
        # Pad if necessary
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        
        return np.array(frames[:self.num_frames])
    
    def _preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Preprocess video frames to target resolution."""
        processed = []
        
        for frame in frames:
            # Convert to PIL and then to tensor
            frame_pil = F.to_pil_image(frame)
            frame_tensor = F.pil_to_tensor(frame_pil)
            
            # Resize and crop
            h, w = frame_tensor.shape[-2:]
            h_w_ratio = float(h) / float(w)
            
            if h_w_ratio < self.h_target / self.w_target:
                h_resize = self.h_target
                w_resize = int(self.h_target / h_w_ratio)
            else:
                w_resize = self.w_target
                h_resize = int(self.w_target * h_w_ratio)
            
            frame_tensor = F.resize(frame_tensor, [h_resize, w_resize])
            frame_tensor = F.center_crop(frame_tensor, [self.h_target, self.w_target])
            processed.append(frame_tensor)
        
        # Stack and normalize
        frames_tensor = torch.stack(processed)
        frames_tensor = frames_tensor.float() / 127.5 - 1.0
        
        return frames_tensor
    
    def _extract_poses(self, frames: np.ndarray) -> torch.Tensor:
        """Extract pose sequences from frames."""
        poses = []
        
        # Use first frame as reference for pose extraction
        ref_frame = frames[0]
        ref_frame_resized = cv2.resize(ref_frame, (self.w_target, self.h_target))
        
        for frame in frames:
            frame_resized = cv2.resize(frame, (self.w_target, self.h_target))
            # Extract pose (this would use DWPose in actual implementation)
            pose = get_image_pose(frame_resized)
            poses.append(pose)
        
        poses = np.array(poses)
        poses_tensor = torch.from_numpy(poses).float() / 127.5 - 1.0
        
        return poses_tensor
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample.
        
        Returns:
            dict with keys:
                - 'ref_image': Reference image tensor [3, H, W]
                - 'pose_sequence': Pose sequence tensor [T, 3, H, W]
                - 'video_frames': Target video frames [T, 3, H, W]
                - 'text': Text description (if available)
        """
        sample = self.samples[idx]
        video_path = self.data_root / sample['video_path']
        
        # Random start frame for temporal augmentation
        if self.split == 'train':
            max_start = max(0, sample.get('num_frames', 1000) - self.num_frames * self.sample_stride)
            start_frame = random.randint(0, max_start)
        else:
            start_frame = 0
        
        # Load video frames
        frames = self._load_video_frames(str(video_path), start_frame)
        
        # Preprocess frames
        frames_tensor = self._preprocess_frames(frames)
        
        # Extract poses
        poses_tensor = self._extract_poses(frames)
        
        # First frame as reference
        ref_image = frames_tensor[0]
        
        sample_dict = {
            'ref_image': ref_image,
            'pose_sequence': poses_tensor,
            'video_frames': frames_tensor,
        }
        
        # Add text if available
        if 'text' in sample:
            sample_dict['text'] = sample['text']
        
        if self.transform:
            sample_dict = self.transform(sample_dict)
        
        return sample_dict


class VideoMotionCollator:
    """Custom collator for VideoMotionDataset."""
    
    def __init__(self, with_text: bool = False):
        self.with_text = with_text
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples."""
        collated = {
            'ref_image': torch.stack([s['ref_image'] for s in batch]),
            'pose_sequence': torch.stack([s['pose_sequence'] for s in batch]),
            'video_frames': torch.stack([s['video_frames'] for s in batch]),
        }
        
        if self.with_text and 'text' in batch[0]:
            collated['text'] = [s['text'] for s in batch]
        
        return collated