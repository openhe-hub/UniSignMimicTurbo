"""
Training dataset for FramerTurbo
Supports video files for frame interpolation training
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from typing import List, Tuple, Optional, Dict
import json


class VideoFrameDataset(Dataset):
    """
    Dataset for video frame interpolation training.
    Extracts frames from video files and creates training samples.

    Args:
        video_dir: Directory containing video files
        num_frames: Number of frames to sample per video (default: 3 for start, middle, end)
        height: Target height for frames
        width: Target width for frames
        sample_stride: Stride for sampling frames from videos
        min_video_frames: Minimum number of frames a video should have
    """

    def __init__(
        self,
        video_dir: str,
        num_frames: int = 3,
        height: int = 320,
        width: int = 512,
        sample_stride: int = 4,
        min_video_frames: int = 16,
        cache_dir: Optional[str] = None,
    ):
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.sample_stride = sample_stride
        self.min_video_frames = min_video_frames
        self.cache_dir = cache_dir

        # Collect all video files
        self.video_files = self._collect_videos()

        if len(self.video_files) == 0:
            raise ValueError(f"No valid video files found in {video_dir}")

        print(f"Found {len(self.video_files)} valid video files")

    def _collect_videos(self) -> List[str]:
        """Collect all video files from the directory"""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []

        for root, dirs, files in os.walk(self.video_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_path = os.path.join(root, file)
                    # Check if video has enough frames
                    cap = cv2.VideoCapture(video_path)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                    if frame_count >= self.min_video_frames:
                        video_files.append(video_path)

        return video_files

    def _extract_frames_from_video(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from a video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        return frames

    def _sample_frame_indices(self, total_frames: int) -> List[int]:
        """
        Sample frame indices for training.
        Returns indices for [start_frame, end_frame] and intermediate frames.
        """
        # Ensure we have enough frames to sample
        max_start_idx = total_frames - (self.num_frames - 1) * self.sample_stride - 1

        if max_start_idx < 0:
            # If video is too short, just sample uniformly
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()
        else:
            # Random start index
            start_idx = random.randint(0, max_start_idx)
            indices = [start_idx + i * self.sample_stride for i in range(self.num_frames)]

        return indices

    def _resize_frame(self, frame: np.ndarray) -> Image.Image:
        """Resize frame to target size"""
        frame_pil = Image.fromarray(frame)
        frame_pil = frame_pil.resize((self.width, self.height), Image.BILINEAR)
        return frame_pil

    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor [-1, 1]"""
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        image = image * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        return image

    def __len__(self) -> int:
        return len(self.video_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a training sample.

        Returns:
            dict with keys:
                - 'pixel_values': tensor of shape (num_frames, C, H, W)
                - 'first_frame': tensor of shape (C, H, W)
                - 'last_frame': tensor of shape (C, H, W)
                - 'video_path': str
        """
        video_path = self.video_files[idx]

        # Extract frames
        frames = self._extract_frames_from_video(video_path)

        # Sample frame indices
        frame_indices = self._sample_frame_indices(len(frames))

        # Get sampled frames
        sampled_frames = [frames[i] for i in frame_indices]

        # Resize frames
        sampled_frames_pil = [self._resize_frame(f) for f in sampled_frames]

        # Convert to tensors
        frames_tensor = torch.stack([self._pil_to_tensor(f) for f in sampled_frames_pil])

        return {
            'pixel_values': frames_tensor,  # (num_frames, 3, H, W)
            'first_frame': frames_tensor[0],  # (3, H, W)
            'last_frame': frames_tensor[-1],  # (3, H, W)
            'video_path': video_path,
        }


class ImagePairDataset(Dataset):
    """
    Dataset for image pair training (start and end frames).
    Useful if you already have extracted frame pairs.

    Expected directory structure:
        data_dir/
            sample_001_start.jpg
            sample_001_end.jpg
            sample_002_start.jpg
            sample_002_end.jpg
            ...
    """

    def __init__(
        self,
        data_dir: str,
        height: int = 320,
        width: int = 512,
        num_frames: int = 3,
    ):
        self.data_dir = data_dir
        self.height = height
        self.width = width
        self.num_frames = num_frames

        # Collect image pairs
        self.pairs = self._collect_pairs()

        if len(self.pairs) == 0:
            raise ValueError(f"No valid image pairs found in {data_dir}")

        print(f"Found {len(self.pairs)} image pairs")

    def _collect_pairs(self) -> List[Tuple[str, str]]:
        """Collect start-end image pairs"""
        files = os.listdir(self.data_dir)
        pairs_dict = {}

        for fname in files:
            if fname.endswith('_start.jpg') or fname.endswith('_start.png'):
                sample_id = fname.rsplit('_start.', 1)[0]
                if sample_id not in pairs_dict:
                    pairs_dict[sample_id] = {}
                pairs_dict[sample_id]['start'] = os.path.join(self.data_dir, fname)
            elif fname.endswith('_end.jpg') or fname.endswith('_end.png'):
                sample_id = fname.rsplit('_end.', 1)[0]
                if sample_id not in pairs_dict:
                    pairs_dict[sample_id] = {}
                pairs_dict[sample_id]['end'] = os.path.join(self.data_dir, fname)

        # Filter complete pairs
        pairs = []
        for sample_id, paths in pairs_dict.items():
            if 'start' in paths and 'end' in paths:
                pairs.append((paths['start'], paths['end']))

        return pairs

    def _load_and_resize(self, image_path: str) -> torch.Tensor:
        """Load image and convert to tensor"""
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.width, self.height), Image.BILINEAR)

        # Convert to tensor [-1, 1]
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = image * 2.0 - 1.0

        return image

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_path, end_path = self.pairs[idx]

        first_frame = self._load_and_resize(start_path)
        last_frame = self._load_and_resize(end_path)

        # Create intermediate frames (will be generated by model)
        # For now, we just use the endpoints
        pixel_values = torch.stack([first_frame] + [first_frame] * (self.num_frames - 2) + [last_frame])

        return {
            'pixel_values': pixel_values,
            'first_frame': first_frame,
            'last_frame': last_frame,
            'start_path': start_path,
            'end_path': end_path,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for dataloader"""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    first_frames = torch.stack([item['first_frame'] for item in batch])
    last_frames = torch.stack([item['last_frame'] for item in batch])

    return {
        'pixel_values': pixel_values,  # (B, num_frames, 3, H, W)
        'first_frames': first_frames,  # (B, 3, H, W)
        'last_frames': last_frames,    # (B, 3, H, W)
    }
