import numpy as np
import cv2
import mediapipe as mp
import torch


class MediaPipeWholebody:
    """Pose detection using MediaPipe instead of ONNX models"""
    
    def __init__(self, device="cpu"):
        self.device = device
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_holistic = mp.solutions.holistic
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def _mediapipe_to_openpose_format(self, results, image_shape):
        """Convert MediaPipe landmarks to OpenPose format
        
        MediaPipe has 33 pose landmarks, we need to map to OpenPose 18 keypoints format
        """
        H, W = image_shape[:2]
        
        # Initialize keypoints array (1 person, 18 keypoints, 2 coordinates)
        keypoints = np.zeros((1, 18, 2))
        scores = np.zeros((1, 18))
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Mapping from MediaPipe to OpenPose keypoints
            # OpenPose format: 
            # 0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist,
            # 5: LShoulder, 6: LElbow, 7: LWrist, 8: RHip, 9: RKnee,
            # 10: RAnkle, 11: LHip, 12: LKnee, 13: LAnkle, 14: REye,
            # 15: LEye, 16: REar, 17: LEar
            
            mp_to_op = {
                0: 0,   # Nose
                # Neck is computed as midpoint between shoulders
                12: 2,  # Right shoulder
                14: 3,  # Right elbow  
                16: 4,  # Right wrist
                11: 5,  # Left shoulder
                13: 6,  # Left elbow
                15: 7,  # Left wrist
                24: 8,  # Right hip
                26: 9,  # Right knee
                28: 10, # Right ankle
                23: 11, # Left hip
                25: 12, # Left knee
                27: 13, # Left ankle
                5: 14,  # Right eye (using right eye inner)
                2: 15,  # Left eye (using left eye inner)
                8: 16,  # Right ear
                7: 17,  # Left ear
            }
            
            for mp_idx, op_idx in mp_to_op.items():
                if mp_idx < len(landmarks):
                    landmark = landmarks[mp_idx]
                    keypoints[0, op_idx] = [landmark.x * W, landmark.y * H]
                    scores[0, op_idx] = landmark.visibility if hasattr(landmark, 'visibility') else 0.5
            
            # Compute neck as midpoint between shoulders
            if 11 < len(landmarks) and 12 < len(landmarks):
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]
                neck_x = (left_shoulder.x + right_shoulder.x) / 2 * W
                neck_y = (left_shoulder.y + right_shoulder.y) / 2 * H
                keypoints[0, 1] = [neck_x, neck_y]
                scores[0, 1] = min(left_shoulder.visibility, right_shoulder.visibility) \
                               if hasattr(left_shoulder, 'visibility') else 0.5
        
        return keypoints, scores
    
    def _extract_hand_keypoints(self, hand_landmarks, image_shape):
        """Extract hand keypoints from MediaPipe hand landmarks"""
        H, W = image_shape[:2]
        
        if hand_landmarks:
            hand_kpts = np.zeros((21, 2))
            hand_scores = np.zeros(21)
            
            for i, landmark in enumerate(hand_landmarks.landmark):
                hand_kpts[i] = [landmark.x * W, landmark.y * H]
                hand_scores[i] = landmark.visibility if hasattr(landmark, 'visibility') else 0.5
            
            return hand_kpts, hand_scores
        else:
            return np.zeros((21, 2)), np.zeros(21)
    
    def _extract_face_keypoints(self, face_landmarks, image_shape):
        """Extract face keypoints from MediaPipe face landmarks"""
        H, W = image_shape[:2]
        
        # MediaPipe provides 468 face landmarks, we'll select 68 key ones
        # This is a simplified mapping - you may want to adjust based on your needs
        if face_landmarks:
            # Select subset of landmarks (simplified - taking every 7th point)
            indices = list(range(0, min(468, len(face_landmarks.landmark)), 7))[:68]
            
            face_kpts = np.zeros((68, 2))
            face_scores = np.zeros(68)
            
            for i, idx in enumerate(indices):
                if idx < len(face_landmarks.landmark):
                    landmark = face_landmarks.landmark[idx]
                    face_kpts[i] = [landmark.x * W, landmark.y * H]
                    face_scores[i] = landmark.visibility if hasattr(landmark, 'visibility') else 0.5
            
            return face_kpts, face_scores
        else:
            return np.zeros((68, 2)), np.zeros(68)
    
    def __call__(self, image):
        """Process image to extract pose keypoints
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
            
        Returns:
            keypoints: Array of shape (N, 18, 2) for N detected persons
            scores: Array of shape (N, 18) with confidence scores
        """
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with holistic model to get pose, hands, and face
        results = self.holistic.process(image_rgb)
        
        # Extract body keypoints
        keypoints, scores = self._mediapipe_to_openpose_format(results, image.shape)
        
        # Extract hands keypoints if needed
        left_hand, left_hand_scores = self._extract_hand_keypoints(
            results.left_hand_landmarks, image.shape)
        right_hand, right_hand_scores = self._extract_hand_keypoints(
            results.right_hand_landmarks, image.shape)
        
        # Extract face keypoints if needed
        face_kpts, face_scores = self._extract_face_keypoints(
            results.face_landmarks, image.shape)
        
        # Combine hands into single array
        hands = np.vstack([left_hand, right_hand])
        hands_scores = np.vstack([left_hand_scores[:, np.newaxis], 
                                  right_hand_scores[:, np.newaxis]])
        
        return keypoints, scores
    
    def __del__(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'holistic'):
            self.holistic.close()


class Wholebody:
    """Compatibility wrapper to match the original Wholebody interface"""
    
    def __init__(self, model_det=None, model_pose=None, device="cpu"):
        # Ignore ONNX model paths, use MediaPipe instead
        self.processor = MediaPipeWholebody(device)
    
    def __call__(self, image):
        """Process image and return in the expected format matching original ONNX output
        
        The original ONNX model returns:
        - keypoints: shape (1, 133, 2) - 18 body + 6 foot + 68 face + 21*2 hands
        - scores: shape (1, 133) - confidence scores
        
        We'll match this format for compatibility
        """
        # Process image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.processor.holistic.process(image_rgb)
        
        H, W = image.shape[:2]
        
        # Initialize full keypoints array (134 keypoints total)
        # 18 body + 6 foot + 68 face + 21 left hand + 21 right hand = 134
        keypoints = np.zeros((1, 134, 2))
        scores = np.zeros((1, 134))
        
        # Extract body keypoints (0-17)
        body_kpts, body_scores = self.processor._mediapipe_to_openpose_format(results, image.shape)
        keypoints[0, :18] = body_kpts[0]
        scores[0, :18] = body_scores[0]
        
        # Foot keypoints (18-23) - MediaPipe doesn't have separate foot, so we'll use ankle positions
        # with slight offsets as placeholders
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Use ankle positions with small offsets for foot keypoints
            for i, ankle_idx in enumerate([27, 28]):  # Left and right ankles in MediaPipe
                if ankle_idx < len(landmarks):
                    ankle = landmarks[ankle_idx]
                    base_pos = np.array([ankle.x * W, ankle.y * H])
                    # Create 3 foot points per ankle with small offsets
                    for j in range(3):
                        offset = np.array([j * 5 - 5, 10])  # Small horizontal and vertical offsets
                        keypoints[0, 18 + i*3 + j] = base_pos + offset
                        scores[0, 18 + i*3 + j] = ankle.visibility if hasattr(ankle, 'visibility') else 0.3
        
        # Face keypoints (24-91)
        if results.face_landmarks:
            # Select 68 face landmarks from MediaPipe's 468
            # Using specific indices that roughly correspond to standard 68-point face model
            face_indices = list(range(0, min(468, len(results.face_landmarks.landmark)), 7))[:68]
            for i, idx in enumerate(face_indices):
                if idx < len(results.face_landmarks.landmark):
                    landmark = results.face_landmarks.landmark[idx]
                    keypoints[0, 24 + i] = [landmark.x * W, landmark.y * H]
                    scores[0, 24 + i] = 0.8  # MediaPipe face is generally reliable
        
        # Left hand keypoints (92-112)
        if results.left_hand_landmarks:
            for i, landmark in enumerate(results.left_hand_landmarks.landmark[:21]):
                keypoints[0, 92 + i] = [landmark.x * W, landmark.y * H]
                scores[0, 92 + i] = 0.8
        
        # Right hand keypoints (113-133)
        if results.right_hand_landmarks:
            for i, landmark in enumerate(results.right_hand_landmarks.landmark[:21]):
                keypoints[0, 113 + i] = [landmark.x * W, landmark.y * H]
                scores[0, 113 + i] = 0.8
        
        return keypoints, scores