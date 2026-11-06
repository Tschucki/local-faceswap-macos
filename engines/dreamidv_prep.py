# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for Local FaceSwap macOS: extracted from the original generation
# script, writes into a caller-chosen folder and limits the frame count.
"""Pose and face-mask conditioning videos for the DreamID-V face model."""
import os

import cv2
import numpy as np
from PIL import Image

from express_adaption.get_video_npy import get_video_npy
from express_adaption.media_pipe import FaceMeshAlign_dreamidv, FaceMeshDetector

CORE_LANDMARK_INDICES = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    1, 2, 5, 6, 48, 64, 94, 98, 168, 195, 197, 278, 294, 324, 327, 4, 24,
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
    468, 473, 55, 65, 52, 53, 46, 285, 295, 282, 283, 276, 70, 63, 105, 66, 107,
    300, 293, 334, 296, 336, 156,
]
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
CORE_LANDMARK_INDICES = list(set(CORE_LANDMARK_INDICES + FACE_OVAL_INDICES))


def _save_landmark_video(landmarks_sequence, path, frame_size, fps, mode):
    width, height = frame_size
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'Could not open video writer for {path}')
    for frame_landmarks in landmarks_sequence:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if mode == 'points':
            for x, y in frame_landmarks[:, :2].astype(int):
                cv2.circle(frame, (int(x), int(y)), radius=2, color=(255, 255, 255), thickness=-1)
        else:
            cv2.fillConvexPoly(frame, frame_landmarks.astype(np.int32), color=(255, 255, 255))
        writer.write(frame)
    writer.release()


def generate_pose_and_mask_videos(ref_video_path, ref_image_path, max_frames=None, out_dir=None):
    """Write <name>_pose.mp4 (aligned landmarks) and <name>_mask.mp4 (face oval); return both paths."""
    detector = FaceMeshDetector()
    align_motion = FaceMeshAlign_dreamidv()
    fps = cv2.VideoCapture(ref_video_path).get(cv2.CAP_PROP_FPS)
    face_results = get_video_npy(ref_video_path, max_frames=max_frames)
    video_name = os.path.splitext(os.path.basename(ref_video_path))[0]
    out_dir = out_dir or os.path.join(os.path.dirname(ref_video_path), 'temp_generated')
    os.makedirs(out_dir, exist_ok=True)

    _, ref_img_lmk = detector(np.array(Image.open(ref_image_path).convert('RGB')))
    _, pose_addvis = align_motion(face_results, ref_img_lmk)
    size = (face_results[0]['width'], face_results[0]['height'])

    pose_path = os.path.join(out_dir, video_name + '_pose.mp4')
    _save_landmark_video(pose_addvis[:, CORE_LANDMARK_INDICES, :], pose_path, size, fps, 'points')
    mask_path = os.path.join(out_dir, video_name + '_mask.mp4')
    _save_landmark_video(pose_addvis[:, FACE_OVAL_INDICES, :], mask_path, size, fps, 'mask')
    return pose_path, mask_path
