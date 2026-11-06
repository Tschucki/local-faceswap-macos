# Copyright 2024-2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Modified for Local FaceSwap macOS: Apple Silicon (MPS) support and pipeline changes.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from tqdm import tqdm
import os
import cv2
import csv
import json
import math
import random
import numpy as np
import imageio
import matplotlib
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image, ImageSequence
from io import BytesIO
# from IPython.display import Video
# from IPython.display import display, Image as IPyImage
import torchvision.transforms as T

import sys
from .media_pipe.mp_utils  import LMKExtractor
from .media_pipe.draw_util import FaceMeshVisualizer
from .media_pipe.pose_util import project_points_with_trans, matrix_to_euler_and_translation, euler_and_translation_to_matrix

lmk_extractor = LMKExtractor()
vis = FaceMeshVisualizer(forehead_edge=False)

def get_video_npy(video_path, max_frames=None):
    frames = imageio.get_reader(video_path)
    face_results = []
    last_valid_face = None
    first_valid_index = -1

    for i, frame in enumerate(frames):
        if max_frames is not None and i >= max_frames:
            break
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        face_result = lmk_extractor(frame_bgr)
        
        if face_result is not None:
            face_result['width'] = frame_bgr.shape[1]
            face_result['height'] = frame_bgr.shape[0]
            last_valid_face = face_result
            if first_valid_index == -1:
                first_valid_index = i
        else:
            print(f"[DreamID-V] Warning: Face detection failed on frame {i}. Using fallback.")
            if last_valid_face is not None:
                face_result = last_valid_face.copy()
            else:
                face_result = None
        
        face_results.append(face_result)
        
    if last_valid_face is None:
        raise AssertionError("Can not detect a face in any frame of the video.")
        
    for i in range(len(face_results)):
        if face_results[i] is None:
            face_results[i] = face_results[first_valid_index].copy()
            
    return face_results
