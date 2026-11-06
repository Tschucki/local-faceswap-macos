#!/usr/bin/env python3
"""Head swap (face + hair) on Apple Silicon with Wan2.2-Animate-14B, replacement mode.

Only the head of the person in the video is replaced by the person in the
reference photo; body, background, camera and audio stay original.

Pipeline
  1. Track the head with MediaPipe face landmarks and follow it with a smoothed,
     constant-size crop, so the model always sees the head large and centred.
  2. In crop space build what Wan-Animate's replacement mode expects:
     pose skeleton (ViTPose), 512x512 face crops (expression driver),
     a head+hair+neck mask and the masked background. The mask also covers
     where the *new* hair will be (the reference photo's hair silhouette is
     projected onto every frame via the face landmarks), so longer hair has room.
     The reference photo is first brought to the scene's exposure/white balance.
  3. Wan2.2-Animate-14B regenerates only the masked region, with its relighting
     LoRA so the new head takes the scene's light. Long clips are generated in
     77-frame windows that continue from the previous window's last frame(s).
  4. Give the new head the original face's exposure, white balance and shading
     (the model tends to keep the photo's flat studio light, the main "AI look"),
     then paste it back into the untouched full-resolution frames with a
     feathered alpha (old head + new head + neck; the original clothes stay),
     match film grain, keep the source fps and copy the original audio.
     --from_work re-runs this step on a --keep_temp folder without regenerating.
"""
import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import warnings

warnings.filterwarnings('ignore')

import cv2
import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(REPO, 'models')
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'engines'))  # bundled model code (Wan, DreamID-V, MediaPipe helpers)

from faceswap import FrameWriter, match_grain, mux_audio, probe, write_sidecar  # noqa: E402

SEG_MODEL = os.path.join(REPO, '.cache', 'models', 'selfie_multiclass_256x256.tflite')
SEG_URL = ('https://storage.googleapis.com/mediapipe-models/image_segmenter/'
           'selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite')
HAND_MODEL = os.path.join(REPO, '.cache', 'models', 'hand_landmarker.task')
HAND_URL = ('https://storage.googleapis.com/mediapipe-models/hand_landmarker/'
            'hand_landmarker/float16/latest/hand_landmarker.task')
# selfie_multiclass classes
BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHER = 0, 1, 2, 3, 4, 5

PRESETS = {
    # size: portrait crop (width x height) fed to the DiT, multiples of 16.
    # Head + hair + shoulders are taller than wide, so portrait framing gives the
    # face more pixels per token than a square crop.
    'fast':     dict(size=(448, 640), steps=20),
    'balanced': dict(size=(512, 720), steps=20),
    'best':     dict(size=(576, 816), steps=25),
}
# MediaPipe face-mesh points used to align the reference head to each frame
ALIGN_IDX = [33, 133, 362, 263, 1, 61, 291, 152, 10, 234, 454]
OVAL_IDX = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
            400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
            54, 103, 67, 109]
INNER_LIP_IDX = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
OUTER_LIP_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
# eyes + brows, lips, nose wings: left out when measuring skin shading
FEATURE_IDX = [
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
    [98, 97, 2, 326, 327, 294, 278, 344, 440, 275, 4, 45, 220, 115, 48, 64],
]


# --------------------------------------------------------------------------- #
# Landmarks and segmentation
# --------------------------------------------------------------------------- #
def download(url, path):
    """urllib with certifi's CA bundle (some macOS Pythons ship without system CAs)."""
    import ssl
    import urllib.request
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(url, context=ctx) as r, open(path + '.part', 'wb') as f:
        shutil.copyfileobj(r, f)
    os.replace(path + '.part', path)


class FaceTools:
    def __init__(self):
        from express_adaption.media_pipe.mp_utils import LMKExtractor
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.lmk = LMKExtractor()
        if not os.path.exists(SEG_MODEL):
            logging.info('Downloading hair/face segmentation (16 MB) ...')
            download(SEG_URL, SEG_MODEL)
        opts = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=SEG_MODEL),
            running_mode=vision.RunningMode.IMAGE, output_category_mask=True)
        self.seg = vision.ImageSegmenter.create_from_options(opts)
        self.mp = mp

    def landmarks(self, rgb):
        """478 face-mesh points in pixels, or None."""
        h, w = rgb.shape[:2]
        res = self.lmk(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if res is None:
            return None
        return res['lmks'][:, :2] * np.array([w, h], dtype=np.float32)

    def categories(self, rgb):
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        return self.seg.segment(img).category_mask.numpy_view().copy()

    def hands(self, rgb):
        """21 landmarks per detected hand, in pixels (list, maybe empty)."""
        if not hasattr(self, 'hand'):
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            if not os.path.exists(HAND_MODEL):
                logging.info('Downloading hand detection (8 MB) ...')
                download(HAND_URL, HAND_MODEL)
            self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL), num_hands=2,
                min_hand_detection_confidence=0.4))
        h, w = rgb.shape[:2]
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        return [np.array([[p.x * w, p.y * h] for p in hand], np.float32)
                for hand in self.hand.detect(img).hand_landmarks]


ID_MODEL = os.path.join(REPO, '.cache', 'models', 'auraface_glintr100.onnx')
ID_REPO = ('fal/AuraFace-v1', 'glintr100.onnx')  # ArcFace-type recognition, Apache-2.0
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


class IdentityScorer:
    """How much a face looks like the person in the photo: cosine similarity of
    AuraFace embeddings (5-point aligned 112 px crops from MediaPipe landmarks).
    Different people score ~0.0-0.2, the same person ~0.9 (frame to frame)."""

    def __init__(self, tools):
        import onnxruntime as ort
        if not os.path.exists(ID_MODEL):
            from huggingface_hub import hf_hub_download
            logging.info('Downloading AuraFace face recognition (260 MB, for the identity score) ...')
            tmp = hf_hub_download(*ID_REPO, local_dir=os.path.join(REPO, '.cache', 'models', '.dl'))
            shutil.move(tmp, ID_MODEL)
            shutil.rmtree(os.path.join(REPO, '.cache', 'models', '.dl'), ignore_errors=True)
        self.sess = ort.InferenceSession(ID_MODEL, providers=['CPUExecutionProvider'])
        self.tools = tools

    def embed(self, rgb):
        p = self.tools.landmarks(rgb)
        if p is None:
            return None
        five = np.array([(p[33] + p[133]) / 2, (p[362] + p[263]) / 2, p[1], p[61], p[291]], np.float32)
        M, _ = cv2.estimateAffinePartial2D(five, ARCFACE_DST)
        if M is None:
            return None
        face = cv2.warpAffine(rgb, M, (112, 112), flags=cv2.INTER_AREA).astype(np.float32)
        e = self.sess.run(None, {'data': ((face - 127.5) / 127.5).transpose(2, 0, 1)[None]})[0][0]
        return e / np.linalg.norm(e)

    def similarity(self, frames, ref_emb, k=12):
        """Mean similarity of up to k evenly spaced frames to ref_emb (None if no face)."""
        idx = np.linspace(0, len(frames) - 1, min(k, len(frames))).astype(int)
        es = [e for e in (self.embed(frames[i]) for i in idx) if e is not None]
        return float(np.mean([e @ ref_emb for e in es])) if es else None


def face_size(pts):
    return float(np.linalg.norm(pts[10] - pts[152]))  # forehead top -> chin


def head_mask(cat, pts, neck=True):
    """Hair + face (+ upper neck) as uint8 {0,1}, cleaned to the head's component."""
    m = np.isin(cat, (HAIR, FACE_SKIN)).astype(np.uint8)
    if pts is not None:
        cv2.fillConvexPoly(m, cv2.convexHull(pts[OVAL_IDX].astype(np.int32)), 1)
        if neck:
            fs = face_size(pts)
            chin_y = pts[152, 1]
            band = np.zeros_like(m)
            x0, x1 = int(pts[172, 0]), int(pts[397, 0])  # jaw width: neck, not hands
            band[int(chin_y - 0.1 * fs):int(chin_y + 0.45 * fs), max(0, x0):max(0, x1)] = 1
            m |= (band & (cat == BODY_SKIN)).astype(np.uint8)
        # keep only components touching the face (drops other people / hands)
        n, lab = cv2.connectedComponents(m)
        if n > 2:
            hull = np.zeros_like(m)
            cv2.fillConvexPoly(hull, cv2.convexHull(pts[OVAL_IDX].astype(np.int32)), 1)
            keep = np.unique(lab[(hull > 0) & (m > 0)])
            keep = keep[keep > 0]
            if len(keep):
                m = np.isin(lab, keep).astype(np.uint8)
    return m


def hand_mask(hands, shape, fs):
    """Detected hands as a mask: palm polygon + finger bones (no hull: keeps the gaps between fingers)."""
    hand = np.zeros(shape, np.uint8)
    for h in hands:
        cv2.fillConvexPoly(hand, cv2.convexHull(h[[0, 1, 5, 9, 13, 17]].astype(np.int32)), 1)
        for finger in ([0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]):
            cv2.polylines(hand, [h[finger].astype(np.int32)], False, 1, max(2, int(fs * 0.08)))
    return hand


def grow_occluder(seed, cat, pts, hands):
    """Occluders are found inside the face oval; a hand at the chin continues below
    it, a cup or sandwich sticks out sideways. Grow them through connected skin /
    object pixels (never hair, face skin or background) up to half a face size
    around the oval. Returns 0 = nothing, 1 = weak (plain body skin: may be the
    neck, so the new head's own chin/hair wins there), 2 = strong (detected hand
    or object: always stays original)."""
    out = seed.astype(np.uint8) * 2
    if pts is None or not seed.any():
        return out
    fs = face_size(pts)
    oval = np.zeros(cat.shape, np.uint8)
    cv2.fillConvexPoly(oval, cv2.convexHull(pts[OVAL_IDX].astype(np.int32)), 1)
    region = cv2.dilate(oval, disk(fs * 0.5)) > 0
    hand = hand_mask(hands, cat.shape, fs) > 0
    cand = ((hand | np.isin(cat, (BODY_SKIN, CLOTHES, OTHER))) & (cat != HAIR) & region) | (seed > 0)
    _, lab = cv2.connectedComponents(cand.astype(np.uint8))
    ids = np.unique(lab[seed > 0])
    grown = np.isin(lab, ids[ids > 0]).astype(np.uint8)
    grown = (cv2.dilate(grown, disk(fs * 0.02)) > 0) & (cat != HAIR) & region
    strong = grown & (hand | (cat == OTHER))
    out[grown] = np.maximum(out[grown], 1)
    out[strong] = 2
    return out


def occluder_mask(cat, pts, hands):
    """Things in front of the face that must stay original: hands and fingers
    (also in the mouth), food, cups, microphones, glasses held up ...
    Inside the (slightly shrunk) face oval: skin that belongs to a detected hand,
    and anything the segmenter does not call face/hair/body skin. The open mouth
    is excluded (dark mouth interior and teeth are not objects)."""
    m = np.zeros(cat.shape, np.uint8)
    if pts is None:
        return m
    fs = face_size(pts)
    oval = np.zeros_like(m)
    cv2.fillConvexPoly(oval, cv2.convexHull(pts[OVAL_IDX].astype(np.int32)), 1)
    core = cv2.erode(oval, disk(fs * 0.05))
    mouth = np.zeros_like(m)
    cv2.fillConvexPoly(mouth, cv2.convexHull(pts[INNER_LIP_IDX].astype(np.int32)), 1)
    mouth = cv2.dilate(mouth, disk(fs * 0.03))
    hand = hand_mask(hands, m.shape, fs)
    # Body skin inside the face oval is a hand/finger even when the hand detector
    # misses it (partly visible hands, fingers at the mouth); the neck lies
    # below the chin, outside the oval. Detected hands add fingers that the
    # segmenter took for face skin.
    # Objects only from the middle of the nose down: food, cups and microphones
    # reach the mouth, while worn glasses (segmented as "other") sit at the eye
    # line and must be swapped with the head, or the original eyes would show
    # through the lenses. t runs along the face axis: 0 at the eyes, 1 at the mouth.
    eyes = (pts[33] + pts[133] + pts[362] + pts[263]) / 4
    axis = (pts[13] + pts[14]) / 2 - eyes
    yy, xx = np.mgrid[0:cat.shape[0], 0:cat.shape[1]].astype(np.float32)
    t = ((xx - eyes[0]) * axis[0] + (yy - eyes[1]) * axis[1]) / max(float(axis @ axis), 1.0)
    objects = np.isin(cat, (BACKGROUND, CLOTHES, OTHER)) & (mouth == 0) & (t > 0.42)
    m = (core > 0) & (((hand > 0) & (cat != HAIR)) | (cat == BODY_SKIN) | objects)
    m = m.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    min_px = max(20, int(0.004 * oval.sum()))
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]
    if not keep:
        return np.zeros_like(m)
    # grow a little into the oval rim (the core is shrunk) and over anti-aliased edges
    m = cv2.dilate(np.isin(lab, keep).astype(np.uint8), disk(fs * 0.03))
    return m & oval & (cat != HAIR).astype(np.uint8)


def original_mouth(gen, src, gpts, spts):
    """Mouth interior (teeth, tongue, food, a finger) from the original frame,
    fitted into the new lips. Returns (rgb, weight) or None when the mouth is
    closed. The new lips stay; only what is between them comes from the video."""
    fs = face_size(gpts)
    inner_g, inner_s = gpts[INNER_LIP_IDX], spts[INNER_LIP_IDX]
    area = cv2.contourArea(cv2.convexHull(inner_g.astype(np.float32)))
    if area < 0.002 * fs * fs:  # closed or nearly closed
        return None
    A, _ = cv2.estimateAffine2D(inner_s, inner_g)
    if A is None:
        return None
    h, w = gen.shape[:2]
    warped = cv2.warpAffine(src.astype(np.float32), A, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT101)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [inner_g.astype(np.int32)], 1)
    m = cv2.erode(m, disk(fs * 0.01))
    weight = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, fs * 0.008))
    return warped, weight


def disk(d):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(d) | 1,) * 2)


def skin_weight(cat, pts):
    """Face skin without eyes, brows, lips and nose, eroded: where shading is measured."""
    m = (cat == FACE_SKIN).astype(np.uint8)
    if pts is None:
        return m.astype(np.float32)
    fs = face_size(pts)
    feat = np.zeros_like(m)
    for idx in FEATURE_IDX:
        cv2.fillConvexPoly(feat, cv2.convexHull(pts[idx].astype(np.int32)), 1)
    m &= 1 - cv2.dilate(feat, disk(fs * 0.06))
    return cv2.erode(m, disk(fs * 0.04)).astype(np.float32)


def masked_blur(img, w, sigma):
    """Normalised convolution: blur of img using only pixels with weight w (+ coverage)."""
    num = cv2.GaussianBlur(img * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return num / np.maximum(den, 1e-4), den


# --------------------------------------------------------------------------- #
# Crop track
# --------------------------------------------------------------------------- #
def smooth_track(values, sigma):
    """Fill missing entries by interpolation, then Gaussian-smooth over time."""
    v = np.array([np.nan if x is None else x for x in values], dtype=np.float64)
    ok = ~np.isnan(v)
    if not ok.any():
        return None
    idx = np.arange(len(v))
    v = np.interp(idx, idx[ok], v[ok])
    if sigma > 0 and len(v) > 1:
        r = int(3 * sigma)
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
        k /= k.sum()
        v = np.convolve(np.pad(v, r, mode='edge'), k, mode='valid')
    return v


def crop_matrices(lmks, width, height, crop_h, size, fps):
    """Per-frame 2x3 affine (full frame -> crop) for a smoothed, fixed-size crop.
    crop_h: wanted crop height in source pixels; size: model (width, height)."""
    cx = [None if p is None else float((p[:, 0].min() + p[:, 0].max()) / 2) for p in lmks]
    cy = [None if p is None else float((p[:, 1].min() + p[:, 1].max()) / 2 - 0.12 * face_size(p)) for p in lmks]
    sigma = 0.25 * fps
    cx, cy = smooth_track(cx, sigma), smooth_track(cy, sigma)
    mw, mh = size
    ch = min(crop_h, height, width * mh / mw)  # keep the model aspect, stay inside the frame
    cw = ch * mw / mh
    s = mh / ch
    mats = []
    for x, y in zip(cx, cy):
        x0 = min(max(0.0, x - cw / 2), width - cw)
        y0 = min(max(0.0, y - ch / 2), height - ch)
        mats.append(np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], dtype=np.float64))
    return mats, ch


def warp(img, M, size, interp=cv2.INTER_CUBIC):
    return cv2.warpAffine(img, M, tuple(size), flags=interp, borderMode=cv2.BORDER_REFLECT101)


def to_crop(pts, M):
    return pts @ M[:, :2].T + M[:, 2]


# --------------------------------------------------------------------------- #
# Reference
# --------------------------------------------------------------------------- #
def prepare_reference(path, tools, size, face_ratio):
    """Portrait of the new person framed exactly like the video crop + its head mask."""
    img = np.array(Image.open(path).convert('RGB'))
    pts = tools.landmarks(img)
    if pts is None:
        raise SystemExit(f'No face found in the photo: {path}')
    fs = face_size(pts)
    if fs < 200:
        logging.warning(f'The face in the photo is only ~{int(fs)}px tall. '
                        'A larger, sharp photo improves identity and hair a lot.')
    mw, mh = size
    ch = fs / face_ratio
    cw = ch * mw / mh
    cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
    cy = (pts[:, 1].min() + pts[:, 1].max()) / 2 - 0.12 * fs
    s = mh / ch
    M = np.array([[s, 0, -(cx - cw / 2) * s], [0, s, -(cy - ch / 2) * s]])
    # Outside the photo: black, like Wan-Animate's own padding_resize (no smeared edges).
    ref = cv2.warpAffine(img, M, (mw, mh), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    rpts = to_crop(pts, M)
    rmask = head_mask(tools.categories(ref), rpts, neck=False)
    return ref, rpts, rmask


# --------------------------------------------------------------------------- #
# Scene light and composite
# --------------------------------------------------------------------------- #
def light_terms(gen, src, gen_cat, src_cat, gpts, spts):
    """How to move the generated head into the scene's light, measured on face skin.

    Wan-Animate tends to keep the reference photo's (studio) exposure and flat
    light, which is what makes the head look pasted in. Returns
      gain  (3,)   RGB factor: exposure + white balance of the original face skin
      shape (H, W) relative shading of the original face (light direction,
                   shadow side), warped onto the new face's geometry
    Only heavily blurred luma ratios are used, so the new face keeps its detail.
    """
    h, w = gen.shape[:2]
    ws, wg = skin_weight(src_cat, spts), skin_weight(gen_cat, gpts)
    if ws.sum() < 200 or wg.sum() < 200:
        return None, None
    s, g = src.astype(np.float32), gen.astype(np.float32)
    gain = (s[ws > 0].mean(0) + 1) / (g[wg > 0].mean(0) + 1)
    ys, yg = s.mean(2), g.mean(2)
    A, _ = cv2.estimateAffinePartial2D(spts[ALIGN_IDX], gpts[ALIGN_IDX])
    if A is not None:
        ys = cv2.warpAffine(ys, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ws = cv2.warpAffine(ws, A, (w, h), flags=cv2.INTER_LINEAR)
    sigma = SHADE_BLUR * face_size(gpts)
    bs, cs = masked_blur(ys, ws, sigma)
    bg, cg = masked_blur(yg, wg, sigma)
    conf = np.clip(np.minimum(cs, cg) / 0.3, 0, 1)  # 1 on the face, 0 away from it
    ratio = (bs * yg[wg > 0.5].mean() + 1e-3) / (bg * ys[ws > 0.5].mean() + 1e-3)
    ratio = np.where(conf > 0, np.clip(ratio, 0.6, 1.6), 1.0)  # far from skin both blurs are ~0
    shape = np.exp(conf * np.log(ratio))
    return gain.astype(np.float32), shape.astype(np.float16)


def apply_light(img, gain, shape):
    """img * gain * shape, capped per pixel below 250 so bright skin keeps its hue
    instead of clipping one channel (red cheeks turning orange)."""
    f = gain[None, None, :] * shape[..., None]
    out = img.astype(np.float32) * f
    peak = out.max(axis=2, keepdims=True)
    return out * np.minimum(1.0, 250.0 / np.maximum(peak, 1e-3))


def frontal_index(pts_list):
    """Frame whose face is most frontal (nose centred between the face sides)."""
    best, score = 0, np.inf
    for i, p in enumerate(pts_list):
        if p is None:
            continue
        left, right = np.linalg.norm(p[1] - p[234]), np.linalg.norm(p[454] - p[1])
        s = abs(left - right) / (left + right)
        if s < score:
            best, score = i, s
    return best


def reference_in_scene_light(ref, rpts, tools, crop, cpts_k, src_cat_k, strength):
    """Bring the reference photo to the video's exposure and white balance
    (measured on face skin of a frontal frame), so the model does not start from
    studio light. Shading is left alone: it would be baked into the identity."""
    gain, _ = light_terms(ref, crop, tools.categories(ref), src_cat_k, rpts, cpts_k)
    if gain is None or strength <= 0:
        return ref, None
    gain = gain ** strength
    out = apply_light(ref, gain, np.ones(ref.shape[:2], np.float32))
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), gain


# Shading blur relative to face height. Finer blur also carries the old face's
# 3D structure (nose/cheekbone shadows) onto the new face: measured on the test
# clip, 0.12 raised the similarity to the original person from 0.05 to 0.10,
# 0.25 keeps the light direction with only 0.07.
SHADE_BLUR = 0.25
DETAIL_SIGMAS = (0.8, 1.6, 3.2, 6.4)  # full-res px; bands 0.8-1.6, 1.6-3.2, 3.2-6.4


def match_detail(gen, orig, w_gen, w_src, prev, strength, lo=0.7, hi=1.8):
    """Give the new face skin the texture spectrum of the original skin in the
    same footage. Generated skin is too smooth in the mid frequencies (pores,
    fine creases) while its edges stay crisp, a typical AI signature. Per band,
    the new face's *own* detail is scaled (nothing is copied from the old face).
    gen/orig: float RGB (same region), w_gen/w_src: skin weights. prev: band
    gains of the previous frame (smoothing). Returns (gen, gains)."""
    mg, ms = w_gen > 0.5, w_src > 0.5
    if mg.sum() < 200 or ms.sum() < 200 or strength <= 0:
        return gen, prev
    w = cv2.GaussianBlur(w_gen, (0, 0), 2.0)[..., None]
    ys = orig.mean(2)
    ls = [ys] + [cv2.GaussianBlur(ys, (0, 0), s) for s in DETAIL_SIGMAS]
    bands = range(1, len(DETAIL_SIGMAS))
    target = np.array([(ls[b] - ls[b + 1])[ms].std() for b in bands])

    def split(img):
        y = img.mean(2)
        lv = [y] + [cv2.GaussianBlur(y, (0, 0), s) for s in DETAIL_SIGMAS]
        return [lv[b] - lv[b + 1] for b in bands]

    # Gaussian bands overlap, so one boost reaches only ~half the target: two passes.
    first = None
    for it in range(2):
        d = split(gen)
        k = np.clip(target / np.maximum([x[mg].std() for x in d], 1e-3), lo, hi)
        if it == 0:
            if prev is not None:
                k = 0.7 * prev + 0.3 * k  # temporal smoothing
            first = k
        else:
            k = np.clip(k, 0.85, 1.4)
        gen = gen + sum((k[j] ** strength - 1) * d[j] for j in range(len(d)))[..., None] * w
    return gen, first


def recolor_clothes(gen, gen_cat, src, src_cat):
    """Wan paints its own top inside the mask (e.g. where long hair was removed);
    give it the colour statistics of the original clothes."""
    gm, sm = gen_cat == CLOTHES, src_cat == CLOTHES
    if gm.sum() < 200 or sm.sum() < 200:
        return gen
    gl = cv2.cvtColor(np.clip(gen, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    sl = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    out = (gl - gl[gm].mean(0)) / (gl[gm].std(0) + 1e-3) * sl[sm].std(0) + sl[sm].mean(0)
    out = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    wgt = cv2.GaussianBlur(gm.astype(np.float32), (0, 0), 2.0)[..., None]
    return gen * (1 - wgt) + out * wgt


def head_alpha(src_cat, gen_cat, src_head, new_head, gen_mask, grow, sigma):
    """Paste-back alpha: old head (must disappear) + new head, inside the region
    the model regenerated. Where the new frame shows clothes, the original
    clothes/neck stay, unless old hair has to be covered there."""
    a = cv2.dilate(src_head | new_head, disk(grow))
    keep_src = (gen_cat == CLOTHES) & ~np.isin(src_cat, (HAIR, FACE_SKIN))
    a = a & gen_mask & (~keep_src).astype(np.uint8)
    return cv2.GaussianBlur(a.astype(np.float32), (0, 0), sigma)


def face_pass(gen, args, work, fps_str):
    """Second pass with DreamID-V (face only) on the generated head crops.

    Wan-Animate gets hair, head shape, motion and light right, but resembles the
    photo less the larger the face is in the model. DreamID-V swaps just the
    face on top and is much closer to the photo (test clip, AuraFace: 0.60 ->
    0.86). It runs before the light/detail step, so that step also corrects
    DreamID-V's brighter, smoother face."""
    import subprocess
    from decord import VideoReader
    from faceswap import write_video
    src, dst = os.path.join(work, 'face_pass_in.mp4'), os.path.join(work, 'face_pass_out.mp4')
    write_video(src, list(gen), fps_str, crf=8)
    cmd = [sys.executable, os.path.join(REPO, 'faceswap.py'), '--ref_image', args.ref_image,
           '--video', src, '--preset', args.face_pass, '--guide', str(args.face_guide),
           '--seed', str(args.seed), '--out', dst]
    t0 = time.time()
    logging.info(f'Face pass with DreamID-V ({args.face_pass}, guide {args.face_guide}) ...')
    subprocess.run(cmd, check=True)
    vr = VideoReader(dst)
    out = vr.get_batch(list(range(min(len(vr), len(gen))))).asnumpy()
    if len(out) < len(gen) or out.shape[1:3] != gen.shape[1:3]:
        raise SystemExit(f'Face pass returned {len(out)} frames {out.shape[2]}x{out.shape[1]}, '
                         f'expected {len(gen)} frames {gen.shape[2]}x{gen.shape[1]}.')
    logging.info(f'Face pass done in {(time.time() - t0) / 60:.1f} min')
    return out


def merge_face_pass(wan, fp, tools, mouth=0.0, eyes=0.0):
    """Combine the face pass with the Wan generation it was run on.
    - DreamID-V only counts inside the face oval: outside it (neck, hair, a hand
      or food at the chin) it sometimes paints things of its own, Wan is right there.
    - mouth > 0: lips and mouth from Wan (opens like the original), eyes > 0: eyes
      and lids from Wan (DreamID-V sometimes opens one eye while the person looks
      down). Both cost a little similarity to the photo.
    Every mask covers both versions' landmarks, so no second edge or mouth shows."""
    out = fp.copy()
    for i in range(len(fp)):
        pts = [p for p in (tools.landmarks(wan[i]), tools.landmarks(fp[i])) if p is not None]
        if not pts:
            continue
        fs = face_size(pts[0])

        def region(idx, grow, soft):
            m = np.zeros(wan.shape[1:3], np.uint8)
            for p in pts:
                for part in idx:
                    cv2.fillConvexPoly(m, cv2.convexHull(p[part].astype(np.int32)), 1)
            m = cv2.dilate(m, disk(fs * grow))
            return cv2.GaussianBlur(m.astype(np.float32), (0, 0), fs * soft)

        w_fp = region([OVAL_IDX], 0.03, 0.03)
        if mouth > 0:
            w_fp *= 1 - mouth * region([OUTER_LIP_IDX], 0.05, 0.025)
        if eyes > 0:
            w_fp *= 1 - eyes * region(FEATURE_IDX[:2], 0.06, 0.025)
        w = w_fp[..., None]
        out[i] = np.clip(fp[i] * w + wan[i].astype(np.float32) * (1 - w) + 0.5, 0, 255).astype(np.uint8)
    return out


def load_generated(work, n):
    npy = os.path.join(work, 'generated.npy')
    if os.path.exists(npy):
        gen = np.load(npy)
    else:
        from decord import VideoReader
        vr = VideoReader(os.path.join(work, 'generated.mp4'))
        gen = vr.get_batch(list(range(len(vr)))).asnumpy()
    if len(gen) < n:
        raise SystemExit(f'{work} holds only {len(gen)} generated frames, {n} needed. '
                         'Use the same --start/--seconds as in the original run.')
    return gen[:n]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
TURBO_LORA = ('Kijai/WanVideo_comfy', 'Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors')


def turbo_lora_path():
    """lightx2v step-distill LoRA (Wan2.1-I2V-14B, works on Wan-Animate), 0.7 GB."""
    path = os.path.join(MODELS, 'loras', os.path.basename(TURBO_LORA[1]))
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        logging.info('Downloading the lightx2v turbo LoRA (0.7 GB) ...')
        tmp = hf_hub_download(TURBO_LORA[0], TURBO_LORA[1], local_dir=os.path.join(MODELS, 'loras', '.dl'))
        shutil.move(tmp, path)
        shutil.rmtree(os.path.join(MODELS, 'loras', '.dl'), ignore_errors=True)
    return path


def parse_loras(specs):
    """['path', 'path:0.8'] -> [(path, strength)]"""
    out = []
    for s in specs or []:
        path, strength = s, 1.0
        head, sep, tail = s.rpartition(':')
        if sep:
            try:
                path, strength = head, float(tail)
            except ValueError:
                pass
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise SystemExit(f'LoRA not found: {path}')
        out.append((path, strength))
    return out


def load_model(args):
    from wan_animate import WanAnimate
    from wan_animate.configs import animate_14B
    t0 = time.time()
    model = WanAnimate(
        config=animate_14B,
        checkpoint_dir=args.ckpt_dir,
        use_relighting_lora=not args.no_relight,
        t5_cache_dir=os.path.join(REPO, '.cache', 't5'),
        extra_loras=args.loras,
    )
    logging.info(f'Wan2.2-Animate-14B loaded in {time.time() - t0:.0f}s')
    return model


def load_pose(args):
    from wan_animate.modules.animate.preprocess.pose2d import Pose2d
    pc = os.path.join(args.ckpt_dir, 'process_checkpoint')
    return Pose2d(checkpoint=os.path.join(pc, 'pose2d', 'vitpose_h_wholebody.onnx'),
                  detector_checkpoint=os.path.join(pc, 'det', 'yolov10m.onnx'), device='cpu')


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description='Head swap (face + hair) with Wan2.2-Animate-14B.')
    p.add_argument('--ref_image', required=True, help='Photo of the new person (head and hair clearly visible).')
    p.add_argument('--video', required=True, help='Target video.')
    p.add_argument('--out', default=None, help='Output .mp4 (default: output/head_<name>_<settings>_<time>.mp4).')
    p.add_argument('--preset', default='balanced', choices=list(PRESETS))
    p.add_argument('--start', type=float, default=0.0, help='Start time in seconds.')
    p.add_argument('--seconds', type=float, default=None, help='Length in seconds (default: until the end).')
    p.add_argument('--size', default=None, help='Overrides the preset: model resolution WxH, e.g. 512x720 (multiples of 16).')
    p.add_argument('--steps', type=int, default=None, help='Overrides the preset: sampling steps.')
    p.add_argument('--solver', default=None, choices=['dpm++', 'euler', 'unipc'],
                   help='Sampler (default: dpm++, with --turbo: euler).')
    p.add_argument('--turbo', action='store_true',
                   help='lightx2v distill LoRA: 6 instead of 20-25 steps (about 4x faster).')
    p.add_argument('--lora', action='append', default=[],
                   help='Mix in an extra LoRA, PATH or PATH:STRENGTH (repeatable).')
    p.add_argument('--shift', type=float, default=5.0)
    p.add_argument('--face_ratio', type=float, default=0.30, help='Face height relative to the crop (smaller = more surroundings/hair).')
    p.add_argument('--hair_room', type=float, default=0.06, help='Extra mask margin around head/hair, relative to the crop size.')
    p.add_argument('--mask_grid', type=int, default=1,
                   help='Block grid of the generation mask as in Wan preprocessing (1 = one box around head/hair).')
    p.add_argument('--refert', type=int, default=1, choices=[1, 5], help='Overlap frames between 77-frame windows.')
    p.add_argument('--grain', type=float, default=1.0, help='Film grain matching (0 = off).')
    p.add_argument('--light', type=float, default=1.0,
                   help='Match exposure and white balance of the new head to the scene (0 = off, 1 = full).')
    p.add_argument('--shade', type=float, default=1.0,
                   help='Carry over the light/shadow gradient of the original face (0 = off, 1 = full).')
    p.add_argument('--detail', type=float, default=1.0,
                   help='Match skin texture (pores, fine creases) of the new face to the original footage (0 = off).')
    p.add_argument('--ref_light', type=float, default=1.0,
                   help='Match the photo to the scene exposure/white balance before generating (0 = off).')
    p.add_argument('--from_work', default=None,
                   help='Re-composite only: reuse the generated heads from a --keep_temp folder.')
    p.add_argument('--no_relight', action='store_true', help='Turn the relighting LoRA off.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--keep_mouth', action='store_true',
                   help='Put the mouth interior (teeth, tongue, food, a finger) of the original video into the new '
                        'lips. Good for eating/drinking; the teeth are then those of the original person.')
    p.add_argument('--no_occlusion', dest='occlusion', action='store_false',
                   help='Do not detect occlusions in front of the face (hand, fingers, food, cup).')
    p.add_argument('--face_pass', default='off', choices=['off', 'fast', 'balanced', 'best'],
                   help='Second, face-only pass with DreamID-V: much closer to the photo, '
                        '+~8 min (balanced) per 77 frames. Needs the DreamID-V weights.')
    p.add_argument('--wan_mouth', type=float, default=0.0,
                   help='After the face pass, take lips/mouth from the Wan generation (0..1): '
                        'the mouth opens like in the original, the rest stays from the face pass.')
    p.add_argument('--wan_eyes', type=float, default=0.0,
                   help='After the face pass, take eyes/lids from the Wan generation (0..1): '
                        'gaze and blinks follow the original, costs a little similarity.')
    p.add_argument('--face_guide', type=float, default=4.0,
                   help='Identity guidance of the face pass (lower = more natural expressions).')
    p.add_argument('--candidates', type=int, default=1,
                   help='Generate this many variants (seeds) and keep the one closest to the photo (time x N).')
    p.add_argument('--ckpt_dir', default=os.path.join(MODELS, 'wan2.2-animate-14B'), help='Wan2.2-Animate-14B folder.')
    p.add_argument('--keep_temp', action='store_true', help='Keep the work folder (masks, crops, generation).')
    args = p.parse_args()
    preset = PRESETS[args.preset]
    if args.size:
        try:
            args.size = tuple(int(v) for v in args.size.lower().split('x'))
        except ValueError:
            p.error('--size expects WxH, e.g. 512x720')
    else:
        args.size = preset['size']
    args.loras = parse_loras(args.lora)
    if args.turbo:
        args.steps = args.steps or 6
        args.solver = args.solver or 'euler'
    args.solver = args.solver or 'dpm++'
    args.steps = args.steps or preset['steps']
    if len(args.size) != 2 or any(v % 16 for v in args.size):
        p.error('--size: width and height must be multiples of 16')
    return args


def settings_tag(args):
    """Short, file-name safe summary of the settings that change the result."""
    preset = PRESETS[args.preset]
    parts = [args.preset if tuple(args.size) == tuple(preset['size']) else f'{args.size[0]}x{args.size[1]}']
    default_steps = 6 if args.turbo else preset['steps']
    if args.turbo:
        parts.append('turbo')
    if args.steps != default_steps:
        parts.append(f'{args.steps}steps')
    parts += [f'lora-{os.path.splitext(os.path.basename(path))[0][:20]}' for path, _ in args.loras]
    if args.face_pass != 'off':
        parts.append(f'facepass-{args.face_pass}' + (f'-wanmouth{args.wan_mouth:g}' if args.wan_mouth > 0 else '') +
                     (f'-waneyes{args.wan_eyes:g}' if args.wan_eyes > 0 else ''))
    if args.keep_mouth:
        parts.append('keepmouth')
    if not args.occlusion:
        parts.append('no-occlusion')
    if args.candidates > 1:
        parts.append(f'{args.candidates}candidates')
    if args.seed != 42:
        parts.append(f'seed{args.seed}')
    if args.from_work:
        parts.append('recomposited')
    return '_'.join(parts)


def save_debug(work, name, frames, fps_str):
    from faceswap import write_video
    write_video(os.path.join(work, name), [np.asarray(f, np.uint8) for f in frames], fps_str, crf=18)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    from decord import VideoReader
    from wan_animate.modules.animate.preprocess.human_visualization import draw_aapose_by_meta_new
    from wan_animate.modules.animate.preprocess.pose2d_utils import AAPoseMeta
    from wan_animate.modules.animate.preprocess.utils import get_aug_mask, get_face_bboxes

    info = probe(args.video)
    vr = VideoReader(args.video)
    total = len(vr)
    start = int(round(args.start * info['fps']))
    end = total if args.seconds is None else min(total, start + int(round(args.seconds * info['fps'])))
    if start >= end:
        raise SystemExit('Empty time range.')
    W, H = info['width'], info['height']
    MW, MH = args.size

    stamp = time.strftime('%Y%m%d_%H%M%S')
    name = os.path.splitext(os.path.basename(args.video))[0]
    out_path = args.out or os.path.join(REPO, 'output', f'head_{name}_{settings_tag(args)}_{stamp}.mp4')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.join(REPO, 'output', '.work'), exist_ok=True)
    # unique even when two runs start in the same second (one would delete the other's folder)
    work = tempfile.mkdtemp(prefix=f'head_{stamp}_', dir=os.path.join(REPO, 'output', '.work'))
    t_all = time.time()

    # ---- 1. track head -------------------------------------------------------
    tools = FaceTools()
    frames = vr.get_batch(list(range(start, end))).asnumpy()
    n = len(frames)
    logging.info(f'{n} frames ({n / info["fps"]:.1f}s @ {info["fps_str"]} fps), preset {args.preset}: '
                 f'{MW}x{MH}px, {args.steps} steps ({args.solver})')
    lmks = [tools.landmarks(f) for f in frames]
    found = [p for p in lmks if p is not None]
    if not found:
        raise SystemExit('No face found in the video.')
    if len(found) < n:
        logging.info(f'Face found in {len(found)}/{n} frames (rest interpolated).')
    fs = float(np.median([face_size(p) for p in found]))
    mats, crop_h = crop_matrices(lmks, W, H, fs / args.face_ratio, args.size, info['fps'])
    logging.info(f'Head ~{fs:.0f}px tall -> tracking crop {crop_h * MW / MH:.0f}x{crop_h:.0f}px '
                 f'-> model {MW}x{MH}px ({MH / crop_h:.2f}x)')
    if MH / crop_h < 0.75:
        logging.warning('The head in the video is larger than the model resolution; preset "best" or '
                        'a larger --size gives more detail.')

    # ---- 2. crop-space inputs -------------------------------------------------
    t0 = time.time()
    ref, rpts, rmask = prepare_reference(args.ref_image, tools, args.size, args.face_ratio)
    crops = [warp(f, M, args.size) for f, M in zip(frames, mats)]
    cpts = [None if p is None else to_crop(p, M) for p, M in zip(lmks, mats)]

    room = max(3, int(MH * args.hair_room)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (room, room))
    src_cats, src_heads, gen_masks, occ_raw, grow_raw = [], [], [], [], []
    for c, p in zip(crops, cpts):
        src_cats.append(tools.categories(c))
        if args.occlusion:
            hands = tools.hands(c)
            occ_raw.append(occluder_mask(src_cats[-1], p, hands))
            grow_raw.append(grow_occluder(occ_raw[-1], src_cats[-1], p, hands))
        else:
            occ_raw.append(np.zeros((MH, MW), np.uint8))
            grow_raw.append(np.zeros((MH, MW), np.uint8))
        hm = head_mask(src_cats[-1], p) & (1 - occ_raw[-1])
        src_heads.append(hm)
        m = hm.copy()
        if p is not None:  # where the new head + hair will be
            A, _ = cv2.estimateAffinePartial2D(rpts[ALIGN_IDX], p[ALIGN_IDX])
            if A is not None:
                m |= (cv2.warpAffine(rmask, A, (MW, MH), flags=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        gen_masks.append(cv2.dilate(m, kernel))
    # Occluders (hand, food, cup ...): 3-frame vote against flicker. They are cut
    # out of the generation mask, so the model sees them as fixed picture content
    # and builds the face behind them; at paste-back they stay original on top.
    occs = [(np.mean(occ_raw[max(0, i - 1):i + 2], axis=0) > 0.5).astype(np.uint8) for i in range(n)]
    # their continuation outside the face oval (hand below the chin, cup at the side): only at paste-back
    occ_strong = [(np.mean([g == 2 for g in grow_raw[max(0, i - 1):i + 2]], axis=0) > 0.5) for i in range(n)]
    occ_weak = [(np.mean([g > 0 for g in grow_raw[max(0, i - 1):i + 2]], axis=0) > 0.5) for i in range(n)]
    del grow_raw
    occ_frames = sum(1 for o in occs if o.any())
    if occ_frames:
        logging.info(f'Occlusions in front of the face (hand, object) in {occ_frames}/{n} frames: kept original')
    # temporal smoothing of the generation mask (3 frames), then Wan's block mask
    masks = []
    for i in range(n):
        avg = np.mean(gen_masks[max(0, i - 1):i + 2], axis=0)
        m = (avg > 0.34).astype(np.uint8)
        m = get_aug_mask(m, w_len=args.mask_grid, h_len=args.mask_grid) if m.any() else m
        masks.append(m & (1 - occs[i]))
    bgs = [c * (1 - m[..., None]) for c, m in zip(crops, masks)]
    k = frontal_index(cpts)
    if cpts[k] is not None:
        ref, ref_gain = reference_in_scene_light(ref, rpts, tools, crops[k], cpts[k], src_cats[k], args.ref_light)
        if ref_gain is not None:
            logging.info(f'Photo brought to the scene light (frame {start + k}): '
                         f'RGB x{" / ".join(f"{v:.2f}" for v in ref_gain)}')
    logging.info(f'Masks done in {time.time() - t0:.0f}s')

    if args.keep_temp:
        save_debug(work, 'crop.mp4', crops, info['fps_str'])
        save_debug(work, 'mask.mp4', [np.repeat(m[..., None] * 255, 3, 2) for m in masks], info['fps_str'])
        save_debug(work, 'bg.mp4', bgs, info['fps_str'])
        Image.fromarray(ref).save(os.path.join(work, 'ref.png'))

    # ---- 3. generate (or reuse an earlier generation) -------------------------
    scorer = IdentityScorer(tools)
    ref_emb = scorer.embed(np.array(Image.open(args.ref_image).convert('RGB')))
    if args.from_work:
        gen = load_generated(args.from_work, n)
        if gen.shape[1:3] != (MH, MW):
            raise SystemExit(f'{args.from_work} was generated at {gen.shape[2]}x{gen.shape[1]}px: '
                             f'pass --size {gen.shape[2]}x{gen.shape[1]}.')
        old_crop = os.path.join(args.from_work, 'crop.mp4')
        if os.path.exists(old_crop):  # same framing? otherwise the head lands in the wrong place
            first = VideoReader(old_crop)[0].asnumpy().astype(np.float32)
            if np.abs(first - crops[0].astype(np.float32)).mean() > 6:
                raise SystemExit('The crop does not match the earlier run: use the same --start/--seconds/'
                                 '--face_ratio as then.')
        logging.info(f'Reusing generated heads from {args.from_work}')
    else:
        t0 = time.time()
        pose2d = load_pose(args)
        metas = pose2d(crops)
        conds, faces = [], []
        for c, meta in zip(crops, metas):
            conds.append(draw_aapose_by_meta_new(np.zeros_like(c), AAPoseMeta.from_humanapi_meta(meta)))
            x1, x2, y1, y2 = get_face_bboxes(meta['keypoints_face'][:, :2], scale=1.3, image_shape=(MH, MW))
            faces.append(cv2.resize(c[y1:y2, x1:x2], (512, 512)))
        del pose2d
        logging.info(f'Pose + face crops done in {time.time() - t0:.0f}s')
        if args.keep_temp:
            save_debug(work, 'pose.mp4', conds, info['fps_str'])
            save_debug(work, 'face.mp4', faces, info['fps_str'])

        if args.turbo:
            args.loras.insert(0, (turbo_lora_path(), 1.0))
        model = load_model(args)
        t_gen = time.time()
        last, cand = [time.time()], [0]

        def progress(step, steps, clip, clips):
            now = time.time()
            if step == 1 or step == steps or now - last[0] > 60:
                last[0] = now
                done = (cand[0] * clips + clip) * steps + step
                eta = (now - t_gen) / done * (args.candidates * clips * steps - done)
                which = f'variant {cand[0] + 1}/{args.candidates}, ' if args.candidates > 1 else ''
                logging.info(f'{which}window {clip + 1}/{clips}, step {step}/{steps}, '
                             f'about {eta / 60:.0f} min left')

        best = None
        for c in range(args.candidates):
            cand[0] = c
            video = model.generate(
                None, replace_flag=True, clip_len=77, refert_num=args.refert, shift=args.shift,
                sample_solver=args.solver, sampling_steps=args.steps, guide_scale=1.0, seed=args.seed + c,
                inputs=dict(cond_images=conds, face_images=faces, refer_image=ref,
                            bg_images=bgs, mask_images=[m.astype(np.float32) for m in masks]),
                progress=progress)
            g = ((video.float().clamp(-1, 1) + 1) * 127.5).round().permute(1, 2, 3, 0).to(torch.uint8).numpy()
            del video
            if args.candidates > 1:
                score = scorer.similarity(g, ref_emb) if ref_emb is not None else None
                logging.info(f'Variant {c + 1} (seed {args.seed + c}): identity vs. photo '
                             + ('?' if score is None else f'{score:.3f}'))
                score = -1.0 if score is None else score
            else:
                score = 0.0
            if best is None or score > best[0]:
                best = (score, args.seed + c, g)
        gen = best[2]
        if args.candidates > 1:
            logging.info(f'Keeping the variant with seed {best[1]}')
        del model, best
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        logging.info(f'Generation done in {(time.time() - t_gen) / 60:.1f} min')
        if args.keep_temp:  # exact frames, so --from_work can re-composite without loss
            np.save(os.path.join(work, 'generated.npy'), gen[:n])
            save_debug(work, 'generated.mp4', gen, info['fps_str'])
            with open(os.path.join(work, 'generation.json'), 'w') as f:  # recorded in later --from_work sidecars
                json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}, f,
                          indent=1, ensure_ascii=False, default=str)

    if args.face_pass != 'off':
        # a --keep_temp run also keeps the face-pass result, so --from_work does not redo those minutes
        fp_name = f'generated_facepass_{args.face_pass}_g{args.face_guide:g}_s{args.seed}.npy'
        fp_cached = os.path.join(args.from_work or work, fp_name)  # next to the generation it belongs to
        wan = gen
        if args.from_work and os.path.exists(fp_cached) and np.load(fp_cached, mmap_mode='r').shape == gen[:n].shape:
            gen = np.load(fp_cached)
            logging.info(f'Reusing the face pass from {args.from_work}')
        else:
            if ref_emb is not None:
                score = scorer.similarity(gen, ref_emb)
                logging.info('Identity before the face pass: ' + ('?' if score is None else f'{score:.3f}'))
            gen = face_pass(gen, args, work, info['fps_str'])
            if args.keep_temp or args.from_work:
                np.save(fp_cached, gen[:n])
        gen = merge_face_pass(wan[:n], gen[:n], tools, args.wan_mouth, args.wan_eyes)
        logging.info('Face pass applied inside the face oval' +
                     (f', mouth from Wan ({args.wan_mouth:g})' if args.wan_mouth > 0 else '') +
                     (f', eyes from Wan ({args.wan_eyes:g})' if args.wan_eyes > 0 else ''))
        del wan
    id_score = None
    if ref_emb is not None:
        score = id_score = scorer.similarity(gen, ref_emb)
        if score is not None:
            logging.info(f'Identity vs. photo (AuraFace): {score:.3f}  '
                         '(other person ~0.0-0.2, same person ~0.9)')

    # ---- 4. scene light + paste back ------------------------------------------
    t0 = time.time()
    sigma = max(1.0, MH * 0.012)
    alphas, gen_cats, gains, shapes, gen_skin, src_skin, gen_pts = [], [], [], [], [], [], []
    for i in range(n):
        gc = tools.categories(gen[i])
        gp = tools.landmarks(gen[i])
        gp = cpts[i] if gp is None else gp
        gen_cats.append(gc)
        gen_skin.append(skin_weight(gc, gp).astype(np.uint8))
        gen_pts.append(gp)
        src_skin.append(skin_weight(src_cats[i], cpts[i]).astype(np.uint8))
        alphas.append(head_alpha(src_cats[i], gc, src_heads[i], head_mask(gc, gp), masks[i], room // 2, sigma))
        g = s = None
        if gp is not None and cpts[i] is not None:
            g, s = light_terms(gen[i], crops[i], gc, src_cats[i], gp, cpts[i])
        gains.append(g)
        shapes.append(s)
    # exposure/white balance: smooth over time (no pumping); shading: 3-frame mean
    if any(g is not None for g in gains):
        gains = np.stack([smooth_track([None if g is None else float(g[c]) for g in gains], 2.0)
                          for c in range(3)], axis=1).astype(np.float32)
        logging.info(f'Light matched: head exposure x{gains.mean():.2f} '
                     f'(RGB {" / ".join(f"{v:.2f}" for v in gains.mean(0))})')
    else:
        gains = np.ones((n, 3), np.float32)
    gains = gains ** args.light

    silent = os.path.join(work, 'video_noaudio.mp4')
    writer = FrameWriter(silent, W, H, info['fps_str'])
    rng = np.random.default_rng(args.seed)
    band_gains, band_log, mouth_frames = None, [], 0
    for i in range(n):
        a = np.mean(alphas[max(0, i - 1):i + 2], axis=0)  # temporal smoothing
        if occs[i].any() or occ_weak[i].any():  # hand/food in front of the face: original, crisp edge
            # outside the oval plain skin (maybe the neck) gives way to the new head's own chin/hair
            new_head = np.isin(gen_cats[i], (FACE_SKIN, HAIR))
            keep = (occs[i] > 0) | occ_strong[i] | (occ_weak[i] & ~new_head)
            a = a * (1 - cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.5))
        near = [s.astype(np.float32) for s in shapes[max(0, i - 1):i + 2] if s is not None]
        shade = np.mean(near, axis=0) ** args.shade if near else np.ones((MH, MW), np.float32)
        g = apply_light(gen[i], gains[i], shade)
        if args.keep_mouth and gen_pts[i] is not None and cpts[i] is not None:
            mouth = original_mouth(g, crops[i], gen_pts[i], cpts[i])
            if mouth is not None:
                g = g * (1 - mouth[1][..., None]) + mouth[0] * mouth[1][..., None]
                mouth_frames += 1
        g = np.clip(recolor_clothes(g, gen_cats[i], crops[i], src_cats[i]), 0, 255)
        M = mats[i]
        a_full = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)[..., None]
        g_full = cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REFLECT101)
        orig = frames[i].astype(np.float32)
        ys, xs = np.where(a_full[..., 0] > 0.01)
        if len(ys):
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            box = (slice(y0, y1), slice(x0, x1))
            if args.detail > 0:
                to_full = lambda m: cv2.warpAffine(m.astype(np.float32), M, (W, H),  # noqa: E731
                                                   flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)[box]
                g_full[box], band_gains = match_detail(g_full[box], orig[box], to_full(gen_skin[i]),
                                                       to_full(src_skin[i]), band_gains, args.detail)
                if band_gains is not None:
                    band_log.append(band_gains)
            g_full[box] = match_grain(orig[box], g_full[box], a_full[box][..., 0], args.grain, rng)
        writer.write(np.clip(orig * (1 - a_full) + g_full * a_full + 0.5, 0, 255).astype(np.uint8))
    writer.close()
    if args.keep_mouth:
        logging.info(f'Original mouth interior in {mouth_frames}/{n} frames (open mouth)')
    if band_log:
        logging.info('Skin texture matched, gain per band (fine -> coarse): '
                     + ' / '.join(f'{v:.2f}' for v in np.mean(band_log, axis=0)))
    logging.info(f'Light + paste-back done in {time.time() - t0:.0f}s')

    if info['has_audio']:
        mux_audio(silent, args.video, start / info['fps'], n / info['fps'], out_path)
    else:
        shutil.move(silent, out_path)
    if not args.keep_temp:
        shutil.rmtree(work, ignore_errors=True)
    else:
        logging.info(f'Work folder: {work}')
    write_sidecar(out_path, args, info, start, n, id_score, (time.time() - t_all) / 60)
    logging.info(f'Done in {(time.time() - t_all) / 60:.1f} min: {out_path} (+ settings in .json)')
    print(f'Save file: {out_path}')


if __name__ == '__main__':
    main()
