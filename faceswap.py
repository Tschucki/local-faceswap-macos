#!/usr/bin/env python3
"""Face-only swap with DreamID-V on Apple Silicon: face crop -> swap -> paste back.

Running the model on the whole frame would shrink it to the model resolution
(1080x1920 -> 464x832): the face ends up ~100 px wide and the entire video is
re-rendered soft, the typical "plastic" look.

This script instead
  1. finds the face in every frame and cuts one square crop per chunk around it,
  2. runs only that crop through DreamID-V, at (or above) the face's native size,
  3. blends the swapped face back into the untouched full-resolution frames
     with a feathered mask and matched film grain,
  4. processes long videos in chunks with a short crossfade, keeps the source
     frame rate and copies the original audio.

Models are loaded once for all chunks. The DiT runs in bf16 (autocast), the
T5 prompt embedding is cached on disk, the solver is Euler (UniPC overexposes
on MPS).
"""
import argparse
import logging
import math
import os
import shutil
import subprocess
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

PROMPT = 'chang face'  # the prompt DreamID-V was trained with (sic)
CHUNK_FRAMES = 81      # model window, 4n+1

PRESETS = {
    # model_side: crop resolution fed to the DiT; steps: Euler steps
    'fast':     dict(model_side=448, steps=20),
    'balanced': dict(model_side=512, steps=25),
    'best':     dict(model_side=640, steps=30),
}
DTYPES = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}


# --------------------------------------------------------------------------- #
# Video I/O
# --------------------------------------------------------------------------- #
def probe(path):
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
         '-of', 'default=noprint_wrappers=1', path],
        capture_output=True, text=True, check=True).stdout
    info = dict(line.split('=', 1) for line in out.strip().splitlines())
    num, den = info['r_frame_rate'].split('/')
    has_audio = bool(subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries',
         'stream=index', '-of', 'csv=p=0', path],
        capture_output=True, text=True).stdout.strip())
    return dict(width=int(info['width']), height=int(info['height']),
                fps_str=info['r_frame_rate'], fps=float(num) / float(den),
                has_audio=has_audio)


class FrameWriter:
    """Pipe RGB frames into ffmpeg (libx264)."""

    def __init__(self, path, width, height, fps_str, crf=16):
        self.proc = subprocess.Popen(
            ['ffmpeg', '-y', '-loglevel', 'error',
             '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
             '-r', fps_str, '-i', '-',
             '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
             '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf),
             '-pix_fmt', 'yuv420p', path],
            stdin=subprocess.PIPE)

    def write(self, frame_rgb):
        self.proc.stdin.write(np.ascontiguousarray(frame_rgb, dtype=np.uint8).tobytes())

    def close(self):
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError('ffmpeg encode failed')


def write_video(path, frames, fps_str, crf=10):
    h, w = frames[0].shape[:2]
    writer = FrameWriter(path, w, h, fps_str, crf=crf)
    for f in frames:
        writer.write(f)
    writer.close()


def mux_audio(video_path, source_path, start_s, dur_s, out_path):
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', video_path,
           '-ss', f'{start_s:.3f}', '-t', f'{dur_s:.3f}', '-i', source_path,
           '-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'copy',
           '-c:a', 'aac', '-b:a', '192k', out_path]  # no -shortest: never cut video frames
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- #
# Face geometry
# --------------------------------------------------------------------------- #
_extractor = None


def face_box(frame_rgb):
    """Landmark bounding box (x0, y0, x1, y1) in pixels, or None."""
    global _extractor
    if _extractor is None:
        from express_adaption.media_pipe.mp_utils import LMKExtractor
        _extractor = LMKExtractor()
    h, w = frame_rgb.shape[:2]
    res = _extractor(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    if res is None:
        return None
    pts = res['lmks'][:, :2] * np.array([w, h])
    return (*pts.min(axis=0), *pts.max(axis=0))


def crop_window(boxes, width, height, face_ratio):
    """One static square crop for the whole chunk: (x0, y0, side)."""
    b = np.array([bx for bx in boxes if bx is not None], dtype=np.float64)
    face = np.median(np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]))
    ux0, uy0 = b[:, 0].min(), b[:, 1].min()
    ux1, uy1 = b[:, 2].max(), b[:, 3].max()
    # face_ratio = face size / crop side; also keep the whole motion range in.
    side = max(face / face_ratio, (ux1 - ux0) * 1.35, (uy1 - uy0) * 1.35)
    side = int(min(side, width, height))
    cx = (ux0 + ux1) / 2
    cy = (uy0 + uy1) / 2 - 0.06 * face  # a little more hair, a little less neck
    x0 = int(round(min(max(0, cx - side / 2), width - side)))
    y0 = int(round(min(max(0, cy - side / 2), height - side)))
    return x0, y0, side


def prepare_ref_image(path, out_path, side):
    """Crop a loose face portrait out of a large photo; keep tight crops as they are."""
    img = np.array(Image.open(path).convert('RGB'))
    h, w = img.shape[:2]
    box = face_box(img)
    if box is None:
        raise SystemExit(f'No face found in the photo: {path}')
    face = max(box[2] - box[0], box[3] - box[1])
    if face / min(w, h) < 0.35:
        crop = int(face / 0.5)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 - 0.06 * face
        x0, y0 = int(cx - crop / 2), int(cy - crop / 2)
        canvas = np.full((crop, crop, 3), 255, np.uint8)  # model pads with white too
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(w, x0 + crop), min(h, y0 + crop)
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
        img = canvas
        logging.info(f'Photo cropped to the face ({crop}px crop).')
    else:
        # Already a face crop: pad to square instead of cutting the face.
        s = max(h, w)
        canvas = np.full((s, s, 3), 255, np.uint8)
        canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
        img = canvas
    if face < 160:
        logging.warning(f'The face in the photo is only ~{int(face)}px wide. '
                        'A sharper, larger photo improves the identity a lot.')
    Image.fromarray(img).resize((side, side), Image.Resampling.LANCZOS).save(out_path, quality=97)
    return out_path


# --------------------------------------------------------------------------- #
# Compositing
# --------------------------------------------------------------------------- #
def resize(img, size):
    interp = cv2.INTER_AREA if size < img.shape[0] else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


def build_alphas(mask_frames, side, n):
    """Binary face-oval masks -> dilated, feathered, temporally smoothed alpha."""
    k = max(3, int(side * 0.06)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    sigma = max(1.0, side * 0.035)
    alphas = []
    for m in mask_frames[:n]:
        m = (m[..., 0] > 127).astype(np.uint8)
        m = cv2.resize(m, (side, side), interpolation=cv2.INTER_NEAREST)
        m = cv2.dilate(m, kernel)
        a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma)
        alphas.append(a)
    smoothed = []
    for i in range(len(alphas)):
        win = alphas[max(0, i - 1):i + 2]
        smoothed.append(np.mean(win, axis=0))
    return smoothed


def match_grain(orig, gen, alpha, strength, rng):
    """Add luma noise so the swapped face has the same grain as the footage."""
    if strength <= 0:
        return gen
    m = alpha > 0.5
    if m.sum() < 100:
        return gen

    def hf_std(img):
        y = img.mean(axis=2)
        return float((y - cv2.GaussianBlur(y, (0, 0), 1.0))[m].std())

    missing = math.sqrt(max(0.0, hf_std(orig) ** 2 - hf_std(gen) ** 2)) * strength
    if missing < 0.2:
        return gen
    noise = rng.normal(0.0, missing, size=gen.shape[:2]).astype(np.float32)
    return gen + noise[..., None]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_model(args):
    import dreamidv_wan
    from dreamidv_wan.configs import WAN_CONFIGS
    cfg = WAN_CONFIGS['swapface']
    if torch.backends.mps.is_available():
        cfg.t5_dtype = torch.float32
        cfg.param_dtype = torch.float32
    t0 = time.time()
    model = dreamidv_wan.DreamIDV(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        dreamidv_ckpt=args.dreamidv_ckpt,
        device_id=0,
        rank=0,
        compute_dtype=DTYPES[args.dtype],
        t5_cache_dir=os.path.join(REPO, '.cache', 't5'),
    )
    logging.info(f'Models loaded in {time.time() - t0:.0f}s')
    return model


def swap_chunk(model, src_frames, ref_path, work, idx, args):
    """Swap one chunk. Returns composited full-res frames (same count as input)."""
    from dreamidv_prep import generate_pose_and_mask_videos
    from dreamidv_wan.configs import SIZE_CONFIGS
    from decord import VideoReader

    n_real = len(src_frames)
    h, w = src_frames[0].shape[:2]
    boxes = [face_box(f) for f in src_frames]
    found = sum(b is not None for b in boxes)
    if found == 0:
        logging.warning(f'Chunk {idx}: no face found, keeping the original frames.')
        return list(src_frames)
    if found < n_real:
        logging.info(f'Chunk {idx}: face found in {found}/{n_real} frames.')

    x0, y0, side = crop_window(boxes, w, h, args.face_ratio)
    ms = args.model_side
    logging.info(f'Chunk {idx}: crop {side}x{side}px at ({x0},{y0}) -> model {ms}x{ms}')

    # Pad to 4n+1 frames by repeating the last frame; trimmed again afterwards.
    n_model = max(5, ((n_real - 1 + 3) // 4) * 4 + 1)
    crops = [resize(f[y0:y0 + side, x0:x0 + side], ms) for f in src_frames]
    crops += [crops[-1]] * (n_model - n_real)

    chunk_dir = os.path.join(work, f'chunk_{idx:03d}')
    os.makedirs(chunk_dir, exist_ok=True)
    crop_path = os.path.join(chunk_dir, 'crop.mp4')
    write_video(crop_path, crops, args.fps_str)

    pose_path, mask_path = generate_pose_and_mask_videos(
        crop_path, ref_path, max_frames=n_model, out_dir=chunk_dir)

    video = model.generate(
        PROMPT,
        [crop_path, mask_path, ref_path, pose_path],
        size=SIZE_CONFIGS['1280*720'],  # area cap only (960^2); the crop sets the size
        frame_num=n_model,
        shift=args.shift,
        sample_solver='euler',
        sampling_steps=args.steps,
        guide_scale_img=args.guide,
        seed=args.seed,
        offload_model=False)

    gen = ((video.clamp(-1, 1) + 1) * 127.5).permute(1, 2, 3, 0).to(torch.uint8).cpu().numpy()
    if gen.shape[1] != ms or gen.shape[2] != ms:
        logging.warning(f'Unexpected model output size {gen.shape[1:3]}, resizing.')

    mvr = VideoReader(mask_path)
    mask_frames = mvr.get_batch(list(range(min(len(mvr), n_real)))).asnumpy()
    alphas = build_alphas(mask_frames, side, n_real)

    rng = np.random.default_rng(args.seed + idx)
    out = []
    for i in range(n_real):
        frame = src_frames[i].copy()
        a = alphas[min(i, len(alphas) - 1)][..., None]
        orig = frame[y0:y0 + side, x0:x0 + side].astype(np.float32)
        g = cv2.resize(gen[i], (side, side), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        g = match_grain(orig, g, a[..., 0], args.grain, rng)
        frame[y0:y0 + side, x0:x0 + side] = np.clip(orig * (1 - a) + g * a, 0, 255).astype(np.uint8)
        out.append(frame)
    return out


def write_sidecar(out_path, args, info, start, n, id_score, minutes):
    """<video>.json next to the result: every setting + identity score, so runs stay traceable."""
    import json
    data = {
        'result': os.path.basename(out_path),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'photo': os.path.abspath(args.ref_image),
        'video': os.path.abspath(args.video),
        'range_s': [round(start / info['fps'], 3), round((start + n) / info['fps'], 3)],
        'frames': n,
        'identity_auraface': None if id_score is None else round(float(id_score), 3),
        'minutes': round(minutes, 1),
        'settings': {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()},
    }
    gen_file = os.path.join(getattr(args, 'from_work', None) or '', 'generation.json')
    if getattr(args, 'from_work', None) and os.path.exists(gen_file):  # re-composite: how it was generated
        with open(gen_file) as g:
            data['generation'] = json.load(g)
    with open(os.path.splitext(out_path)[0] + '.json', 'w') as f:
        json.dump(data, f, indent=1, ensure_ascii=False, default=str)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description='Face-only swap with DreamID-V (crop, swap, paste back).')
    p.add_argument('--ref_image', required=True, help='Photo of the new face.')
    p.add_argument('--video', required=True, help='Target video.')
    p.add_argument('--out', default=None, help='Output .mp4 (default: output/face_<name>_<settings>_<time>.mp4).')
    p.add_argument('--preset', default='balanced', choices=list(PRESETS))
    p.add_argument('--start', type=float, default=0.0, help='Start time in seconds.')
    p.add_argument('--seconds', type=float, default=None, help='Length in seconds (default: until the end).')
    p.add_argument('--model_side', type=int, default=None, help='Overrides the preset: crop resolution (multiple of 16).')
    p.add_argument('--steps', type=int, default=None, help='Overrides the preset: Euler steps.')
    p.add_argument('--guide', type=float, default=4.0, help='Identity guidance (CFG). Lower = more natural, higher = closer to the photo.')
    p.add_argument('--shift', type=float, default=5.0)
    p.add_argument('--face_ratio', type=float, default=0.45, help='Face size relative to the crop.')
    p.add_argument('--grain', type=float, default=1.0, help='Film grain matching (0 = off).')
    p.add_argument('--overlap', type=int, default=4, help='Crossfade frames between chunks.')
    p.add_argument('--chunk', type=int, default=CHUNK_FRAMES, help='Frames per chunk (4n+1, max 81).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dtype', default='bf16', choices=list(DTYPES))
    p.add_argument('--ckpt_dir', default=os.path.join(MODELS, 'wan2.1-1.3B'), help='Wan2.1-T2V-1.3B folder.')
    p.add_argument('--dreamidv_ckpt', default=os.path.join(MODELS, 'dreamidv.pth'), help='DreamID-V weights.')
    p.add_argument('--keep_temp', action='store_true', help='Keep the work folder (crops, masks).')
    args = p.parse_args()
    preset = PRESETS[args.preset]
    args.model_side = args.model_side or preset['model_side']
    args.steps = args.steps or preset['steps']
    if args.model_side % 16:
        p.error('--model_side must be a multiple of 16')
    if args.chunk % 4 != 1 or not 5 <= args.chunk <= CHUNK_FRAMES:
        p.error('--chunk must be 4n+1 (5..81)')
    if not 0 <= args.overlap < args.chunk // 2:
        p.error('invalid --overlap')
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    from decord import VideoReader

    info = probe(args.video)
    args.fps_str = info['fps_str']
    vr = VideoReader(args.video)
    total = len(vr)
    start = int(round(args.start * info['fps']))
    end = total if args.seconds is None else min(total, start + int(round(args.seconds * info['fps'])))
    if start >= end:
        raise SystemExit('Empty time range.')

    stamp = time.strftime('%Y%m%d_%H%M%S')
    name = os.path.splitext(os.path.basename(args.video))[0]
    tag = '_'.join([args.preset] + ([f'guide{args.guide:g}'] if args.guide != 4.0 else [])
                   + ([f'seed{args.seed}'] if args.seed != 42 else []))
    out_path = args.out or os.path.join(REPO, 'output', f'face_{name}_{tag}_{stamp}.mp4')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.join(REPO, 'output', '.work'), exist_ok=True)
    # unique even when two runs start in the same second (one would delete the other's folder)
    work = tempfile.mkdtemp(prefix=f'{stamp}_', dir=os.path.join(REPO, 'output', '.work'))

    step = args.chunk - args.overlap
    starts = list(range(start, end, step))
    if len(starts) > 1 and end - starts[-1] <= args.overlap:
        starts.pop()  # last chunk would only contain overlap frames
    logging.info(f'{end - start} frames ({(end - start) / info["fps"]:.1f}s @ {info["fps_str"]} fps) '
                 f'in {len(starts)} chunk(s), preset {args.preset}: '
                 f'{args.model_side}px, {args.steps} steps, {args.dtype}')

    ref_path = prepare_ref_image(args.ref_image, os.path.join(work, 'ref.jpg'), args.model_side)
    model = load_model(args)

    silent_path = os.path.join(work, 'video_noaudio.mp4')
    writer = FrameWriter(silent_path, info['width'], info['height'], info['fps_str'])
    tail = []  # last `overlap` frames of the previous chunk, not yet written
    t_start = time.time()
    for ci, s in enumerate(starts):
        e = min(end, s + args.chunk)
        t0 = time.time()
        src = list(vr.get_batch(list(range(s, e))).asnumpy())
        frames = swap_chunk(model, src, ref_path, work, ci, args)

        ov = len(tail)
        for k in range(ov):  # linear crossfade over the overlap
            wgt = (k + 1) / (ov + 1)
            mixed = tail[k].astype(np.float32) * (1 - wgt) + frames[k].astype(np.float32) * wgt
            writer.write(np.clip(mixed + 0.5, 0, 255).astype(np.uint8))
        last = ci == len(starts) - 1
        keep = 0 if last else min(args.overlap, len(frames) - ov)
        for f in frames[ov:len(frames) - keep]:
            writer.write(f)
        tail = frames[len(frames) - keep:] if keep else []
        del src, frames
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        took = time.time() - t0
        left = (len(starts) - ci - 1) * took
        logging.info(f'Chunk {ci + 1}/{len(starts)} done in {took / 60:.1f} min. '
                     f'About {left / 60:.0f} min left.')
    writer.close()

    if info['has_audio']:
        mux_audio(silent_path, args.video, start / info['fps'], (end - start) / info['fps'], out_path)
    else:
        shutil.move(silent_path, out_path)
    if not args.keep_temp:
        shutil.rmtree(work, ignore_errors=True)
    write_sidecar(out_path, args, info, start, end - start, None, (time.time() - t_start) / 60)
    logging.info(f'Done in {(time.time() - t_start) / 60:.1f} min: {out_path} (+ settings in .json)')
    print(f'Save file: {out_path}')


if __name__ == '__main__':
    main()
