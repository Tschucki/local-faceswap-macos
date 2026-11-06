"""Side-by-side comparison video: original | result, cropped to the head, labelled,
with the original audio. Works for head swaps and face swaps.

    python compare.py --original target.mp4 --result output/head_x.mp4 --start 2.5 --label "Turbo"

Writes <result>__compare.mp4 unless --out is given."""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

FONTS = ['/System/Library/Fonts/SFNS.ttf', '/System/Library/Fonts/Helvetica.ttc', '/Library/Fonts/Arial.ttf']
PANEL_W, PANEL_H, BAR_H = 640, 800, 44


def _font(size=26):
    for f in FONTS:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def _label(text, w, font):
    bar = Image.new('RGB', (w, BAR_H), (20, 20, 20))
    ImageDraw.Draw(bar).text((12, 7), text, font=font, fill=(240, 240, 240))
    return np.array(bar)


def _head_box(points, shape):
    """Portrait (4:5) box around the head, the same for the whole clip."""
    from headswap import face_size
    pts = np.concatenate(points)
    fs = max(face_size(p) for p in points)
    cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1]) - 0.1 * fs
    h = min(shape[0], 2.4 * fs)
    w = min(shape[1], h * PANEL_W / PANEL_H)
    x0 = int(np.clip(cx - w / 2, 0, shape[1] - w))
    y0 = int(np.clip(cy - h / 2, 0, shape[0] - h))
    return x0, y0, int(w), int(h)


def make_comparison(original, result, start=0.0, out=None, label='Result', tools=None):
    """Returns the written path, or None when no face was found in the original."""
    from decord import VideoReader

    from faceswap import mux_audio, probe, write_video
    from headswap import FaceTools
    tools = tools or FaceTools()
    out = out or os.path.splitext(result)[0] + '__compare.mp4'
    rv = VideoReader(result)
    sv = VideoReader(original)
    fps = sv.get_avg_fps()
    s0 = int(round(start * fps))
    n = min(len(rv), len(sv) - s0)
    src = sv.get_batch(list(range(s0, s0 + n))).asnumpy()
    res = rv.get_batch(list(range(n))).asnumpy()
    if res.shape[1:3] != src.shape[1:3]:
        res = np.stack([cv2.resize(r, (src.shape[2], src.shape[1]), interpolation=cv2.INTER_AREA) for r in res])
    points = [p for p in (tools.landmarks(src[i]) for i in range(0, n, 4)) if p is not None]
    if not points:
        return None
    x0, y0, w, h = _head_box(points, src.shape[1:3])
    font = _font()
    left, right = _label('Original', PANEL_W, font), _label(label, PANEL_W, font)
    frames = []
    for s, r in zip(src, res):
        a = cv2.resize(s[y0:y0 + h, x0:x0 + w], (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
        b = cv2.resize(r[y0:y0 + h, x0:x0 + w], (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
        frames.append(np.hstack([np.vstack([left, a]), np.vstack([right, b])]))
    tmp = out + '.tmp.mp4'
    write_video(tmp, frames, probe(result)['fps_str'], crf=16)
    if probe(original)['has_audio']:
        mux_audio(tmp, original, s0 / fps, n / fps, out)
        os.remove(tmp)
    else:
        os.replace(tmp, out)
    return out


def main():
    p = argparse.ArgumentParser(description='Comparison video original | result (head crop).')
    p.add_argument('--original', required=True, help='Target video (the original).')
    p.add_argument('--result', required=True, help='Result video.')
    p.add_argument('--start', type=float, default=0.0, help='Second in the original where the result starts.')
    p.add_argument('--label', default='Result', help='Label of the right side.')
    p.add_argument('--out', default=None, help='Output (default: <result>__compare.mp4).')
    args = p.parse_args()
    path = make_comparison(args.original, args.result, args.start, args.out, args.label)
    if path is None:
        raise SystemExit('No face found in the original.')
    print(f'Comparison: {path}')


if __name__ == '__main__':
    main()
