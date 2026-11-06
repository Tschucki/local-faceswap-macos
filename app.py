"""Local FaceSwap macOS: terminal app for head and face swaps on Apple Silicon.

    python app.py            (or double-click "Local FaceSwap.command")

Arrow keys + Enter in every menu, Tab completes paths, files can be dragged into
the terminal. The app checks photo and video first, estimates the run time,
shows live progress, writes a comparison video and keeps a history of all runs
(with their settings and identity score). The heavy lifting is done by
headswap.py (head + hair, Wan2.2-Animate-14B) and faceswap.py (face only,
DreamID-V); the app only builds their command line."""
import json
import math
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

import questionary
from questionary import Choice, Separator, Style
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

APP_NAME = 'Local FaceSwap macOS'
REPO = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(REPO, 'models')
LORAS = os.path.join(MODELS, 'loras')
OUTPUT = os.path.join(REPO, 'output')
LOGS = os.path.join(OUTPUT, 'logs')
SETTINGS_PATH = os.path.join(REPO, '.cache', 'app_settings.json')
IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
VIDEO_EXT = ('.mp4', '.mov', '.m4v', '.mkv', '.avi', '.webm')

console = Console(highlight=False)
STYLE = Style([
    ('qmark', 'fg:#e5a50a bold'),
    ('question', 'bold'),
    ('answer', 'fg:#62a0ea bold'),
    ('pointer', 'fg:#e5a50a bold'),
    ('highlighted', 'fg:#e5a50a bold'),
    ('selected', 'fg:#62a0ea'),
    ('instruction', 'fg:#888888 italic'),
    ('text', ''),
    ('disabled', 'fg:#777777 italic'),
])

# Measured on an M5 Max (128 GB), 77 frames = 3.2 s at 24 fps per window/chunk.
HEAD_PROFILES = {
    'fast': dict(title='Fast', args=['--preset', 'fast', '--turbo'], gen=4.5, face=0,
                 desc='448×640, turbo (6 steps). For trying out photo and clip range.'),
    'standard': dict(title='Standard', args=['--preset', 'best', '--turbo'], gen=9.5, face=0,
                     desc='576×816, turbo. Good expressions and hair, identity ~0.6.'),
    'identity': dict(title='Maximum likeness (recommended)',
                     args=['--preset', 'best', '--turbo', '--face_pass', 'balanced'], gen=9.5, face=8.0,
                     desc='Standard + DreamID-V face pass: identity 0.61 → 0.79 '
                          '(eyes and mouth follow the original), 0.81 without that option.'),
    'best3': dict(title='Best of 3 variants', args=['--preset', 'best', '--turbo', '--face_pass', 'balanced',
                                                     '--candidates', '3'], gen=28.5, face=8.0,
                  desc='Generates 3 variants (seeds), keeps the one closest to the photo, then the face pass.'),
    'classic': dict(title='Classic without turbo', args=['--preset', 'best'], gen=36.0, face=0,
                    desc='25 steps instead of 6: ~4x slower and less similar in tests. For comparison only.'),
}
FACE_PROFILES = {
    'fast': dict(title='Fast', args=['--preset', 'fast'], chunk=4.8, desc='448 px, 20 steps.'),
    'balanced': dict(title='Balanced (recommended)', args=['--preset', 'balanced'], chunk=7.9, desc='512 px, 25 steps.'),
    'best': dict(title='Best quality', args=['--preset', 'best'], chunk=15.0, desc='640 px, 30 steps.'),
}
MODEL_FILES = [
    ('Wan2.2-Animate-14B (head + hair)', 'wan2.2-animate-14B', 'head'),
    ('Wan2.1-T2V-1.3B (base of DreamID-V)', 'wan2.1-1.3B', 'face'),
    ('DreamID-V weights', 'dreamidv.pth', 'face'),
    ('lightx2v turbo LoRA (downloads itself otherwise, 0.7 GB)',
     'loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors', 'turbo'),
]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def load_settings():
    try:
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(s):
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, 'w') as f:
        json.dump(s, f, indent=1, ensure_ascii=False)


def clean_path(p):
    """Paths dragged into the terminal come quoted or with escaped spaces."""
    p = (p or '').strip()
    if len(p) > 1 and p[0] == p[-1] and p[0] in '\'"':
        p = p[1:-1]
    p = p.replace('\\ ', ' ').replace('\\(', '(').replace('\\)', ')').replace("\\'", "'")
    return os.path.abspath(os.path.expanduser(p)) if p else ''


def mmss(minutes):
    if minutes is None:
        return '?'
    m = max(0, int(round(minutes)))
    return f'{m // 60} h {m % 60:02d} min' if m >= 60 else f'{m} min'


def secs(s):
    return f'{s:.1f} s'


def number(text):
    """Accepts 2.5 as well as 2,5."""
    return float(text.replace(',', '.'))


def select(message, choices, **kw):
    kw.setdefault('style', STYLE)
    kw.setdefault('instruction', '(↑/↓ to choose, Enter to confirm)')
    return questionary.select(message, choices=choices, **kw)


class quiet:
    """Silences native stderr (MediaPipe/TensorFlow C++ warnings) while analysing."""

    def __enter__(self):
        sys.stderr.flush()
        self.saved = os.dup(2)
        self.null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.null, 2)

    def __exit__(self, *exc):
        sys.stderr.flush()
        os.dup2(self.saved, 2)
        os.close(self.saved)
        os.close(self.null)


def nice_path(p):
    p = os.path.abspath(p)
    if p.startswith(REPO + os.sep):
        return os.path.relpath(p, REPO)
    home = os.path.expanduser('~')
    return '~' + p[len(home):] if p.startswith(home) else p


def yes_no(message, default=False):
    yes, no = Choice('Yes', value=True), Choice('No', value=False)
    return select(message, [yes, no], default=yes if default else no)


def ask(q):
    """Ask a questionary question; Ctrl+C / Esc returns None instead of crashing."""
    try:
        return q.unsafe_ask()
    except KeyboardInterrupt:
        return None


def press_key():
    ask(questionary.press_any_key_to_continue('Press any key …', style=STYLE))


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                          'stream=codec_type,width,height,r_frame_rate:format=duration', '-of', 'json', path],
                         capture_output=True, text=True)
    try:
        d = json.loads(out.stdout)
    except ValueError:
        return None
    v = next((s for s in d.get('streams', []) if s.get('codec_type') == 'video'), None)
    if v is None:
        return None
    num, den = (v.get('r_frame_rate') or '0/1').split('/')
    fps = float(num) / max(float(den), 1)
    return dict(width=v['width'], height=v['height'], fps=fps,
                duration=float(d.get('format', {}).get('duration', 0) or 0),
                audio=any(s.get('codec_type') == 'audio' for s in d['streams']))


_tools = None


def face_tools():
    global _tools
    if _tools is None:
        sys.path.insert(0, REPO)
        from headswap import FaceTools  # noqa: E402  (imports torch: only when needed)
        _tools = FaceTools()
    return _tools


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_photo(path):
    """Face found? How large? Hair cut off at the top? Returns (ok, lines)."""
    import numpy as np
    from PIL import Image, ImageOps
    from headswap import face_size
    img = np.array(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))
    pts = face_tools().landmarks(img)
    lines = [f'Size {img.shape[1]}×{img.shape[0]} px']
    if pts is None:
        return False, lines + ['[red]No face found[/] – please choose a photo with a clearly visible face.']
    fs = face_size(pts)
    top = pts[:, 1].min()
    lines.append(f'Face found, about {fs:.0f} px tall')
    if fs < 300:
        lines.append(f'[yellow]Face rather small ({fs:.0f} px)[/] – from ~300 px on it gets much more detailed.')
    if top < 0.35 * fs:
        lines.append('[yellow]Little room above the head[/] – hair may be cut off (matters for head swaps).')
    yaw = abs(pts[1, 0] - (pts[33, 0] + pts[263, 0]) / 2) / max(abs(pts[263, 0] - pts[33, 0]), 1)
    if yaw > 0.15:
        lines.append('[yellow]Head turned[/] – a frontal photo works best.')
    return True, lines


def check_video(path, info, start=0.0, length=None):
    """Samples a few frames: face found, face size. Returns (ok, lines)."""
    from decord import VideoReader
    import numpy as np
    from headswap import face_size
    vr = VideoReader(path)
    end = info['duration'] if length is None else min(info['duration'], start + length)
    ts = np.linspace(start, max(start, end - 0.05), 6)
    found, sizes = 0, []
    for t in ts:
        f = vr[min(len(vr) - 1, int(t * info['fps']))].asnumpy()
        p = face_tools().landmarks(f)
        if p is not None:
            found += 1
            sizes.append(face_size(p))
    lines = [f"{info['width']}×{info['height']} px, {info['fps']:.2f} fps, {secs(info['duration'])}, "
             + ('with audio' if info['audio'] else 'no audio')]
    if not found:
        lines.append('[red]No face found in the sampled frames.[/]')
    else:
        lines.append(f'Face in {found}/{len(ts)} samples, about {np.median(sizes):.0f} px tall')
        if found < len(ts):
            lines.append('[yellow]Face not visible everywhere[/] – those frames are skipped or interpolated.')
    return found > 0, lines


# --------------------------------------------------------------------------- #
# job description -> command line + time estimate
# --------------------------------------------------------------------------- #
def frames_of(job, info):
    length = job['length'] if job['length'] is not None else info['duration'] - job['start']
    return max(1, int(round(length * info['fps'])))


def estimate(job, info):
    n = frames_of(job, info)
    if job['mode'] == 'head':
        prof = HEAD_PROFILES[job['profile']]
        windows = 1 + max(0, math.ceil((n - 77) / 76))
        chunks = 1 + max(0, math.ceil((n - 81) / 77))
        return 0.6 + windows * (prof['gen'] + 0.5) + chunks * prof['face']
    chunks = 1 + max(0, math.ceil((n - 81) / 77))
    return 0.6 + chunks * FACE_PROFILES[job['profile']]['chunk']


def has_face_pass(job):
    return job['mode'] == 'head' and '--face_pass' in HEAD_PROFILES[job['profile']]['args']


def build_command(job):
    if job['mode'] == 'head':
        cmd = [sys.executable, '-u', os.path.join(REPO, 'headswap.py')] + HEAD_PROFILES[job['profile']]['args']
        if not job.get('occlusion', True):
            cmd.append('--no_occlusion')
        if job.get('keep_mouth'):
            cmd.append('--keep_mouth')
        if job.get('natural', True) and has_face_pass(job):
            cmd += ['--wan_mouth', '0.6', '--wan_eyes', '1']
        if job.get('keep_temp'):
            cmd.append('--keep_temp')
        for lora in job.get('loras', []):
            cmd += ['--lora', lora]
    else:
        cmd = [sys.executable, '-u', os.path.join(REPO, 'faceswap.py')] + FACE_PROFILES[job['profile']]['args']
        cmd += ['--guide', str(job.get('guide', 4.0))]
    cmd += ['--ref_image', job['photo'], '--video', job['video'], '--start', f"{job['start']:g}"]
    if job['length'] is not None:
        cmd += ['--seconds', f"{job['length']:g}"]
    if job.get('seed', 42) != 42:
        cmd += ['--seed', str(job['seed'])]
    return cmd


# --------------------------------------------------------------------------- #
# live run
# --------------------------------------------------------------------------- #
HEAD_STAGES = [('prep', 'Track head, masks, pose'), ('load', 'Load model'), ('gen', 'Generate head'),
               ('face', 'Face pass (DreamID-V)'), ('paste', 'Match light and paste back')]
FACE_STAGES = [('load', 'Load models'), ('gen', 'Swap face'), ('paste', 'Save')]
RE_GEN = re.compile(r'(?:variant (\d+)/(\d+), )?window (\d+)/(\d+), step (\d+)/(\d+)')
RE_TQDM = re.compile(r'(\d+)/(\d+) \[')
RE_CHUNK = re.compile(r'Chunk (\d+)/(\d+) done')
RE_CHUNK_START = re.compile(r'Chunk (\d+): crop')
RE_CHUNKS = re.compile(r'in (\d+) chunk\(s\)')
RE_SCORE = re.compile(r'Identity vs\. photo \(AuraFace\): ([0-9.]+)')
NOISE = re.compile(r'^(W0000|I0000|WARNING: All log|INFO: Created TensorFlow|\s*\d+%\|)|absl::|inference_feedback|'
                   r'gl_context|NORM_RECT|^\s*$')
ABORTED = 'Stopped by the user.'


class Run:
    """Parses the tool's log into stages and a progress fraction."""

    def __init__(self, job, est_min):
        self.job = job
        self.stages = HEAD_STAGES if job['mode'] == 'head' else FACE_STAGES
        if job['mode'] == 'head' and HEAD_PROFILES[job['profile']]['face'] == 0:
            self.stages = [s for s in self.stages if s[0] != 'face']
        self.keys = [k for k, _ in self.stages]
        prof = HEAD_PROFILES.get(job['profile']) if job['mode'] == 'head' else None
        # rough weight of each stage in minutes (only for the progress bar)
        if job['mode'] == 'head':
            face = est_min * prof['face'] / max(prof['gen'] + 0.5 + prof['face'], 1)
            self.weights = dict(prep=0.4, load=0.3, gen=max(0.5, est_min - 1.1 - face), face=face, paste=0.4)
        else:
            self.weights = dict(load=0.4, gen=max(0.5, est_min - 0.6), paste=0.2)
        self.stage, self.frac, self.detail = self.keys[0], 0.0, ''
        self.score, self.result, self.work, self.error = None, None, None, []
        self.lines = []
        self.t0 = time.time()
        self.est = est_min
        self.gen_pos = (1, 1, 1, 1)  # variant, variants, window, windows
        self.chunks = (1, 1)

    def set_stage(self, key, detail=''):
        if key in self.keys and self.keys.index(key) >= self.keys.index(self.stage):
            if key != self.stage:
                self.frac = 0.0
            self.stage, self.detail = key, detail

    def feed(self, line):
        line = line.rstrip()
        if not line or NOISE.search(line):
            m = RE_TQDM.search(line)
            if m and self.stage in ('gen', 'face'):
                self._step(int(m.group(1)), int(m.group(2)))
            return
        msg = re.sub(r'^\[[0-9 :,-]+\]\s*', '', line)
        self.lines = (self.lines + [msg])[-6:]
        head = self.job['mode'] == 'head'
        if 'Traceback' in line or 'Error' in line or line.startswith('SystemExit'):
            self.error.append(msg)
        if head and 'Masks done' in line:
            self.frac = 0.6
        elif head and 'Pose + face crops done' in line:
            self.set_stage('load')
        elif head and 'Wan2.2-Animate-14B loaded in' in line:
            self.set_stage('gen')
        elif head and 'Face pass with DreamID-V' in line:
            self.set_stage('face')
        elif head and 'Reusing generated heads' in line:  # re-run on a kept generation
            self.set_stage('face' if 'face' in self.keys else 'paste')
        elif head and ('Light matched' in line or 'Face pass done' in line
                       or 'Reusing the face pass' in line
                       or ('Generation done' in line and 'face' not in self.keys)):
            self.set_stage('paste')
        elif not head and 'Models loaded' in line:
            self.set_stage('gen')
        elif not head and line.split('] ')[-1].startswith('Done in'):
            self.set_stage('paste')
        # the face pass is a faceswap.py run with its own 'Save file:' line: not the final result
        final = line.startswith('Save file:') and not (head and self.stage == 'face')
        if final:  # done
            self.stage, self.frac, self.detail = self.keys[-1], 1.0, ''
        m = RE_GEN.search(line)
        if m and head:
            v, vs = int(m.group(1) or 1), int(m.group(2) or 1)
            self.gen_pos = (v, vs, int(m.group(3)), int(m.group(4)))
            self._step(int(m.group(5)), int(m.group(6)))
        m = RE_CHUNKS.search(line)
        if m:
            self.chunks = (1, int(m.group(1)))
        m = RE_CHUNK_START.search(line)
        if m:
            self.chunks = (int(m.group(1)), self.chunks[1])
        m = RE_CHUNK.search(line)
        if m:
            self.chunks = (int(m.group(1)) + 1, int(m.group(2)))
            if self.stage == 'face' or (not head and self.stage == 'gen'):
                self.frac = min(1.0, int(m.group(1)) / int(m.group(2)))
        m = RE_TQDM.search(line)
        if m and self.stage in ('gen', 'face'):
            self._step(int(m.group(1)), int(m.group(2)))
        m = RE_SCORE.search(line)
        if m:
            self.score = float(m.group(1))
        if final:
            self.result = line.split(':', 1)[1].strip()
        if 'Work folder:' in line:
            self.work = line.split('Work folder:', 1)[1].strip()

    def _step(self, step, steps):
        if self.stage == 'gen' and self.job['mode'] == 'head':
            v, vs, w, ws = self.gen_pos
            self.frac = min(1.0, (((v - 1) * ws + (w - 1)) * steps + step) / max(vs * ws * steps, 1))
            self.detail = (f'variant {v}/{vs} · ' if vs > 1 else '') + f'window {w}/{ws} · step {step}/{steps}'
        elif self.stage in ('gen', 'face'):
            c, cs = self.chunks
            self.frac = min(1.0, ((c - 1) + step / max(steps, 1)) / max(cs, 1))
            self.detail = f'part {c}/{max(cs, c)} · step {step}/{steps}'

    def progress(self):
        if self.stage == self.keys[-1] and self.frac >= 1.0:
            return 1.0
        total = sum(self.weights[k] for k in self.keys)
        idx = self.keys.index(self.stage)
        done = sum(self.weights[k] for k in self.keys[:idx]) + self.weights[self.stage] * self.frac
        return min(0.99, done / total)

    def eta(self):
        el = (time.time() - self.t0) / 60
        p = self.progress()
        if p < 0.08:
            return max(0.0, self.est - el)
        return el / p * (1 - p)

    def render(self):
        p = self.progress()
        el = (time.time() - self.t0) / 60
        width = 46
        filled = int(p * width)
        bar = Text('━' * filled, style='bold #e5a50a') + Text('━' * (width - filled), style='#444444')
        head = Text.assemble(bar, f'  {p * 100:4.0f} %   ', ('elapsed ', 'dim'), mmss(el),
                             ('  ·  about ', 'dim'), mmss(self.eta()), (' left', 'dim'))
        st = Table.grid(padding=(0, 1))
        idx = self.keys.index(self.stage)
        for i, (k, label) in enumerate(self.stages):
            if i < idx:
                st.add_row(Text('✔', style='green'), Text(label, style='dim'), Text(''))
            elif i == idx:
                st.add_row(Text('▶', style='bold #e5a50a'), Text(label, style='bold'), Text(self.detail, style='#62a0ea'))
            else:
                st.add_row(Text('·', style='dim'), Text(label, style='dim'), Text(''))
        log = Text('\n'.join(l[:110] for l in self.lines[-5:]), style='dim')
        return Panel(Group(head, Text(''), st, Text(''), log),
                     title=f"[bold]{os.path.basename(self.job['video'])}[/]  →  {self.job_title()}",
                     subtitle='[dim]Ctrl+C = stop[/]', border_style='#e5a50a', box=box.ROUNDED, padding=(1, 2))

    def job_title(self):
        if self.job['mode'] == 'head':
            return f"Head + hair · {HEAD_PROFILES[self.job['profile']]['title']}"
        return f"Face only · {FACE_PROFILES[self.job['profile']]['title']}"


def _reader(proc, q):
    buf = b''
    while True:
        chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, 'read1') else proc.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        parts = re.split(rb'[\r\n]', buf)
        buf = parts.pop()
        for p in parts:
            q.put(p.decode('utf-8', 'replace'))
    if buf:
        q.put(buf.decode('utf-8', 'replace'))
    q.put(None)


def run_job(job, est_min, cmd=None):
    """Runs the tool with a live progress panel. Returns the Run (result/score/error)."""
    cmd = cmd or build_command(job)
    os.makedirs(LOGS, exist_ok=True)
    log_path = os.path.join(LOGS, time.strftime('run_%Y%m%d_%H%M%S.log'))
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    # own session: Ctrl+C reaches only the app, which asks before stopping the run
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
    q = queue.Queue()
    threading.Thread(target=_reader, args=(proc, q), daemon=True).start()
    run = Run(job, est_min)
    try:
        _follow(proc, q, run, cmd, log_path)
    finally:  # never leave a 40 GB model run behind when the app itself stops
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
    run.returncode = proc.returncode
    run.log_path = log_path
    if run.result:  # keep the log next to the history: same name as the video
        final = os.path.join(LOGS, os.path.splitext(os.path.basename(run.result))[0] + '.log')
        shutil.move(log_path, final)
        run.log_path = final
    return run


def _follow(proc, q, run, cmd, log_path):
    from rich.live import Live
    with open(log_path, 'w') as log:
        log.write('$ ' + ' '.join(shlex.quote(c) for c in cmd) + '\n')
        finished = False
        while not finished:
            try:
                with Live(run.render(), console=console, refresh_per_second=4, transient=False) as live:
                    while True:
                        try:
                            line = q.get(timeout=0.25)
                        except queue.Empty:
                            live.update(run.render())
                            continue
                        if line is None:
                            finished = True
                            break
                        log.write(line + '\n')
                        run.feed(line)
                        live.update(run.render())
            except KeyboardInterrupt:
                if ask(yes_no('Really stop the run?', default=False)):
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    run.error.append(ABORTED)
                    finished = True
                    while q.get() is not None:  # drain the reader
                        pass


# --------------------------------------------------------------------------- #
# screens
# --------------------------------------------------------------------------- #
def header():
    console.clear()
    console.print(Panel(Text.assemble((APP_NAME, 'bold #e5a50a'), '   ',
                                      ('Head and face swaps, locally on Apple Silicon', 'dim')),
                        box=box.ROUNDED, border_style='#e5a50a', padding=(0, 2)))


def info_panel(title, lines, ok=True):
    console.print(Panel('\n'.join(lines), title=title, title_align='left',
                        border_style='green' if ok else 'red', box=box.ROUNDED, padding=(0, 1)))


def pick_file(message, ext, last, kind):
    def valid(t):
        p = clean_path(t)
        if not p:
            return 'Please enter a path or drag the file into the window.'
        if not os.path.isfile(p):
            return 'File not found.'
        if not p.lower().endswith(ext):
            return f'Not a {kind} file ({", ".join(ext)}).'
        return True

    default = last if last and os.path.isfile(last) else ''
    r = ask(questionary.path(f'{message} [Tab completes, dragging the file in works too]', default=default,
                             validate=valid, style=STYLE,
                             file_filter=lambda f: os.path.isdir(f) or f.lower().endswith(ext)))
    return clean_path(r) if r else None


def pick_range(info, settings):
    dur = info['duration']
    choices = [Choice(f'Whole video ({secs(dur)})', value='all')]
    if dur > 4:
        choices.append(Choice('Quick test: the first 3.2 s', value='test',
                              description='One model window (77 frames). Ideal for checking photo and settings.'))
    choices.append(Choice('Choose a range …', value='custom'))
    r = ask(select('Which part of the video?', choices, style=STYLE,
                   default=next((c for c in choices if c.value == settings.get('range')), None)))
    if r is None:
        return None
    settings['range'] = r
    if r == 'all':
        return 0.0, None
    if r == 'test':
        return 0.0, 3.2

    def num(lo, hi):
        def v(t):
            try:
                x = number(t)
            except ValueError:
                return 'Please enter a number, e.g. 2.5'
            return True if lo <= x <= hi else f'Between {lo:g} and {hi:g}.'
        return v

    s = ask(questionary.text('Start at second:', default='0', validate=num(0, max(0.0, dur - 0.2)), style=STYLE))
    if s is None:
        return None
    start = number(s)
    rest = dur - start
    ln = ask(questionary.text(f'Length in seconds (max {secs(rest)}):', default=f'{min(rest, 3.2):g}',
                              validate=num(0.2, rest), style=STYLE))
    if ln is None:
        return None
    return start, number(ln)


def pick_extras(job, settings):
    if job['mode'] == 'head':
        prev = settings.get('extras', ['occlusion', 'natural', 'compare'])
        options = [
            Choice('Keep occlusions original (hands, food, glass in front of the face)', value='occlusion',
                   checked='occlusion' in prev),
        ]
        if has_face_pass(job):
            options.append(Choice('Eyes and mouth follow the original (gaze, blinks, mouth opening; '
                                  'costs ~0.02 identity)', value='natural', checked='natural' in prev))
        options += [
            Choice('Mouth interior from the original (teeth, tongue, food in the mouth)', value='keep_mouth',
                   checked='keep_mouth' in prev),
            Choice('Create a comparison video (original | result)', value='compare', checked='compare' in prev),
            Choice('Keep intermediate results (change light/mouth later without regenerating)', value='keep_temp',
                   checked='keep_temp' in prev),
        ]
        extra_loras = [f for f in sorted(os.listdir(LORAS)) if f.endswith('.safetensors')
                       and 'lightx2v' not in f] if os.path.isdir(LORAS) else []
        if extra_loras:
            options.append(Choice('Load an extra LoRA …', value='lora', checked='lora' in prev))
    else:
        prev = settings.get('extras_face', ['compare'])
        options = [Choice('Create a comparison video (original | result)', value='compare', checked='compare' in prev)]
    r = ask(questionary.checkbox('Extras:', choices=options, style=STYLE,
                                 instruction='(Space = on/off, Enter = continue)'))
    if r is None:
        return False
    if job['mode'] == 'head':
        if not has_face_pass(job) and 'natural' in prev:
            r.append('natural')  # not offered for this profile: keep the remembered choice
        settings['extras'] = r
        job['occlusion'] = 'occlusion' in r
        job['natural'] = 'natural' in r
        job['keep_mouth'] = 'keep_mouth' in r
        job['keep_temp'] = 'keep_temp' in r
        job['loras'] = []
        if 'lora' in r:
            f = ask(select('Which LoRA?', extra_loras, style=STYLE))
            st = ask(select('Strength:', ['0.5', '0.8', '1.0'], default='1.0', style=STYLE))
            if f and st:
                job['loras'] = [f'{os.path.join(LORAS, f)}:{st}']
    else:
        settings['extras_face'] = r
        g = ask(select('Identity strength:', style=STYLE, default=settings.get('guide', '4.0'), choices=[
            Choice('more natural (3.0)', value='3.0'), Choice('normal (4.0)', value='4.0'),
            Choice('closer to the photo (5.0)', value='5.0')]))
        if g is None:
            return False
        settings['guide'] = g
        job['guide'] = float(g)
    job['compare'] = 'compare' in r
    return True


def summary(job, info, est):
    t = Table(box=box.SIMPLE_HEAD, show_header=False, padding=(0, 1))
    t.add_column(style='dim', no_wrap=True)
    t.add_column()
    t.add_row('Photo', os.path.basename(job['photo']))
    rng = (f"{secs(job['start'])} – {secs(job['start'] + job['length'])}" if job['length'] is not None
           else f"whole video ({secs(info['duration'])})")
    t.add_row('Video', f"{os.path.basename(job['video'])}  ·  {rng}  ·  {frames_of(job, info)} frames")
    if job['mode'] == 'head':
        p = HEAD_PROFILES[job['profile']]
        t.add_row('Mode', 'Head + hair (Wan2.2-Animate-14B)')
        t.add_row('Profile', f"{p['title']}  [dim]{p['desc']}[/]")
        ex = [n for k, n in (('occlusion', 'occlusions kept original'), ('keep_mouth', 'mouth from original'),
                             ('keep_temp', 'keep intermediate results')) if job.get(k)]
        if has_face_pass(job) and job.get('natural', True):
            ex.insert(1, 'eyes/mouth follow the original')
        ex += [f'LoRA {os.path.basename(l)}' for l in job.get('loras', [])]
    else:
        p = FACE_PROFILES[job['profile']]
        t.add_row('Mode', 'Face only (DreamID-V)')
        t.add_row('Profile', f"{p['title']}  [dim]{p['desc']}[/]")
        ex = [f"identity strength {job['guide']:g}"]
    if job.get('compare'):
        ex.append('comparison video')
    t.add_row('Extras', ', '.join(ex) or '–')
    t.add_row('Time', f'[bold]about {mmss(est)}[/]  [dim](M5 Max; the first start takes longer to load models)[/]')
    t.add_row('Output', 'output/  [dim](the name contains the settings, plus a .json with every value)[/]')
    console.print(Panel(t, title='Summary', title_align='left', border_style='#62a0ea', box=box.ROUNDED))


def rating(score):
    if score is None:
        return '[dim]not measured[/]'
    s = f'{score:.2f}'
    if score >= 0.75:
        return f'[green bold]{s}[/] [green]very similar[/]'
    if score >= 0.6:
        return f'[green]{s}[/] similar'
    if score >= 0.45:
        return f'[yellow]{s}[/] recognisable, the "Maximum likeness" profile helps'
    return (f'[red]{s}[/] not very similar – try another photo or profile '
            '[dim](sunglasses or heavy occlusion lower the score without the swap being bad)[/]')


def open_file(path, reveal=False):
    subprocess.run(['open', '-R', path] if reveal else ['open', path])


def result_screen(job, run, info):
    console.print()
    if run.returncode != 0 or not run.result:
        aborted = ABORTED in run.error
        msg = '\n'.join(run.error[-6:] or run.lines[-6:]) or 'Unknown error.'
        info_panel('Stopped' if aborted else 'Failed', [msg, '', f'Log: {nice_path(run.log_path)}'], ok=False)
        press_key()
        return
    comp = None
    if job.get('compare'):
        with console.status('Creating the comparison video …', spinner='dots'), quiet():
            try:
                from compare import make_comparison
                label = run.job_title().split(' · ', 1)[1]
                comp = make_comparison(job['video'], run.result, job['start'], label=label, tools=face_tools())
            except Exception as e:  # comparison is a bonus, never fail the run for it
                console.print(f'[yellow]Comparison video not possible: {e}[/]')
    sidecar = os.path.splitext(run.result)[0] + '.json'
    score = run.score
    if score is None and os.path.exists(sidecar):
        score = json.load(open(sidecar)).get('identity_auraface')
    lines = [f'[bold]{nice_path(run.result)}[/]',
             f'Took {mmss((time.time() - run.t0) / 60)}   ·   Identity vs. photo: {rating(score)}']
    if comp:
        lines.append(f'Comparison: {nice_path(comp)}')
    lines.append(f'[dim]Settings: {nice_path(sidecar)}[/]')
    lines.append(f'[dim]Log: {nice_path(run.log_path)}[/]')
    info_panel('Done', lines)
    while True:
        choices = [Choice('▶  Play result', value='play')]
        if comp:
            choices.append(Choice('▶  Play comparison (original | result)', value='cmp'))
        choices.append(Choice('📂 Show in Finder', value='finder'))
        if run.work and job['mode'] == 'head':
            choices.append(Choice('🎚  Fine-tune without regenerating (light, mouth)', value='tune'))
        choices.append(Choice('⌂  Back to the main menu', value='back'))
        a = ask(select('What next?', choices, style=STYLE))
        if a in (None, 'back'):
            return
        if a == 'play':
            open_file(run.result)
        elif a == 'cmp':
            open_file(comp)
        elif a == 'finder':
            open_file(run.result, reveal=True)
        elif a == 'tune':
            tune(job, run, info)
            return


def tune(job, run, info):
    """Re-run only light matching + paste-back on the kept generation (about a minute)."""
    light = ask(select('Match the light to the scene:', style=STYLE, default='1', choices=[
        Choice('full (default)', value='1'), Choice('half', value='0.5'), Choice('off (light as generated)', value='0')]))
    if light is None:
        return
    mouth = ask(yes_no('Use the mouth interior of the original?', default=bool(job.get('keep_mouth'))))
    if mouth is None:
        return
    natural = job.get('natural', True)
    if has_face_pass(job):  # the face pass is kept next to the generation, so this costs no extra minutes
        natural = ask(yes_no('Eyes and mouth like in the original (instead of all from the face pass)?',
                             default=natural))
        if natural is None:
            return
    new_job = dict(job, keep_mouth=mouth, natural=natural, keep_temp=False)
    cmd = build_command(new_job) + ['--from_work', run.work, '--light', light, '--shade', light]
    est = 0.6 + 0.5 * (1 + frames_of(job, info) // 77)
    header()
    r = run_job(new_job, est, cmd=cmd)
    if r.result and not r.work:
        r.work = run.work  # same kept generation: allows tuning again
    result_screen(new_job, r, info)


def new_swap(settings):
    header()
    photo = pick_file('Photo of the new person:', IMAGE_EXT, settings.get('photo'), 'image')
    if not photo:
        return
    with console.status('Checking the photo …', spinner='dots'), quiet():
        ok, lines = check_photo(photo)
    info_panel(f'Photo: {os.path.basename(photo)}', lines, ok)
    if not ok:
        press_key()
        return
    settings['photo'] = photo

    video = pick_file('Target video:', VIDEO_EXT, settings.get('video'), 'video')
    if not video:
        return
    info = probe(video)
    if info is None:
        info_panel('Video', ['[red]The video cannot be read.[/]'], ok=False)
        press_key()
        return
    with console.status('Checking the video …', spinner='dots'), quiet():
        ok, lines = check_video(video, info)
    info_panel(f'Video: {os.path.basename(video)}', lines, ok)
    if not ok and not ask(yes_no('Continue anyway?', default=False)):
        return
    settings['video'] = video

    mode = ask(select('What should be swapped?', style=STYLE, default=settings.get('mode', 'head'), choices=[
        Choice('Head + hair', value='head',
               description='Wan2.2-Animate-14B: the whole head including hair. Body, clothes, background, audio stay.'),
        Choice('Face only', value='face',
               description='DreamID-V 1.3B: only the face. Hair and head shape stay from the original.'),
    ]))
    if mode is None:
        return
    settings['mode'] = mode
    rng = pick_range(info, settings)
    if rng is None:
        return
    job = dict(mode=mode, photo=photo, video=video, start=rng[0], length=rng[1])

    profiles = HEAD_PROFILES if mode == 'head' else FACE_PROFILES
    default_key = settings.get(f'profile_{mode}', 'identity' if mode == 'head' else 'balanced')
    choices = []
    for key, p in profiles.items():
        est = estimate(dict(job, profile=key), info)
        choices.append(Choice(f"{p['title']:<34} about {mmss(est)}", value=key, description=p['desc']))
    prof = ask(select('Quality:', choices, style=STYLE,
                      default=next((c for c in choices if c.value == default_key), None)))
    if prof is None:
        return
    settings[f'profile_{mode}'] = prof
    job['profile'] = prof
    if not pick_extras(job, settings):
        return
    save_settings(settings)

    est = estimate(job, info)
    header()
    summary(job, info, est)
    go = ask(select('Start?', style=STYLE, choices=[
        Choice('▶  Start now', value='go'), Choice('✎  Change something', value='edit'), Choice('✕  Cancel', value='no')]))
    if go == 'edit':
        return new_swap(settings)
    if go != 'go':
        return
    header()
    run = run_job(job, est)
    result_screen(job, run, info)


def describe(e):
    """Short settings summary from a sidecar's 'settings'."""
    parts = [e.get('preset', '?')]
    if e.get('from_work') and isinstance(e.get('_generation'), dict):
        g = e['_generation']
        return describe(dict(g, keep_mouth=e.get('keep_mouth'), occlusion=e.get('occlusion', True),
                             from_work=None)) + ', fine-tuned'
    if 'face_pass' in e:  # head swap
        if e.get('turbo'):
            parts.append('turbo')
        if e.get('face_pass', 'off') != 'off':
            parts.append('face pass' + (' (eyes/mouth orig.)' if e.get('wan_eyes') or e.get('wan_mouth') else ''))
        if (e.get('candidates') or 1) > 1:
            parts.append(f"{e['candidates']} variants")
        if e.get('keep_mouth'):
            parts.append('mouth orig.')
        if not e.get('occlusion', True):
            parts.append('no occlusion')
        if e.get('lora'):
            parts.append('+LoRA')
        if e.get('from_work'):
            parts.append('fine-tuned')
    else:
        parts.append(f"guide {e.get('guide', 4.0):g}")
    if e.get('seed', 42) != 42:
        parts.append(f"seed {e['seed']}")
    return ', '.join(parts)


def history(settings):
    header()
    items = []
    for root, dirs, files in os.walk(OUTPUT):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'logs']
        for f in files:
            if f.endswith('.json'):
                try:
                    d = json.load(open(os.path.join(root, f)))
                except (OSError, ValueError):
                    continue
                if 'result' in d and 'settings' in d:
                    video = os.path.join(root, d['result'])
                    if os.path.exists(video):
                        items.append((d, video))
    if not items:
        info_panel('History', ['No results with settings (.json) in output/ yet.'])
        press_key()
        return
    items.sort(key=lambda x: x[0].get('created', ''), reverse=True)
    t = Table(box=box.SIMPLE_HEAD, header_style='bold')
    for c in ('No', 'Date', 'Mode', 'Video', 'Range', 'Settings', 'Identity', 'Time'):
        t.add_column(c, no_wrap=c not in ('Settings', 'Video'))
    for i, (d, video) in enumerate(items[:30], 1):
        e = dict(d['settings'], _generation=d.get('generation'))
        a, b = d.get('range_s', [0, 0])
        sc = d.get('identity_auraface')
        t.add_row(str(i), d.get('created', '')[:16], 'head' if 'face_pass' in e else 'face',
                  os.path.basename(e.get('video', '')), f'{a:g}–{b:g} s', describe(e),
                  '–' if sc is None else f'{sc:.2f}', mmss(d.get('minutes')))
    console.print(t)
    pick = ask(select('Open a result:', style=STYLE, choices=[
        Choice(f'{i}. {os.path.basename(v)}', value=i - 1) for i, (_, v) in enumerate(items[:30], 1)] + [
        Separator(), Choice('⌂  Back', value=None)]))
    if pick is None:
        return
    d, video = items[pick]
    comp = os.path.splitext(video)[0] + '__compare.mp4'
    a = ask(select(os.path.basename(video), style=STYLE, choices=[
        Choice('▶  Play', value='play'),
        Choice('▶  Play comparison' if os.path.exists(comp) else '▶  Create and play a comparison video', value='cmp'),
        Choice('📂 Show in Finder', value='finder'),
        Choice('⚙  Show all settings', value='json'),
        Choice('⌂  Back', value=None)]))
    if a == 'play':
        open_file(video)
    elif a == 'finder':
        open_file(video, reveal=True)
    elif a == 'json':
        console.print_json(data=d)
        press_key()
    elif a == 'cmp':
        if not os.path.exists(comp):
            with console.status('Creating the comparison video …', spinner='dots'), quiet():
                from compare import make_comparison
                comp = make_comparison(d['settings']['video'], video, d.get('range_s', [0])[0],
                                       label='Result', tools=face_tools())
        if comp:
            open_file(comp)


def system_info():
    header()
    import psutil
    t = Table(box=box.SIMPLE_HEAD, header_style='bold')
    t.add_column('Model')
    t.add_column('Status')
    for label, rel, _ in MODEL_FILES:
        t.add_row(label, '[green]found[/]' if os.path.exists(os.path.join(MODELS, rel)) else '[red]missing[/]')
    console.print(t)
    mem = psutil.virtual_memory()
    du = shutil.disk_usage(REPO)
    info_panel('System', [
        f'Memory: {mem.total / 1024 ** 3:.0f} GB, {mem.available / 1024 ** 3:.0f} GB free '
        '[dim](head swaps need ~45 GB)[/]',
        f'Free disk space: {du.free / 1024 ** 3:.0f} GB',
        f'Models: {MODELS}',
        f'Results: {OUTPUT}',
        f'Logs: {LOGS}',
        f'Python: {sys.executable}',
    ])
    press_key()


def main():
    if not sys.stdin.isatty():
        raise SystemExit('Please start it in a terminal: python app.py')
    if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:
        raise SystemExit('ffmpeg/ffprobe are missing (brew install ffmpeg).')
    sys.path.insert(0, REPO)
    settings = load_settings()
    while True:
        header()
        a = ask(select('What would you like to do?', style=STYLE, choices=[
            Choice('▶  New swap', value='new'),
            Choice('☰  History (previous results)', value='history'),
            Choice('ⓘ  Models and system', value='info'),
            Choice('✕  Quit', value='quit')]))
        if a in (None, 'quit'):
            console.print('[dim]Bye.[/]')
            return
        if a == 'new':
            new_swap(settings)
        elif a == 'history':
            history(settings)
        elif a == 'info':
            system_info()


if __name__ == '__main__':
    main()
