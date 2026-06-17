# ============================================================
# FINANCE EXPLAINER - AI Video Generator v1
# New niche, character-free. Reuses the Vaults v3 architecture:
# Two-call GPT (Story Beats -> Render Decisions), OpenCV + Pillow
# renderer, Whisper word-timestamp-driven timing, validation pass.
#
# What's NEW vs Vaults: the render decision vocabulary adds
# number_counter (animated count-up/down) and grid (repeated-glyph
# walls, e.g. a column of zeros) element types, suited to numbers/
# stats/comparisons instead of dramatic single-word captions. Topic
# styles, music map, and GPT prompts are finance-specific. No
# recurring character, no procedural illustration shapes (planet/
# brain/etc) -- this niche is character-free by design.
# ============================================================

import subprocess
import os
import json
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import traceback
import glob
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import cv2
from PIL import Image

load_dotenv()

from openai import OpenAI
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Finance Explainer v1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

current_job = {"status": "idle", "progress": 0, "output": None, "error": None, "started_at": None}

OUTPUT_WIDTH  = 1920
OUTPUT_HEIGHT = 1080
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Shared GPT-4o rate limiter ──────────────────────────────────────
# Call 2 (render decisions) and Call 3 (freeform motion design) both
# batch against gpt-4o with their own ThreadPoolExecutors. Those pools
# have no idea about each other, but they share the SAME org-level
# 30k-tokens-per-minute ceiling -- back-to-back Call 2 then Call 3
# batches were colliding mid-window and getting 429'd. A single
# process-wide semaphore + a minimum gap between calls keeps total
# concurrent GPT-4o requests bounded regardless of which function
# issued them.
import threading
import time as _time

_GPT4O_CONCURRENCY = threading.Semaphore(3)   # max simultaneous gpt-4o requests, any caller
_GPT4O_LOCK = threading.Lock()
_GPT4O_LAST_CALL_TS = [0.0]
_GPT4O_MIN_GAP_SECONDS = 1.5  # stagger request starts so bursts don't all land in the same TPM window

def gpt4o_call(client, **kwargs):
    """Wrapper around client.chat.completions.create for gpt-4o that
    throttles across ALL callers (Call 2 and Call 3 batches alike) so
    they can't collectively exceed the shared TPM budget."""
    with _GPT4O_CONCURRENCY:
        with _GPT4O_LOCK:
            wait = _GPT4O_MIN_GAP_SECONDS - (_time.time() - _GPT4O_LAST_CALL_TS[0])
            if wait > 0:
                _time.sleep(wait)
            _GPT4O_LAST_CALL_TS[0] = _time.time()
        return client.chat.completions.create(**kwargs)


def _call_with_retry(fn, label="gpt-4o call", max_retries=3):
    """Retry on 429 (rate limit) with exponential backoff. OpenAI's 429
    error includes a suggested wait time in its message; this is a
    simple fixed-backoff fallback since parsing that out reliably isn't
    worth the fragility. Re-raises on the final attempt so a genuinely
    persistent failure still surfaces instead of silently vanishing."""
    delay = 2.0
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if is_rate_limit and attempt < max_retries - 1:
                print(f"  ⏳ {label}: rate limited, retrying in {delay:.0f}s (attempt {attempt+1}/{max_retries})...")
                _time.sleep(delay)
                delay *= 2
                continue
            raise

if not OPENAI_API_KEY:
    print("⚠  WARNING: OPENAI_API_KEY not set.")

# Toggle: True = generate animated background procedurally (no broll, no
# clip failures, zero cost). False = use the old broll-clip pipeline.
USE_PROCEDURAL_BACKGROUND = True

MUSIC_MAP = {
    "markets":   "bg_musics/finance_ambient.mp3",
    "growth":    "bg_musics/finance_ambient.mp3",
    "warning":   "bg_musics/dark_ambient.mp3",
    "history":   "bg_musics/finance_ambient.mp3",
    "default":   "bg_musics/finance_ambient.mp3",
}

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")

FONT_BLACK_CANDIDATES = [
    os.path.join(FONTS_DIR, "Anton-Regular.ttf"),
    os.path.join(FONTS_DIR, "Montserrat-Black.ttf"),
    os.path.join(FONTS_DIR, "Montserrat-ExtraBold.ttf"),
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
FONT_BOLD_CANDIDATES = [
    os.path.join(FONTS_DIR, "Montserrat-ExtraBold.ttf"),
    os.path.join(FONTS_DIR, "Montserrat-Bold.ttf"),
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
FONT_REGULAR_CANDIDATES = [
    os.path.join(FONTS_DIR, "Montserrat-Bold.ttf"),
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

def find_font(candidates):
    for path in candidates:
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print(f"  ✓ Font: {path}")
            return path
    return None

FONT_BLACK   = find_font(FONT_BLACK_CANDIDATES)
FONT_BOLD    = find_font(FONT_BOLD_CANDIDATES)
FONT_REGULAR = find_font(FONT_REGULAR_CANDIDATES)

def get_primary_font_path(bold: bool = True) -> str:
    """Return best available font: Black > ExtraBold > Bold > anything."""
    if FONT_BLACK:   return FONT_BLACK
    if FONT_BOLD:    return FONT_BOLD
    if FONT_REGULAR: return FONT_REGULAR
    return None

def _probe_clip_health(filepath: str) -> tuple[bool, str]:
    """Quick ffprobe check: can this file actually be decoded?
    Returns (is_healthy, reason_if_not)."""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=width,height,duration,codec_name',
           '-of', 'json', filepath]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "ffprobe failed"
        return False, err[:120]
    try:
        data = json.loads(result.stdout)
        streams = data.get('streams', [])
        if not streams:
            return False, "no video stream found"
        s = streams[0]
        if not s.get('width') or not s.get('height'):
            return False, "missing width/height"
        return True, ""
    except Exception as e:
        return False, f"parse error: {e}"


@app.on_event("startup")
async def startup_event():
    print("🚀 Vaults of History v3 starting...")
    # Audit broll folders so missing clips are immediately visible
    broll_dirs = ['space_vids','ancient_ruins_vids','cosmic_vids',
                  'dark_sky_vids','temple_vids']
    print("📁 Broll folder audit:")
    bad_clips = []
    for d in broll_dirs:
        if os.path.exists(d):
            files = [f for f in os.listdir(d) if f.lower().endswith(('.mp4','.mov','.avi'))]
            status = f"✅ {len(files)} clips" if files else "❌ EMPTY -- add Seedance clips here"
            print(f"  {d}: {status}")
            # Health-check each clip so broken files are caught before a run
            for f in files:
                fpath = os.path.join(d, f)
                healthy, reason = _probe_clip_health(fpath)
                if not healthy:
                    bad_clips.append((fpath, reason))
        else:
            print(f"  {d}: ❌ MISSING -- folder doesn't exist")

    if bad_clips:
        print("⚠️  BROKEN CLIPS DETECTED (these will render as black filler):")
        for fpath, reason in bad_clips:
            print(f"    ✗ {fpath} -- {reason}")
        print(f"  → Replace or remove these {len(bad_clips)} file(s) to eliminate black segments.")
    else:
        print("  ✅ All clips passed health check")



# ============================================================
# PROCEDURAL BACKGROUND GENERATOR
# Replaces broll entirely. No clip failures, no black fillers,
# zero cost, and a visual identity matching a finance/numbers
# explainer channel -- clean, data-forward, not cinematic-horror.
#
# One primary animated style per topic, with intensity (from
# GPT Call 1's per-beat "intensity" field) smoothly driving
# animation speed/density/glow over the course of the video.
# ============================================================

def _bgr(r, g, b):
    """Convenience: define colors in RGB, return BGR for OpenCV."""
    return (b, g, r)


TOPIC_STYLES = {
    'markets': {
        'bg':      _bgr(8, 10, 14),
        'accent':  _bgr(120, 230, 170),    # market green
        'accent2': _bgr(255, 215, 130),    # gold/currency
        'styles':  ['particles', 'geometric'],
    },
    'growth': {
        'bg':      _bgr(6, 12, 14),
        'accent':  _bgr(120, 230, 170),
        'accent2': _bgr(160, 210, 255),
        'styles':  ['geometric', 'particles'],
    },
    'warning': {
        'bg':      _bgr(10, 6, 6),
        'accent':  _bgr(230, 90, 80),       # warning red
        'accent2': _bgr(255, 200, 90),
        'styles':  ['particles', 'geometric'],
    },
    'history': {
        'bg':      _bgr(10, 10, 14),
        'accent':  _bgr(255, 215, 130),     # gold
        'accent2': _bgr(170, 175, 190),
        'styles':  ['geometric', 'particles'],
    },
    'default': {
        'bg':      _bgr(8, 9, 12),
        'accent':  _bgr(255, 255, 255),
        'accent2': _bgr(120, 230, 170),
        'styles':  ['particles', 'geometric'],
    },
}


class _Starfield:
    """Deterministic starfield: positions fixed, twinkle + slow horizontal drift."""
    def __init__(self, width, height, n_stars=220, seed=42):
        rng = random.Random(seed)
        self.stars = []
        for _ in range(n_stars):
            self.stars.append({
                'x': rng.uniform(0, width),
                'y': rng.uniform(0, height),
                'r': rng.uniform(0.6, 2.4),
                'speed': rng.uniform(2, 10),
                'phase': rng.uniform(0, 6.283),
                'tw_speed': rng.uniform(0.8, 2.5),
            })
        self.width, self.height = width, height

    def draw(self, frame, t, intensity, color):
        w = self.width
        bright_base = 0.35 + 0.04 * intensity
        for s in self.stars:
            x = (s['x'] + t * s['speed'] * (0.5 + 0.08 * intensity)) % w
            tw = 0.5 + 0.5 * math.sin(t * s['tw_speed'] + s['phase'])
            brightness = bright_base + 0.5 * tw
            r = max(1, int(round(s['r'] * (0.8 + 0.5 * tw))))
            col = tuple(int(c * min(brightness, 1.0)) for c in color)
            cv2.circle(frame, (int(x), int(s['y'])), r, col, -1, lineType=cv2.LINE_AA)
        return frame


def _draw_nebula(frame, t, intensity, color):
    """Soft slow-moving glow blobs, rendered at low-res and upscaled for a
    cheap painterly blur (full-res GaussianBlur on 1920x1080 every frame is
    too slow for 2500+ frames)."""
    h, w = frame.shape[:2]
    sw, sh = max(w // 3, 8), max(h // 3, 8)
    small = np.zeros((sh, sw, 3), dtype=np.float32)
    n_blobs = 3 + int(min(intensity, 10) // 3)
    for i in range(n_blobs):
        bx = sw * (0.2 + 0.6 * ((i * 0.37 + 0.06 * t + 0.5 * math.sin(t * 0.04 + i)) % 1))
        by = sh * (0.2 + 0.6 * ((i * 0.61 + 0.04 * t + 0.5 * math.cos(t * 0.035 + i * 1.3)) % 1))
        radius = int(min(sw, sh) * (0.35 + 0.08 * math.sin(t * 0.08 + i)))
        cv2.circle(small, (int(bx), int(by)), max(radius, 4), color, -1, lineType=cv2.LINE_AA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sw * 0.18)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    strength = 0.10 + 0.012 * intensity
    out = np.clip(frame.astype(np.float32) + big * strength, 0, 255).astype(np.uint8)
    return out


def _draw_geometric(frame, t, intensity, color):
    """Slowly rotating concentric hexagons -- 'sacred geometry' motif."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    base_r = int(min(w, h) * 0.32)
    n_shapes = 3
    speed = 4 + intensity * 1.2  # degrees/sec
    for i in range(n_shapes):
        angle0 = math.radians(t * speed * (1 if i % 2 == 0 else -1) + i * 40)
        r = base_r - i * int(base_r * 0.22)
        sides = 6
        pts = []
        for k in range(sides):
            a = angle0 + 2 * math.pi * k / sides
            pts.append((int(cx + r * math.cos(a)), int(cy + r * math.sin(a))))
        pts = np.array(pts, dtype=np.int32)
        alpha = 0.10 + 0.01 * intensity
        overlay = frame.copy()
        cv2.polylines(overlay, [pts], True, color, 2, lineType=cv2.LINE_AA)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return frame


def _draw_aurora(frame, t, intensity, color):
    """Flowing horizontal energy bands, low-res + blur for performance."""
    h, w = frame.shape[:2]
    sw, sh = max(w // 4, 8), max(h // 4, 8)
    small = np.zeros((sh, sw, 3), dtype=np.float32)
    n_bands = 3
    for b in range(n_bands):
        y_center = sh * (0.25 + 0.22 * b) + 4 * math.sin(t * 0.3 + b)
        for x in range(sw):
            y_off = 3 * math.sin(x * 0.25 + t * (0.4 + 0.05 * intensity) + b * 2)
            y = int(y_center + y_off)
            if 0 <= y < sh:
                cv2.line(small, (x, max(0, y - 1)), (x, min(sh, y + 1)), color, 1)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=sw * 0.06)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    strength = 0.18 + 0.015 * intensity
    out = np.clip(frame.astype(np.float32) + big * strength, 0, 255).astype(np.uint8)
    return out


def _draw_particles(frame, t, intensity, color, seed=21, n=70):
    h, w = frame.shape[:2]
    rng = random.Random(seed)
    bright_base = 0.25 + 0.04 * intensity
    for i in range(n):
        sx = rng.uniform(0, w)
        sy = rng.uniform(0, h)
        speed = rng.uniform(8, 25) * (0.5 + 0.08 * intensity)
        phase = rng.uniform(0, 6.283)
        x = (sx + t * speed) % w
        y = (sy + 25 * math.sin(t * 0.6 + phase)) % h
        tw = 0.5 + 0.5 * math.sin(t * 1.5 + phase)
        r = 1 + int(2 * tw)
        col = tuple(int(c * min(bright_base + 0.5 * tw, 1.0)) for c in color)
        cv2.circle(frame, (int(x), int(y)), r, col, -1, lineType=cv2.LINE_AA)
    return frame


_BG_DRAW_FNS = {
    'starfield': lambda frame, t, intensity, color, sf: sf.draw(frame, t, intensity, color),
    'nebula':    lambda frame, t, intensity, color, sf: _draw_nebula(frame, t, intensity, color),
    'geometric': lambda frame, t, intensity, color, sf: _draw_geometric(frame, t, intensity, color),
    'aurora':    lambda frame, t, intensity, color, sf: _draw_aurora(frame, t, intensity, color),
    'particles': lambda frame, t, intensity, color, sf: _draw_particles(frame, t, intensity, color),
}


# ============================================================
# PROCEDURAL ILLUSTRATIONS -- "drawing" system
#
# A small library of line-art subjects, each defined as a list of
# STROKES (point paths in 0-1 normalized space). When a beat's
# content clearly evokes one of these (GPT Call 1 sets
# beat["visual_subject"]), the renderer progressively "draws" the
# shape -- a pen-reveal animation -- then holds it with a gentle
# pulse for the rest of the beat. Sits centered, low-opacity,
# beneath the text layer.
# ============================================================

def _circle_pts(cx, cy, r, n=36, a0=0.0, a1=2*math.pi):
    return [(cx + r*math.cos(a0 + (a1-a0)*i/(n-1)),
             cy + r*math.sin(a0 + (a1-a0)*i/(n-1))) for i in range(n)]


def _ellipse_pts(cx, cy, rx, ry, n=36, a0=0.0, a1=2*math.pi, rot=0.0):
    pts = []
    for i in range(n):
        a = a0 + (a1-a0)*i/(n-1)
        x, y = rx*math.cos(a), ry*math.sin(a)
        xr = x*math.cos(rot) - y*math.sin(rot)
        yr = x*math.sin(rot) + y*math.cos(rot)
        pts.append((cx+xr, cy+yr))
    return pts


def _lumpy_circle_pts(cx, cy, r, bumps=5, bump_amt=0.15, n=48):
    pts = []
    for i in range(n):
        a = 2*math.pi*i/(n-1)
        rr = r * (1 + bump_amt*math.sin(bumps*a))
        pts.append((cx + rr*math.cos(a), cy + rr*math.sin(a)))
    return pts


def _build_illustration_shapes():
    shapes = {}

    # UPTREND: rising line + arrowhead, classic "stock going up" glyph
    line_pts = [(0.18, 0.78), (0.36, 0.58), (0.50, 0.66), (0.82, 0.24)]
    arrow_tip = line_pts[-1]
    ang = math.atan2(line_pts[-1][1] - line_pts[-2][1], line_pts[-1][0] - line_pts[-2][0])
    head_len, head_w = 0.09, 0.05
    back_x = arrow_tip[0] - head_len * math.cos(ang)
    back_y = arrow_tip[1] - head_len * math.sin(ang)
    perp = ang + math.pi / 2
    left  = (back_x + head_w * math.cos(perp), back_y + head_w * math.sin(perp))
    right = (back_x - head_w * math.cos(perp), back_y - head_w * math.sin(perp))
    shapes['uptrend'] = [
        line_pts,
        [left, arrow_tip, right],
    ]

    # DOWNTREND: mirror of uptrend
    dline_pts = [(0.18, 0.24), (0.36, 0.46), (0.50, 0.38), (0.82, 0.78)]
    dang = math.atan2(dline_pts[-1][1] - dline_pts[-2][1], dline_pts[-1][0] - dline_pts[-2][0])
    dback_x = dline_pts[-1][0] - head_len * math.cos(dang)
    dback_y = dline_pts[-1][1] - head_len * math.sin(dang)
    dperp = dang + math.pi / 2
    dleft  = (dback_x + head_w * math.cos(dperp), dback_y + head_w * math.sin(dperp))
    dright = (dback_x - head_w * math.cos(dperp), dback_y - head_w * math.sin(dperp))
    shapes['downtrend'] = [
        dline_pts,
        [dleft, dline_pts[-1], dright],
    ]

    # COIN STACK: 3 stacked ellipses (top view edge), side rim lines
    shapes['coin_stack'] = [
        _ellipse_pts(0.5, 0.70, 0.22, 0.07, n=28),
        _ellipse_pts(0.5, 0.58, 0.22, 0.07, n=28),
        _ellipse_pts(0.5, 0.46, 0.22, 0.07, n=28),
        [(0.28, 0.46), (0.28, 0.70)],
        [(0.72, 0.46), (0.72, 0.70)],
    ]

    # CLOCK: circle face + two hands (time value of money / countdown)
    hour_ang = math.radians(-60)
    min_ang  = math.radians(60)
    shapes['clock'] = [
        _circle_pts(0.5, 0.5, 0.30, n=40),
        [(0.5, 0.5), (0.5 + 0.14 * math.cos(hour_ang), 0.5 + 0.14 * math.sin(hour_ang))],
        [(0.5, 0.5), (0.5 + 0.22 * math.cos(min_ang),  0.5 + 0.22 * math.sin(min_ang))],
    ]

    # PERCENT: two circles + diagonal slash, the % glyph as line art
    diag = [(0.24, 0.76), (0.76, 0.24)]
    shapes['percent'] = [
        _circle_pts(0.30, 0.30, 0.10, n=22),
        _circle_pts(0.70, 0.70, 0.10, n=22),
        diag,
    ]

    # SCALE: balance/comparison glyph -- center post + tilted beam + two pans
    tilt = math.radians(-8)

    def _rot(px, py, cx, cy, a):
        dx, dy = px - cx, py - cy
        return (cx + dx * math.cos(a) - dy * math.sin(a),
                cy + dx * math.sin(a) + dy * math.cos(a))

    beam_l = _rot(0.22, 0.32, 0.5, 0.32, tilt)
    beam_r = _rot(0.78, 0.32, 0.5, 0.32, tilt)
    shapes['scale'] = [
        [(0.5, 0.20), (0.5, 0.82)],          # center post
        [(0.34, 0.82), (0.66, 0.82)],        # base
        [beam_l, beam_r],                     # beam
        _ellipse_pts(beam_l[0], beam_l[1] + 0.10, 0.09, 0.035, n=18),  # left pan
        _ellipse_pts(beam_r[0], beam_r[1] + 0.10, 0.09, 0.035, n=18),  # right pan
    ]

    return shapes



ILLUSTRATION_SHAPES = _build_illustration_shapes()

_SHAPE_LEN_CACHE: dict = {}


def _stroke_length(pts):
    return sum(math.hypot(pts[i+1][0]-pts[i][0], pts[i+1][1]-pts[i][1])
               for i in range(len(pts)-1))


def _draw_illustration(frame, t, beat_start, beat_dur, subject, color):
    """Progressively 'draw' the named shape (pen-reveal), then hold with a
    gentle pulse for the remainder of the beat. No-op if subject unknown."""
    strokes = ILLUSTRATION_SHAPES.get(subject)
    if not strokes:
        return frame

    h, w = frame.shape[:2]
    size = min(w, h) * 0.55
    cx, cy = w * 0.5, h * 0.5

    def to_px(p):
        return (int(cx + (p[0]-0.5)*size), int(cy + (p[1]-0.5)*size))

    if subject not in _SHAPE_LEN_CACHE:
        lens = [_stroke_length(s) for s in strokes]
        _SHAPE_LEN_CACHE[subject] = (lens, sum(lens) or 1.0)
    stroke_lens, total_len = _SHAPE_LEN_CACHE[subject]

    el_t = t - beat_start
    reveal_dur = min(1.2, max(0.4, beat_dur * 0.6))
    progress = max(0.0, min(1.0, el_t / reveal_dur))
    target = progress * total_len

    overlay = frame.copy()
    remaining = target
    for stroke, slen in zip(strokes, stroke_lens):
        if slen <= 1e-6:
            continue
        if remaining >= slen:
            pts = np.array([to_px(p) for p in stroke], dtype=np.int32)
            cv2.polylines(overlay, [pts], False, color, 2, lineType=cv2.LINE_AA)
            remaining -= slen
        elif remaining > 0:
            frac_len = remaining
            acc = 0.0
            pts_px = []
            for i in range(len(stroke)-1):
                seg_len = math.hypot(stroke[i+1][0]-stroke[i][0], stroke[i+1][1]-stroke[i][1])
                pts_px.append(to_px(stroke[i]))
                if acc + seg_len >= frac_len:
                    seg_frac = (frac_len - acc) / seg_len if seg_len > 0 else 0
                    ix = stroke[i][0] + (stroke[i+1][0]-stroke[i][0]) * seg_frac
                    iy = stroke[i][1] + (stroke[i+1][1]-stroke[i][1]) * seg_frac
                    pts_px.append(to_px((ix, iy)))
                    break
                acc += seg_len
            if len(pts_px) >= 2:
                pts = np.array(pts_px, dtype=np.int32)
                cv2.polylines(overlay, [pts], False, color, 2, lineType=cv2.LINE_AA)
            remaining = 0
        else:
            break

    if progress < 1.0:
        alpha = 0.35
    else:
        pulse = 0.5 + 0.5 * math.sin((el_t - reveal_dur) * 2.2)
        alpha = 0.28 + 0.12 * pulse

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def _build_subject_timeline(beats, total_duration, fps):
    """Per-frame (visual_subject, beat_start, beat_dur), discrete (not
    interpolated) so the illustration matches whichever beat is speaking."""
    n_frames = max(1, int(total_duration * fps))
    out = []
    bi = 0
    nb = len(beats)
    for f in range(n_frames):
        t = f / fps
        while bi + 1 < nb and float(beats[bi+1].get('start_time', 0.0)) <= t:
            bi += 1
        b = beats[bi] if nb else {}
        subj = (b.get('visual_subject') or 'none').strip().lower()
        bs = float(b.get('start_time', 0.0))
        be = float(b.get('end_time', bs + 1.0))
        out.append((subj, bs, max(be - bs, 0.1)))
    return out



def _build_intensity_curve(beats, total_duration, fps):
    """Per-frame intensity (1-10), linearly interpolated between beat
    midpoints and smoothed slightly so it drifts rather than jumps."""
    n_frames = max(1, int(total_duration * fps))
    control_t = []
    control_v = []
    for b in beats:
        s = float(b.get('start_time', 0.0))
        e = float(b.get('end_time', s + 1.0))
        mid = (s + e) / 2.0
        val = float(b.get('intensity', 5))
        control_t.append(mid)
        control_v.append(val)
    if not control_t:
        return [5.0] * n_frames

    curve = np.interp(
        [f / fps for f in range(n_frames)],
        control_t, control_v,
        left=control_v[0], right=control_v[-1]
    )
    # Light smoothing (moving average) so intensity drifts, not snaps
    if len(curve) > 5:
        kernel = np.ones(5) / 5
        curve = np.convolve(curve, kernel, mode='same')
    return curve.tolist()


def generate_procedural_background(beats: list, topic: str, total_duration: float,
                                     output_path: str, width: int = 1920,
                                     height: int = 1080, fps: int = 30) -> str:
    """Generate a fully procedural animated background video. No broll, no
    clip failures, no black fillers. One visual identity per topic, with
    intensity smoothly tracking the narration's emotional arc."""
    import cv2

    style_cfg = TOPIC_STYLES.get(topic, TOPIC_STYLES['default'])
    bg_color      = style_cfg['bg']
    accent        = style_cfg['accent']
    accent2       = style_cfg['accent2']
    style_names   = style_cfg['styles']

    n_frames = max(1, int(total_duration * fps))
    print(f"  🎨 Procedural background: topic={topic}, styles={style_names}, {n_frames} frames")

    intensity_curve = _build_intensity_curve(beats, total_duration, fps)
    subject_timeline = _build_subject_timeline(beats, total_duration, fps)
    n_with_subject = sum(1 for s, _, _ in subject_timeline if s != 'none' and s in ILLUSTRATION_SHAPES)
    if n_with_subject:
        print(f"  ✏️  Illustrations active on {n_with_subject}/{n_frames} frames")

    # Render at half-resolution then upscale -- the background is intentionally
    # soft/abstract (it sits behind text), so this is ~4x faster with no
    # visible quality loss after upscale + video compression.
    rw, rh = width // 2, height // 2
    starfield = _Starfield(rw, rh, n_stars=220, seed=hash(topic) & 0xffff)

    raw_path = output_path.replace('.mp4', '_bg_raw.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))

    # Precompute a subtle vignette mask once (multiplicative, darkens edges)
    yv, xv = np.mgrid[0:rh, 0:rw].astype(np.float32)
    cx, cy = rw / 2, rh / 2
    dist = np.sqrt(((xv - cx) / (rw / 2)) ** 2 + ((yv - cy) / (rh / 2)) ** 2)
    vignette = np.clip(1.0 - 0.35 * np.clip(dist - 0.5, 0, 1), 0.55, 1.0)
    vignette3 = vignette[:, :, None]

    for f in range(n_frames):
        t = f / fps
        intensity = intensity_curve[f]

        frame = np.full((rh, rw, 3), bg_color, dtype=np.uint8)

        # Primary style at full strength
        frame = _BG_DRAW_FNS[style_names[0]](frame, t, intensity, accent, starfield)
        # Secondary style as subtle accent layer
        if len(style_names) > 1:
            frame = _BG_DRAW_FNS[style_names[1]](frame, t, intensity * 0.7, accent2, starfield)

        # Content-aware illustration (drawing system) -- if this beat
        # evokes a known subject (planet, brain, dna, etc.), draw it with
        # a pen-reveal animation, centered, beneath the text layer
        subj, b_start, b_dur = subject_timeline[f]
        if subj in ILLUSTRATION_SHAPES:
            frame = _draw_illustration(frame, t, b_start, b_dur, subj, accent)

        # Vignette
        frame = (frame.astype(np.float32) * vignette3).astype(np.uint8)

        # Upscale to final output resolution
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        writer.write(frame)

        if f % (fps * 5) == 0:
            print(f"    {f}/{n_frames} frames...", end='\r')

    writer.release()
    print(f"    {n_frames}/{n_frames} frames... done")

    # Re-encode to H.264 yuv420p for downstream ffmpeg compatibility
    r = subprocess.run([
        'ffmpeg', '-y', '-i', raw_path,
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20',
        '-pix_fmt', 'yuv420p', '-r', str(fps), '-an', output_path
    ], capture_output=True)
    os.remove(raw_path)
    if r.returncode != 0:
        raise Exception(f"Background re-encode failed: {r.stderr.decode()[-200:]}")

    print(f"  ✅ Procedural background: {output_path}")
    return output_path



# ============================================================
# GPT CALL 1 -- STORY BEAT ANALYZER
# Sends Whisper segments (with timestamps) to GPT so it can
# produce accurate timing without word-level lookup
# ============================================================
def build_whisper_word_list(whisper_segments: list) -> list:
    """Flatten Whisper segments into an ordered word list with timestamps."""
    words = []
    for seg in whisper_segments:
        for we in seg.get('words', []):
            wc = we.get('word', '').upper().strip('.,!?;:\'"()[]- ')
            if not wc:
                continue
            words.append({
                'word':  wc,
                'start': float(we.get('start', 0.0)),
                'end':   float(we.get('end',   0.0)),
            })
    return words


def realign_beat_times(beats: list, whisper_word_list: list) -> list:
    """Recompute start_time/end_time for every beat by sequentially matching
    each beat's verbatim text against Whisper's word-level timestamps.

    GPT Call 1 is only given segment-level [start-end] brackets. When it splits
    one segment into multiple beats, it INVENTS the split-point timestamps --
    it has no word-level data. Those guessed boundaries cause every downstream
    word-matching step to look in the wrong time window, producing words that
    appear far too early or too late.

    Walk through the Whisper word list with a single forward-only pointer.
    Bounded lookahead handles normal drift; if that fails, fall back to an
    UNBOUNDED search from the global pointer so one bad match can't strand
    every subsequent beat. If a beat truly can't be matched, estimate its
    timing sequentially rather than keeping GPT's possibly-wild guess.
    """
    ptr = 0
    n = len(whisper_word_list)
    LOOKAHEAD = 20

    def norm(w):
        return w.upper().strip('.,!?;:\'"()[]- ')

    for beat in beats:
        text = (beat.get("text") or "").strip()
        words = [norm(w) for w in text.split() if norm(w)]

        if not words:
            continue

        start_idx = None
        end_idx = None
        local_ptr = ptr

        for w in words:
            found = None
            for look in range(local_ptr, min(local_ptr + LOOKAHEAD, n)):
                ww = whisper_word_list[look]['word']
                if ww == w or w in ww or ww in w:
                    found = look
                    break
            if found is None:
                for look in range(ptr, n):
                    ww = whisper_word_list[look]['word']
                    if ww == w or w in ww or ww in w:
                        found = look
                        break
            if found is None:
                continue
            if start_idx is None:
                start_idx = found
            end_idx = found
            local_ptr = found + 1

        if start_idx is not None and end_idx is not None:
            beat["start_time"] = whisper_word_list[start_idx]['start']
            beat["end_time"]   = whisper_word_list[end_idx]['end']
            ptr = end_idx + 1
        else:
            if ptr < n:
                est_start = whisper_word_list[ptr]['start']
            elif n > 0:
                est_start = whisper_word_list[-1]['end']
            else:
                est_start = float(beat.get("start_time", 0.0))
            est_dur = max(0.3, 0.35 * len(words))
            beat["start_time"] = est_start
            beat["end_time"]   = est_start + est_dur
            print(f"    ⚠ Could not align beat text '{text[:40]}' -- estimated timing")
            ptr = min(ptr + max(1, len(words)), n)

    for i in range(1, len(beats)):
        prev_end = float(beats[i-1].get("end_time", 0.0))
        cur_start = float(beats[i].get("start_time", 0.0))
        if cur_start < prev_end:
            beats[i]["start_time"] = prev_end
            if float(beats[i].get("end_time", 0.0)) <= prev_end:
                beats[i]["end_time"] = prev_end + 0.3

    return beats


def analyze_story_beats(transcript_text: str, whisper_segments: list,
                        topic_hint: str, total_duration: float) -> dict:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set.")

    print(f"  🎭 Call 1: Story beats ({len(transcript_text)} chars, {total_duration:.1f}s)...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Build timed transcript — each Whisper segment on its own line with [start - end]
    timed_lines = []
    for seg in whisper_segments:
        s = float(seg.get('start', 0))
        e = float(seg.get('end', 0))
        t = seg.get('text', '').strip()
        if t:
            timed_lines.append(f"[{s:.2f}s - {e:.2f}s] {t}")
    timed_transcript = "\n".join(timed_lines)

    system_prompt = f"""You are the producer for a finance/numbers explainer channel. Style: clear, dynamic, data-forward -- think "explain this number visually" rather than dramatic horror-story captions. Audience wants to actually understand the number, not just feel a jump-scare.
Total audio duration: {total_duration:.1f} seconds.

You will receive a transcript with EXACT timestamps from Whisper speech recognition.
Each line is formatted as: [start - end] spoken words

YOUR JOB: Segment the transcript into beats for visual data-explainer editing.

RULES:
- Use the Whisper timestamps directly -- they are accurate. Copy start_time and end_time from the brackets.
- Beat text MUST be copied VERBATIM from the transcript. Exact words, exact spelling. No paraphrasing.
- Keep beats 2-12 words -- natural spoken phrases or short clauses. Numbers/stats often need a slightly longer beat to land (e.g. "that's a four hundred percent increase").
- A single Whisper segment can become 1-3 beats if it contains multiple natural phrases.
- Cover the ENTIRE transcript -- every word must appear in some beat.
- "pause" beats only for clear silence gaps (>0.5s) between segments.

beat_type: "hook"|"setup"|"data_point"|"comparison"|"insight"|"warning"|"resolution"|"outro"
- "data_point": beat states a specific number/stat/dollar amount/percentage
- "comparison": beat contrasts two numbers or two things (X vs Y, before vs after)
- "warning": beat flags risk, loss, a downside, a mistake to avoid
- "insight": beat draws a conclusion or "here's what that means" takeaway

DATA EXTRACTION (critical -- this is what makes the visuals possible):
If a beat states an actual quantity, extract it into structured fields so the renderer can animate it precisely instead of guessing from prose:
- "has_data": true/false -- true only if this beat states a concrete number/stat/amount
- "data_value": the numeric value as a plain number (e.g. 400000, 4.5, 23). No currency symbols, no commas, no words.
- "data_unit": "percent"|"dollars"|"years"|"times"|"count"|"none" -- what the number represents
- "data_label": a SHORT (1-4 word) label for what the number IS, verbatim-ish from the beat (e.g. "AVERAGE RETURN", "COMPOUND INTEREST", "MARKET CAP")
- "data_direction": "up"|"down"|"neutral" -- only relevant for comparison/trend beats (does the number represent growth, loss, or neither)
- "compare_value": for "comparison" beats only -- the second number being compared against (numeric, same rules as data_value), else null

VISUAL_SUBJECT (icon drawing system): if this beat CLEARLY evokes one of these
concepts, set visual_subject to it -- the renderer draws it as line-art.
Options: "none"|"uptrend"|"downtrend"|"coin_stack"|"clock"|"percent"|"scale".
Be CONSERVATIVE -- most beats should be "none". Only set when the beat is
genuinely about that concept (e.g. "uptrend" for beats about growth/gains,
"downtrend" for losses/declines, "coin_stack" for savings/wealth/money itself,
"clock" for time-value-of-money/waiting/compounding over time, "percent" for
beats centrally about a rate/percentage, "scale" for risk/reward tradeoffs or
weighing two options). Never force a match.

Return ONLY valid JSON:
{{
  "topic": "markets|growth|warning|history|default",
  "music_mood": "driving|tense|optimistic|neutral|serious",
  "beats": [
    {{
      "beat_type": "hook|setup|data_point|comparison|insight|warning|resolution|outro",
      "text": "verbatim words from transcript",
      "start_time": 0.0,
      "end_time": 2.5,
      "intensity": 8,
      "has_data": true,
      "data_value": 400000,
      "data_unit": "dollars",
      "data_label": "RETIREMENT SAVINGS",
      "data_direction": "up",
      "compare_value": null,
      "visual_subject": "none|uptrend|downtrend|coin_stack|clock|percent|scale"
    }}
  ]
}}"""

    try:
        response = gpt4o_call(client,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Topic hint: {topic_hint}\n\nTimed transcript:\n{timed_transcript}\n\nSegment every line into beats. Use the timestamps shown. Copy text verbatim. Extract data fields wherever a beat states a real number."}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            timeout=90,
        )
        result = json.loads(response.choices[0].message.content)
        beats  = result.get('beats', [])
        print(f"  ✅ {len(beats)} beats, topic={result.get('topic')}")
        return result
    except Exception as e:
        print(f"  ❌ Story beats failed: {e}")
        raise


# ============================================================
# GPT CALL 2 -- RENDER DECISION GENERATOR
# ============================================================

def _build_batch_prompt(topic: str, batch: list) -> str:
    """Build the GPT Call 2 user prompt, annotating each beat with its real duration
    so GPT can set start_offset values that actually fit within the beat window."""
    annotated = []
    for b in batch:
        dur = round(float(b.get("end_time", 0)) - float(b.get("start_time", 0)), 2)
        entry = dict(b)
        entry["_duration_seconds"] = dur  # injected so GPT knows the real budget
        annotated.append(entry)
    return (
        f"Topic: {topic}\n\n"
        f"Beats ({len(batch)} total -- output exactly {len(batch)} scenes):\n"
        f"{json.dumps(annotated, indent=2)}\n\n"
        f"IMPORTANT: Each beat has a _duration_seconds field. "
        f"All start_offset values for elements in that beat MUST be less than _duration_seconds. "
        f"If _duration_seconds is 0.8s, valid start_offsets are 0.0, 0.2, 0.4 -- NOT 0.6 or higher (element would never show). "
        f"For beats shorter than 0.5s: use only 1 element with start_offset 0.0. "
        f"For beats 0.5-1.0s: max 2 elements, stagger by 0.2s. "
        f"For beats >1.0s: up to 3 elements, stagger by 0.3s. "
        f"Compose each scene to make the number/concept understandable. Vary layouts. White dominant; use number_counter for every real value."
    )

def generate_render_decisions(beats: list, topic: str) -> list:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set.")

    print(f"  🎨 Call 2: Scene compositions for {len(beats)} beats...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = f"""You are an elite short-form video editor for a finance/numbers explainer channel. You compose every frame like a motion designer -- choosing position, size, color, animation, and timing for each visual element. You are not picking from preset templates. You are designing each scene to make a NUMBER or CONCEPT visually understandable, not just dramatic.

Channel: a finance/numbers explainer. Audience wants to actually grasp the number -- a stat, a comparison, a growth curve, a cost. The aesthetic is CLEAN and DATA-FORWARD -- like a sharp explainer video, not a horror-trailer caption stack. Still punchy and fast-paced, just clearer.

=== FONT BEHAVIOR ===
The renderer uses Anton (ultra-condensed) as the primary font. This font is TALL and NARROW. All text is automatically rendered in ALL CAPS -- so write content in ALL CAPS.
SIZE RULES (strictly enforced by renderer):
- Single impact word or number: 120-160px max. Centered or slightly off-center.
- Sentence words (2+ words in a beat): 70-110px each. Cascade across canvas.
- number_counter elements: 140-220px (numbers need to be the visual anchor of a data beat).
- DO NOT go above 220px for any element -- it will be clamped.
- Fewer elements per scene is better. 2-4 elements max. Dense scenes are unreadable.

=== YOUR RENDERING ENGINE ===
Python OpenCV + Pillow on a {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} canvas.

For each beat, you output a SCENE -- a list of ELEMENTS placed and animated however you want. Each beat in the input includes data fields (has_data, data_value, data_unit, data_label, data_direction, compare_value) extracted from the transcript. USE THESE FIELDS when has_data is true -- they're the actual numbers you should visualize, not something to re-derive from the text.

=== ELEMENT TYPES ===

TEXT element:
{{
  "type": "text",
  "content": "WORD",
  "x": 0.5, "y": 0.4,
  "anchor": "center",              // "center" | "left" | "right"
  "size": 120,
  "color": "#FFFFFF",
  "weight": "black",               // "regular" | "bold" | "black"
  "outline": 4,
  "anim": "fade_in",
  "start_offset": 0.0,
  "duration": null,
  "anim_duration": 0.15,
  "effect": "none"
}}

NUMBER_COUNTER element (NEW -- use this for any beat with has_data=true):
{{
  "type": "number_counter",
  "target_value": 400000,          // copy from the beat's data_value
  "prefix": "$",                   // "$" for dollars, "" otherwise
  "suffix": "%",                   // "%" for percent, "" otherwise -- never put both prefix and suffix unless the unit genuinely needs it
  "decimals": 0,                   // 0 for whole numbers, 1-2 for precise stats
  "x": 0.5, "y": 0.42,
  "anchor": "center",
  "size": 180,
  "color": "#FFFFFF",
  "weight": "black",
  "outline": 5,
  "count_from": 0,                 // where the count-up animation starts (usually 0, or a lower number for a "before" value)
  "count_duration": 0.8,           // seconds for the number to animate from count_from to target_value
  "start_offset": 0.0,
  "duration": null
}}
The renderer animates this counting UP (or down) from count_from to target_value over count_duration seconds, formatted with prefix/suffix/decimals/comma separators automatically. This is your primary tool for making a statistic feel alive instead of just appearing.

GRID element (NEW -- use for beats about scale/quantity/repetition, e.g. "thousands of dollars" or visualizing a large count):
{{
  "type": "grid",
  "glyph": "0",                    // the single character or short string repeated in the grid
  "rows": 4,
  "cols": 14,
  "cell_size": 60,                 // pixel size of each glyph
  "color": "#FBC02D",
  "x": 0.5, "y": 0.55,             // CENTER of the whole grid
  "anim": "fill_sequential",        // "fill_sequential" reveals cell by cell, "fade_in" reveals all at once
  "fill_duration": 1.2,             // total seconds for fill_sequential to complete
  "start_offset": 0.0,
  "duration": null
}}
Use this sparingly -- it's for the rare beat where "a LOT of something" is the point (e.g. visualizing thousands as a wall of repeated digits/symbols). Keep rows*cols under 80 total cells or it gets visually noisy.

LINE element (for dividers, underlines, comparison axes):
{{
  "type": "line",
  "x1": 0.3, "y1": 0.5, "x2": 0.7, "y2": 0.5,
  "thickness": 8,
  "color": "#FFFFFF",
  "anim": "draw_horizontal",
  "start_offset": 0.2,
  "duration": null,
  "anim_duration": 0.3
}}

RECT element (for boxes, comparison bars, highlight bars):
{{
  "type": "rect",
  "x": 0.4, "y": 0.5, "w": 0.2, "h": 0.1,
  "color": "#FBC02D",
  "filled": true,
  "thickness": 4,
  "anim": "fade_in",
  "start_offset": 0.0,
  "duration": null
}}
For a COMPARISON beat (data_direction or compare_value set): two RECT bars side by side, heights proportional to the two values (taller bar = bigger number), is a strong visual. Pair with a TEXT label under each.

CIRCLE element:
{{
  "type": "circle",
  "x": 0.5, "y": 0.5, "radius": 0.05,
  "color": "#FFFFFF",
  "filled": false,
  "thickness": 4,
  "anim": "fade_in",
  "start_offset": 0.0,
  "duration": null
}}

=== ANIMATIONS ===
- "none": appears instantly
- "fade_in": opacity 0→100% over anim_duration
- "slide_in_left" / "slide_in_right" / "slide_in_top" / "slide_in_bottom"
- "scale_in": starts at 1.3x scale and snaps to 1.0x (punch effect)
- "snap": appears instantly with a 1-frame white flash
- "draw_horizontal": (lines only) draws progressively left-to-right
- "fill_sequential": (grid only) reveals cells one at a time

=== EFFECTS (applied during display, not just entrance) ===
- "none": static
- "flicker": rapid on/off blinking for first 0.3s (for warning/shock numbers)
- "shake": position jitters slightly (for impact)
- "glow": adds soft colored glow halo around element

=== HOW TO COMPOSE SCENES ===

ELEMENT LIMIT: Maximum 4 elements per scene. Less is more. 2-3 elements is ideal.

STAGGER ALL ELEMENTS: start_offset must be less than the beat's _duration_seconds or the element will NEVER appear.
- Beat <0.5s: 1 element only, start_offset 0.0
- Beat 0.5-1.0s: max 2 elements, offsets 0.0 and 0.3
- Beat >1.0s: up to 3 elements, offsets 0.0 / 0.35 / 0.7
NEVER set start_offset >= _duration_seconds.

=== POSITIONING GRID (1920x1080 canvas) ===
Safe zone: x: 0.08-0.92, y: 0.12-0.88.

Three vertical bands:
- UPPER band:  y: 0.20-0.35
- CENTER band: y: 0.42-0.58
- LOWER band:  y: 0.65-0.80

=== BEAT-TYPE -> COMPOSITION MAPPING ===

For a "data_point" beat (has_data=true, no compare_value): ONE number_counter element in CENTER band showing the actual value (use prefix/suffix from data_unit), plus ONE text element in LOWER band with the data_label. 2 elements. This is your bread-and-butter scene type.

For a "comparison" beat (has_data=true AND compare_value set): two RECT bars side by side (proportional heights, taller = larger value) OR two number_counter elements side by side (x: 0.28 and x: 0.72), each with a short text label beneath. 3-4 elements.

For a "warning" beat: text in CENTER band, color #E85D4A or similar warning-red (still pass through _ensure_bright_color), "shake" or "flicker" effect. 1-2 elements.

For an "insight"/"resolution" beat (the takeaway, usually no raw number): SPOKEN SENTENCE treatment -- pick the 1-2 most important words, place across UPPER/CENTER bands, 90-130px, fade_in or slide_in.

For a "hook" or "setup" beat: 1-2 elements, CENTER or UPPER band, sets up the number that's about to land -- don't put the actual data_value here unless the beat itself states it.

For a beat with visual_subject set (uptrend/downtrend/coin_stack/clock/percent/scale): the renderer draws that icon automatically in the background -- you do NOT need to add an element for it. Just compose the text/number elements as normal; they'll appear on top of the icon.

VARY bands across consecutive beats so the video doesn't feel static.

=== HARD RULES ===
1. Output exactly {len(beats)} scenes, one per beat, in order.
2. Every "content" in TEXT elements must be ALL CAPS and use words VERBATIM from the beat text.
3. For pause beats: output {{"elements": []}} (empty scene).
4. start_offset values must fit within the beat duration. STAGGER them -- never all 0.0.
5. x, y values are 0.0-1.0. NEVER use percentages or pixels.
6. MAX 4 elements per scene.
7. Never repeat the same word twice in one scene.
8. Content must be a SINGLE WORD or SHORT PHRASE -- never a full sentence in one element.
9. When has_data is true, ALWAYS use a number_counter element for the actual value -- never spell the number out as a TEXT word (e.g. use number_counter with target_value=400000, not a text element saying "FOUR HUNDRED THOUSAND").

=== COLOR DISCIPLINE ===
White (#FFFFFF) is your dominant color. Market green (#78E6AA) for positive/growth numbers. Warning red (#E85D4A) for losses/risk. Gold (#FFD782) for ONE key highlight per scene maximum.

Return ONLY valid JSON:
{{
  "scenes": [
    {{
      "beat_index": <int>,
      "beat_type": "<hook|setup|data_point|comparison|insight|warning|resolution|outro>",
      "elements": [
        // list of element objects as specified above
      ]
    }}
    // ... exactly {len(beats)} scenes
  ]
}}"""

    # Batch to stay safely under GPT-4o's 16384 output token limit.
    # 6 beats per batch ~ 5-7k tokens output, well within limit even with long scenes.
    BATCH_SIZE = 3
    all_scenes = []
    batches = [beats[i:i+BATCH_SIZE] for i in range(0, len(beats), BATCH_SIZE)]

    def _run_batch(batch_idx, batch):
        start_beat = batch_idx * BATCH_SIZE
        print(f"  🎨 Batch {batch_idx+1}/{len(batches)}: beats {start_beat}-{start_beat+len(batch)-1}...")
        response = _call_with_retry(lambda: gpt4o_call(client,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _build_batch_prompt(topic, batch)}
            ],
            response_format={"type": "json_object"},
            temperature=0.85,
            max_tokens=8000,
            timeout=120,
        ), label=f"Call 2 batch {batch_idx+1}")
        result = json.loads(response.choices[0].message.content)
        batch_scenes = result.get('scenes', [])
        if len(batch_scenes) > len(batch):
            # GPT split a beat into 2 scenes -- if we keep the extra,
            # EVERY subsequent scene shifts by +1 relative to its beat,
            # causing cascading "hallucination" drops in the validator.
            # Trim to guarantee scenes[i] always corresponds to beats[i].
            print(f"  ⚠️  Batch {batch_idx+1}: expected {len(batch)} scenes, got {len(batch_scenes)} -- trimming extras")
            batch_scenes = batch_scenes[:len(batch)]
        elif len(batch_scenes) < len(batch):
            print(f"  ⚠️  Batch {batch_idx+1}: expected {len(batch)} scenes, got {len(batch_scenes)} -- padding with empty scenes")
            while len(batch_scenes) < len(batch):
                batch_scenes.append({"beat_index": start_beat + len(batch_scenes), "elements": []})
        print(f"  ✅ Batch {batch_idx+1} done: {len(batch_scenes)} scenes")
        return batch_idx, batch_scenes

    # Concurrency is bounded by the shared _GPT4O_CONCURRENCY semaphore
    # (process-wide, shared with Call 3), not by this pool size -- the
    # pool just needs enough workers to keep the semaphore saturated.
    results = [None] * len(batches)
    MAX_WORKERS = 3
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run_batch, i, b): i for i, b in enumerate(batches)}
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                idx, batch_scenes = future.result()
                results[idx] = batch_scenes
            except Exception as e:
                print(f"  ❌ Batch {batch_idx+1} failed: {e}")
                raise

    for batch_scenes in results:
        all_scenes.extend(batch_scenes)

    print(f"  ✅ {len(all_scenes)} total scenes composed")
    return all_scenes


# ============================================================
# GPT CALL 3 -- FREEFORM MOTION DESIGNER + CODER
#
# Captions are handled by YouTube itself in this niche, so on-screen
# TEXT is now the exception (a rare emphasis label), not the default.
# The primary visual language is bespoke motion graphics: an object
# (dot, shape outline, label pill, card) moving along a path the
# Designer/Coder choose freely, timed to the beat's real audio window.
#
# GPT chooses WHAT happens and WHEN (full creative freedom over
# composition, path shape, timing, easing CHOICE). The actual curve
# math, easing FUNCTIONS, and pixel rendering stay deterministic on
# the renderer side -- GPT never emits raw coordinate math, only
# parameters into primitives that can't NaN/crash.
#
# No automated vision-QA loop. A scene that renders but looks wrong
# is caught by the person reviewing output, not by another GPT call.
# A scene that's malformed enough to fail validation just gets that
# one object dropped (logged), never the whole video.
# ============================================================

FREEFORM_EASINGS = {"linear", "ease_out", "ease_in_out", "spring"}
FREEFORM_OBJECT_TYPES = {"dot", "path_shape", "label_pill", "card"}

def _build_freeform_batch_prompt(batch: list) -> str:
    annotated = []
    for b in batch:
        dur = round(float(b.get("end_time", 0)) - float(b.get("start_time", 0)), 2)
        entry = dict(b)
        entry["_duration_seconds"] = dur
        annotated.append(entry)
    return (
        f"Beats ({len(batch)} total -- output exactly {len(batch)} freeform scenes, one per beat, in order):\n"
        f"{json.dumps(annotated, indent=2)}\n\n"
        f"Each beat's _duration_seconds is its real spoken-audio window. All start_offset + duration "
        f"for objects in that beat MUST fit inside _duration_seconds -- an object that starts after the "
        f"beat ends will never be seen. "
        f"Design a small bespoke motion graphic for each beat -- a metaphor or motion idea that matches "
        f"what's being said, not a generic template repeated every beat. Vary the object type, path shape, "
        f"and easing across consecutive beats so the video doesn't feel like the same animation looping."
    )

def generate_freeform_scenes(beats: list, topic: str) -> list:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY not set.")

    print(f"  🎬 Call 3: Freeform motion design for {len(beats)} beats...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = f"""You are a motion graphics designer AND the engineer who builds it, in one step. You invent a small bespoke animated moment for each beat of narration -- the kind of thing seen in sharp finance/explainer YouTube videos: a dot tracing an arc while a label springs in, a card tilting into view, a shape drawing itself stroke by stroke. You choose the concept. You choose the exact motion. You do NOT write raw math or code -- you express your choice as parameters into a small set of primitives the renderer already knows how to draw deterministically, so nothing you output can produce broken coordinates or crash the renderer.

Captions are handled by YouTube itself -- you are NOT writing on-screen text. Your job is the animation, full stop. Do not include any text/caption elements here; that's a separate system.

=== YOUR CANVAS ===
{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}. All positions are 0.0-1.0 normalized (0,0 = top-left, 1,1 = bottom-right). NEVER use pixels.

=== OBJECT TYPES (the only primitives you may use) ===

DOT -- a small circle that travels along a path. Good for: tracing an arc, marking a moving point on a curve, a "play head" style marker.
{{
  "type": "dot",
  "path": [[0.2, 0.7], [0.5, 0.3], [0.8, 0.2]],   // 2-5 points, the dot moves through them in order
  "radius": 0.012,                                 // 0.005-0.03 typical
  "color": "#FFD782",
  "start_offset": 0.0,
  "duration": 1.0,                                 // seconds to travel the whole path
  "easing": "ease_out"
}}

PATH_SHAPE -- a line/curve that draws itself progressively (pen-reveal), then can hold or fade. Good for: an arc, a rising/falling trend gesture, an abstract underline-the-point stroke.
{{
  "type": "path_shape",
  "path": [[0.15, 0.75], [0.5, 0.25], [0.85, 0.2]], // 2-6 points defining the curve (rendered as a smooth spline)
  "color": "#78E6AA",
  "thickness": 4,
  "start_offset": 0.0,
  "duration": 0.8,                                  // seconds for the draw-on to complete
  "easing": "ease_out",
  "hold": true                                       // true = stays visible after drawing, false = fades out after drawing completes
}}

LABEL_PILL -- a small rounded-rect badge with a short label (use ONLY for a number/value/very short tag, e.g. "13HR", "+started", "22%" -- never a sentence). Springs or fades into a fixed position.
{{
  "type": "label_pill",
  "content": "13HR",
  "x": 0.62, "y": 0.28,           // center position, normalized
  "color": "#3A2E0A",             // pill background
  "text_color": "#FFD782",
  "start_offset": 0.4,
  "anim_duration": 0.3,
  "easing": "spring"               // springs in with overshoot-and-settle, matching the screenshots' bouncy feel
}}

CARD -- a rounded rect "panel" that can tilt in pseudo-3D and hold other objects conceptually behind/in front of it (you don't nest objects inside it -- just place a card as its own background panel for a moment).
{{
  "type": "card",
  "x": 0.5, "y": 0.5, "w": 0.32, "h": 0.22,   // x,y is CENTER, w,h are normalized size
  "color": "#1A1A1A",
  "tilt_deg": 8,                                // -15 to 15, perspective tilt amount
  "start_offset": 0.0,
  "anim_duration": 0.4,
  "easing": "ease_out"
}}

=== EASING (choose the feel, the renderer computes the actual curve) ===
- "linear": constant speed, no easing personality -- use sparingly, feels mechanical
- "ease_out": fast start, settles smoothly -- good default for most motion
- "ease_in_out": smooth start and end -- good for slow contemplative beats
- "spring": overshoots past the target then settles back -- bouncy, energetic. Use for label_pill reveals and anything that should feel like it "pops" in. This is what makes a reveal feel alive instead of a flat fade.

=== HOW TO DESIGN A BEAT ===
1. Read the beat text and its data fields. Ask: what's the simplest VISUAL IDEA that represents this moment? (e.g. "13 hours of work" -> a dot traveling along a rising arc with a "13HR" pill appearing at the peak. "the market went up" -> a path_shape drawing an upward stroke. "two options, pick one" -> two cards side by side.)
2. Use 1-3 objects per scene. More than 3 gets visually noisy at this scale.
3. Stagger start_offset so objects don't all begin at once -- let the dot/path move FIRST, then the label_pill appears once the motion lands somewhere meaningful (e.g. pill appears when the dot reaches the path's end).
4. For "data_point" or "comparison" beats with a real number, that number can be a label_pill content (e.g. "$400K") riding alongside a path_shape/dot -- this is your strongest combo for stat beats.
5. For "warning" beats: tighter, faster motion (shorter durations, ease_out) reads as more urgent than slow ease_in_out.
6. For pause beats: output {{"freeform_objects": []}}.
7. Vary path shapes across consecutive beats -- don't repeat the same arc shape every time. Mix rising arcs, falling arcs, S-curves, straight diagonals, simple horizontal sweeps.

=== HARD RULES ===
1. Output exactly {len(beats)} scenes, one per beat, in order.
2. All path point coordinates are 0.0-1.0. NEVER pixels, NEVER values outside 0-1.
3. start_offset + duration (or + anim_duration) MUST be less than the beat's _duration_seconds.
4. label_pill content is SHORT (1-4 characters/words) -- a number, a tag, never a sentence. No other object type carries text.
5. Max 3 objects per scene.
6. easing must be exactly one of: linear, ease_out, ease_in_out, spring.

Return ONLY valid JSON:
{{
  "scenes": [
    {{
      "beat_index": <int>,
      "concept": "<one short phrase describing the visual idea, for human review only -- not rendered>",
      "freeform_objects": [
        // 0-3 objects as specified above
      ]
    }}
    // ... exactly {len(beats)} scenes
  ]
}}"""

    BATCH_SIZE = 3
    all_scenes = []
    batches = [beats[i:i+BATCH_SIZE] for i in range(0, len(beats), BATCH_SIZE)]

    def _run_batch(batch_idx, batch):
        start_beat = batch_idx * BATCH_SIZE
        print(f"  🎬 Freeform batch {batch_idx+1}/{len(batches)}: beats {start_beat}-{start_beat+len(batch)-1}...")
        response = _call_with_retry(lambda: gpt4o_call(client,
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _build_freeform_batch_prompt(batch)}
            ],
            response_format={"type": "json_object"},
            temperature=0.9,
            max_tokens=6000,
            timeout=120,
        ), label=f"Call 3 batch {batch_idx+1}")
        result = json.loads(response.choices[0].message.content)
        batch_scenes = result.get('scenes', [])
        if len(batch_scenes) > len(batch):
            batch_scenes = batch_scenes[:len(batch)]
        elif len(batch_scenes) < len(batch):
            while len(batch_scenes) < len(batch):
                batch_scenes.append({"beat_index": start_beat + len(batch_scenes),
                                      "concept": "", "freeform_objects": []})
        print(f"  ✅ Freeform batch {batch_idx+1} done: {len(batch_scenes)} scenes")
        return batch_idx, batch_scenes

    # Same shared semaphore as Call 2 -- the two calls run back-to-back
    # in the pipeline but the TPM window doesn't care about function
    # boundaries, only elapsed time, so they still need to share it.
    results = [None] * len(batches)
    MAX_WORKERS = 3
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run_batch, i, b): i for i, b in enumerate(batches)}
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                idx, batch_scenes = future.result()
                results[idx] = batch_scenes
            except Exception as e:
                print(f"  ❌ Freeform batch {batch_idx+1} failed: {e}")
                raise

    for batch_scenes in results:
        all_scenes.extend(batch_scenes)

    print(f"  ✅ {len(all_scenes)} total freeform scenes composed")
    return all_scenes


def validate_freeform_scenes(scenes: list, beats: list) -> list:
    """Crash guard, not an aesthetics gate. Every numeric param is clamped
    or the object is dropped outright if it can't be made safe. One bad
    object never takes down the rest of its scene or the video."""
    print(f"  🔍 Validating {len(scenes)} freeform scenes...")
    dropped = 0

    for scene_pos, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            scenes[scene_pos] = {"beat_index": scene_pos, "concept": "", "freeform_objects": []}
            continue
        scene["beat_index"] = scene_pos
        beat_dur = 1.0
        if scene_pos < len(beats):
            beat_dur = max(0.1, float(beats[scene_pos].get("end_time", 1.0)) -
                                float(beats[scene_pos].get("start_time", 0.0)))

        objects = scene.get("freeform_objects", [])
        if not isinstance(objects, list):
            scene["freeform_objects"] = []
            continue

        cleaned = []
        for obj in objects:
            if not isinstance(obj, dict):
                dropped += 1
                continue
            otype = obj.get("type")
            if otype not in FREEFORM_OBJECT_TYPES:
                dropped += 1
                continue

            try:
                start_offset = max(0.0, float(obj.get("start_offset", 0.0)))
                dur_key = "duration" if "duration" in obj else "anim_duration"
                duration = max(0.05, float(obj.get(dur_key, 0.4)))
                if start_offset >= beat_dur:
                    dropped += 1
                    continue
                # Clamp duration so it never extends past the beat window
                duration = min(duration, max(0.05, beat_dur - start_offset))
                obj[dur_key] = round(duration, 3)
                obj["start_offset"] = round(start_offset, 3)
            except (TypeError, ValueError):
                dropped += 1
                continue

            easing = obj.get("easing", "ease_out")
            if easing not in FREEFORM_EASINGS:
                easing = "ease_out"
            obj["easing"] = easing

            if otype in ("dot", "path_shape"):
                path = obj.get("path", [])
                if not isinstance(path, list) or len(path) < 2:
                    dropped += 1
                    continue
                safe_path = []
                ok = True
                for pt in path[:6]:
                    try:
                        x, y = float(pt[0]), float(pt[1])
                        safe_path.append([max(0.0, min(1.0, x)), max(0.0, min(1.0, y))])
                    except (TypeError, ValueError, IndexError):
                        ok = False
                        break
                if not ok or len(safe_path) < 2:
                    dropped += 1
                    continue
                obj["path"] = safe_path
                obj.setdefault("color", "#FFFFFF")
                if otype == "dot":
                    obj["radius"] = max(0.004, min(float(obj.get("radius", 0.012)), 0.05))
                else:
                    obj["thickness"] = max(1, min(int(obj.get("thickness", 4)), 12))
                    obj["hold"] = bool(obj.get("hold", True))

            elif otype == "label_pill":
                content = str(obj.get("content", "")).strip()
                if not content or len(content) > 12:
                    dropped += 1
                    continue
                obj["content"] = content
                obj["x"] = max(0.05, min(0.95, float(obj.get("x", 0.5))))
                obj["y"] = max(0.05, min(0.95, float(obj.get("y", 0.5))))
                obj.setdefault("color", "#1A1A1A")
                obj.setdefault("text_color", "#FFD782")

            elif otype == "card":
                obj["x"] = max(0.1, min(0.9, float(obj.get("x", 0.5))))
                obj["y"] = max(0.1, min(0.9, float(obj.get("y", 0.5))))
                obj["w"] = max(0.05, min(0.9, float(obj.get("w", 0.3))))
                obj["h"] = max(0.05, min(0.9, float(obj.get("h", 0.2))))
                obj["tilt_deg"] = max(-15.0, min(15.0, float(obj.get("tilt_deg", 0.0))))
                obj.setdefault("color", "#1A1A1A")

            cleaned.append(obj)

        if len(cleaned) > 3:
            cleaned = cleaned[:3]

        scene["freeform_objects"] = cleaned

    print(f"  ✅ Validated {len(scenes)} freeform scenes, dropped {dropped} unsafe objects")
    return scenes


# ============================================================
# VALIDATION PASS - validates scene compositions
# ============================================================
def _ensure_bright_color(hex_color: str, min_luminance: float = 130.0) -> str:
    """If a color is too dark to read against the near-black procedural
    background, brighten it. White and the brand yellow (#FBC02D) pass
    through unchanged -- they're already bright. Dark/muted colors get
    scaled up toward white while preserving hue, so 'dark grey' becomes
    'light grey' rather than just snapping to pure white for everything."""
    try:
        h = hex_color.strip().lstrip('#')
        if len(h) != 6:
            return "#FFFFFF"
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except Exception:
        return "#FFFFFF"

    # Perceived luminance (standard weights)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    if luminance >= min_luminance:
        return hex_color

    if luminance < 1.0:
        # Pure/near black -- no hue to preserve, just go white
        return "#FFFFFF"

    # Scale up toward white, preserving hue/ratio
    scale = min_luminance / luminance
    r = min(255, int(r * scale))
    g = min(255, int(g * scale))
    b = min(255, int(b * scale))
    return f"#{r:02X}{g:02X}{b:02X}"


def validate_decisions(scenes: list, beats: list) -> list:
    print(f"  🔍 Validating {len(scenes)} scenes...")
    fixed = 0

    for scene_pos, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            scenes[scene_pos] = {"beat_index": scene_pos, "elements": []}
            continue

        # Use enumeration position -- ignore GPT's beat_index
        scene["beat_index"] = scene_pos
        beat = beats[scene_pos] if scene_pos < len(beats) else {}
        beat_text = beat.get("text", "").strip().lower()
        beat_words = set()
        for w in beat_text.split():
            beat_words.add(w.strip('.,!?;:\'"()[]- '))

        elements = scene.get("elements", [])
        if not isinstance(elements, list):
            scene["elements"] = []
            continue

        cleaned = []
        for el in elements:
            if not isinstance(el, dict):
                continue
            etype = el.get("type", "text")

            # Validate TEXT content against beat text
            if etype == "text":
                content = (el.get("content") or "").strip()
                if not content:
                    continue
                # Strip punctuation, lowercase for comparison
                check_words = [w.strip('.,!?;:\'"()[]- ').lower()
                               for w in content.split()
                               if len(w.strip('.,!?;:\'"()[]- ')) > 2]
                if check_words and beat_words:
                    matches = sum(1 for w in check_words if w in beat_words)
                    if matches == 0 and len(check_words) > 0:
                        # Pure hallucination -- skip element
                        print(f"  ⚠ Scene {scene_pos}: dropped hallucinated text '{content[:30]}'")
                        fixed += 1
                        continue

                # Default text properties
                el.setdefault("x", 0.5)
                el.setdefault("y", 0.5)
                el.setdefault("anchor", "center")
                el.setdefault("size", 90)
                el.setdefault("color", "#FFFFFF")
                el.setdefault("weight", "black")
                el.setdefault("outline", 4)
                el.setdefault("anim", "fade_in")
                el.setdefault("start_offset", 0.0)
                el.setdefault("duration", None)
                el.setdefault("anim_duration", 0.15)
                el.setdefault("effect", "none")

                # Enforce minimum brightness -- against the dark procedural
                # background, a muted/dark fill color (e.g. dark grey, olive)
                # combined with a black outline becomes nearly invisible.
                el["color"] = _ensure_bright_color(el["color"])

            elif etype == "line":
                el.setdefault("x1", 0.3)
                el.setdefault("y1", 0.5)
                el.setdefault("x2", 0.7)
                el.setdefault("y2", 0.5)
                el.setdefault("thickness", 6)
                el.setdefault("color", "#FFFFFF")
                el.setdefault("anim", "draw_horizontal")
                el.setdefault("start_offset", 0.0)
                el.setdefault("duration", None)
                el.setdefault("anim_duration", 0.3)

            elif etype == "rect":
                el.setdefault("x", 0.4)
                el.setdefault("y", 0.4)
                el.setdefault("w", 0.2)
                el.setdefault("h", 0.1)
                el.setdefault("color", "#FFFFFF")
                el.setdefault("filled", True)
                el.setdefault("thickness", 3)
                el.setdefault("anim", "fade_in")
                el.setdefault("start_offset", 0.0)
                el.setdefault("duration", None)
                el.setdefault("anim_duration", 0.2)

            elif etype == "circle":
                el.setdefault("x", 0.5)
                el.setdefault("y", 0.5)
                el.setdefault("radius", 0.05)
                el.setdefault("color", "#FFFFFF")
                el.setdefault("filled", False)
                el.setdefault("thickness", 4)
                el.setdefault("anim", "fade_in")
                el.setdefault("start_offset", 0.0)
                el.setdefault("duration", None)
                el.setdefault("anim_duration", 0.2)

            elif etype == "number_counter":
                # target_value is required and must be numeric -- if GPT
                # sent something unparseable, drop the element entirely
                # rather than render garbage.
                try:
                    el["target_value"] = float(el.get("target_value", 0))
                except (TypeError, ValueError):
                    print(f"  ⚠ Scene {scene_pos}: dropped number_counter with bad target_value")
                    fixed += 1
                    continue
                el.setdefault("prefix", "")
                el.setdefault("suffix", "")
                el.setdefault("decimals", 0)
                el.setdefault("x", 0.5)
                el.setdefault("y", 0.42)
                el.setdefault("anchor", "center")
                el.setdefault("size", 180)
                el.setdefault("color", "#FFFFFF")
                el.setdefault("weight", "black")
                el.setdefault("outline", 5)
                el.setdefault("count_from", 0)
                el.setdefault("count_duration", 0.8)
                el.setdefault("start_offset", 0.0)
                el.setdefault("duration", None)
                el["color"] = _ensure_bright_color(el["color"])
                el["size"] = max(60, min(int(el.get("size", 180)), 220))

            elif etype == "grid":
                glyph = str(el.get("glyph", "0")).strip()
                if not glyph:
                    print(f"  ⚠ Scene {scene_pos}: dropped grid with empty glyph")
                    fixed += 1
                    continue
                el["glyph"] = glyph[:3]  # keep cells readable -- short glyphs only
                el.setdefault("rows", 4)
                el.setdefault("cols", 10)
                el.setdefault("cell_size", 60)
                el.setdefault("color", "#FBC02D")
                el.setdefault("x", 0.5)
                el.setdefault("y", 0.55)
                el.setdefault("anim", "fill_sequential")
                el.setdefault("fill_duration", 1.2)
                el.setdefault("start_offset", 0.0)
                el.setdefault("duration", None)
                el["color"] = _ensure_bright_color(el["color"])
                # Cap total cells so the grid never becomes visual noise
                rows = max(1, min(int(el.get("rows", 4)), 10))
                cols = max(1, min(int(el.get("cols", 10)), 16))
                while rows * cols > 80:
                    if cols > rows:
                        cols -= 1
                    else:
                        rows -= 1
                el["rows"], el["cols"] = rows, cols

            else:
                # Unknown type, skip
                continue

            # Clamp coordinates to safe range
            for k in ("x", "y", "x1", "y1", "x2", "y2", "w", "h", "radius"):
                if k in el and isinstance(el[k], (int, float)):
                    el[k] = max(0.0, min(1.0, float(el[k])))

            cleaned.append(el)

        # HARD CAP: max 4 elements per scene
        if len(cleaned) > 4:
            print(f"  ⚠ Scene {scene_pos}: trimmed {len(cleaned)} elements to 4")
            cleaned = cleaned[:4]
            fixed += 1

        # AUTO-STAGGER: if all text elements have start_offset 0.0, spread them out
        text_els = [e for e in cleaned if e.get("type", "text") == "text" and isinstance(e.get("content", ""), str)]
        all_zero = all(float(e.get("start_offset", 0.0)) < 0.05 for e in text_els)
        if all_zero and len(text_els) > 1:
            beat_dur = max(0.5, float(beats[scene_pos].get("end_time", 2.0)) - float(beats[scene_pos].get("start_time", 0.0))) if scene_pos < len(beats) else 1.0
            step = min(0.25, beat_dur / (len(text_els) + 1))
            for i, e in enumerate(text_els):
                e["start_offset"] = round(i * step, 2)
            fixed += 1

        scene["elements"] = cleaned

    print(f"  ✅ Validated {len(scenes)} scenes, fixed {fixed} issues")
    return scenes



# ============================================================
# SCENE-BASED RENDERER v5
# GPT outputs scene compositions (lists of elements).
# This renderer executes any combination of text/line/rect/circle
# with per-element animation and timing.
# ============================================================
def render_text_overlay_opencv(video_path: str, scenes: list, beats: list,
                               whisper_segments: list, output_path: str,
                               freeform_scenes: list = None):
    print(f"🎨 Scene renderer v5: {len(scenes)} scenes...")
    freeform_scenes = freeform_scenes or []

    try:
        import cv2
        from PIL import Image, ImageDraw, ImageFont
        print(f"  ✓ OpenCV {cv2.__version__} + Pillow ready")
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        subprocess.run(['ffmpeg', '-y', '-i', video_path, '-c', 'copy', output_path],
                       check=True, capture_output=True)
        return

    if not os.path.exists(video_path):
        raise Exception(f"Video not found: {video_path}")

    # ── basic helpers ──────────────────────────────────────────────
    def load_pil_font(path, size, weight="black"):
        # weight selects between Black / ExtraBold / Bold
        try:
            if weight == "regular":
                p = FONT_BOLD or path
            elif weight == "black":
                p = FONT_BLACK or FONT_BOLD or path
            else:
                p = FONT_BOLD or FONT_BLACK or path
            if p and os.path.exists(p):
                return ImageFont.truetype(p, size)
        except: pass
        return ImageFont.load_default()

    def hex_to_rgb(hex_str):
        h = (hex_str or "#FFFFFF").lstrip('#')
        try: return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        except: return (255, 255, 255)

    def apply_vignette(frame):
        rows, cols = frame.shape[:2]
        X = cv2.getGaussianKernel(cols, cols * 0.6)
        Y = cv2.getGaussianKernel(rows, rows * 0.6)
        mask = (Y * X.T) / (Y * X.T).max()
        out = frame.copy().astype(np.float32)
        for i in range(3): out[:,:,i] *= mask
        return np.clip(out, 0, 255).astype(np.uint8)

    def apply_warm_grade(frame):
        out = frame.copy().astype(np.float32)
        out[:,:,2] = np.clip(out[:,:,2] * 1.04, 0, 255)
        return out.astype(np.uint8)

    def to_pil(frame):
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def to_frame(pil_img):
        return cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)

    def composite_layer(frame, layer):
        pil = to_pil(frame).convert('RGBA')
        merged = Image.alpha_composite(pil, layer)
        return to_frame(merged)

    # ── video metadata ─────────────────────────────────────────────
    def _ffprobe_dur(path):
        try:
            r = subprocess.run(['ffprobe','-v','error','-show_entries','format=duration',
                                '-of','default=noprint_wrappers=1:nokey=1', path],
                               capture_output=True, text=True)
            return float(r.stdout.strip())
        except: return 0.0

    TARGET_FPS = 30.0
    vid_dur = _ffprobe_dur(video_path)

    # Transcode to CFR for predictable frame timing
    cfr_video = output_path.replace(".mp4", "_cfr_tmp.mp4")
    subprocess.run(['ffmpeg','-y','-i',video_path,
                    '-vf',f'fps={TARGET_FPS:.0f}',
                    '-c:v','libx264','-preset','ultrafast','-crf','18',
                    '-an', cfr_video],
                   capture_output=True, check=True)

    # ── Build ordered Whisper word list — exact timestamps in speech order ──
    whisper_word_list = []
    for seg in whisper_segments:
        for we in seg.get('words', []):
            raw = we.get('word', '')
            wc = raw.upper().strip('.,!?;:\'"()[]- ')
            if not wc:
                continue
            whisper_word_list.append({
                'word':  wc,
                'start': float(we.get('start', 0.0)),
                'end':   float(we.get('end',   0.0)),
            })

    def get_beat_whisper_words(beat_start, beat_end):
        """All Whisper words whose start falls within this beat window."""
        return [w for w in whisper_word_list
                if beat_start - 0.15 <= w['start'] <= beat_end + 0.15]

    def match_word_in_list(word, candidates):
        """Find the best matching Whisper word entry for `word` within candidates."""
        wc = word.upper().strip('.,!?;:\'"()[]- ')
        for w in candidates:
            if w['word'] == wc:
                return w
        for w in candidates:
            if wc in w['word'] or w['word'] in wc:
                return w
        return None

    def clamp(v, lo, hi): return max(lo, min(v, hi))

    # ── Build flat timeline — pure Whisper, scoped per beat ─────────────────
    #
    # SINGLE-WORD BEAT: 1 text element → IMPACT mode
    #   - Appears at exact Whisper word start
    #   - Lasts up to 1.0s with bzzt flicker, but never overlaps the next beat
    #
    # MULTI-WORD BEAT: multiple text elements → CAPTION mode
    #   - Each word appears at its Whisper timestamp
    #   - Disappears when next word is spoken
    #   - For multi-word content (e.g. "THE WORLD"), anim_end accounts for
    #     the LAST word's end time, not just the first word's start
    #
    timeline = []
    for scene_pos, scene in enumerate(scenes):
        beat = beats[scene_pos] if scene_pos < len(beats) else {}
        beat_start = clamp(float(beat.get("start_time", 0.0)), 0, vid_dur - 0.1)
        beat_end   = clamp(float(beat.get("end_time", beat_start + 2.0)),
                           beat_start + 0.05, vid_dur)
        next_beat_start = None
        if scene_pos + 1 < len(beats):
            next_beat_start = float(beats[scene_pos + 1].get("start_time", beat_end))
            beat_end = min(beat_end, next_beat_start)

        elements = scene.get("elements", [])
        if not elements:
            continue

        text_els  = [e for e in elements if e.get("type", "text") == "text"]
        other_els = [e for e in elements if e.get("type", "text") != "text"]

        # Whisper words available in this beat's time window (beat-scoped to
        # avoid matching the wrong occurrence of common words like THE/IS/A)
        beat_words = get_beat_whisper_words(beat_start, beat_end)

        # Resolve each text element to Whisper timestamps within this beat
        resolved = []  # [(whisper_start, whisper_end, el)]
        used_indices = set()
        for el in text_els:
            raw = (el.get("content") or "").strip()
            if not raw:
                continue
            words_in_content = raw.split()
            available = [w for i, w in enumerate(beat_words) if i not in used_indices]

            first_match = match_word_in_list(words_in_content[0], available)
            if first_match:
                idx = beat_words.index(first_match)
                used_indices.add(idx)
                el_start = first_match['start']
                el_end   = first_match['end']

                # If content has multiple words, extend el_end to cover the
                # LAST word's Whisper end time too (so "THE WORLD" stays
                # visible until "WORLD" is actually spoken, not just "THE")
                if len(words_in_content) > 1:
                    available2 = [w for i, w in enumerate(beat_words) if i not in used_indices]
                    last_match = match_word_in_list(words_in_content[-1], available2)
                    if last_match:
                        idx2 = beat_words.index(last_match)
                        used_indices.add(idx2)
                        el_end = max(el_end, last_match['end'])

                resolved.append((el_start, el_end, el))
            else:
                # No Whisper match — fallback to beat_start
                resolved.append((beat_start, beat_end, el))

        resolved.sort(key=lambda x: x[0])
        is_single_word = len(resolved) == 1

        for i, (ws, we_t, el) in enumerate(resolved):
            anim_start = clamp(ws, 0.0, vid_dur - 0.1)

            if is_single_word:
                # IMPACT: up to 1.0s with bzzt flicker, but never bleed into
                # the next beat's start time
                impact_end = anim_start + 1.0
                if next_beat_start is not None:
                    impact_end = min(impact_end, next_beat_start)
                anim_end = clamp(impact_end, anim_start + 0.1, vid_dur)
                impact = True
            else:
                # CAPTION: hold until next word's Whisper start, but at least
                # until this element's own Whisper end time (covers multi-word content)
                min_end = max(we_t, anim_start + 0.08)
                if i + 1 < len(resolved):
                    anim_end = clamp(max(resolved[i + 1][0], min_end), anim_start + 0.08, vid_dur)
                else:
                    anim_end = clamp(max(beat_end, min_end), anim_start + 0.08, vid_dur)
                impact = False

            timeline.append({
                "el":            el,
                "start":         anim_start,
                "end":           anim_end,
                "anim_duration": 0.06 if impact else float(el.get("anim_duration", 0.10)),
                "impact":        impact,
            })

        for el in other_els:
            timeline.append({
                "el":            el,
                "start":         beat_start,
                "end":           beat_end,
                "anim_duration": float(el.get("anim_duration", 0.2)),
                "impact":        False,
            })

    timeline.sort(key=lambda x: x["start"])
    print(f"  📊 Timeline: {len(timeline)} elements")

    # ── Freeform object timeline -- beat-scoped, not word-scoped. These
    # are independent animated objects (dots/paths/pills/cards), not
    # spoken captions, so they're timed to the beat's real audio window
    # directly rather than matched against individual Whisper words.
    freeform_timeline = []
    for scene_pos, fscene in enumerate(freeform_scenes):
        beat = beats[scene_pos] if scene_pos < len(beats) else {}
        beat_start = clamp(float(beat.get("start_time", 0.0)), 0, vid_dur - 0.1)
        beat_end   = clamp(float(beat.get("end_time", beat_start + 2.0)),
                           beat_start + 0.05, vid_dur)
        if scene_pos + 1 < len(beats):
            beat_end = min(beat_end, float(beats[scene_pos + 1].get("start_time", beat_end)))

        for obj in fscene.get("freeform_objects", []):
            dur_key = "duration" if "duration" in obj else "anim_duration"
            start_offset = float(obj.get("start_offset", 0.0))
            duration = float(obj.get(dur_key, 0.4))
            # path_shape with hold=True and label_pill/card stay visible
            # until the beat ends; dot and non-held path_shape end with
            # their own motion duration.
            stays_until_beat_end = obj.get("type") in ("label_pill", "card") or \
                                    (obj.get("type") == "path_shape" and obj.get("hold", True))
            obj_start = clamp(beat_start + start_offset, 0.0, vid_dur)
            if stays_until_beat_end:
                obj_end = beat_end
            else:
                obj_end = clamp(obj_start + duration + 0.3, obj_start + 0.1, vid_dur)
            freeform_timeline.append({"obj": obj, "start": obj_start, "end": obj_end})

    freeform_timeline.sort(key=lambda x: x["start"])
    print(f"  🎬 Freeform timeline: {len(freeform_timeline)} objects")

    # ── Frame-by-frame render ─────────────────────────────────────
    cap = cv2.VideoCapture(cfr_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_vid = TARGET_FPS

    temp_video = output_path.replace(".mp4", "_noaudio_tmp.mp4")
    out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps_vid, (fw, fh))

    print(f"  🎬 {total_frames} frames @ {fps_vid:.0f}fps...")
    frame_idx = 0
    prev_pct = -1

    # ── Element drawing functions ──────────────────────────────────
    def get_anim_progress(el_t, start, end, anim_dur):
        """Return (entrance_progress, exit_progress) both 0..1.
        entrance_progress: 0=not started, 1=fully appeared
        exit_progress: 1=visible, 0=fully gone (only at end)
        """
        if anim_dur <= 0:
            anim_dur = 0.001
        entrance = clamp((el_t - start) / anim_dur, 0.0, 1.0)
        # No exit fade by default - just snap out at end
        return entrance

    def draw_text_element(layer, el, el_t, anim_t):
        """Draw a TEXT element with animation."""
        draw = ImageDraw.Draw(layer)
        content = el.get("content", "").upper().strip()  # Always uppercase for Anton
        if not content:
            return
        x_pct = float(el.get("x", 0.5))
        y_pct = float(el.get("y", 0.5))
        # Hard cap: 130px max for sentence words, 160px for single-word impact,
        # 220px for number_counter values (numbers/currency need to read big)
        raw_size = int(el.get("size", 90))
        word_count = len(content.split())
        if el.get("_is_counter"):
            size_cap = 220
        else:
            size_cap = 160 if word_count == 1 else 110
        size = max(20, min(raw_size, size_cap))
        color = hex_to_rgb(el.get("color", "#FFFFFF"))
        weight = el.get("weight", "black")
        outline = max(0, min(int(el.get("outline", 4)), 12))
        anim = el.get("anim", "fade_in")
        anchor = el.get("anchor", "center")
        effect = el.get("effect", "none")

        font = load_pil_font(get_primary_font_path(), size, weight)
        try:
            bbox = font.getbbox(content)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except:
            tw = size * len(content) * 0.55
            th = size

        # Compute base position
        target_x = int(OUTPUT_WIDTH * x_pct)
        target_y = int(OUTPUT_HEIGHT * y_pct)
        if anchor == "center":
            base_x = target_x - tw // 2
            base_y = target_y - th // 2
        elif anchor == "left":
            base_x = target_x
            base_y = target_y - th // 2
        elif anchor == "right":
            base_x = target_x - tw
            base_y = target_y - th // 2
        else:
            base_x = target_x - tw // 2
            base_y = target_y - th // 2

        # Clamp to screen with padding so text never goes off-edge
        pad = 30
        max_x = OUTPUT_WIDTH - tw - pad
        max_y = OUTPUT_HEIGHT - th - pad
        base_x = max(pad, min(base_x, max_x))
        base_y = max(pad, min(base_y, max_y))

        # Apply entrance animation
        draw_x, draw_y = base_x, base_y
        alpha = 1.0
        scale = 1.0

        if anim == "fade_in":
            alpha = anim_t
        elif anim == "slide_in_left":
            slide_dist = int(OUTPUT_WIDTH * 0.3)
            draw_x = base_x - int(slide_dist * (1.0 - anim_t))
            alpha = anim_t
        elif anim == "slide_in_right":
            slide_dist = int(OUTPUT_WIDTH * 0.3)
            draw_x = base_x + int(slide_dist * (1.0 - anim_t))
            alpha = anim_t
        elif anim == "slide_in_top":
            slide_dist = int(OUTPUT_HEIGHT * 0.2)
            draw_y = base_y - int(slide_dist * (1.0 - anim_t))
            alpha = anim_t
        elif anim == "slide_in_bottom":
            slide_dist = int(OUTPUT_HEIGHT * 0.2)
            draw_y = base_y + int(slide_dist * (1.0 - anim_t))
            alpha = anim_t
        elif anim == "scale_in":
            scale = 1.3 - 0.3 * anim_t
            alpha = anim_t
        elif anim == "snap":
            alpha = 1.0
        elif anim == "none":
            alpha = 1.0

        # Effects (applied during display)
        if effect == "flicker":
            # Blink during first 0.3s of element life
            if el_t - 0 < 0.3:
                frame_no = int(el_t * 30)
                if frame_no % 2 == 1:
                    return
        elif effect == "shake":
            import random as _r
            draw_x += _r.randint(-3, 3)
            draw_y += _r.randint(-3, 3)

        # Re-render text at scaled size if scale != 1.0
        render_font = font
        if abs(scale - 1.0) > 0.02:
            new_size = max(20, int(size * scale))
            render_font = load_pil_font(get_primary_font_path(), new_size, weight)
            try:
                bbox = render_font.getbbox(content)
                tw2 = bbox[2] - bbox[0]
                th2 = bbox[3] - bbox[1]
                draw_x = base_x + (tw - tw2) // 2
                draw_y = base_y + (th - th2) // 2
            except: pass

        # Clamp draw position AFTER animation offsets (slide/shake can push text off-screen)
        draw_x = max(pad, min(draw_x, OUTPUT_WIDTH - tw - pad))
        draw_y = max(pad, min(draw_y, OUTPUT_HEIGHT - th - pad))

        a_int = max(0, min(int(255 * alpha), 255))
        if a_int < 5:
            return

        # Draw outline (multiple offsets for thick outline)
        if outline > 0:
            for ox in range(-outline, outline + 1):
                for oy in range(-outline, outline + 1):
                    if ox * ox + oy * oy <= outline * outline:
                        if ox == 0 and oy == 0:
                            continue
                        draw.text((draw_x + ox, draw_y + oy), content,
                                  font=render_font, fill=(0, 0, 0, a_int))
        # Draw fill
        draw.text((draw_x, draw_y), content, font=render_font,
                  fill=(color[0], color[1], color[2], a_int))

    def draw_line_element(layer, el, el_t, anim_t):
        """Draw a LINE element with animation."""
        draw = ImageDraw.Draw(layer)
        x1 = int(OUTPUT_WIDTH * float(el.get("x1", 0.3)))
        y1 = int(OUTPUT_HEIGHT * float(el.get("y1", 0.5)))
        x2 = int(OUTPUT_WIDTH * float(el.get("x2", 0.7)))
        y2 = int(OUTPUT_HEIGHT * float(el.get("y2", 0.5)))
        thickness = max(1, int(el.get("thickness", 6)))
        color = hex_to_rgb(el.get("color", "#FFFFFF"))
        anim = el.get("anim", "draw_horizontal")

        alpha = 1.0
        end_x, end_y = x2, y2

        if anim == "fade_in":
            alpha = anim_t
        elif anim == "draw_horizontal":
            end_x = x1 + int((x2 - x1) * anim_t)
            end_y = y1 + int((y2 - y1) * anim_t)
            alpha = 1.0
        elif anim == "none":
            alpha = 1.0

        a_int = max(0, min(int(255 * alpha), 255))
        if a_int < 5:
            return

        draw.line([(x1, y1), (end_x, end_y)],
                  fill=(color[0], color[1], color[2], a_int),
                  width=thickness)

    def draw_rect_element(layer, el, el_t, anim_t):
        """Draw a RECT element."""
        draw = ImageDraw.Draw(layer)
        x = int(OUTPUT_WIDTH * float(el.get("x", 0.4)))
        y = int(OUTPUT_HEIGHT * float(el.get("y", 0.4)))
        w = int(OUTPUT_WIDTH * float(el.get("w", 0.2)))
        h = int(OUTPUT_HEIGHT * float(el.get("h", 0.1)))
        color = hex_to_rgb(el.get("color", "#FFFFFF"))
        filled = bool(el.get("filled", True))
        thickness = max(1, int(el.get("thickness", 3)))
        anim = el.get("anim", "fade_in")

        alpha = 1.0
        if anim == "fade_in":
            alpha = anim_t
        elif anim == "scale_in":
            scale = anim_t
            cx, cy = x + w // 2, y + h // 2
            w = int(w * scale); h = int(h * scale)
            x = cx - w // 2; y = cy - h // 2
            alpha = anim_t

        a_int = max(0, min(int(255 * alpha), 255))
        if a_int < 5:
            return

        rgba = (color[0], color[1], color[2], a_int)
        if filled:
            draw.rectangle([x, y, x + w, y + h], fill=rgba)
        else:
            draw.rectangle([x, y, x + w, y + h], outline=rgba, width=thickness)

    def draw_circle_element(layer, el, el_t, anim_t):
        """Draw a CIRCLE element."""
        draw = ImageDraw.Draw(layer)
        cx = int(OUTPUT_WIDTH * float(el.get("x", 0.5)))
        cy = int(OUTPUT_HEIGHT * float(el.get("y", 0.5)))
        r = int(min(OUTPUT_WIDTH, OUTPUT_HEIGHT) * float(el.get("radius", 0.05)))
        color = hex_to_rgb(el.get("color", "#FFFFFF"))
        filled = bool(el.get("filled", False))
        thickness = max(1, int(el.get("thickness", 4)))
        anim = el.get("anim", "fade_in")

        alpha = 1.0
        if anim == "fade_in":
            alpha = anim_t
        elif anim == "scale_in":
            r = int(r * anim_t)
            alpha = anim_t

        a_int = max(0, min(int(255 * alpha), 255))
        if a_int < 5 or r <= 0:
            return

        rgba = (color[0], color[1], color[2], a_int)
        bbox = [cx - r, cy - r, cx + r, cy + r]
        if filled:
            draw.ellipse(bbox, fill=rgba)
        else:
            draw.ellipse(bbox, outline=rgba, width=thickness)

    def _format_counter_value(value, decimals, prefix, suffix):
        """Format a number with comma separators, fixed decimals, and
        prefix/suffix -- e.g. 400000 -> '$400,000', 23.5 -> '23.5%'."""
        if decimals > 0:
            text = f"{value:,.{decimals}f}"
        else:
            text = f"{int(round(value)):,}"
        return f"{prefix}{text}{suffix}"

    def draw_number_counter_element(layer, el, el_t, anim_t):
        """Draw a NUMBER_COUNTER element -- animates from count_from to
        target_value over count_duration, then holds at target_value.
        Reuses draw_text_element's rendering by building a synthetic
        text element each frame with the current counted value."""
        target = float(el.get("target_value", 0))
        count_from = float(el.get("count_from", 0))
        count_dur = max(0.05, float(el.get("count_duration", 0.8)))
        decimals = max(0, int(el.get("decimals", 0)))
        prefix = el.get("prefix", "")
        suffix = el.get("suffix", "")

        progress = clamp(el_t / count_dur, 0.0, 1.0)
        # Ease-out so the count settles rather than stopping abruptly
        eased = 1.0 - (1.0 - progress) ** 3
        current_value = count_from + (target - count_from) * eased
        content = _format_counter_value(current_value, decimals, prefix, suffix)

        synthetic = dict(el)
        synthetic["type"] = "text"
        synthetic["content"] = content
        # Counter elements should not be re-uppercased/word-counted like
        # spoken captions -- force single-word sizing path (no cap at 110px)
        synthetic["_is_counter"] = True
        draw_text_element(layer, synthetic, el_t, 1.0 if progress > 0 else anim_t)

    def draw_grid_element(layer, el, el_t, anim_t):
        """Draw a GRID element -- rows x cols of a repeated glyph, either
        all at once (fade_in) or revealed cell-by-cell left-to-right,
        top-to-bottom (fill_sequential)."""
        draw = ImageDraw.Draw(layer)
        glyph = el.get("glyph", "0")
        rows = max(1, int(el.get("rows", 4)))
        cols = max(1, int(el.get("cols", 10)))
        cell = max(10, int(el.get("cell_size", 60)))
        color = hex_to_rgb(el.get("color", "#FBC02D"))
        anim = el.get("anim", "fill_sequential")
        fill_dur = max(0.05, float(el.get("fill_duration", 1.2)))

        cx = int(OUTPUT_WIDTH * float(el.get("x", 0.5)))
        cy = int(OUTPUT_HEIGHT * float(el.get("y", 0.55)))
        grid_w = cols * cell
        grid_h = rows * cell
        ox = cx - grid_w // 2
        oy = cy - grid_h // 2

        font = load_pil_font(get_primary_font_path(), int(cell * 0.8), "black")

        total_cells = rows * cols
        if anim == "fill_sequential":
            progress = clamp(el_t / fill_dur, 0.0, 1.0)
            visible_cells = int(total_cells * progress)
            cell_alpha = 1.0
        else:
            visible_cells = total_cells
            cell_alpha = anim_t

        a_int = max(0, min(int(255 * cell_alpha), 255))
        if a_int < 5:
            return

        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= visible_cells:
                    return
                gx = ox + c * cell + cell // 2
                gy = oy + r * cell + cell // 2
                try:
                    bbox = font.getbbox(glyph)
                    gw, gh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except Exception:
                    gw, gh = cell // 2, cell // 2
                draw.text((gx - gw // 2, gy - gh // 2), glyph, font=font,
                          fill=(color[0], color[1], color[2], a_int))
                idx += 1

    # ── Freeform motion primitives ──────────────────────────────────
    # Deterministic easing + path math. GPT only ever chooses an easing
    # NAME and a list of points -- this is the only code that turns those
    # choices into an actual on-screen position, so nothing GPT emits can
    # produce NaNs, runaway coordinates, or broken interpolation.

    def _ease(progress, name):
        p = clamp(progress, 0.0, 1.0)
        if name == "linear":
            return p
        elif name == "ease_out":
            return 1.0 - (1.0 - p) ** 3
        elif name == "ease_in_out":
            return 0.5 - 0.5 * math.cos(math.pi * p)
        elif name == "spring":
            # Damped oscillation: overshoots past 1.0 then settles back.
            # Critically-damped-ish spring, tuned to feel like a quick pop.
            if p >= 1.0:
                return 1.0
            omega = 14.0
            zeta = 0.55
            return 1.0 - math.exp(-zeta * omega * p) * math.cos(omega * math.sqrt(max(1e-6, 1 - zeta**2)) * p)
        return p

    def _point_along_path(path_px, progress):
        """path_px: list of (x,y) pixel points. Walks the polyline at
        normalized progress 0..1 by arc length, so motion speed is even
        across uneven point spacing."""
        if len(path_px) == 1:
            return path_px[0]
        seg_lens = [math.hypot(path_px[i+1][0]-path_px[i][0], path_px[i+1][1]-path_px[i][1])
                    for i in range(len(path_px)-1)]
        total = sum(seg_lens) or 1.0
        target = clamp(progress, 0.0, 1.0) * total
        acc = 0.0
        for i, seg_len in enumerate(seg_lens):
            if acc + seg_len >= target or i == len(seg_lens) - 1:
                seg_progress = (target - acc) / seg_len if seg_len > 1e-6 else 0.0
                seg_progress = clamp(seg_progress, 0.0, 1.0)
                x0, y0 = path_px[i]
                x1, y1 = path_px[i+1]
                return (x0 + (x1-x0)*seg_progress, y0 + (y1-y0)*seg_progress)
            acc += seg_len
        return path_px[-1]

    def _path_to_px(path_norm):
        return [(p[0] * OUTPUT_WIDTH, p[1] * OUTPUT_HEIGHT) for p in path_norm]

    def draw_freeform_dot(layer, obj, el_t, _anim_t_unused):
        draw = ImageDraw.Draw(layer)
        duration = max(0.05, float(obj.get("duration", 1.0)))
        progress = _ease(el_t / duration, obj.get("easing", "ease_out"))
        path_px = _path_to_px(obj.get("path", [[0.5, 0.5], [0.5, 0.5]]))
        x, y = _point_along_path(path_px, progress)
        r = max(2, int(obj.get("radius", 0.012) * min(OUTPUT_WIDTH, OUTPUT_HEIGHT)))
        color = hex_to_rgb(obj.get("color", "#FFFFFF"))
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(color[0], color[1], color[2], 255))

    def draw_freeform_path_shape(layer, obj, el_t, _anim_t_unused):
        draw = ImageDraw.Draw(layer)
        duration = max(0.05, float(obj.get("duration", 0.8)))
        hold = bool(obj.get("hold", True))
        progress = _ease(el_t / duration, obj.get("easing", "ease_out"))
        path_px = _path_to_px(obj.get("path", [[0.2, 0.5], [0.8, 0.5]]))
        color = hex_to_rgb(obj.get("color", "#FFFFFF"))
        thickness = max(1, int(obj.get("thickness", 4)))

        if progress >= 1.0 and not hold:
            fade_t = clamp((el_t - duration) / 0.3, 0.0, 1.0)
            alpha = max(0, int(255 * (1.0 - fade_t)))
            if alpha < 5:
                return
        else:
            alpha = 255

        # Walk the path up to `progress` arc-length fraction, drawing
        # only the revealed portion (pen-reveal effect).
        seg_lens = [math.hypot(path_px[i+1][0]-path_px[i][0], path_px[i+1][1]-path_px[i][1])
                    for i in range(len(path_px)-1)]
        total = sum(seg_lens) or 1.0
        target = progress * total
        acc = 0.0
        revealed = [path_px[0]]
        for i, seg_len in enumerate(seg_lens):
            if acc + seg_len <= target:
                revealed.append(path_px[i+1])
            else:
                seg_progress = (target - acc) / seg_len if seg_len > 1e-6 else 0.0
                x0, y0 = path_px[i]
                x1, y1 = path_px[i+1]
                revealed.append((x0 + (x1-x0)*seg_progress, y0 + (y1-y0)*seg_progress))
                break
            acc += seg_len

        if len(revealed) >= 2:
            draw.line(revealed, fill=(color[0], color[1], color[2], alpha),
                       width=thickness, joint="curve")

    def draw_freeform_label_pill(layer, obj, el_t, _anim_t_unused):
        draw = ImageDraw.Draw(layer)
        content = obj.get("content", "")
        anim_dur = max(0.05, float(obj.get("anim_duration", 0.3)))
        easing = obj.get("easing", "spring")
        progress = _ease(el_t / anim_dur, easing)
        # Spring overshoots past 1.0 -- use that for a scale-pop, clamp
        # alpha separately so it doesn't go negative during the overshoot.
        scale = max(0.3, progress)
        alpha = max(0, min(255, int(255 * clamp(el_t / max(0.05, anim_dur * 0.6), 0.0, 1.0))))

        cx = float(obj.get("x", 0.5)) * OUTPUT_WIDTH
        cy = float(obj.get("y", 0.5)) * OUTPUT_HEIGHT
        font_size = max(18, int(28 * scale))
        font = load_pil_font(get_primary_font_path(), font_size, "bold")
        try:
            bbox = font.getbbox(content)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        except Exception:
            tw, th = font_size * len(content) * 0.6, font_size

        pad_x, pad_y = 18, 10
        w, h = tw + pad_x*2, th + pad_y*2
        bg = hex_to_rgb(obj.get("color", "#1A1A1A"))
        txt_color = hex_to_rgb(obj.get("text_color", "#FFD782"))

        x0, y0 = cx - w/2, cy - h/2
        x1, y1 = cx + w/2, cy + h/2
        radius = min(h/2, 16)
        draw.rounded_rectangle([x0, y0, x1, y1], radius=radius,
                                 fill=(bg[0], bg[1], bg[2], alpha))
        draw.text((cx - tw/2, cy - th/2), content, font=font,
                   fill=(txt_color[0], txt_color[1], txt_color[2], alpha))

    def draw_freeform_card(layer, obj, el_t, _anim_t_unused):
        anim_dur = max(0.05, float(obj.get("anim_duration", 0.4)))
        progress = _ease(el_t / anim_dur, obj.get("easing", "ease_out"))
        alpha = max(0, min(255, int(255 * progress)))
        if alpha < 5:
            return

        cx = float(obj.get("x", 0.5)) * OUTPUT_WIDTH
        cy = float(obj.get("y", 0.5)) * OUTPUT_HEIGHT
        w = float(obj.get("w", 0.3)) * OUTPUT_WIDTH
        h = float(obj.get("h", 0.2)) * OUTPUT_HEIGHT
        tilt_deg = float(obj.get("tilt_deg", 0.0))
        color = hex_to_rgb(obj.get("color", "#1A1A1A"))

        card_layer = Image.new('RGBA', (OUTPUT_WIDTH, OUTPUT_HEIGHT), (0, 0, 0, 0))
        cdraw = ImageDraw.Draw(card_layer)
        x0, y0 = cx - w/2, cy - h/2
        x1, y1 = cx + w/2, cy + h/2
        cdraw.rounded_rectangle([x0, y0, x1, y1], radius=min(20, h*0.1),
                                  fill=(color[0], color[1], color[2], alpha))

        if abs(tilt_deg) > 0.5:
            # Perspective tilt via a simple shear + scale approximation --
            # cheap and stable, avoids a full warpPerspective per frame.
            card_arr = np.array(card_layer)
            shear = math.tan(math.radians(tilt_deg)) * 0.3
            M = np.array([[1, shear, -shear * cy], [0, 1, 0]], dtype=np.float32)
            card_arr = cv2.warpAffine(card_arr, M, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
            card_layer = Image.fromarray(card_arr)

        layer.alpha_composite(card_layer)

    FREEFORM_DRAW_FNS = {
        "dot": draw_freeform_dot,
        "path_shape": draw_freeform_path_shape,
        "label_pill": draw_freeform_label_pill,
        "card": draw_freeform_card,
    }

    # ── Main frame loop ──────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret: break

        t = frame_idx / fps_vid
        frame = apply_vignette(frame)
        frame = apply_warm_grade(frame)

        # Freeform motion objects (dots/paths/pills/cards) -- composited
        # before the text-element layer. Each object is wrapped in its
        # own try/except: if one object's data is malformed in a way
        # validation didn't catch, that object is skipped for this frame
        # and logged once, never crashing the render or affecting any
        # other object/beat.
        ff_active = [item for item in freeform_timeline
                     if item["start"] <= t < item["end"]]
        if ff_active:
            ff_layer = Image.new('RGBA', (OUTPUT_WIDTH, OUTPUT_HEIGHT), (0, 0, 0, 0))
            for item in ff_active:
                obj = item["obj"]
                otype = obj.get("type")
                obj_t = t - item["start"]
                draw_fn = FREEFORM_DRAW_FNS.get(otype)
                if not draw_fn:
                    continue
                try:
                    draw_fn(ff_layer, obj, obj_t, None)
                except Exception as e:
                    if not obj.get("_warned"):
                        print(f"  ⚠ freeform object render error ({otype}): {e}")
                        obj["_warned"] = True
            frame = composite_layer(frame, ff_layer)

        # Find all elements active at time t
        raw_active = [item for item in timeline
                      if item["start"] <= t < item["end"]]

        # Deduplicate: if two elements share the same content + near-same x/y position,
        # keep only the one whose start time is closest to t (most recently activated).
        # This prevents "WE WE" doubles when GPT repeats a word across adjacent beats.
        seen_keys = {}
        for item in sorted(raw_active, key=lambda x: x["start"], reverse=True):
            el = item["el"]
            if el.get("type") == "text":
                content_key = (el.get("content", "").upper().strip(),
                               round(float(el.get("x", 0.5)), 1),
                               round(float(el.get("y", 0.5)), 1))
                if content_key not in seen_keys:
                    seen_keys[content_key] = item
            else:
                seen_keys[id(item["el"])] = item
        active = list(seen_keys.values())

        if active:
            # Slight darkening overlay when text is on screen
            frame = cv2.addWeighted(frame, 0.82, np.zeros_like(frame), 0.18, 0)

            # Build composite layer
            layer = Image.new('RGBA', (OUTPUT_WIDTH, OUTPUT_HEIGHT), (0, 0, 0, 0))

            for item in active:
                el    = item["el"]
                el_t  = t - item["start"]   # seconds since element appeared
                el_dur = max(item["end"] - item["start"], 0.01)
                impact = item.get("impact", False)
                etype  = el.get("type", "text")

                if impact and etype == "text":
                    # BZZT FLICKER: flash-flash-flash-hold pattern over 1.0s
                    # 0.00-0.08s: ON  (flash 1)
                    # 0.08-0.16s: OFF
                    # 0.16-0.24s: ON  (flash 2)
                    # 0.24-0.32s: OFF
                    # 0.32-0.40s: ON  (flash 3)
                    # 0.40s+    : HOLD fully visible with slow fade-out at end
                    in_flash = (
                        (0.00 <= el_t < 0.08) or
                        (0.16 <= el_t < 0.24) or
                        (0.32 <= el_t < 0.40)
                    )
                    in_off = (0.08 <= el_t < 0.16) or (0.24 <= el_t < 0.32)
                    if in_off:
                        continue  # skip drawing — element is "off"
                    # During flashes use full alpha; after 0.4s hold then fade out near end
                    if el_t >= 0.40:
                        fade_window = 0.15
                        time_left = el_dur - el_t
                        if time_left < fade_window:
                            anim_t = max(0.0, time_left / fade_window)
                        else:
                            anim_t = 1.0
                    else:
                        anim_t = 1.0  # instant full alpha during flashes
                    try:
                        draw_text_element(layer, el, el_t, anim_t)
                    except Exception as e:
                        print(f"  ⚠ impact render error: {e}")
                else:
                    anim_t = get_anim_progress(el_t, 0, el_dur, item["anim_duration"])
                    try:
                        if etype == "text":
                            draw_text_element(layer, el, el_t, anim_t)
                        elif etype == "line":
                            draw_line_element(layer, el, el_t, anim_t)
                        elif etype == "rect":
                            draw_rect_element(layer, el, el_t, anim_t)
                        elif etype == "circle":
                            draw_circle_element(layer, el, el_t, anim_t)
                        elif etype == "number_counter":
                            draw_number_counter_element(layer, el, el_t, anim_t)
                        elif etype == "grid":
                            draw_grid_element(layer, el, el_t, anim_t)
                    except Exception as e:
                        print(f"  ⚠ element render error: {e}")

            frame = composite_layer(frame, layer)

        out.write(frame)
        frame_idx += 1
        pct = int(frame_idx / max(total_frames, 1) * 20)
        if pct != prev_pct:
            print(f"  [{'█' * pct}{'░' * (20 - pct)}] {frame_idx}/{total_frames}",
                  end='\r')
            prev_pct = pct

    cap.release(); out.release()
    if os.path.exists(cfr_video): os.remove(cfr_video)
    print(f"\n  ✓ Frames done")

    result = subprocess.run([
        'ffmpeg', '-y', '-i', temp_video, '-i', video_path,
        '-map', '0:v', '-map', '1:a',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-c:a', 'copy', output_path
    ], capture_output=True)

    if os.path.exists(temp_video): os.remove(temp_video)
    if result.returncode != 0:
        raise Exception(f"Audio merge failed: {result.stderr.decode()[-200:]}")

    print(f"  ✅ Render complete: {output_path}")



# ============================================================
# MAIN GENERATOR
# ============================================================
class FinanceGenerator:
    def __init__(self, audio_path: str, output_path: str = "output.mp4", niche_config: dict = None):
        self.audio_path  = audio_path
        self.output_path = output_path

        if niche_config:
            self.broll_dirs  = niche_config.get('broll_dirs', {})
            self.keyword_map = niche_config.get('keyword_map', {})
        else:
            self.broll_dirs = {
                'space':   'space_vids',
                'ancient': 'ancient_ruins_vids',
                'cosmic':  'cosmic_vids',
                'sky':     'dark_sky_vids',
                'temple':  'temple_vids',
            }
            self.keyword_map = {
                'space':   ['universe', 'galaxy', 'black hole', 'star', 'planet', 'cosmos'],
                'ancient': ['ancient', 'civilization', 'pyramid', 'ruins', 'lost', 'forgotten'],
                'cosmic':  ['time', 'reality', 'dimension', 'quantum', 'existence', 'consciousness'],
                'sky':     ['sky', 'atmosphere', 'above', 'beyond', 'vast', 'endless'],
                'temple':  ['religion', 'god', 'sacred', 'ritual', 'belief', 'worship'],
            }

    def get_audio_duration(self) -> float:
        cmd    = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                  '-of', 'default=noprint_wrappers=1:nokey=1', self.audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"ffprobe failed: {result.stderr}")
        return float(result.stdout.strip())

    def get_video_info(self, filepath: str):
        cmd    = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                  '-show_entries', 'stream=width,height', '-of', 'json', filepath]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            data = json.loads(result.stdout)
            w    = data['streams'][0]['width']
            h    = data['streams'][0]['height']
            return w, h, w / h
        except:
            return None, None, None

    def get_all_files_from_dir(self, directory: str) -> list:
        if not os.path.exists(directory):
            return []
        files = [os.path.join(directory, f) for f in os.listdir(directory)
                 if f.lower().endswith(('.mp4', '.mov', '.avi'))]
        if not files:
            print(f"  ⚠ Folder exists but is EMPTY: {directory}")
        return files

    def transcribe_with_whisper(self, model: str = "base") -> dict | None:
        cache_file = f"{os.path.splitext(self.audio_path)[0]}_transcription.json"
        if os.path.exists(cache_file):
            print(f"  ✅ Cached transcription")
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        try:
            import whisper
            if not hasattr(whisper, 'load_model'):
                raise ImportError("Wrong whisper. Run: pip install openai-whisper")
            print(f"  🎤 Transcribing ({model})...")
            wm     = whisper.load_model(model)
            result = wm.transcribe(self.audio_path, word_timestamps=True, language="en")
            with open(cache_file, 'w') as f:
                json.dump(result, f, indent=2)
            return result
        except Exception as e:
            print(f"  ❌ Whisper error: {e}")
            return None

    def match_broll_categories(self, full_text: str) -> list:
        text   = full_text.lower()
        scores = {cat: sum(text.count(k) for k in kws)
                  for cat, kws in self.keyword_map.items()}
        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = [self.broll_dirs[c] for c, s in sorted_cats if s > 0 and c in self.broll_dirs]

        # FIX 2: Filter to only folders that actually exist and have clips
        valid_top = []
        for folder in top:
            files = self.get_all_files_from_dir(folder)
            if files:
                valid_top.append(folder)
            else:
                print(f"  ⚠ Skipping empty/missing broll folder: {folder}")

        if not valid_top:
            # Fallback: scan ALL configured broll dirs and use any that have clips
            print(f"  ⚠ No keyword-matched folders had clips -- scanning all broll dirs...")
            for folder in self.broll_dirs.values():
                files = self.get_all_files_from_dir(folder)
                if files:
                    valid_top.append(folder)
                    print(f"  ✓ Found clips in: {folder} ({len(files)} files)")

        if not valid_top:
            raise Exception(
                "No broll clips found in ANY configured folder.\n"
                f"Configured dirs: {list(self.broll_dirs.values())}\n"
                "Add your Seedance space/ancient/cosmic clips to these folders."
            )

        return valid_top

    def create_segment_plan(self, duration: float, beats: list, top_categories: list) -> list:
        segments = []
        # Build complete pool of all folders that have clips
        all_folders = []
        for folder in self.broll_dirs.values():
            if self.get_all_files_from_dir(folder):
                all_folders.append(folder)
        if not all_folders:
            raise Exception("No broll clips found in any folder.")

        # Per-folder clip pools with used tracking
        folder_pools = {}
        for folder in all_folders:
            folder_pools[folder] = list(self.get_all_files_from_dir(folder))

        # Beat category → preferred folder (best-effort, falls back to rotation)
        broll_cat_to_folder = {
            'space':   self.broll_dirs.get('space',   'space_vids'),
            'ancient': self.broll_dirs.get('ancient', 'ancient_ruins_vids'),
            'cosmic':  self.broll_dirs.get('cosmic',  'cosmic_vids'),
            'sky':     self.broll_dirs.get('sky',     'dark_sky_vids'),
            'temple':  self.broll_dirs.get('temple',  'temple_vids'),
        }

        # Per-folder shuffle indices so we cycle without repeating
        folder_idx = {f: 0 for f in all_folders}
        for f in all_folders:
            random.shuffle(folder_pools[f])

        base_dur   = 4.0
        n_segs     = max(int(duration / base_dur), 1)
        folder_rot = 0

        for i in range(n_segs):
            seg_dur = float(beats[i].get('clip_duration', base_dur)) if i < len(beats) else base_dur
            # Pure round-robin across all folders -- guaranteed variety
            target_folder = all_folders[folder_rot % len(all_folders)]
            folder_rot += 1

            # Pick next clip from folder, cycling
            pool = folder_pools[target_folder]
            idx  = folder_idx[target_folder]
            if idx >= len(pool):
                random.shuffle(pool)
                idx = 0
            chosen = pool[idx]
            folder_idx[target_folder] = idx + 1

            segments.append({
                'type':     'broll',
                'category': target_folder,
                'file':     chosen,
                'duration': seg_dur,
            })
            print(f"    seg {i+1}: {os.path.basename(chosen)} [{os.path.basename(target_folder)}]")

        if not segments:
            raise Exception("No segments created.")

        total = sum(s['duration'] for s in segments)
        if total < duration:
            segments[-1]['duration'] += (duration - total)

        return segments

    def _make_black_filler(self, output_file: str, dur: float, fps: int = 30) -> str:
        """Generate a black video segment of exact duration — used when broll clip fails."""
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', f'color=c=black:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:r={fps}:d={dur}',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-r', str(fps), '-an', '-t', str(dur),
            output_file
        ]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            raise Exception(f"Black filler failed: {r.stderr.decode()[-200:]}")
        return output_file

    def process_segment_to_file(self, segment: dict, output_file: str,
                                fps: int = 30, progress_callback=None) -> str:
        """Process one broll segment. ALWAYS returns a valid file — never skips.
        If the clip fails, falls back to a black filler of the correct duration
        so total video length is preserved and text timestamps stay in sync."""
        dur = segment['duration']
        source_file = segment['file']
        w, h, aspect = self.get_video_info(source_file)

        cmd = ['ffmpeg', '-y', '-progress', 'pipe:1', '-nostats',
               '-i', source_file, '-t', str(dur)]

        vf = []
        if aspect and aspect < (OUTPUT_WIDTH / OUTPUT_HEIGHT):
            vf += [f"scale={OUTPUT_WIDTH}:-2:force_original_aspect_ratio=decrease",
                   f"pad={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:(ow-iw)/2:(oh-ih)/2:black"]
        else:
            vf += [f"scale=-2:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase",
                   f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}"]

        # fps=30 MUST come first to convert VFR → CFR before any other filter
        vf = [f"fps={fps}"] + vf
        vf += ["eq=brightness=0.02:contrast=1.05:saturation=1.1", "format=yuv420p"]
        cmd += ['-vf', ','.join(vf), '-c:v', 'libx264', '-preset', 'ultrafast',
                '-crf', '23', '-pix_fmt', 'yuv420p', '-r', str(fps), '-an', output_file]

        success  = False
        err_text = ""
        if progress_callback:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True, bufsize=1)
            total_f = int(dur * fps)
            last_f  = 0
            for line in proc.stdout:
                if line.startswith('frame='):
                    try:
                        cf = int(line.split('=')[1].strip())
                        if cf > last_f:
                            last_f = cf
                            progress_callback(cf, total_f)
                    except:
                        pass
            stderr_out = proc.stderr.read() if proc.stderr else ""
            proc.wait()
            success = proc.returncode == 0
            err_text = stderr_out
        else:
            r = subprocess.run(cmd, capture_output=True)
            success = r.returncode == 0
            err_text = r.stderr.decode(errors='replace')

        if not success:
            err_line = err_text.strip().splitlines()[-1] if err_text.strip() else "unknown error"
            print(f"\n  ⚠ Clip failed ({os.path.basename(source_file)}): {err_line[:150]}")
            print(f"  ⚠ Using black filler ({dur:.2f}s)")
            return self._make_black_filler(output_file, dur, fps)

        return output_file

    def _add_cta_overlay(self, video_input: str, output_path: str, duration: float):
        end_time = round(max(duration - 4, 1), 3)
        vf = (
            f"drawtext=text='Vaults of History'"
            f":fontcolor=yellow:fontsize=42:font=Arial"
            f":borderw=2:bordercolor=black:shadowx=2:shadowy=2"
            f":x=(w-text_w)/2:y=h*0.91:enable='gt(t\\,{end_time})'"
        )
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', video_input, '-vf', vf, '-c:a', 'copy', output_path],
            capture_output=True
        )
        if result.returncode != 0:
            subprocess.run(['ffmpeg', '-y', '-i', video_input, '-c', 'copy', output_path],
                           check=True, capture_output=True)
        else:
            print(f"  ✨ CTA added")

    def create_finance_video(self, bg_volume: float = 0.12, fps: int = 30) -> bool:
        import time
        t0 = time.time()

        print(f"\n{'='*70}")
        print(f"📊  FINANCE EXPLAINER v1")
        print(f"{'='*70}")

        try:
            duration = self.get_audio_duration()
            print(f"⏱  {duration:.2f}s")
        except Exception as e:
            raise Exception(f"STEP 1 FAILED: {e}")

        print(f"\n[STEP 2] Transcribing...")
        transcription = self.transcribe_with_whisper()
        if not transcription:
            raise Exception("Transcription failed")

        full_text        = transcription.get('text', '').strip()
        whisper_segments = transcription.get('segments', [])
        print(f"  ✅ {len(full_text)} chars, {len(whisper_segments)} segments")

        print(f"\n[STEP 3] GPT Call 1: Story Beats...")
        try:
            topic_hint   = list(self.broll_dirs.keys())[0] if self.broll_dirs else "space"
            beats_result = analyze_story_beats(full_text, whisper_segments, topic_hint, duration)
            topic        = beats_result.get('topic', 'default')
            beats        = beats_result.get('beats', [])

            # CRITICAL: GPT Call 1 only sees segment-level [start-end] brackets.
            # When it splits one segment into multiple beats, it INVENTS the
            # split-point timestamps. Recompute every beat's start_time/end_time
            # from actual Whisper word timestamps for frame-accurate rendering.
            _whisper_words = build_whisper_word_list(whisper_segments)
            beats = realign_beat_times(beats, _whisper_words)
            print(f"  🎯 Realigned {len(beats)} beat timestamps to Whisper word boundaries")
        except Exception as e:
            raise Exception(f"STEP 3 FAILED: {e}")

        print(f"\n[STEP 4] GPT Call 2: Render Decisions...")
        try:
            decisions = generate_render_decisions(beats, topic)
        except Exception as e:
            raise Exception(f"STEP 4 FAILED: {e}")

        print(f"\n[STEP 4b] GPT Call 3: Freeform Motion Design...")
        try:
            freeform_scenes = generate_freeform_scenes(beats, topic)
        except Exception as e:
            # Freeform is the new, riskier layer -- if it fails entirely,
            # fall back to no freeform objects rather than failing the
            # whole video. number_counter/text/etc still render fine.
            print(f"  ⚠ Freeform generation failed, continuing without it: {e}")
            freeform_scenes = []

        print(f"\n[STEP 5] Validating...")
        decisions = validate_decisions(decisions, beats)
        if freeform_scenes:
            freeform_scenes = validate_freeform_scenes(freeform_scenes, beats)

        print(f"\n[STEP 6] Music...")
        bg_music = MUSIC_MAP.get(topic, MUSIC_MAP['default'])
        if not os.path.exists(bg_music):
            bg_music = MUSIC_MAP['default']
            if not os.path.exists(bg_music):
                for fname in (os.listdir('bg_musics') if os.path.exists('bg_musics') else []):
                    if fname.endswith('.mp3'):
                        bg_music = os.path.join('bg_musics', fname)
                        break
                else:
                    bg_music = None
        print(f"  🎵 {bg_music}")

        temp_files    = []
        concat_list   = "concat_list.txt"
        concat_output = "concatenated_video.mp4"
        audio_output  = "audio_mixed.mp4"

        try:
            if USE_PROCEDURAL_BACKGROUND:
                print(f"\n[STEP 7-10] Procedural background...")
                generate_procedural_background(beats, topic, duration, concat_output,
                                                 width=OUTPUT_WIDTH, height=OUTPUT_HEIGHT, fps=fps)
            else:
                print(f"\n[STEP 7] B-roll matching...")
                top_categories = self.match_broll_categories(full_text)
                print(f"  📊 {top_categories}")

                print(f"\n[STEP 8] Segment plan...")
                try:
                    video_segments = self.create_segment_plan(duration, beats, top_categories)
                    print(f"  ✅ {len(video_segments)} segments")
                except Exception as e:
                    raise Exception(f"STEP 8 FAILED: {e}")

                print(f"\n[STEP 9] Processing segments...")
                try:
                    from tqdm import tqdm
                    use_tqdm = True
                except:
                    use_tqdm = False

                for i, seg in enumerate(video_segments):
                    temp_file = f"temp_segment_{i:02d}.mp4"
                    t_seg     = time.time()

                    if use_tqdm:
                        total_f = int(seg['duration'] * fps)
                        pbar    = tqdm(total=total_f,
                                       desc=f"  Seg {i+1}/{len(video_segments)}: {os.path.basename(seg['file'])[:28]}",
                                       unit='frame')
                        def upd(c, t, pb=pbar):
                            pb.n = min(c, t); pb.refresh()
                        result = None
                        try:
                            result = self.process_segment_to_file(seg, temp_file, fps, upd)
                        finally:
                            pbar.n = pbar.total; pbar.refresh(); pbar.close()
                            print(f"    ✓ {time.time()-t_seg:.1f}s")
                    else:
                        print(f"  {i+1}/{len(video_segments)}: {os.path.basename(seg['file'])}", end='', flush=True)
                        result = self.process_segment_to_file(seg, temp_file, fps)
                        print(f" ✓ ({time.time()-t_seg:.1f}s)")

                    if os.path.exists(temp_file):
                        temp_files.append(temp_file)

                if not temp_files:
                    raise Exception("No segments processed")

                print(f"\n[STEP 10] Concatenating {len(temp_files)} segments...")
                with open(concat_list, 'w') as f:
                    for tf in temp_files:
                        f.write(f"file '{tf}'\n")
                r = subprocess.run(['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                                     '-i', concat_list, '-c', 'copy', concat_output],
                                    capture_output=True)
                if r.returncode != 0:
                    raise Exception(f"Concat failed: {r.stderr.decode()[-200:]}")
                print(f"  ✅ Done")

            print(f"\n[STEP 11] Audio mix...")
            cmd = ['ffmpeg', '-y', '-i', concat_output, '-i', self.audio_path]
            if bg_music and os.path.exists(bg_music):
                cmd += ['-i', bg_music]
                fc = (
                    f'[1:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,volume=1.0[voice];'
                    f'[2:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,'
                    f'volume={bg_volume},aloop=loop=-1:size=2e+09[bg];'
                    f'[voice][bg]amix=inputs=2:duration=first:dropout_transition=2,aresample=48000[aout]'
                )
                cmd += ['-filter_complex', fc, '-map', '0:v', '-map', '[aout]']
            else:
                cmd += ['-map', '0:v', '-map', '1:a']
            cmd += ['-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                    '-ar', '48000', '-ac', '2', '-t', str(duration), audio_output]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                raise Exception(f"Audio failed: {r.stderr.decode()[-200:]}")
            print(f"  ✅ Mixed")

            print(f"\n[STEP 12] OpenCV text render...")
            try:
                render_text_overlay_opencv(audio_output, decisions, beats, whisper_segments,
                                            self.output_path, freeform_scenes=freeform_scenes)
            except Exception as e:
                print(f"  ❌ Render failed: {e}")
                traceback.print_exc()
                subprocess.run(['ffmpeg', '-y', '-i', audio_output, '-c', 'copy', self.output_path],
                               check=True, capture_output=True)

            if os.path.exists(audio_output):
                os.remove(audio_output)

            print(f"\n[STEP 13] CTA...")
            cta_output = self.output_path.replace(".mp4", "_cta.mp4")
            self._add_cta_overlay(self.output_path, cta_output, duration)
            self.output_path = cta_output

            if not os.path.exists(self.output_path):
                raise Exception(f"Output missing: {self.output_path}")

            file_size  = os.path.getsize(self.output_path) / (1024 * 1024)
            total_time = time.time() - t0

            print(f"\n{'='*70}")
            print(f"✅ COMPLETE!")
            print(f"📁 {self.output_path}")
            print(f"💾 {file_size:.2f} MB | ⏱ {duration:.1f}s | ⚡ {total_time:.0f}s")
            print(f"🎭 {len(beats)} beats | 📝 {len(decisions)} decisions")
            print(f"{'='*70}\n")
            return True

        except Exception as e:
            print(f"\n❌ Pipeline error: {e}")
            traceback.print_exc()
            return False

        finally:
            print(f"\n🧹 Cleanup...")
            for tf in temp_files:
                if os.path.exists(tf):
                    try: os.remove(tf)
                    except: pass
            for f in [concat_list, concat_output, audio_output]:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass
            for tmp in glob.glob("*TEMP_MPY*.mp4") + glob.glob("*_noaudio_tmp.mp4") + glob.glob("*_cfr_tmp.mp4") + glob.glob("temp_segment_*.mp4"):
                try: os.remove(tmp)
                except: pass


# ============================================================
# NICHE TEMPLATES
# Only relevant if USE_PROCEDURAL_BACKGROUND is False (broll-clip
# fallback path). With procedural backgrounds on, this is unused.
# ============================================================
NICHE_TEMPLATES = {
    'finance': {
        'broll_dirs': {},
        'keyword_map': {}
    },
}



# ============================================================
# FASTAPI
# ============================================================
@app.get("/")
def root():
    return {"service": "Finance Explainer v1", "status": "running",
            "openai_key": bool(OPENAI_API_KEY)}

@app.post("/generate")
async def generate_video_api(background_tasks: BackgroundTasks, niche: str = "finance"):
    global current_job
    if current_job["status"] == "processing":
        return {"message": "Already processing", "status": "processing"}
    current_job = {"status": "processing", "progress": 0, "output": None,
                   "error": None, "started_at": datetime.now().isoformat(), "niche": niche}
    background_tasks.add_task(process_video, niche)
    return {"message": f"Started niche={niche}", "status": "processing"}

def process_video(niche: str = "finance"):
    global current_job
    try:
        current_job["progress"] = 5
        audio_url   = "https://raw.githubusercontent.com/RandomSci/Automation_For_Love_Niche/main/Audio_Voice/vaults_narration.mp3"
        audio_file  = "Audio_Voice/vaults_narration.mp3"
        output_file = "vaults_output.mp4"
        trans_file  = f"{os.path.splitext(audio_file)[0]}_transcription.json"

        print(f"\n📥 Downloading audio...")
        os.makedirs("Audio_Voice", exist_ok=True)
        resp = requests.get(audio_url, timeout=30)
        if resp.status_code != 200:
            raise Exception(f"HTTP {resp.status_code}")
        with open(audio_file, "wb") as f:
            f.write(resp.content)
        print(f"  ✅ {len(resp.content)//1024}KB")

        current_job["progress"] = 10

        for old in [output_file, output_file.replace(".mp4", "_cta.mp4"),
                    "audio_mixed.mp4", trans_file]:
            if os.path.exists(old):
                os.remove(old)

        current_job["progress"] = 15
        niche_config = NICHE_TEMPLATES.get(niche, NICHE_TEMPLATES['finance'])
        gen = FinanceGenerator(audio_path=audio_file, output_path=output_file,
                              niche_config=niche_config)

        current_job["progress"] = 20
        success = gen.create_finance_video(bg_volume=0.12, fps=30)
        current_job["progress"] = 95

        final = output_file.replace(".mp4", "_cta.mp4")
        if success and os.path.exists(final):
            current_job.update({"status": "completed", "progress": 100, "output": final})
            print(f"\n🎉 DONE: {final}")
        else:
            raise Exception("Pipeline failed or output missing")

    except Exception as e:
        current_job.update({"status": "error", "error": str(e), "progress": 0})
        print(f"\n❌ FAILED: {e}")
        traceback.print_exc()

@app.get("/status")
def check_status():
    return {**current_job, "ready": current_job["status"] == "completed",
            "niche": current_job.get("niche", "finance")}

@app.get("/download")
def download_video():
    if current_job["status"] != "completed":
        raise HTTPException(400, f"Not ready: {current_job['status']}")
    if not current_job["output"] or not os.path.exists(current_job["output"]):
        raise HTTPException(404, "File not found")
    return FileResponse(current_job["output"], media_type="video/mp4",
                        filename=f"finance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Finance Explainer v1 on :{port} | Key: {'set' if OPENAI_API_KEY else 'MISSING'}")
    uvicorn.run(app, host="0.0.0.0", port=port)