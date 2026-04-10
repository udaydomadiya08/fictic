import time
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Record start time
start_time = time.time()

import shutil
import os
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, concatenate_videoclips
from moviepy.config import change_settings
from moviepy.editor import TextClip
# Use ImageMagick 7 binary
change_settings({"IMAGEMAGICK_BINARY": "magick"})


def safe_clear_pycache(start_path='.'):
    for root, dirs, files in os.walk(start_path):
        for d in dirs:
            if d == '__pycache__':
                full_path = os.path.join(root, d)
                try:
                    shutil.rmtree(full_path)
                    print(f"Deleted {full_path}")
                except PermissionError as e:
                    print(f"Permission denied to delete {full_path}: {e}")

safe_clear_pycache()

import shutil

bad = shutil.which("demucs")
if bad:
    raise RuntimeError(
        f"❌ System demucs detected at {bad}. "
        "Use demucs310 python -m demucs.separate only."
    )


import shutil
from pathlib import Path
def delete_path(path):
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            path.unlink()
        print(f"✅ Deleted: {path}")
    except Exception as e:
        print(f"❌ Error deleting {path}: {e}")

# Cache directories
cache_dirs = [
    Path.home() / "Library" / "Caches",
    Path("/Library/Caches"),
]

# Log directories
log_dirs = [
    Path.home() / "Library" / "Logs",
    Path("/Library/Logs")
]





dirs_to_clean = cache_dirs + log_dirs 
print("🧹 Cleaning up...")

for d in dirs_to_clean:
    if d.exists():
        for item in d.iterdir():
            delete_path(item)
    else:
        print(f"⚠️ Directory not found: {d}")

print("✅ Done cleaning up!")


#!/usr/bin/env python3
"""
fictic_punch_intro.py

Music-driven Fictic-style punch-intro edit.
- Transcribes audio directly from input videos (Whisper per-video).
- Picks single strongest dialogue from inputs (avoids CTAs).
- Places that dialogue first (full original audio), with ducked music under it.
- Fills the rest of the music duration with unique creative filler clips (no repeats).
- Final duration == music duration. No black/static end.
Usage:
 python3 fictic_punch_intro.py --input_videos ./input_vid --music ./music --out ./out/final.mp4 --whisper_model small --language en
"""
import os, sys, argparse, random, tempfile, shutil, math, warnings
from tqdm import tqdm
import numpy as np
import librosa
import cv2
from moviepy.editor import *
from moviepy.video.fx import all as vfx

import os
import glob
import argparse
import tempfile
import shutil
import random
import math
import warnings
from tqdm import tqdm

import numpy as np

import librosa

from moviepy.editor import (
    VideoFileClip, AudioFileClip, concatenate_videoclips,
    CompositeVideoClip, TextClip, vfx, CompositeAudioClip
)
from moviepy.audio.fx.all import audio_loop
from moviepy.audio.AudioClip import concatenate_audioclips

from transit import stitching_with_audio_visual_cues


warnings.filterwarnings("ignore")

# ---------- Config ----------
OUT_W, OUT_H = 1080, 1920
FPS = 30
FONT_NAME = "/Users/uday/Downloads/VIDEOYT/Anton-Regular.ttf"
FONT_SIZE = 78
STROKE_WIDTH = 3
TEXT_COLOR = "white"
GLOW_COLORS = ["#ffb86b", "#f1fa8c"]
WHISPER_DEFAULT = "tiny"

MIN_CLIP_DUR = 0.6 
GLITCH_PROB = 0.08
GLITCH_MAX_DUR = 0.10

# phrases to avoid as punch-lines (outros, CTAs)
BAD_WORDS = ["thanks for watching", "subscribe", "like", "follow", "please subscribe", "hit the bell", "youtube", "check out", "visit"]

# ---------- Utilities ----------

from ultralytics import YOLO
import numpy as np
import cv2

# Load YOLOv8 Nano instance segmentation model once at start
segmentation_model = YOLO('yolov8n-seg.pt')

def get_object_mask(frame, conf_threshold=0.25):
    results = segmentation_model.predict(frame, conf=conf_threshold, verbose=False)
    if results[0].masks is None:
        return None
    masks = results[0].masks.data.cpu().numpy()  # (N, H, W)
    if masks.shape[0] == 0:
        return None
    mask_areas = [m.sum() for m in masks]
    largest_idx = np.argmax(mask_areas)
    return masks[largest_idx].astype(np.uint8)


def find_videos(folder):
    pats = ["*.mp4","*.mov","*.mkv","*.avi","*.webm","*.mpg","*.mpeg"]
    files = []
    for p in pats:
        files += glob.glob(os.path.join(folder, p))
    return sorted(files)

def find_music_file(arg):
    if os.path.isdir(arg):
        pats = ["*.mp3","*.wav","*.m4a","*.flac","*.aac"]
        files = []
        for p in pats:
            files += glob.glob(os.path.join(arg, p))
        files = sorted(files)
        if not files:
            raise FileNotFoundError("No music in folder: " + arg)
        return files[0]
    elif os.path.isfile(arg):
        return arg
    else:
        raise FileNotFoundError("Music path not found: " + arg)
def detect_beats(music_path, fps=44100):
    audio = AudioFileClip(music_path)
    arr = audio_to_mono_safe(audio, fps=22050)

    
    # Ensure mono
    if arr.ndim == 2:
        arr_mono = arr.mean(axis=1)
    else:
        arr_mono = arr

    # compute frame-wise energy
    frame_size = 2048
    hop_size = 512
    energies = []
    times = []

    for i in range(0, len(arr_mono) - frame_size, hop_size):
        frame = arr_mono[i:i+frame_size]
        energies.append(np.sum(frame**2))
        times.append(i / fps)

    from scipy.signal import find_peaks
    peaks, _ = find_peaks(np.array(energies), distance=20)
    beat_times = np.array(times)[peaks]
    return beat_times.tolist()




def center_crop_vertical(clip, out_w=OUT_W, out_h=OUT_H):
    scale = max(out_w/clip.w, out_h/clip.h)
    nw, nh = int(clip.w*scale), int(clip.h*scale)
    clip = clip.resize((nw, nh))
    x1 = (nw - out_w)//2; y1 = (nh - out_h)//2
    return clip.crop(x1=x1, y1=y1, x2=x1+out_w, y2=y1+out_h)

def quick_glitch_overlay(clip):
    overlays = []
    rng = np.random.RandomState(int(random.random()*1e6))
    t = 0.0; step = 0.07
    while t < clip.duration:
        if rng.rand() < GLITCH_PROB:
            g_d = float(rng.rand()*GLITCH_MAX_DUR) + 0.03
            t0 = t; t1 = min(clip.duration, t+g_d)
            sub = clip.subclip(t0, t1).fx(vfx.colorx, 1.5).set_opacity(0.45)
            dx = int(rng.randint(-8,8)); dy = int(rng.randint(-8,8))
            sub = sub.set_position((dx,dy)).set_start(t0)
            overlays.append(sub)
            t += g_d
        t += step
    if overlays:
        return CompositeVideoClip([clip] + overlays).set_duration(clip.duration)
    return clip

def shift_masked_object(frame, mask, shift_x):
    """
    Shift object (frame * mask) horizontally by shift_x pixels.
    Pads empty space with black (zeros), no wrap-around.
    Positive shift_x shifts right, negative shifts left.
    """
    h, w = frame.shape[:2]
    shifted = np.zeros_like(frame)
    mask_exp = mask[..., None]

    if shift_x > 0:
        # Shift right: vacated left area black
        shifted[:, shift_x:, :] = (frame[:, :w - shift_x, :] * mask_exp[:, :w - shift_x, :]).astype(np.uint8)
    elif shift_x < 0:
        # Shift left: vacated right area black
        shifted[:, :w + shift_x, :] = (frame[:, -shift_x:, :] * mask_exp[:, -shift_x:, :]).astype(np.uint8)
    else:
        # No shift
        shifted = (frame * mask_exp).astype(np.uint8)
    return shifted


def masked_object_transition(c1, c2, dur=0.4):
    frame1 = c1.get_frame(0)
    frame2 = c2.get_frame(0)
    mask1 = get_object_mask(frame1)
    mask2 = get_object_mask(frame2)

    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]

    if mask1 is None:
        mask1 = np.zeros((h1, w1), dtype=np.uint8)
    else:
        mask1 = cv2.resize(mask1, (w1, h1))

    if mask2 is None:
        mask2 = np.zeros((h2, w2), dtype=np.uint8)
    else:
        mask2 = cv2.resize(mask2, (w2, h2))

    OUT_W, OUT_H = c1.w, c1.h

    def frame_fn(t):
        alpha = t / dur
        bg = (1 - alpha) * frame1 + alpha * frame2

        dx1 = int(OUT_W * alpha)
        obj1 = shift_masked_object(frame1, mask1, -dx1)

        dx2 = int(OUT_W * (1 - alpha))
        obj2 = shift_masked_object(frame2, mask2, dx2)

        comp = bg * (1 - (mask1 + mask2)[..., None]) + obj1 + obj2
        return np.clip(comp, 0, 255).astype(np.uint8)

    return VideoClip(frame_fn, duration=dur).set_fps(FPS).resize((OUT_W, OUT_H))

def simple_object_transition(c1, c2, dur=0.4):
    frame1 = c1.get_frame(0)
    frame2 = c2.get_frame(0)
    mask1 = get_object_mask(frame1)
    mask2 = get_object_mask(frame2)

    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]

    if mask1 is None:
        mask1 = np.zeros((h1, w1), dtype=np.uint8)
    else:
        mask1 = cv2.resize(mask1, (w1, h1))

    if mask2 is None:
        mask2 = np.zeros((h2, w2), dtype=np.uint8)
    else:
        mask2 = cv2.resize(mask2, (w2, h2))

    OUT_W, OUT_H = c1.w, c1.h

    def frame_fn(t):
        alpha = t / dur
        # Crossfade backgrounds
        bg = (1 - alpha)**2 * frame1 + (alpha**2) * frame2

        # Cross-slide objects horizontally
        dx1 = int(OUT_W * alpha)
        obj1 = shift_masked_object(frame1, mask1, -dx1)

        dx2 = int(OUT_W * (1 - alpha))
        obj2 = shift_masked_object(frame2, mask2, dx2)

        # Combine with masking to avoid overlap
        combined_mask = np.clip(mask1 + mask2, 0, 1)[..., None]
        comp = bg * (1 - combined_mask) + obj1 + obj2
        return np.clip(comp, 0, 255).astype(np.uint8)

    return VideoClip(frame_fn, duration=dur).set_fps(FPS).resize((OUT_W, OUT_H))

def simple_object_fade_transition(c1, c2, dur=0.4):
    frame1 = c1.get_frame(0)
    frame2 = c2.get_frame(0)
    mask1 = get_object_mask(frame1)
    mask2 = get_object_mask(frame2)

    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]

    if mask1 is None:
        mask1 = np.zeros((h1, w1), dtype=np.uint8)
    else:
        mask1 = cv2.resize(mask1, (w1, h1))

    if mask2 is None:
        mask2 = np.zeros((h2, w2), dtype=np.uint8)
    else:
        mask2 = cv2.resize(mask2, (w2, h2))

    OUT_W, OUT_H = c1.w, c1.h

    def frame_fn(t):
        alpha = t / dur
        # Crossfade backgrounds
        bg = (1 - alpha) * frame1 + alpha * frame2
        
        # Current object fully visible
        obj1 = frame1 * mask1[..., None]

        # Next object fades in
        obj2 = (frame2 * mask2[..., None]) * alpha

        # Combine masks carefully to avoid overlap artifacts
        comp = bg * (1 - np.clip(mask1 + mask2, 0, 1)[..., None]) + obj1 + obj2

        return np.clip(comp, 0, 255).astype(np.uint8)

    return VideoClip(frame_fn, duration=dur).set_fps(FPS).resize((OUT_W, OUT_H))


def simple_object_slide_in_transition(c1, c2, dur=0.4, direction='right'):
    frame1 = c1.get_frame(0)
    frame2 = c2.get_frame(0)
    mask1 = get_object_mask(frame1)
    mask2 = get_object_mask(frame2)

    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]

    if mask1 is None:
        mask1 = np.zeros((h1, w1), dtype=np.uint8)
    else:
        mask1 = cv2.resize(mask1, (w1, h1))

    if mask2 is None:
        mask2 = np.zeros((h2, w2), dtype=np.uint8)
    else:
        mask2 = cv2.resize(mask2, (w2, h2))

    OUT_W, OUT_H = c1.w, c1.h

    def frame_fn(t):
        alpha = t / dur
        bg = (1 - alpha) * frame1 + alpha * frame2

        # Current object static
        obj1 = frame1 * mask1[..., None]

        # Incoming object slides in
        if direction == 'right':
            dx = int(OUT_W * (1 - alpha))
            obj2 = shift_masked_object(frame2, mask2, dx)
        elif direction == 'left':
            dx = int(-OUT_W * (1 - alpha))
            obj2 = shift_masked_object(frame2, mask2, dx)
        else:
            dx = 0
            obj2 = frame2 * mask2[..., None]

        combined_mask = np.clip(mask1 + mask2, 0, 1)[..., None]
        comp = bg * (1 - combined_mask) + obj1 + obj2
        return np.clip(comp, 0, 255).astype(np.uint8)

    return VideoClip(frame_fn, duration=dur).set_fps(FPS).resize((OUT_W, OUT_H))

from smtcro import smart_full_crop  # import from your module
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx, CompositeVideoClip
import random
import numpy as np

GLITCH_PROB = 0.1    # chance of glitch per step
GLITCH_MAX_DUR = 0.06  # max glitch duration in seconds
FPS = 30

def safe_glitch_overlay(clip):
    """Add tiny, safe glitch overlays without causing stutter."""
    overlays = []
    rng = np.random.RandomState(int(random.random()*1e6))
    t = 0.0
    step = 0.07
    while t < clip.duration:
        if rng.rand() < GLITCH_PROB:
            g_d = float(rng.rand()*GLITCH_MAX_DUR) + 0.02
            t0 = t
            t1 = min(clip.duration, t + g_d)
            sub = clip.subclip(t0, t1).fx(vfx.colorx, 1.3).set_opacity(0.35)
            # minimal pixel shift for safe glitch
            dx = int(rng.randint(-2,2))
            dy = int(rng.randint(-2,2))
            sub = sub.set_position((dx,dy)).set_start(t0)
            overlays.append(sub)
            t += g_d
        t += step
    if overlays:
        return CompositeVideoClip([clip] + overlays).set_duration(clip.duration)
    return clip

def make_variant(src_path, start, dur, kind="normal"):
    src = VideoFileClip(src_path)
    end = min(src.duration, start + dur)
    clip = src.subclip(start, end).set_fps(FPS)

    # --- SMART CROP ---
    clip = smart_crop_clip(clip)  # your ultra-smooth crop function

    # --- SMOOTH ZOOM ---
    zoom = 1.02 + random.random()*0.03
    clip = clip.fx(vfx.resize, lambda t: 1.0 + (zoom-1.0)*(t/max(1e-6, clip.duration)))

    # --- COLOR ENHANCEMENT ---
    clip = clip.fx(vfx.colorx, 1.05)

    # --- SAFE GLITCH (tiny shifts only, won't stutter) ---
    clip = safe_glitch_overlay(clip)

    return clip


# ---------- Whisper per-video transcription ----------
# def transcribe_each_video(video_paths, whisper_model_name="small", language=None, tmpdir=None):
#     """
#     Transcribe each video separately. Return list of segments:
#     [{'video': path, 'start': s, 'end': e, 'text': t}, ...]
#     """
#     if tmpdir is None:
#         tmpdir = tempfile.mkdtemp(prefix="fictic_whisper_")
#     model = whisper.load_model(whisper_model_name)
#     all_segs = []
#     for p in tqdm(video_paths, desc="Whisper (per-video)"):
#         try:
#             a_out = os.path.join(tmpdir, os.path.basename(p) + ".wav")
#             # extract short audio (entire audio) for whisper
#             try:
#                 ac = AudioFileClip(p)
#                 ac.write_audiofile(a_out, fps=16000, verbose=False, logger=None)
#                 ac.close()
#             except Exception:
#                 # fallback: use moviepy extraction via video clip
#                 v = VideoFileClip(p)
#                 if v.audio:
#                     v.audio.write_audiofile(a_out, fps=16000, verbose=False, logger=None)
#                 v.reader.close(); v.audio = None

#             opts = {}
#             if language:
#                 opts['language'] = language
#                 opts['task'] = 'transcribe'
#             res = model.transcribe(a_out, **opts)
#             for seg in res.get('segments', []):
#                 all_segs.append({'video': p, 'start': float(seg['start']), 'end': float(seg['end']), 'text': seg['text'].strip()})
#         except Exception as e:
#             # just skip failures for single file
#             # print("Whisper failed for", p, e)
#             continue
#     return all_segs, tmpdir

# ---------- Score segments and pick best punch ----------
def score_segments(segments, tmp_audio_dir):
    """
    For each segment compute a score based on RMS/centroid/duration and filter CTAs.
    segments: list of {'video','start','end','text'}
    tmp_audio_dir used to load original audio for segment scoring (we wrote .wav earlier)
    """
    scored = []
    for seg in segments:
        text = (seg.get('text') or "").strip()
        lowtxt = text.lower()
        # filter CTAs/outros
        if any(b in lowtxt for b in BAD_WORDS):
            continue
        duration = max(0.01, seg['end'] - seg['start'])
        # load a short slice from the source file using librosa by reading the file we wrote earlier
        base_wav = os.path.join(tmp_audio_dir, os.path.basename(seg['video']) + ".wav")
        if os.path.exists(base_wav):
            try:
                # offset equals seg['start']
                y, sr = librosa.load(base_wav, sr=16000, offset=max(0, seg['start']), duration=duration)
                rms = float(np.sqrt(np.mean(y**2))) if y.size else 0.0
                cent = librosa.feature.spectral_centroid(y+1e-16, sr=sr) if y.size else np.zeros((1,1))
                cent_var = float(np.var(cent)) if cent.size else 0.0
            except Exception:
                rms, cent_var = 0.0, 0.0
        else:
            # fallback: light scoring by duration only
            rms, cent_var = 0.0, 0.0
        score = rms * 120.0 + cent_var * 2.0 + duration * 0.4 + (len(text.split())*0.8)
        scored.append((score, seg))
    if not scored:
        return None
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1]

# ---------- Subtitle rendering ----------
import math
from moviepy.editor import TextClip
import moviepy.video.fx.all as vfx

import math
import numpy as np
from moviepy.editor import TextClip
import moviepy.video.fx.all as vfx

from moviepy.editor import TextClip, CompositeVideoClip
from moviepy.video.fx.all import fadein, fadeout, resize

import numpy as np
import random
from moviepy.editor import VideoClip, CompositeVideoClip, TextClip
from moviepy.video.fx.all import fadein, fadeout, resize
import colorsys

def create_gradient_frame(w, h, offset, direction, left_color, right_color):
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(x, y)

    if direction == "diagonal_tl_br":
        pos = (X + Y) / 2
        ratio = (pos + offset) % 1

    elif direction == "diagonal_br_tl":
        pos = (X + Y) / 2
        ratio = (1 - pos + offset) % 1

    else:
        ratio = np.full((h, w), offset)

    gradient = (1 - ratio[..., None]) * left_color + ratio[..., None] * right_color
    return gradient.astype(np.uint8)

def hsv_to_rgb_array(h, s, v):
    """Convert arrays of HSV to RGB arrays (values 0-1)"""
    import colorsys
    rgb = np.array([colorsys.hsv_to_rgb(h_, s, v) for h_ in h.flatten()])  # (N,3)
    return rgb.reshape(h.shape + (3,))

def random_bright_color():
    # Return bright HSV (random hue, full saturation & value)
    h = random.random()
    s = 1.0
    v = 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return np.array([r, g, b]) * 255

def group_words(text, n=3):
    words = text.split()
    return [" ".join(words[i:i+n]) for i in range(0, len(words), n)]

import random
from moviepy.editor import *
from moviepy.video.fx.all import fadein, fadeout, resize

def group_words(text, n=3):
    words = text.split()
    return [" ".join(words[i:i+n]) for i in range(0, len(words), n)]

def swipe_direction():
    return random.choice([
        lambda clip: clip.set_position(lambda t: (OUT_W/2, OUT_H*0.75 - 120*(1-t))),   # up
        lambda clip: clip.set_position(lambda t: (OUT_W/2, OUT_H*0.75 + 120*(1-t))),   # down
        lambda clip: clip.set_position(lambda t: (OUT_W/2 - 200*(1-t), OUT_H*0.75)),   # left swipe
        lambda clip: clip.set_position(lambda t: (OUT_W/2 + 200*(1-t), OUT_H*0.75)),   # right swipe
        lambda clip: clip.set_position(lambda t: (OUT_W/2 + 160*(1-t), OUT_H*0.75 - 160*(1-t))),  # diag up-left
        lambda clip: clip.set_position(lambda t: (OUT_W/2 - 160*(1-t), OUT_H*0.75 + 160*(1-t)))   # diag down-right
    ])
# === SHINE THROUGH STROKE (text stays still) ===


def segment_subtitle_clip(text, start, dur):
    TARGET_Y = int(OUT_H * 0.72)  # move slightly up for bigger font ✅

    neon_palette = [
        "#00FFFF", "#FF00FF", "#39FF14", "#FF3131",
        "#FFD700", "#00BFFF", "#9400D3", "#00FF7F"
    ]

    words = text.split()
    per_word_dur = dur / max(len(words), 1)

    clips = []

    for i, word in enumerate(words):
        w_start = start + i * per_word_dur
        w_dur = per_word_dur

        neon = random.choice(neon_palette)

        font_big = int(FONT_SIZE * 1.15)  # ✅ made bigger

        # === Base WHITE text (dominant ✅) ===
        base = TextClip(
            word,
            fontsize=font_big,
            font=FONT_NAME,
            color="white",
            stroke_color=neon,
            stroke_width=int(font_big * 0.035),  # ✅ proportional stroke
            method="label"
        ).set_start(w_start).set_duration(w_dur)

        base = base.set_position(lambda t: ("center", TARGET_Y - base.h))

        # === Subtle outer glow ===
        glow = TextClip(
            word,
            fontsize=int(font_big * 1.10),  # ✅ glow also scaled
            font=FONT_NAME,
            color=neon,
            stroke_width=0,
            method="label", 
            transparent=True  
        ).set_start(w_start).set_duration(w_dur)

        glow = glow.set_opacity(0.5)
        glow = glow.set_position(lambda t: ("center", TARGET_Y - glow.h))

        # === Shine-through effect ===
        def shine(get_frame, t):
            frame = get_frame(t).astype(float)
            h, w, _ = frame.shape
            sweep = np.linspace(0.65, 1.25, w).reshape(1, w, 1)
            offset = int((t * 240) % w)
            sweep = np.roll(sweep, offset, axis=1)
            return np.clip(frame * sweep, 0, 255).astype(np.uint8)

        base_animated = base.fl(shine)

        # Fade subtly
        for layer in (glow, base_animated):
            layer = layer.fx(fadein, 0.06).fx(fadeout, 0.06)

        clips.extend([glow, base_animated])

    return clips



def overlay_single_subtitle(video_clip, text, dur):
    start = 0.0
    layers = [video_clip]
    for layer in segment_subtitle_clip(text, start, dur):
        layers.append(layer)
    return CompositeVideoClip(layers).set_duration(video_clip.duration)

# ---------- Build filler schedule guided by beats ----------
def build_fillers(video_paths, candidates, target_duration, used_key, beat_gaps=None):
    """
    Build filler clips list (unique keys) until target_duration filled.
    used_key can be a tuple or a set of tuples marking dialogue clips to avoid repetition.
    candidates: list of dicts with path,start,dur,motion,rms,idx (from extract_candidates)
    """
    pool = sorted(
        candidates,
        key=lambda x: (x.get("motion", 0) * 0.7 + x.get("rms", 0) * 50.0 + random.random() * 0.01),
        reverse=True,
    )
    fillers = []

    # Handle both tuple and set for used_key
    if isinstance(used_key, set):
        used = set(used_key)  # copy
    else:
        used = set([used_key])

    remaining = target_duration
    bi = 0

    while remaining > 0.35 and pool:
        gap = (beat_gaps[bi % len(beat_gaps)] if beat_gaps else 1.2)
        dur = min(gap * random.uniform(0.9, 1.12), remaining)
        bi += 1
        found = False

        for i, c in enumerate(pool):
            key = (c["path"], round(c["start"], 2))
            if key in used:
                continue

            take = min(c["dur"], max(0.35, dur))
            try:
                clip = make_variant(c["path"], c["start"], take, kind=random.choice(["normal"]))
                clip = clip.crossfadein(0.06)
                fillers.append((clip, key))
                used.add(key)
                remaining -= clip.duration
                found = True
                pool.pop(i)
                break
            except Exception:
                continue

        if not found:
            break

    # Add random unique snippets if still short
    attempts = 0
    while remaining > 0.25 and attempts < 40:
        p = random.choice(video_paths)
        try:
            v = VideoFileClip(p)
            start = random.uniform(0, max(0, v.duration - 0.6))
            dur = min(remaining, random.uniform(0.5, 1.4))
            key = (p, round(start, 2))
            if key in used:
                v.reader.close()
                v.audio = None
                attempts += 1
                continue

            add = make_variant(p, start, dur, kind=random.choice(["normal"]))
            add = add.crossfadein(0.06)
            fillers.append((add, key))
            used.add(key)
            remaining -= add.duration
            v.reader.close()
            v.audio = None
        except Exception:
            attempts += 1
            continue

    return [f[0] for f in fillers]


# ---------- Extract candidates (to be used for filler selection) ----------
def extract_candidates(video_paths):
    candidates = []
    for idx, p in enumerate(video_paths):
        try:
            v = VideoFileClip(p)
            dur = v.duration
            steps = max(3, min(8, int(dur//0.5) + 1))
            starts = list(np.linspace(0, max(0, dur-0.5), steps))
            starts += [random.uniform(0, max(0, dur-0.5)) for _ in range(2)]
            starts = sorted(set([max(0, s) for s in starts]))
            for s in starts:
                seg_dur = min(1.6, max(0.6, min(1.4, dur - s))) if dur - s > 0.2 else min(0.6, dur - s)
                # audio energy estimate
                try:
                    ac = AudioFileClip(p).subclip(s, min(s + seg_dur, dur))
                    arr = ac.to_soundarray(fps=16000)
                    rms = float(np.sqrt((arr**2).mean())) if arr.size else 0.0
                    ac.close()
                except Exception:
                    rms = 0.0
                motion = visual_motion_estimate(p)
                candidates.append({"path": p, "start": float(s), "dur": float(seg_dur), "rms": float(rms), "motion": float(motion), "idx": idx})
            v.reader.close(); v.audio = None
        except Exception:
            continue
    return candidates

def visual_motion_estimate(path, samples=6):
    try:
        clip = VideoFileClip(path)
        dur = max(0.5, clip.duration)
        times = np.linspace(0, dur, min(samples, max(2, int(dur*2))))
        prev = None; diffs = []
        for t in times:
            f = clip.get_frame(min(t, clip.duration-1e-3))
            gray = np.mean(f.astype(np.float32), axis=2)
            if prev is not None:
                diffs.append(np.mean(np.abs(gray - prev)))
            prev = gray
        clip.reader.close(); clip.audio = None
        return float(np.mean(diffs)) if diffs else 0.0
    except Exception:
        return 0.0




import numpy as np
from moviepy.editor import AudioFileClip
import numpy as np
from moviepy.editor import AudioFileClip
import numpy as np
from moviepy.editor import AudioFileClip

import numpy as np
from moviepy.editor import AudioFileClip

def score_segment(energies_segment):
    """Score bars based on average energy and positive slope (energy buildup)."""
    energies = np.array(energies_segment)
    avg_energy = np.mean(energies)
    x = np.arange(len(energies))
    slope = np.polyfit(x, energies, 1)[0]  # Linear slope
    score = avg_energy * 0.5 + max(0, slope) * 0.5  # Equal weight for energy and buildup
    return score

def find_complete_segment(music_path, bars_per_segment=8, beats_per_bar=4, desired_duration=30.0):
    audio = AudioFileClip(music_path)
    beats = detect_beats(music_path)  # Should return list of beat times in seconds

    if beats is None or len(beats) < beats_per_bar:
        seg_duration = min(audio.duration, desired_duration)
        return 0.0, seg_duration

    # Group beats into bars (each bar has beats_per_bar beats)
    bars = [beats[i:i + beats_per_bar] for i in range(0, len(beats) - beats_per_bar + 1, beats_per_bar)]
    bars = [bar for bar in bars if len(bar) == beats_per_bar]

    # Calculate RMS energy per bar
    energies = []
    for bar in bars:
        start, end = bar[0], bar[-1]
        segment = audio.subclip(start, end)
        try:
            sound_array = segment.to_soundarray(fps=22050)
            rms = np.sqrt(np.mean(sound_array ** 2))
        except Exception:
            rms = 0
        energies.append(rms)

    avg_bar_duration = np.mean([bar[-1] - bar[0] for bar in bars])
    max_bars = max(1, int(desired_duration // avg_bar_duration))
    max_bars = min(max_bars, bars_per_segment, len(bars))

    best_start_idx = 0
    best_score = -np.inf
    best_window_size = 1

    for window_size in range(max_bars, 0, -1):  # Try largest segment size down to 1 bar
        for i in range(len(energies) - window_size + 1):
            segment_energies = energies[i:i + window_size]
            s = score_segment(segment_energies)
            segment_start = bars[i][0]
            segment_end = bars[i + window_size - 1][-1]
            duration = segment_end - segment_start

            if duration <= desired_duration and s > best_score:
                best_score = s
                best_start_idx = i
                best_window_size = window_size

        if best_score > -np.inf:
            # if found a valid segment, break early to prioritize longer highest scoring segment
            break

    segment_start = bars[best_start_idx][0]
    segment_end = bars[best_start_idx + best_window_size - 1][-1]

    return segment_start, segment_end
from moviepy.editor import AudioFileClip
import numpy as np

import numpy as np
from moviepy.editor import AudioFileClip
from scipy.signal import find_peaks

import numpy as np
from moviepy.editor import AudioFileClip

import numpy as np
from moviepy.editor import AudioFileClip
from scipy.signal import find_peaks

import numpy as np
from moviepy.editor import AudioFileClip
from scipy.signal import find_peaks


def audio_to_mono_safe(audio_clip, fps=22050):
    """
    Convert AudioFileClip to mono numpy array safely.
    Returns a 1D numpy array, even if audio is silent or broken.
    """
    try:
        frames = list(audio_clip.iter_frames(fps=fps))  # no quantize/nbytes
        if not frames:
            return np.zeros(int(audio_clip.duration * fps), dtype=np.float32)
        arr = np.array(frames, dtype=np.float32)
        if arr.ndim == 3:  # (n, samples, channels) or (n, channels)?
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.ndim == 2:
            arr = arr.mean(axis=1)  # stereo → mono
        if arr.size == 0:
            arr = np.zeros(int(audio_clip.duration * fps), dtype=np.float32)
        return arr
    except Exception:
        duration = audio_clip.duration if getattr(audio_clip, "duration", None) else 1.0
        return np.zeros(int(duration * fps), dtype=np.float32)


def to_mono_safe(audio_clip, fps=44100):
    """
    Convert AudioFileClip to a safe mono numpy array.
    Returns a zero array if audio is missing or fails.
    """
    try:
        frames = list(audio_clip.iter_frames(fps=fps))
        if not frames:
            duration = audio_clip.duration if getattr(audio_clip, "duration", None) else 1.0
            return np.zeros(int(duration * fps), dtype=np.float32)
        arr = np.array(frames, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        return arr
    except Exception:
        duration = audio_clip.duration if getattr(audio_clip, "duration", None) else 1.0
        return np.zeros(int(duration * fps), dtype=np.float32)


def detect_beats(
    music_path,
    fps=44100,
    frame_size=2048,
    hop_size=512,
    cluster_gap_sec=0.3,
    min_transition_gap_sec=1.5,
):
    """
    Returns transition-ready beat times (seconds) for video editing.
    """
    audio = AudioFileClip(music_path)
    try:
        # --- Load audio and convert to mono safely ---
        frames = list(audio.iter_frames(fps=fps))
        if not frames:
            print("⚠️ detect_beats: empty audio, returning no beats.")
            return []

        arr = np.array(frames, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        arr_mono = arr.mean(axis=1) if arr.ndim > 1 else arr

        # --- Compute frame-wise energy ---
        energies = []
        times = []
        for i in range(0, len(arr_mono) - frame_size, hop_size):
            frame = arr_mono[i:i + frame_size]
            energies.append(np.sum(frame ** 2))
            times.append(i / fps)

        energies = np.array(energies)
        times = np.array(times)

        # --- Initial peak detection (small min distance to detect clusters) ---
        min_peak_distance_frames = max(1, int(cluster_gap_sec * fps / hop_size))
        peaks, _ = find_peaks(energies, distance=min_peak_distance_frames)
        beat_times = times[peaks].tolist()
        beat_energies = energies[peaks]

        if not beat_times:
            return []

        # --- Merge very close beats (keep strongest in cluster) ---
        merged = [beat_times[0]]
        merged_energies = [beat_energies[0]]
        cluster = [beat_times[0]]
        cluster_energy = [beat_energies[0]]

        for t, e in zip(beat_times[1:], beat_energies[1:]):
            if t - cluster[-1] < cluster_gap_sec:
                cluster.append(t)
                cluster_energy.append(e)
            else:
                strongest_idx = np.argmax(cluster_energy)
                merged.append(cluster[strongest_idx])
                merged_energies.append(cluster_energy[strongest_idx])
                cluster = [t]
                cluster_energy = [e]

        if cluster and cluster[-1] != merged[-1]:
            strongest_idx = np.argmax(cluster_energy)
            merged.append(cluster[strongest_idx])
            merged_energies.append(cluster_energy[strongest_idx])

        # --- Apply minimum transition gap ---
        final_beats = [merged[0]]
        for t in merged[1:]:
            if t - final_beats[-1] >= min_transition_gap_sec:
                final_beats.append(t)

        return final_beats
    finally:
        try:
            audio.close()
        except Exception:
            pass


from moviepy.editor import AudioFileClip
import numpy as np
def compute_bar_energies(music_path, beats_per_bar=4, fps=22050):
    beats = detect_beats(music_path)
    audio = AudioFileClip(music_path)
    try:
        arr = audio_to_mono_safe(audio, fps=fps)
        n_samples = len(arr)

        # No beats at all → treat whole track as one bar
        if not beats:
            bars = [[0, audio.duration]]
            energies = [float(np.sqrt(np.mean(arr**2)))] if n_samples else [0.0]
            return bars, energies

        if len(beats) < beats_per_bar:
            bars = [[0, audio.duration]]
            energies = [float(np.sqrt(np.mean(arr**2)))] if n_samples else [0.0]
            return bars, energies

        bars = [
            beats[i:i + beats_per_bar]
            for i in range(0, len(beats) - beats_per_bar + 1, beats_per_bar)
        ]
        energies = []

        for bar in bars:
            start_t = bar[0]
            end_t = bar[-1]
            start_idx = int(start_t * fps)
            end_idx = int(end_t * fps)
            start_idx = max(0, min(start_idx, n_samples))
            end_idx = max(start_idx + 1, min(end_idx, n_samples))

            seg = arr[start_idx:end_idx]
            if seg.size == 0:
                energies.append(0.0)
            else:
                rms = float(np.sqrt(np.mean(seg**2)))
                energies.append(rms)

        return bars, energies
    finally:
        try:
            audio.close()
        except Exception:
            pass


def find_continuous_segment(music_path, best_dur, filler_dur, min_energy_threshold=0.3):
    """
    Finds CONTINUOUS segment where:
    - BEST part: Rising energy (start of drop/chorus)
    - FILLER part: Constant HIGH energy (no drop)
    """
    y, sr = librosa.load(music_path, sr=None)
    
    # Trim silence
    yt, idx = librosa.effects.trim(y, top_db=25)
    music_start_trimmed = idx[0] / sr
    
    hop_length = 512
    rms = librosa.feature.rms(y=yt, hop_length=hop_length)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
    
    best_frames = int(best_dur * sr / hop_length)
    filler_frames = int(filler_dur * sr / hop_length)
    
    best_score = 0
    best_start_idx = 0
    
    # Slide window across entire track
    for i in range(len(rms) - best_frames - filler_frames):
        # BEST part: Must be RISING energy
        best_rms = rms[i:i + best_frames]
        best_rise_score = np.mean(np.diff(best_rms))  # Positive = rising
        
        # FILLER part: Must be HIGH + CONSTANT energy  
        filler_rms = rms[i + best_frames:i + best_frames + filler_frames]
        filler_avg = np.mean(filler_rms)
        filler_stable = np.std(filler_rms) < np.std(rms) * 0.7  # More stable than avg
        
        # Combined score
        score = (best_rise_score * 2 if best_rise_score > 0 else 0) + \
                (filler_avg if filler_stable else 0)
        
        if score > best_score and filler_avg > min_energy_threshold:
            best_score = score
            best_start_idx = i
    
    # Convert back to seconds
    best_start = times[best_start_idx]
    best_end = times[best_start_idx + best_frames + filler_frames]
    
    print(f"🎵 BEST RISE: {best_start:.2f}s→{best_start+best_dur:.2f}s")
    print(f"🎵 FILLER HIGH: {best_start+best_dur:.2f}s→{best_end:.2f}s")
    print(f"   Rise score: {np.mean(np.diff(rms[best_start_idx:best_start_idx+best_frames])):.3f}")
    print(f"   Filler avg: {np.mean(rms[best_start_idx+best_frames:best_start_idx+best_frames+filler_frames]):.3f}")
    
    return best_start, best_end



    
import cv2

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_visual_scores(clips, frame_time=0.1, sample_count=3, resize_dim=(320, 180)):
    """
    Compute robust visual difference scores between consecutive clips.
    
    Parameters:
    - clips: list of VideoFileClip objects
    - frame_time: seconds from start/end to grab frames
    - sample_count: number of frames to sample near start/end for averaging
    - resize_dim: tuple to resize frames consistently
    
    Returns:
    - visual_scores: list of floats (higher = more visual change)
    """
    visual_scores = []
    if len(clips) < 2:
        return [0.0]

    for i in range(len(clips) - 1):
        prev_clip = clips[i]
        next_clip = clips[i + 1]

        prev_scores = []
        next_scores = []

        # Sample multiple frames from the end of prev_clip
        for j in range(sample_count):
            t = max(0, prev_clip.duration - frame_time * (j + 1)/sample_count)
            frame = prev_clip.get_frame(t)
            frame = cv2.resize(frame, resize_dim)
            prev_scores.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))

        # Sample multiple frames from the start of next_clip
        for j in range(sample_count):
            t = min(frame_time * (j + 1)/sample_count, next_clip.duration)
            frame = next_clip.get_frame(t)
            frame = cv2.resize(frame, resize_dim)
            next_scores.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))

        # Compute average visual difference over all sampled frames
        diffs = []
        for f1, f2 in zip(prev_scores, next_scores):
            abs_diff = np.mean(cv2.absdiff(f1, f2))
            ssim_score = ssim(f1, f2)
            combined_score = abs_diff * (1 - ssim_score)  # higher = more change
            diffs.append(combined_score)

        visual_scores.append(float(np.mean(diffs)))

    # Normalize scores to 0-1
    max_score = max(visual_scores) if visual_scores else 1.0
    visual_scores = [s / max_score for s in visual_scores]

    return visual_scores



def compute_beat_data(clips, beat_times, tolerance=0.15):
    """
    For each transition (between clips), check if it's close to a merged beat time.
    Returns 1.0 for close-to-beat, 0.0 otherwise.
    """
    beat_data = []
    cumulative_time = 0
    for i in range(len(clips) - 1):
        cumulative_time += clips[i].duration
        # Check if any beat is within tolerance
        close_to_beat = any(abs(cumulative_time - bt) <= tolerance for bt in beat_times)
        beat_data.append(1.0 if close_to_beat else 0.0)
    return beat_data


import os
from moviepy.editor import VideoFileClip

def load_clips_from_folder(folder_path):
    """Load all video clips from a folder sorted by filename."""
    clips = []
    if os.path.exists(folder_path):
        files = sorted(os.listdir(folder_path))
        for f in files:
            if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                path = os.path.join(folder_path, f)
                clips.append(VideoFileClip(path))
    else:
        print(f"⚠️ First clip folder not found: {folder_path}")
    return clips



# ---------- Main ----------
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, AudioFileClip
import shutil
import os
import random
import numpy as np

import os

def choose_first_clips(files, prompt="first clips"):
    """
    Let user select one or more first clips interactively.
    Returns a list of selected file paths.
    """
    print(f"\nAvailable {prompt}:")
    for i, f in enumerate(files):
        print(f"[{i}] {os.path.basename(f)}")

    choices = input(f"Enter numbers of {prompt} to use (comma-separated): ")
    selected = []
    try:
        indices = [int(x.strip()) for x in choices.split(",")]
        selected = [files[i] for i in indices if 0 <= i < len(files)]
    except Exception as e:
        print("Invalid input. No clips selected.")

    if not selected and files:
        print("No valid clips selected. Using the first clip by default.")
        selected = [files[0]]

    return selected


def list_files(folder, extensions=None):
    files = []
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path) and (extensions is None or any(f.lower().endswith(ext) for ext in extensions)):
            files.append(path)
    return files

def choose_files_interactively(files, prompt):
    print(f"\nAvailable {prompt}:")
    for i, f in enumerate(files):
        print(f"[{i}] {os.path.basename(f)}")
    choices = input(f"Enter numbers of {prompt} to use (comma-separated): ")
    selected = []
    try:
        indices = [int(x.strip()) for x in choices.split(",")]
        selected = [files[i] for i in indices if 0 <= i < len(files)]
    except Exception as e:
        print("Invalid input, using empty selection.")
    return selected

def choose_single_file_interactively(files, prompt):
    print(f"\nAvailable {prompt}:")
    for i, f in enumerate(files):
        print(f"[{i}] {os.path.basename(f)}")
    choice = input(f"Enter number of {prompt} to use: ")
    selected = None
    try:
        idx = int(choice.strip())
        if 0 <= idx < len(files):
            selected = files[idx]
    except:
        pass
    return selected

import os
import random
import numpy as np
import shutil
from pathlib import Path
from datetime import datetime
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    CompositeVideoClip,
    TextClip,
    concatenate_videoclips,
)


def list_files(folder, extensions=None):
    files = []
    for entry in os.listdir(folder):
        if extensions is None or any(entry.lower().endswith(ext) for ext in extensions):
            files.append(os.path.join(folder, entry))
    return files

def delete_path(path):
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            path.unlink()
        print(f"✅ Deleted: {path}")
    except Exception as e:
        print(f"❌ Error deleting {path}: {e}")

def safe_clear_pycache(start_path="."):
    for root, dirs, files in os.walk(start_path):
        for d in dirs:
            if d == "__pycache__":
                full_path = os.path.join(root, d)
                try:
                    shutil.rmtree(full_path)
                    print(f"Deleted {full_path}")
                except PermissionError as e:
                    print(f"Permission denied to delete {full_path}: {e}")

from bstscene1 import find_best_scene # Your scene detection module
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeVideoClip, TextClip
import random
import numpy as np
import os
from datetime import datetime

import os
from moviepy.editor import *
import numpy as np
from datetime import datetime
import random

# ============================================================
# 1) CREATE BEST SCENE AS SEPARATE FILE
# ============================================================
import subprocess
from moviepy.editor import VideoFileClip, AudioFileClip

def extract_audio(video_path, audio_out="temp_raw.wav"):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "2",
        "-ar", "44100",
        audio_out
    ]
    subprocess.run(cmd)
    return audio_out


def separate_vocals(audio_path):
    print("🎤 Running AI vocal separation (Demucs)…")
    cmd = ["demucs", audio_path]
    subprocess.run(cmd)

    base = audio_path.replace(".wav", "")
    vocals_path = f"separated/htdemucs/{base}/vocals.wav"
    return vocals_path

from smtcro import smart_full_crop  # import from your module

from objtrans import read_frame_rgb, build_object_rgba, save_rgba_temp_image, compose_and_concat
from objtrans import YoloSamSeg

# ---------------- REPLACE ensure_size ----------------
def smart_crop_clip(moviepy_clip):
    """
    Apply smart full-scene crop to a MoviePy clip (frame by frame)
    using the SAME logic as smart_crop_video
    """
    import numpy as np
    import cv2
    from moviepy.editor import ImageSequenceClip

    frames = []
    prev_box = None
    history = []
    prev_frame_for_flow = None

    fps = moviepy_clip.fps

    for frame in moviepy_clip.iter_frames(fps=fps, dtype='uint8'):
        # MoviePy gives RGB → convert to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        crop_frame, prev_box, prev_frame_for_flow = smart_full_crop(
            frame_bgr,
            prev_box=prev_box,
            history=history,
            prev_frame_for_flow=prev_frame_for_flow
        )

        # Back to RGB for MoviePy
        frames.append(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB))

    new_clip = ImageSequenceClip(frames, fps=fps)
    return new_clip


from moviepy.editor import (
    VideoFileClip, AudioFileClip, CompositeAudioClip,
    TextClip, CompositeVideoClip, concatenate_videoclips
)
import os
from objtrans import make_batch_or_single

# ---------------------------
# 1️⃣ Create best scene with clean, denoised audio
# ---------------------------
from moviepy.editor import VideoFileClip, AudioFileClip
from smtaud import remove_noise, compute_speech_loudness
from ficint1 import process_video_return_clip

from coleff import auto_apply_filter_to_clip_dynamic
import os
from moviepy.editor import VideoFileClip, AudioFileClip

from moviepy.editor import VideoFileClip, AudioFileClip
import os

from moviepy.editor import VideoFileClip, AudioFileClip
from ficint1 import process_video_return_clip

from coleff import auto_apply_filter_to_clip_dynamic
import os

import os
from moviepy.editor import VideoFileClip, AudioFileClip
from ficint1 import process_video_return_clip


from coleff import auto_apply_filter_to_clip_dynamic

import os
from moviepy.editor import VideoFileClip, AudioFileClip
from ficint1 import process_video_return_clip

from coleff import auto_apply_filter_to_clip_dynamic

import os
from moviepy.editor import VideoFileClip, AudioFileClip
# ... imports ...

from video_effect import apply_best_effect, EFFECT_PACK
from moviepy.editor import VideoFileClip, AudioFileClip
import os
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

from video_effect import apply_best_effect
from ficint1 import process_video_return_clip
import os
from filters import fast_apply_filter, fast_auto_choose
# ============================================================
# CREATE BEST SCENE FILE
# ============================================================
import os
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip, TextClip, CompositeVideoClip
from datetime import datetime
from smtaud import compute_speech_loudness, smart_duck_music

# ---------------------------
# CREATE BEST SCENE
import os
import random
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip, TextClip, CompositeVideoClip
# (keep your other imports and helper functions)
#!/usr/bin/env python3
# optimized_pipeline_fictic.py
"""
Optimized pipeline:
- fast intermediate exports using hardware encoder when available
- lossless concat helper
- final HQ export only at the end (export_hq)
- minimal changes to your flow; just faster + safer
"""
import os
import shlex
import subprocess
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip, TextClip, CompositeVideoClip
import random
from datetime import datetime




import librosa
import numpy as np

def find_smart_music_start(music_path):
    """
    Finds a smart music start:
    - trims silence
    - skips boring intro
    - lands near first energy rise
    """
    y, sr = librosa.load(music_path, sr=None)

    # Trim silence
    yt, idx = librosa.effects.trim(y, top_db=25)
    base_start = idx[0] / sr

    # RMS energy
    rms = librosa.feature.rms(y=yt)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr)

    threshold = np.percentile(rms, 75)

    for i, e in enumerate(rms):
        if e >= threshold:
            return base_start + times[i]

    return base_start

# ------------------------------
# YOUR MAIN PIPELINE FUNCTIONS
# (I replaced heavy intermediate write_videofile usage with fast_export/lossless_concat)
# ------------------------------

# --- export_hq kept above; now apply to create_best_scene_file, create_fillers_file, merge_final_video

# Note: functions like extract_audio, separate_vocals, remove_noise, smart_crop_clip, process_video_return_clip,
# compute_visual_scores, detect_beats, compute_beat_data, apply_best_effect, fast_auto_choose, fast_apply_filter,
# extract_candidates, build_fillers, stitching_with_audio_visual_cues, make_batch_or_single, smart_duck_music,
# compute_speech_loudness are expected to be defined elsewhere in your project.
from musicrem import separate_audio_with_demucs
from musicrem import separate_audio_with_demucs
import os
import shlex
import subprocess
import random
from datetime import datetime
import gc

from moviepy.editor import (
    VideoFileClip, AudioFileClip, concatenate_videoclips,
    CompositeAudioClip, TextClip, CompositeVideoClip, ColorClip
)

import librosa
import numpy as np
from musicrem import separate_audio_with_demucs
from transit import YoloSamSeg

# ----------------- HW + ffmpeg helpers -----------------

def run_ffmpeg(cmd):
    """Run ffmpeg command (list or string). Raises on failure."""
    if isinstance(cmd, (list, tuple)):
        cmd_list = cmd
    else:
        cmd_list = shlex.split(cmd)
    proc = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={proc.returncode}):\n{' '.join(cmd_list)}\nSTDERR:\n{proc.stderr.decode()}"
        )
    return proc.stdout, proc.stderr


def detect_hw_accel():
    """Detect common hardware encoders (ffmpeg encoders list)."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        enc = out.stdout.lower()
        hw = {}
        if "h264_videotoolbox" in enc:
            hw["h264"] = "h264_videotoolbox"
        if "hevc_videotoolbox" in enc or "hevc_videotoolboxenc" in enc:
            hw["hevc"] = "hevc_videotoolbox"
        return hw
    except Exception:
        return {}


HW = detect_hw_accel()  # global hardware encoder map (may be {})





def lossless_concat(file_list, out_path):
    """Concatenate files without re-encoding if they share codec/format."""
    listfile = "ffconcat_list.txt"
    with open(listfile, "w", encoding="utf-8") as f:
        for p in file_list:
            f.write(f"file '{os.path.abspath(p)}'\n")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile, "-c", "copy", out_path]
    try:
        run_ffmpeg(cmd)
    finally:
        if os.path.exists(listfile):
            try:
                os.remove(listfile)
            except Exception:
                pass
    return out_path


import glob
import shutil

def disk_space_check(min_gb=1.0):
    """Check disk space and auto-cleanup temps."""
    free_gb = shutil.disk_usage(".").free / (1024**3)
    if free_gb < min_gb:
        cleaned = 0
        for pattern in ["temp_*.mp4", "best_*.mp4", "fillers_*.mp4", "*temp_filtered*.mp4"]:
            for f in glob.glob(pattern):
                try:
                    os.remove(f)
                    cleaned += 1
                except:
                    pass
        print(f"🧹 Cleaned {cleaned} temp files. {free_gb:.1f}GB free.")
        if free_gb < min_gb:
            raise RuntimeError(f"❌ DISK FULL: {free_gb:.1f}GB free!")
    return free_gb

def fast_export(
    input_clip_or_path,
    out_path,
    fps=24,      # LOWER for temp files
    crf=26,      # HIGHER = smaller temp files
    preset="fast",
    pix_fmt="yuv420p",
    use_hw=False,  # CPU only for stability
    threads=2,
):
    disk_space_check(0.5)
    
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except:
            pass

    if isinstance(input_clip_or_path, str):
        cmd = [
            "ffmpeg", "-y", "-i", input_clip_or_path,
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-r", str(fps),
            "-pix_fmt", pix_fmt,
            "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "96k",
            "-threads", str(threads),
            out_path,
        ]
        run_ffmpeg(cmd)
    else:
        input_clip_or_path.write_videofile(
            out_path,
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            preset=preset,
            ffmpeg_params=["-crf", str(crf), "-pix_fmt", pix_fmt, "-movflags", "+faststart"],
            threads=threads,
            logger=None,
        )
    return out_path

def export_hq(clip, out_path, use_hw_final=False, encoder_choice=None, threads=2):
    """Final HQ export."""
    disk_space_check(2.0)
    
    if isinstance(clip, str):
        cmd = [
            "ffmpeg", "-y", "-i", clip,
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "21",           # ✅ HQ final (your sweet spot)
            "-pix_fmt", "yuv420p",  # ✅ Safer than yuv422p
            "-profile:v", "high",
            "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-threads", str(threads),
            out_path,
        ]
        run_ffmpeg(cmd)
        return out_path
    else:
        clip.write_videofile(
            out_path,
            codec="libx264",
            audio_codec="aac",
            fps=clip.fps,
            preset="slow",
            ffmpeg_params=[
                "-crf", "21",           # ✅ HQ final
                "-pix_fmt", "yuv420p",
                "-profile:v", "high",
                "-movflags", "+faststart",
                "-b:a", "192k",
                "-ar", "48000",
            ],
            threads=threads,
            logger=None,
        )
        return out_path


# ----------------- Music helper -----------------

def find_smart_music_start(music_path):
    """
    Finds a smart music start:
    - trims silence
    - skips boring intro
    - lands near first energy rise
    """
    y, sr = librosa.load(music_path, sr=None)
    yt, idx = librosa.effects.trim(y, top_db=25)
    base_start = idx[0] / sr

    rms = librosa.feature.rms(y=yt)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr)

    threshold = np.percentile(rms, 75)

    for i, e in enumerate(rms):
        if e >= threshold:
            return base_start + times[i]

    return base_start

# ----------------- External deps placeholders -----------------
# These must exist in your project:
# extract_audio, remove_noise, smart_crop_clip, process_video_return_clip,
# compute_visual_scores, detect_beats, compute_beat_data, apply_best_effect,
# fast_auto_choose, fast_apply_filter, extract_candidates, build_fillers,
# make_batch_or_single, smart_duck_music, compute_speech_loudness,
# find_continuous_segment, list_files, choose_first_clips,
# choose_single_file_interactively, choose_files_interactively

# ----------------- Main pipeline funcs -----------------

segmenter = YoloSamSeg(
    yolo_model="yolov8n.pt",
    sam_checkpoint="sam_b.pth"
)

def create_best_scene_file(
    first_clip_path,
    out_path="best_scene.mp4",
    intensity_main=0.85,
    history=None,
    music_path=None,
):
    history = history or []
    clip = VideoFileClip(first_clip_path)
    temp_audio_video = "temp_scene_for_audio.mp4"

    try:
        # Fast intermediate export for audio extraction
        fast_export(
            clip,
            temp_audio_video,
            fps=30,
            crf=20,
            preset="medium",
            pix_fmt="yuv420p",
            use_hw=True,
        )

        try:
            raw_audio = extract_audio(temp_audio_video)
            clean_vocals = separate_audio_with_demucs(raw_audio)
            denoised_audio_path = remove_noise(clean_vocals)
        except Exception:
            denoised_audio_path = None

    except Exception:
        denoised_audio_path = None
    finally:
        if os.path.exists(temp_audio_video):
            try:
                os.remove(temp_audio_video)
            except Exception:
                pass

    # Smart crop (returns a MoviePy VideoClip)
    resized_clip = smart_crop_clip(clip)

    # Attach cleaned audio if available
    audio_clean = None
    if denoised_audio_path and os.path.exists(denoised_audio_path):
        audio_clean = AudioFileClip(denoised_audio_path).volumex(3.0)
        clip_with_audio = resized_clip.set_audio(audio_clean)
    else:
        clip_with_audio = resized_clip.set_audio(clip.audio)

    processed_clip = process_video_return_clip(clip_with_audio)

    visual_score = compute_visual_scores([processed_clip])[0]
    beat_times = detect_beats(music_path) if music_path else []
    beat_data = compute_beat_data([processed_clip], beat_times)
    beat_strength = beat_data[0] if beat_data else 0.5

    processed_clip_with_effect, history, duration = apply_best_effect(
        processed_clip,
        visual_score=visual_score,
        beat_strength=beat_strength,
        history=history,
    )

    global chosen_filter
    chosen_filter = fast_auto_choose(processed_clip_with_effect)

    filtered_clip, applied_filter = fast_apply_filter(
        processed_clip_with_effect,
        out_path=None,
        filter_name=chosen_filter,
    )

    try:
        export_hq(filtered_clip, out_path, use_hw_final=False)
    finally:
        try:
            if hasattr(filtered_clip, "close"):
                filtered_clip.close()
        except Exception:
            pass
        try:
            if hasattr(processed_clip_with_effect, "close"):
                processed_clip_with_effect.close()
        except Exception:
            pass
        try:
            if hasattr(processed_clip, "close"):
                processed_clip.close()
        except Exception:
            pass
        try:
            if hasattr(resized_clip, "close"):
                resized_clip.close()
        except Exception:
            pass
        try:
            clip.close()
        except Exception:
            pass
        try:
            if audio_clean:
                audio_clean.close()
        except Exception:
            pass

    gc.collect()

    print(f"✅ FINAL: best scene saved with FAST filter '{chosen_filter}' and smart effect applied")
    history.append({"file": out_path, "filter": chosen_filter})
    return out_path, history


def create_fillers_file(
    video_files,
    target_duration,
    out_path="fillers.mp4",
    music_path=None,
    history=None,
    music_offset=0.0,
):
    print("🔍 Running create_fillers_file...")
    history = history or []
    global chosen_filter

    candidates = extract_candidates(video_files)
    if not candidates:
        print("❌ No candidates for fillers.")
        return None

    fillers = build_fillers(video_files, candidates, target_duration, used_key=None)
    if fillers is None:
        print("❌ build_fillers returned None")
        return None
    if not isinstance(fillers, (list, tuple)):
        fillers = [fillers]

    seq_clips = list(fillers)
    stitched_rest = []

    from transit4 import replace_with_reverse_safe, transition_person_static

    for i in range(len(seq_clips) - 1):
        c1 = seq_clips[i]
        c2 = seq_clips[i + 1]

        if random.random() < 0.8:  # 80% chance for forward-backward
            new_c2 = replace_with_reverse_safe(c1, c2)
        else:
            # build a small bg list of up to 3 clips
            bg_list = [c1, c2]
            if i + 2 < len(seq_clips):
                bg_list.append(seq_clips[i + 2])

            dummy_clip = ColorClip(
                size=c2.size,
                color=(0, 0, 0),
                duration=c1.duration
            ).set_fps(c1.fps)

            new_c2 = transition_person_static(
                dummy_clip,
                c2,
                bg_clips=bg_list,   # <= only 2–3 clips, not all seq_clips
                segmenter=segmenter,
                bg_blur_k=85,
            )

            try:
                dummy_clip.close()
            except Exception:
                pass

        stitched_rest.append(new_c2)

    stitched = [seq_clips[0]] + stitched_rest


    # ---- Beat + Visual Analysis ----
    raw_beats = detect_beats(music_path) if music_path else []
    beat_times = [b - music_offset for b in raw_beats if b >= music_offset]

    visual_scores = compute_visual_scores(seq_clips)
    beat_data = compute_beat_data(seq_clips, beat_times)

    for idx, c in enumerate(stitched):
        if hasattr(c, "audio") and c.audio:
            c = c.volumex(0.0)
        vs = visual_scores[idx] if idx < len(visual_scores) else 0
        bs = beat_data[idx] if idx < len(beat_data) else 0
        c, history, _ = apply_best_effect(
            c,
            visual_score=vs,
            beat_strength=bs,
            history=history,
        )
        stitched[idx] = c

    fillers_clip = concatenate_videoclips(stitched, method="compose").set_fps(30)

    temp_combined_path = "temp_combined_fillers.mp4"
    fast_export(
        fillers_clip,
        temp_combined_path,
        fps=30,
        crf=21,
        preset="medium",
        pix_fmt="yuv420p",
        use_hw=True,
    )

    try:
        fillers_clip.close()
    except Exception:
        pass

    for c in seq_clips:
        try:
            c.close()
        except Exception:
            pass
    for c in stitched_rest:
        try:
            if hasattr(c, "close"):
                c.close()
        except Exception:
            pass

    gc.collect()

    temp_filtered_path = "temp_filtered_fillers.mp4"
    filtered_clip, applied_filter = fast_apply_filter(
        temp_combined_path,
        out_path=temp_filtered_path,
        filter_name=chosen_filter,
    )

    try:
        final_clip = VideoFileClip(temp_filtered_path)
        export_hq(final_clip, out_path, use_hw_final=False)
    finally:
        try:
            final_clip.close()
        except Exception:
            pass
        try:
            if hasattr(filtered_clip, "close"):
                filtered_clip.close()
        except Exception:
            pass

    for f in [temp_combined_path, temp_filtered_path]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    gc.collect()

    print(f"✅ Filler file created: {out_path} with filter '{applied_filter}'")
    return out_path, history


def merge_final_video(
    best_path,
    fillers_path,
    music_path,
    speech_loudness,
    out_path,
    music_start,
    music_end,  # continuous segment end
    use_hw_final=False,
):
    if isinstance(best_path, (tuple, list)):
        best_path = best_path[0]
    if isinstance(fillers_path, (tuple, list)):
        fillers_path = fillers_path[0]

    temp_transition_file = "temp_transition.mp4"
    make_batch_or_single(best_path, fillers_path, "slide", 0.7, temp_transition_file)

    final_video = VideoFileClip(temp_transition_file)
    best_clip = VideoFileClip(best_path)
    fillers_clip = VideoFileClip(fillers_path)

    # Load full music once
    full_music = AudioFileClip(music_path)

    # BEST CLIP: use start of continuous segment
    best_music_start = music_start
    best_music_end = music_start + best_clip.duration
    best_clip_duration = best_clip.duration

    # Safety clamp inside [music_start, music_end]
    best_music_end = min(best_music_end, music_end)
    best_music = full_music.subclip(best_music_start, best_music_end)

    # Ducked music for BEST part (uses same window)
    combined_best_audio = smart_duck_music(
        vocal_audio_clip=best_clip.audio,
        music_audio_path=music_path,
        speech_loudness=speech_loudness,
        duration=best_clip_duration,
        music_start=best_music_start,
    )

    # FILLER CLIP: immediately after BEST, still continuous
    filler_music_start = best_music_end
    filler_music_end = min(music_end, filler_music_start + fillers_clip.duration)

    print(f"🎵 Best music (cont): {best_music_start:.2f}→{best_music_end:.2f}")
    print(f"🎵 Filler music (cont): {filler_music_start:.2f}→{filler_music_end:.2f}")

    filler_music = full_music.subclip(
        filler_music_start,
        filler_music_end
    ).volumex(0.2).set_start(best_clip_duration)

    # Final continuous music bed = best-part ducked + continuous filler under rest
    final_audio = CompositeAudioClip([combined_best_audio, filler_music])
    final_video = final_video.set_audio(final_audio)

    watermark = TextClip(
        "Uday_editZ",
        fontsize=40,
        color="white",
        font="Arial",
        method="label",
    ).set_position("center").set_opacity(0.4).set_duration(final_video.duration)
    final_video = CompositeVideoClip([final_video, watermark])

    donation = TextClip(
        "If you like my videos\nplease like, comment and subscribe!\n\nThank you :)\nUday_editZ",
        fontsize=62,
        color="white",
        font="Arial",
        method="label",
        size=(1080, 1920),
    ).set_duration(5).on_color(
        size=(1080, 1920),
        color=(0, 0, 0),
        col_opacity=0,
    ).set_position("center")

    final_video = concatenate_videoclips([final_video, donation])

    export_hq(final_video, out_path, use_hw_final=use_hw_final)

    for clip in [best_clip, fillers_clip, final_video]:
        try:
            clip.close()
        except Exception:
            pass
    try:
        full_music.close()
    except Exception:
        pass
    if os.path.exists(temp_transition_file):
        try:
            os.remove(temp_transition_file)
        except Exception:
            pass

    gc.collect()

    print(f"🎉 FINAL VIDEO EXPORT COMPLETE: {out_path}")
    return out_path


# ----------------- MAIN -----------------

def main(args):
    try:
        num_videos = int(input("Enter total number of videos to produce: "))
    except ValueError:
        print("Invalid input. Using 1 video by default.")
        num_videos = 1

    video_inputs = []
    for i in range(num_videos):
        print(f"\n--- Inputs for video {i+1}/{num_videos} ---")

        first_clip_files = list_files(args.first_clip_folder, [".mp4", ".mov", ".mkv"])
        selected_first_clips = choose_first_clips(first_clip_files)

        music_files = list_files(args.music, [".mp3", ".wav", ".m4a", ".ogg"])
        music_path = choose_single_file_interactively(music_files, "music")

        video_files = list_files(args.input_videos, [".mp4", ".mov", ".mkv"])
        selected_videos = choose_files_interactively(video_files, "filler videos")

        video_inputs.append(
            {
                "first_clips": selected_first_clips,
                "music": music_path,
                "fillers": selected_videos,
            }
        )

    for i, v in enumerate(video_inputs):
        print(f"\n🎬 Producing video {i+1}/{num_videos}")

        if len(v["first_clips"]) > 1:
            merged_path = f"merged_first_clips_{i}.mp4"
            try:
                lossless_concat(v["first_clips"], merged_path)
            except Exception:
                clips = [VideoFileClip(f) for f in v["first_clips"]]
                concatenated_clip = concatenate_videoclips(clips, method="compose")
                fast_export(
                    concatenated_clip,
                    merged_path,
                    fps=30,
                    crf=20,
                    preset="medium",
                    pix_fmt="yuv420p",
                    use_hw=True,
                )
                concatenated_clip.close()
                for c in clips:
                    try:
                        c.close()
                    except Exception:
                        pass
        else:
            merged_path = v["first_clips"][0]

        history = []
        best_scene_file, history = create_best_scene_file(
            merged_path,
            out_path=f"best_scene_{i}.mp4",
            history=history,
            music_path=v["music"],
        )

        fillers_file, history = create_fillers_file(
            v["fillers"],
            target_duration=21.0,
            out_path=f"fillers_{i}.mp4",
            music_path=v["music"],
            history=history,
        )

        # drop references to original lists ASAP
        v["first_clips"] = []
        v["fillers"] = []
        gc.collect()

        best_audio_clip = AudioFileClip(best_scene_file)
        temp_audio_path = f"temp_best_scene_audio_{i}.wav"
        best_audio_clip.write_audiofile(temp_audio_path, fps=44100)
        best_audio_clip.close()

        speech_loudness = compute_speech_loudness(temp_audio_path)
        try:
            os.remove(temp_audio_path)
        except Exception:
            pass

        best_clip = VideoFileClip(best_scene_file)
        fill_clip = VideoFileClip(fillers_file)
        segment_start, segment_end = find_continuous_segment(
            music_path=v["music"],
            best_dur=best_clip.duration,
            filler_dur=fill_clip.duration,
        )
        best_clip.close()
        fill_clip.close()
        gc.collect()

        music_start = segment_start
        music_end = segment_end  # ← ADD THIS

        os.makedirs("out", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"out/final_video_{i}_{timestamp}.mp4"

        merge_final_video(
            best_scene_file,
            fillers_file,
            v["music"],
            speech_loudness,
            out_path,
            music_start=music_start,
            music_end=music_end,  # ← ADD THIS
            use_hw_final=False,
        )


        print("🎉 FINAL VIDEO READY:", out_path)






# End of updated HQ-optimized code snippet



    # video_paths = find_videos(args.input_videos)
    # if not video_paths:
    #     print("No input videos found:", args.input_videos); return

    # music_path = find_music_file(args.music)
    # # Extract musically complete best segment of music
    # desired_music_duration = None  # or set to your preferred length, e.g. target_duration if known
    # start, end = find_complete_segment(music_path, desired_duration=desired_music_duration)
    # print(f"Selected music segment from {start:.2f}s to {end:.2f}s for editing.")

    # music_audio = AudioFileClip(music_path).subclip(start, end)
    # target_duration = music_audio.duration  # update global target to match segment length



    # # # 1) transcribe each input video individually with Whisper
    # # print("Transcribing each input video with Whisper (this may take time)...")
    # # whisper_segs, tmp_wdir = transcribe_each_video(video_paths, whisper_model_name=args.whisper_model, language=args.language)

    # # print(f"Whisper produced {len(whisper_segs)} segments across inputs.")

    # # # 2) score segments and pick best punch (from input audio directly)
    # # best_seg = score_segments(whisper_segs, tmp_wdir)
    # # if best_seg:
    # #     print("Picked punch dialogue:", best_seg.get('text','')[:120])
    # # else:
    # #     print("No suitable dialogue found in Whisper output; will fall back to visual candidate.")

    # # # 3) prepare filler candidates
    # # print("Extracting visual candidates for fillers...")
    # candidates = extract_candidates(video_paths)
    # # if not candidates:
    # #     print("No visual candidates found; aborting.")
    # #     try:
    # #         shutil.rmtree(tmp_wdir)
    # #     except Exception:
    # #         pass
    # #     return

    # # 4) build first dialogue clip from best_seg (or fallback)
    # # if best_seg:
    # #     src = best_seg['video']; s0 = max(0.0, best_seg['start']); s1 = best_seg['end']
    # #     # extend a little for context
    # #     pre = max(0.0, s0 - 0.45); post = min(VideoFileClip(src).duration, s1 + 0.21)
    # #     try:
    # #         first_clip = VideoFileClip(src).subclip(pre, post).set_fps(FPS)
    # #         first_clip = center_crop_vertical(first_clip)
    # #         first_clip = first_clip.fx(vfx.colorx, 1.12)
    # #         subtitle_text = best_seg.get('text','')
    # #         dialogue_duration = max(0.25, s1 - s0)
    # #     except Exception as e:
    # #         print("Failed to build dialogue clip from best_seg:", e)
    # #         best_seg = None

    # # if not best_seg:
    #     # fallback: choose top visual candidate by RMS+motion

    # def calculate_score(candidate, existing_segments_paths):
    #     # Normalize RMS and motion to [0,1] roughly (example)
    #     rms_norm = min(candidate['rms'] / 0.1, 1.0)  # tune divisor for your data range
    #     motion_norm = min(candidate['motion'] / 1.0, 1.0)  # tune divisor for your data range
        
    #     # Novelty penalty if from an already used source video to encourage diversity
    #     novelty_penalty = 1.0 if candidate['path'] not in existing_segments_paths else 0.7
        
    #     # Weighted sum with novelty to prioritize unique, strong scenes
    #     score = (rms_norm * 0.6 + motion_norm * 0.3) * novelty_penalty
    #     return score

    # def pick_best_scene(candidates, used_paths):
    #     # Pick candidate with highest improved score
    #     best = max(candidates, key=lambda c: calculate_score(c, used_paths))
    #     return best

    # # Example usage in your main flow:
    # used_paths = set()
    # fallback = pick_best_scene(candidates, used_paths)
    # used_paths.add(fallback['path'])


    # src = fallback['path']; pre = max(0.0, fallback['start']-0.18)
    # post = min(VideoFileClip(src).duration, fallback['start'] + min(2.0, fallback['dur']) + 0.18)
    # first_clip = VideoFileClip(src).subclip(pre, post).set_fps(FPS)
    # first_clip = center_crop_vertical(first_clip)
    # # first_clip = first_clip.fx(vfx.colorx, 1.12)
    # # subtitle_text = fallback.get('text','')
    # dialogue_duration = min(first_clip.duration, 1.8)

    # # add subtitle overlay to first clip (whole-segment style)
    # # first_with_subs = overlay_single_subtitle(first_clip, subtitle_text, dialogue_duration)

    # # 5) Build fillers guided by beats (unique, no repetition) and excluding the dialogue source/time
    # beats = detect_beats(music_path)
    # beat_gaps = None
    # if beats is not None and len(beats) > 1:
    #     gaps = np.diff(beats)
    #     beat_gaps = [max(0.45, min(2.6, g)) for g in gaps] if len(gaps) > 0 else None
    #     print(f"Detected {len(beats)} beats; using beat gaps for filler durations.")
    # else:
    #     print("No beats detected; using adaptive filler durations.")

    # # mark used key for dialogue to avoid repeating same exact start
    # used_key = (src, round(pre,2))

    # # remaining time after the first clip
    # remaining_time = max(0.0, target_duration - first_clip.duration)

    # fillers = build_fillers(video_paths, candidates, remaining_time, used_key, beat_gaps=beat_gaps)

    # # final sequence: first_with_subs + fillers
    # seq_clips =  fillers
    # def choose_transition_maybe(prev, nxt, apply_prob=0.5, min_clip_dur=1.0):
    #     if random.random() > apply_prob or prev.duration < min_clip_dur or nxt.duration < min_clip_dur:
    #         return None
    #     r = random.random()
    #     if r < 0.4:
    #         return impact_zoom_transition(prev, nxt, dur=0.28, strength=0.18)
    #     elif r < 0.75:
    #         return whip_transition(prev, nxt, dur=0.30, direction=random.choice(['left', 'right']))
    #     elif r < 0.9:
    #         return flash_transition(prev, nxt, dur=0.18)
    #     else:
    #         return masked_object_transition(prev, nxt, dur=0.3)  # use new masked object transition
    #     # return simple_forward_slide_object_transition(prev, nxt, dur=0.4)



    # def stitching_conditionally(clips, apply_prob=0.8, min_clip_dur=0.8):
    #     seq = [clips[0]]
    #     for nxt in clips[1:]:
    #         prev = seq[-1]
    #         trans = choose_transition_maybe(prev, nxt, apply_prob=apply_prob, min_clip_dur=min_clip_dur)
    #         if trans is not None:
    #             seq.append(trans)
    #         seq.append(nxt)
    #     clean = [safe_clip(s) for s in seq if s is not None]
    #     return concatenate_videoclips(clean, method="compose")



    # final = stitching_conditionally(seq_clips, apply_prob=0.8, min_clip_dur=0.8)




    # # ensure full coverage until music end — if short, append unique snippets (no static image)
    # if final.duration < target_duration - 0.02:
    #     print("Final shorter than music; appending more dynamic fillers...")
    #     extra_needed = target_duration - final.duration
    #     more = build_fillers(video_paths, candidates, extra_needed, used_key, beat_gaps=beat_gaps)
    #     if more:
    #         final = concatenate_videoclips([final] + more, method="compose")

    # # Trim or extend carefully as before
    # if final.duration < target_duration - 0.02:
    #     attempts = 0
    #     while final.duration < target_duration - 0.02 and attempts < 30:
    #         p = random.choice(video_paths)
    #         try:
    #             v = VideoFileClip(p)
    #             start = random.uniform(0, max(0, v.duration - 0.6))
    #             dur = min(1.2, target_duration - final.duration)
    #             add = make_variant(p, start, dur, kind=random.choice(["normal","speedup"]))
    #             final = concatenate_videoclips([final, add], method="compose")
    #             v.reader.close(); v.audio = None
    #         except Exception:
    #             pass
    #         attempts += 1

    # if final.duration > target_duration:
    #     final = final.subclip(0, target_duration)
    # final = final.set_fps(FPS).set_duration(target_duration)

    # # 6) Audio mixing:
    # # - Dialogue: video audio full volume, music very low
    # # - After dialogue: music full volume, video muted

    # print("Mixing audio: ULTRA CLEAR DIALOGUE, music ducked only during dialogue...")

    # dialogue_start = 0.0
    # dialogue_end = dialogue_duration

    # # Load + match music duration
    # music_orig = AudioFileClip(music_path)
    # if music_orig.duration > final.duration:
    #     music_orig = music_orig.subclip(0, final.duration)
    # else:
    #     music_orig = audio_loop(music_orig, duration=final.duration)

    # # 1️⃣ Music segments
    # music_very_low = music_orig.subclip(dialogue_start, dialogue_end).volumex(0.05)
    # music_very_low = music_very_low.audio_fadein(0.20).audio_fadeout(0.20)

    # music_after = None
    # if music_orig.duration > dialogue_end:
    #     music_after = music_orig.subclip(dialogue_end, final.duration).volumex(1.0)

    # # 2️⃣ Video audio: full for dialogue, muted after
    # orig_audio = final.audio if final.audio else music_orig.volumex(0.0)

    # video_loud = orig_audio.subclip(dialogue_start, dialogue_end).volumex(1.8)
    # video_mute = orig_audio.subclip(dialogue_end, final.duration).volumex(0.0)

    # # 3️⃣ Combine audio layers
    # audio_layers = [
    #     music_very_low.set_start(dialogue_start),
    #     video_loud.set_start(dialogue_start),
    #     video_mute.set_start(dialogue_end)
    # ]

    # if music_after:
    #     audio_layers.append(music_after.set_start(dialogue_end))

    # combined_audio = CompositeAudioClip(audio_layers).set_duration(final.duration)

    # # Set final audio with smooth end fade
    # final = final.set_audio(combined_audio).audio_fadeout(0.8)


    # # 7) export
    # os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # print("Rendering final video (this will take a while)...")
    # final.write_videofile(args.out, codec="libx264", audio_codec="aac", fps=FPS, preset="medium", threads=4, bitrate="6M", ffmpeg_params=["-movflags","+faststart"])
    # print("Cleaning tmp...")
    # try:
    #     shutil.rmtree(tmp_wdir)
    # except Exception:
    #     pass
    # print("Done ->", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fictic punch-intro (music-driven, input-audio whisper).")
    p.add_argument("--input_videos", required=True, help="Folder with input clips")
    p.add_argument("--first_clip_folder", required=True, help="Folder with input clips")
    p.add_argument("--music", required=True, help="Music file or folder (music length used)")
    p.add_argument("--out", default="./out/final_edit_9x16.mp4", help="Output path")
    p.add_argument("--whisper_model", default=WHISPER_DEFAULT, help="Whisper model (tiny/base/small/medium/large)")
    p.add_argument("--language", default=None, help="Whisper language code (optional)")
    args = p.parse_args()
    main(args)
    import numpy as np
    import sounddevice as sd

    fs = 44100  # Sample rate
    seconds = 20  # Duration of note
    frequency = 440  # Frequency in Hz

    # Generate beep
    t = np.linspace(0, seconds, int(fs * seconds), False)
    note = np.sin(frequency * 2 * np.pi * t)
    sd.play(note, fs)
    sd.wait()
    # Record end time
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"⏱️ Total execution time: {elapsed_time:.2f} seconds")

    #to call: python3 fictic20.py --first_clip_folder "firstclip" --input_videos ./input_vid --music ./music --out "out/" --whisper_model tiny 