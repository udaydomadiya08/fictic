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
import whisper
import librosa

from moviepy.editor import (
    VideoFileClip, AudioFileClip, concatenate_videoclips,
    CompositeVideoClip, TextClip, vfx, CompositeAudioClip
)
from moviepy.audio.fx.all import audio_loop
from moviepy.audio.AudioClip import concatenate_audioclips

warnings.filterwarnings("ignore")

# ---------- Config ----------
OUT_W, OUT_H = 1080, 1920
FPS = 30
FONT_NAME = "/Users/uday/Downloads/VIDEOYT/Anton-Regular.ttf"
FONT_SIZE = 78
STROKE_WIDTH = 3
TEXT_COLOR = "white"
GLOW_COLORS = ["#ffb86b", "#f1fa8c"]
WHISPER_DEFAULT = "small"

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

def detect_beats(music_path, sr=22050, hop_length=512):
    try:
        y, sr = librosa.load(music_path, sr=sr, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        return beats
    except Exception:
        return None

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


def make_variant(src_path, start, dur, kind="normal"):
    src = VideoFileClip(src_path)
    end = min(src.duration, start + dur)
    clip = src.subclip(start, end).set_fps(FPS)
    clip = center_crop_vertical(clip)
    if kind == "speedup":
        clip = clip.fx(vfx.speedx, 1.25)
    elif kind == "reverse":
        clip = clip.fx(vfx.time_mirror)
    elif kind == "stutter":
        small = clip.subclip(0, min(0.12, clip.duration))
        try:
            clip = concatenate_videoclips([small]*3).set_duration(min(clip.duration, 0.36))
        except Exception:
            pass
    zoom = 1.03 + random.random()*0.05
    clip = clip.fx(vfx.resize, lambda t: 1.0 + (zoom-1.0)*(t/max(1e-6, clip.duration)))
    ramp = min(0.09, clip.duration*0.12)
    if ramp > 0.02 and clip.duration > ramp + 0.01:
        try:
            a = clip.subclip(0, clip.duration - ramp)
            b = clip.subclip(clip.duration - ramp, clip.duration).fx(vfx.speedx, 1.08)
            clip = concatenate_videoclips([a,b])
        except Exception:
            pass
    clip = clip.fx(vfx.colorx, 1.05)
    clip = quick_glitch_overlay(clip)
    return clip

# ---------- Whisper per-video transcription ----------
def transcribe_each_video(video_paths, whisper_model_name="small", language=None, tmpdir=None):
    """
    Transcribe each video separately. Return list of segments:
    [{'video': path, 'start': s, 'end': e, 'text': t}, ...]
    """
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="fictic_whisper_")
    model = whisper.load_model(whisper_model_name)
    all_segs = []
    for p in tqdm(video_paths, desc="Whisper (per-video)"):
        try:
            a_out = os.path.join(tmpdir, os.path.basename(p) + ".wav")
            # extract short audio (entire audio) for whisper
            try:
                ac = AudioFileClip(p)
                ac.write_audiofile(a_out, fps=16000, verbose=False, logger=None)
                ac.close()
            except Exception:
                # fallback: use moviepy extraction via video clip
                v = VideoFileClip(p)
                if v.audio:
                    v.audio.write_audiofile(a_out, fps=16000, verbose=False, logger=None)
                v.reader.close(); v.audio = None

            opts = {}
            if language:
                opts['language'] = language
                opts['task'] = 'transcribe'
            res = model.transcribe(a_out, **opts)
            for seg in res.get('segments', []):
                all_segs.append({'video': p, 'start': float(seg['start']), 'end': float(seg['end']), 'text': seg['text'].strip()})
        except Exception as e:
            # just skip failures for single file
            # print("Whisper failed for", p, e)
            continue
    return all_segs, tmpdir

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
            method="label"
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
    used_key is tuple marking the dialogue clip to avoid repetition.
    candidates: list of dicts with path,start,dur,motion,rms,idx (from extract_candidates)
    """
    pool = sorted(candidates, key=lambda x: (x.get("motion",0)*0.7 + x.get("rms",0)*50.0 + random.random()*0.01), reverse=True)
    fillers = []
    used = set([used_key])
    remaining = target_duration
    # if beat_gaps available, cycle through them for durations
    bi = 0
    while remaining > 0.35 and pool:
        gap = (beat_gaps[bi % len(beat_gaps)] if beat_gaps else 1.2)
        dur = min(gap * random.uniform(0.9,1.12), remaining)
        bi += 1
        # pick next unused pool item
        found = False
        for i, c in enumerate(pool):
            key = (c['path'], round(c['start'],2))
            if key in used:
                continue
            take = min(c['dur'], max(0.35, dur))
            try:
                clip = make_variant(c['path'], c['start'], take, kind=random.choice(["normal","speedup","stutter"]))
                clip = clip.crossfadein(0.06)
                fillers.append((clip, key))
                used.add(key)
                remaining -= clip.duration
                found = True
                # remove used candidate from pool
                pool.pop(i)
                break
            except Exception:
                continue
        if not found:
            # if no candidate suits, break and we'll append random short parts later
            break

    # if still short, append random unique snippets
    attempts = 0
    while remaining > 0.25 and attempts < 40:
        p = random.choice(video_paths)
        try:
            v = VideoFileClip(p)
            start = random.uniform(0, max(0, v.duration - 0.6))
            dur = min(remaining, random.uniform(0.5, 1.4))
            key = (p, round(start,2))
            if key in used:
                v.reader.close(); v.audio = None
                attempts += 1
                continue
            add = make_variant(p, start, dur, kind=random.choice(["normal","speedup"]))
            add = add.crossfadein(0.06)
            fillers.append((add, key))
            used.add(key)
            remaining -= add.duration
            v.reader.close(); v.audio = None
        except Exception:
            attempts += 1
            continue

    # return only clips
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

def safe_clip(clip, min_dur=MIN_CLIP_DUR):
    if clip is None:
        return None
    if clip.duration >= min_dur - 1e-6:
        return clip
    try:
        return clip.fx(vfx.loop, duration=min_dur)
    except Exception:
        # last resort: extend by freezing last frame via ImageClip
        try:
            frm = clip.get_frame(max(0, clip.duration - 1e-3))
            imc = ImageClip(frm).set_duration(min_dur - clip.duration)
            return concatenate_videoclips([clip, imc], method="compose")
        except Exception:
            return clip.set_duration(min_dur)


def safe_frame_get(clip, t):
    t = np.clip(t, 0, max(0, clip.duration - 1e-3))
    f = clip.get_frame(t)
    # ✅ Force exact output size
    return cv2.resize(f, (OUT_W, OUT_H))

def ensure_size(c):
    return c.resize((OUT_W, OUT_H)).set_position(("center", "center"))

def impact_zoom_transition(c1, c2, dur=0.12, strength=0.18):
    c1 = ensure_size(safe_clip(c1))
    c2 = ensure_size(safe_clip(c2))
    a = c1.subclip(max(0, c1.duration - dur), c1.duration).set_duration(dur)
    b = c2.subclip(0, min(c2.duration, dur)).set_duration(dur)

    def frame_fn(t):
        f1 = safe_frame_get(a, t).astype(np.float32)
        f2 = safe_frame_get(b, t).astype(np.float32)

        if t < dur*0.55:
            frac = t/(dur*0.55)
            zoom = 1.0 + strength*(frac**1.7)

            hh, ww = OUT_H, OUT_W
            nw = int(ww/zoom); nh = int(hh/zoom)
            cx, cy = ww//2, hh//2
            x1 = max(0, cx - nw//2); y1 = max(0, cy - nh//2)
            crop = f1[y1:y1+nh, x1:x1+nw]
            crop = cv2.resize(crop, (ww, hh))
            return crop.astype(np.uint8)
        else:
            frac = (t - dur*0.55)/(dur*0.45 + 1e-8)
            comp = (1-frac)*f1 + frac*f2
            return np.clip(comp, 0, 255).astype(np.uint8)

    return VideoClip(lambda tt: frame_fn(tt), duration=dur).set_fps(FPS)


def whip_transition(c1, c2, dur=0.12, direction='left'):
    c1 = ensure_size(safe_clip(c1))
    c2 = ensure_size(safe_clip(c2))
    a = c1.subclip(max(0, c1.duration - dur), c1.duration).set_duration(dur)
    b = c2.subclip(0, min(c2.duration, dur)).set_duration(dur)

    def frame_fn(t):
        f1 = safe_frame_get(a, t).astype(np.float32)
        f2 = safe_frame_get(b, t).astype(np.float32)

        frac = t/dur
        shift = int((OUT_W + 200)*(frac**1.2))

        if direction == 'left':
            f1s = np.roll(f1, -shift, axis=1)
            f2s = np.roll(f2, OUT_W-shift, axis=1)
        else:
            f1s = np.roll(f1, shift, axis=1)
            f2s = np.roll(f2, -OUT_W+shift, axis=1)

        mask = np.linspace(0, 1, OUT_W) < frac
        mask = np.repeat(mask[np.newaxis, :], OUT_H, axis=0)[:,:,None]
        comp = (1-mask)*f1s + mask*f2s
        return np.clip(comp, 0,255).astype(np.uint8)

    return VideoClip(lambda tt: frame_fn(tt), duration=dur).set_fps(FPS)


def flash_transition(c1, c2, dur=0.12):
    c1 = ensure_size(safe_clip(c1))
    c2 = ensure_size(safe_clip(c2))
    a = c1.subclip(max(0, c1.duration - dur), c1.duration).set_duration(dur)
    b = c2.subclip(0, min(c2.duration, dur)).set_duration(dur)

    def frame_fn(t):
        frac = t/dur
        f1 = safe_frame_get(a, t).astype(np.float32)
        f2 = safe_frame_get(b, t).astype(np.float32)

        if frac < 0.35:
            alpha = 1.0 + 3.0*(frac/0.35)
            return np.clip(f1*alpha,0,255).astype(np.uint8)
        else:
            r = (frac-0.35)/0.65
            comp = (1-r)*f1 + r*f2
            if 0.45 < frac < 0.55:
                comp = np.clip(comp + 40*(1 - abs(frac-0.5)*4), 0,255)
            return comp.astype(np.uint8)

    return VideoClip(lambda tt: frame_fn(tt), duration=dur).set_fps(FPS)

animated_clips = set()

def object_in_transition(c1, c2, dur=0.4):
    """
    Animate object of c2 sliding in from right, 
    c1 object stays fixed (no slide out).
    Background crossfades smoothly.
    """
    frame1 = c1.get_frame(0)
    frame2 = c2.get_frame(0)
    mask2 = get_object_mask(frame2)

    h2, w2 = frame2.shape[:2]
    if mask2 is None:
        mask2 = np.zeros((h2, w2), dtype=np.uint8)
    else:
        mask2 = cv2.resize(mask2, (w2, h2))

    OUT_W, OUT_H = c1.w, c1.h

    def frame_fn(t):
        alpha = t / dur
        bg = (1 - alpha) * frame1 + alpha * frame2

        # Incoming clip object slides in from right
        dx2 = int(OUT_W * (1 - alpha))
        obj2 = shift_masked_object(frame2, mask2, dx2)

        # Outgoing clip object stays fixed - fully visible
        obj1 = frame1

        # Combine masks for blending background
        comp = bg * (1 - mask2[..., None]) + obj1 * (mask2[..., None] == 0) + obj2
        return np.clip(comp, 0, 255).astype(np.uint8)

    # Mark c2 as animated in
    animated_clips.add(c2.filename if hasattr(c2, 'filename') else id(c2))

    return VideoClip(frame_fn, duration=dur).set_fps(FPS).resize((OUT_W, OUT_H))

def object_slide_in_transition(c1, c2, dur=0.4, direction='right'):
    """
    Incoming c2 object slides in from specified direction from right,
    c1 object stays fixed and fully visible,
    background crossfades smoothly.
    """
    frame1 = c1.get_frame(0)
    frame2 = c2.get_frame(0)
    mask2 = get_object_mask(frame2)

    h2, w2 = frame2.shape[:2]
    if mask2 is None:
        mask2 = np.zeros((h2, w2), dtype=np.uint8)
    else:
        mask2 = cv2.resize(mask2, (w2, h2))

    OUT_W, OUT_H = c1.w, c1.h

    def frame_fn(t):
        alpha = t / dur
        # Background crossfade
        bg = (1 - alpha) * frame1 + alpha * frame2

        # Object from c1 stays fixed
        obj1 = frame1

        # Object from c2 slides in from right
        if direction == 'right':
            dx = int(OUT_W * (1 - alpha))
        elif direction == 'left':
            dx = -int(OUT_W * (1 - alpha))
        else:
            dx = 0

        obj2 = shift_masked_object(frame2, mask2, dx)

        # Composite with mask to avoid overlaps
        combined_mask = np.clip(mask2, 0, 1)[..., None]
        comp = bg * (1 - combined_mask) + obj1 * (combined_mask == 0) + obj2
        return np.clip(comp, 0, 255).astype(np.uint8)

    return VideoClip(frame_fn, duration=dur).set_fps(FPS).resize((OUT_W, OUT_H))

def object_in_fade_transition(c1, c2, dur=0.4):
    """
    Incoming object and scene smoothly fade in over current fixed scene and object.
    No sliding or motion, just clean fade.
    """
    frame1 = c1.get_frame(0)
    frame2 = c2.get_frame(0)
    mask2 = get_object_mask(frame2)

    h2, w2 = frame2.shape[:2]
    if mask2 is None:
        mask2 = np.zeros((h2, w2), dtype=np.uint8)
    else:
        mask2 = cv2.resize(mask2, (w2, h2))

    OUT_W, OUT_H = c1.w, c1.h

    def frame_fn(t):
        alpha = t / dur
        
        # Crossfade full scenes
        bg = (1 - alpha) * frame1 + alpha * frame2
        
        # Current object fully visible (no change)
        obj1 = frame1
        
        # Incoming object fades in
        obj2 = (frame2 * mask2[..., None]) * alpha
        
        # Combined mask to avoid overlap issues
        combined_mask = np.clip(mask2, 0, 1)[..., None]

        comp = bg * (1 - combined_mask) + obj1 * (combined_mask == 0) + obj2
        
        return np.clip(comp, 0, 255).astype(np.uint8)

    # Mark c2 as animated in
    animated_clips.add(c2.filename if hasattr(c2, 'filename') else id(c2))

    return VideoClip(frame_fn, duration=dur).set_fps(FPS).resize((OUT_W, OUT_H))


def object_in_no_out_transition(c1, c2, dur=0.4):
    """
    No slide out for c1 object (no animation),
    crossfade backgrounds,
    animate c2 object sliding in.
    """
    frame1 = c1.get_frame(0)
    frame2 = c2.get_frame(0)
    mask2 = get_object_mask(frame2)

    h2, w2 = frame2.shape[:2]
    if mask2 is None:
        mask2 = np.zeros((h2, w2), dtype=np.uint8)
    else:
        mask2 = cv2.resize(mask2, (w2, h2))

    OUT_W, OUT_H = c1.w, c1.h

    def frame_fn(t):
        alpha = t / dur
        bg = (1 - alpha) * frame1 + alpha * frame2

        dx2 = int(OUT_W * (1 - alpha))
        obj2 = shift_masked_object(frame2, mask2, dx2)

        # Outgoing object static fully visible
        obj1 = frame1

        comp = bg * (1 - mask2[..., None]) + obj1 * (mask2[..., None] == 0) + obj2
        return np.clip(comp, 0, 255).astype(np.uint8)

    # Mark c2 as animated in (if needed)
    animated_clips.add(c2.filename if hasattr(c2, 'filename') else id(c2))

    return VideoClip(frame_fn, duration=dur).set_fps(FPS).resize((OUT_W, OUT_H))

from moviepy.editor import VideoClip
import numpy as np
import cv2

from ultralytics import YOLO
import numpy as np
import cv2

# Load YOLOv8 Nano instance segmentation model once at start
segmentation_model = YOLO('yolov8n-seg.pt')

def get_object_mask(frame, conf_threshold=0.25):
    """
    Return the largest object mask in the frame as a binary mask.
    """
    results = segmentation_model.predict(frame, conf=conf_threshold, verbose=False)
    if results[0].masks is None:
        return None
    masks = results[0].masks.data.cpu().numpy()  # shape (N, H, W)
    if masks.shape[0] == 0:
        return None
    mask_areas = [m.sum() for m in masks]
    largest_idx = np.argmax(mask_areas)
    return masks[largest_idx].astype(np.uint8)

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


def simple_forward_slide_object_transition(c1, c2, dur=0.4):
    frame1 = c1.get_frame(0)
    frame2 = c2.get_frame(0)
    mask2 = get_object_mask(frame2)

    h2, w2 = frame2.shape[:2]
    if mask2 is None:
        mask2 = np.zeros((h2, w2), dtype=np.uint8)
    else:
        mask2 = cv2.resize(mask2, (w2, h2))

    OUTW, OUTH = c1.w, c1.h  # output size

    def frame_fn(t):
        # Linear alpha from 0 to 1
        alpha = t / dur
        # dx moves from OUTW (start) linearly down to 0 (end)
        dx = int(OUTW * (1 - alpha))
        # shift object frame by dx pixels to right -> simple forward sliding from right to left
        obj2_shifted = shift_masked_object(frame2, mask2, dx)
        # Compose: background + masked shifted object without fades
        comp = frame1 * (1 - mask2[..., None]) + obj2_shifted * (mask2[..., None] > 0)
        return np.clip(comp, 0, 255).astype(np.uint8)

    return VideoClip(frame_fn, duration=dur).set_fps(FPS).resize(OUTW, OUTH)




def object_in_no_out_slide_fade_transition(c1, c2, dur=0.4, direction='right'):
    """
    No slide out for c1 object (no animation),
    crossfade backgrounds,
    animate c2 object sliding in with fade.
    """
    frame1 = c1.get_frame(0)
    frame2 = c2.get_frame(0)
    mask2 = get_object_mask(frame2)

    h2, w2 = frame2.shape[:2]
    if mask2 is None:
        mask2 = np.zeros((h2, w2), dtype=np.uint8)
    else:
        mask2 = cv2.resize(mask2, (w2, h2))

    OUT_W, OUT_H = c1.w, c1.h

    def frame_fn(t):
        alpha = t / dur
        bg = (1 - alpha) * frame1 + alpha * frame2

        # Calculate horizontal shift based on direction
        if direction == 'right':
            dx = int(OUT_W * (1 - alpha))
        elif direction == 'left':
            dx = -int(OUT_W * (1 - alpha))
        else:
            dx = 0

        # Slide and fade in incoming object's masked area
        slid_obj = shift_masked_object(frame2, mask2, dx)
        fade_mask = (mask2 * alpha).astype(np.float32)[..., None]
        obj2 = (slid_obj.astype(np.float32) * fade_mask).astype(np.uint8)

        # Outgoing object static fully visible
        obj1 = frame1

        # Combine with masks to avoid overlap issues
        combined_mask = np.clip(mask2, 0, 1)[..., None]
        comp = bg * (1 - combined_mask) + obj1 * (combined_mask == 0) + obj2

        return np.clip(comp, 0, 255).astype(np.uint8)

    animated_clips.add(c2.filename if hasattr(c2, 'filename') else id(c2))

    return VideoClip(frame_fn, duration=dur).set_fps(FPS).resize((OUT_W, OUT_H))


def choose_transition_with_state(prev, nxt, dur=0.4):
    prev_id = prev.filename if hasattr(prev, 'filename') else id(prev)
    nxt_id = nxt.filename if hasattr(nxt, 'filename') else id(nxt)

    # If prev clip not animated yet, animate incoming object in with no out; mark animated
    if prev_id not in animated_clips:
        animated_clips.add(prev_id)
        return object_in_transition(prev, nxt, dur)
    else:
        # Previous clip already animated, so no out animation on it
        return object_in_no_out_transition(prev, nxt, dur)




def choose_transition(prev, nxt):
    r = random.random()

    if r < 0.46:
        return impact_zoom_transition(prev, nxt, dur=0.28, strength=0.18)
    elif r < 0.86:
        return whip_transition(prev, nxt, dur=0.3, direction=random.choice(['left', 'right']))
    else:
        return flash_transition(prev, nxt, dur=0.18)  # Slightly shorter for flash to keep impact but not abrupt

# ---------- Main ----------
def main(args):
    video_paths = find_videos(args.input_videos)
    if not video_paths:
        print("No input videos found:", args.input_videos); return

    music_path = find_music_file(args.music)
    print(f"Found {len(video_paths)} video(s). Music: {music_path}")

    music_audio = AudioFileClip(music_path)
    target_duration = music_audio.duration
    print(f"Target final duration (music length) = {target_duration:.2f}s")

    # 1) transcribe each input video individually with Whisper
    print("Transcribing each input video with Whisper (this may take time)...")
    whisper_segs, tmp_wdir = transcribe_each_video(video_paths, whisper_model_name=args.whisper_model, language=args.language)

    print(f"Whisper produced {len(whisper_segs)} segments across inputs.")

    # 2) score segments and pick best punch (from input audio directly)
    best_seg = score_segments(whisper_segs, tmp_wdir)
    if best_seg:
        print("Picked punch dialogue:", best_seg.get('text','')[:120])
    else:
        print("No suitable dialogue found in Whisper output; will fall back to visual candidate.")

    # 3) prepare filler candidates
    print("Extracting visual candidates for fillers...")
    candidates = extract_candidates(video_paths)
    if not candidates:
        print("No visual candidates found; aborting.")
        try:
            shutil.rmtree(tmp_wdir)
        except Exception:
            pass
        return

    # 4) build first dialogue clip from best_seg (or fallback)
    if best_seg:
        src = best_seg['video']; s0 = max(0.0, best_seg['start']); s1 = best_seg['end']
        # extend a little for context
        pre = max(0.0, s0 - 0.45); post = min(VideoFileClip(src).duration, s1 + 0.21)
        try:
            first_clip = VideoFileClip(src).subclip(pre, post).set_fps(FPS)
            first_clip = center_crop_vertical(first_clip)
            first_clip = first_clip.fx(vfx.colorx, 1.12)
            subtitle_text = best_seg.get('text','')
            dialogue_duration = max(0.25, s1 - s0)
        except Exception as e:
            print("Failed to build dialogue clip from best_seg:", e)
            best_seg = None

    if not best_seg:
        # fallback: choose top visual candidate by RMS+motion
        fallback = max(candidates, key=lambda c: (c['rms']*50 + c['motion']*0.6))
        src = fallback['path']; pre = max(0.0, fallback['start']-0.18)
        post = min(VideoFileClip(src).duration, fallback['start'] + min(2.0, fallback['dur']) + 0.18)
        first_clip = VideoFileClip(src).subclip(pre, post).set_fps(FPS)
        first_clip = center_crop_vertical(first_clip)
        first_clip = first_clip.fx(vfx.colorx, 1.12)
        subtitle_text = fallback.get('text','')
        dialogue_duration = min(first_clip.duration, 1.8)

    # add subtitle overlay to first clip (whole-segment style)
    first_with_subs = overlay_single_subtitle(first_clip, subtitle_text, dialogue_duration)

    # 5) Build fillers guided by beats (unique, no repetition) and excluding the dialogue source/time
    beats = detect_beats(music_path)
    beat_gaps = None
    if beats is not None and len(beats) > 1:
        gaps = np.diff(beats)
        beat_gaps = [max(0.45, min(2.6, g)) for g in gaps] if len(gaps) > 0 else None
        print(f"Detected {len(beats)} beats; using beat gaps for filler durations.")
    else:
        print("No beats detected; using adaptive filler durations.")

    # mark used key for dialogue to avoid repeating same exact start
    used_key = (src, round(pre,2))

    # remaining time after the first clip
    remaining_time = max(0.0, target_duration - first_with_subs.duration)

    fillers = build_fillers(video_paths, candidates, remaining_time, used_key, beat_gaps=beat_gaps)

    # final sequence: first_with_subs + fillers
    seq_clips = [first_with_subs] + fillers
    def choose_transition_maybe(prev, nxt, apply_prob=0.5, min_clip_dur=1.0):
        if random.random() > apply_prob or prev.duration < min_clip_dur or nxt.duration < min_clip_dur:
            return None
        r = random.random()
        if r < 0.4:
            return impact_zoom_transition(prev, nxt, dur=0.28, strength=0.18)
        elif r < 0.75:
            return whip_transition(prev, nxt, dur=0.30, direction=random.choice(['left', 'right']))
        elif r < 0.9:
            return flash_transition(prev, nxt, dur=0.18)
        else:
            return masked_object_transition(prev, nxt, dur=0.3)  # use new masked object transition
        # return simple_forward_slide_object_transition(prev, nxt, dur=0.4)



    def stitching_conditionally(clips, apply_prob=0.8, min_clip_dur=0.8):
        seq = [clips[0]]
        for nxt in clips[1:]:
            prev = seq[-1]
            trans = choose_transition_maybe(prev, nxt, apply_prob=apply_prob, min_clip_dur=min_clip_dur)
            if trans is not None:
                seq.append(trans)
            seq.append(nxt)
        clean = [safe_clip(s) for s in seq if s is not None]
        return concatenate_videoclips(clean, method="compose")



    final = stitching_conditionally(seq_clips, apply_prob=0.8, min_clip_dur=0.8)




    # ensure full coverage until music end — if short, append unique snippets (no static image)
    if final.duration < target_duration - 0.02:
        print("Final shorter than music; appending more dynamic fillers...")
        extra_needed = target_duration - final.duration
        more = build_fillers(video_paths, candidates, extra_needed, used_key, beat_gaps=beat_gaps)
        if more:
            final = concatenate_videoclips([final] + more, method="compose")

    # trim or extend carefully — we must not loop video audio; so if slightly short, append last small moving snippet
    if final.duration < target_duration - 0.02:
        # append snippets from source videos (small moving parts)
        attempts = 0
        while final.duration < target_duration - 0.02 and attempts < 30:
            p = random.choice(video_paths)
            try:
                v = VideoFileClip(p)
                start = random.uniform(0, max(0, v.duration - 0.6))
                dur = min(1.2, target_duration - final.duration)
                add = make_variant(p, start, dur, kind=random.choice(["normal","speedup"]))
                final = concatenate_videoclips([final, add], method="compose")
                v.reader.close(); v.audio = None
            except Exception:
                pass
            attempts += 1

    if final.duration > target_duration:
        final = final.subclip(0, target_duration)
    final = final.set_fps(FPS).set_duration(target_duration)

    # 6) Audio mixing:
    # - Dialogue: video audio full volume, music very low
    # - After dialogue: music full volume, video muted

    print("Mixing audio: ULTRA CLEAR DIALOGUE, music ducked only during dialogue...")

    dialogue_start = 0.0
    dialogue_end = dialogue_duration

    # Load + match music duration
    music_orig = AudioFileClip(music_path)
    if music_orig.duration > final.duration:
        music_orig = music_orig.subclip(0, final.duration)
    else:
        music_orig = audio_loop(music_orig, duration=final.duration)

    # 1️⃣ Music segments
    music_very_low = music_orig.subclip(dialogue_start, dialogue_end).volumex(0.05)
    music_very_low = music_very_low.audio_fadein(0.20).audio_fadeout(0.20)

    music_after = None
    if music_orig.duration > dialogue_end:
        music_after = music_orig.subclip(dialogue_end, final.duration).volumex(1.0)

    # 2️⃣ Video audio: full for dialogue, muted after
    orig_audio = final.audio if final.audio else music_orig.volumex(0.0)

    video_loud = orig_audio.subclip(dialogue_start, dialogue_end).volumex(1.8)
    video_mute = orig_audio.subclip(dialogue_end, final.duration).volumex(0.0)

    # 3️⃣ Combine audio layers
    audio_layers = [
        music_very_low.set_start(dialogue_start),
        video_loud.set_start(dialogue_start),
        video_mute.set_start(dialogue_end)
    ]

    if music_after:
        audio_layers.append(music_after.set_start(dialogue_end))

    combined_audio = CompositeAudioClip(audio_layers).set_duration(final.duration)

    # Set final audio with smooth end fade
    final = final.set_audio(combined_audio).audio_fadeout(0.8)


    # 7) export
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print("Rendering final video (this will take a while)...")
    final.write_videofile(args.out, codec="libx264", audio_codec="aac", fps=FPS, preset="medium", threads=4, bitrate="6M", ffmpeg_params=["-movflags","+faststart"])
    print("Cleaning tmp...")
    try:
        shutil.rmtree(tmp_wdir)
    except Exception:
        pass
    print("Done ->", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fictic punch-intro (music-driven, input-audio whisper).")
    p.add_argument("--input_videos", required=True, help="Folder with input clips")
    p.add_argument("--music", required=True, help="Music file or folder (music length used)")
    p.add_argument("--out", default="./out/final_edit_9x16.mp4", help="Output path")
    p.add_argument("--whisper_model", default=WHISPER_DEFAULT, help="Whisper model (tiny/base/small/medium/large)")
    p.add_argument("--language", default=None, help="Whisper language code (optional)")
    args = p.parse_args()
    main(args)
    #to call: python3 fictic6.py --input_videos ./input_vid --music ./music --out "./out/final_edit_9x16.mp4" --whisper_model tiny 