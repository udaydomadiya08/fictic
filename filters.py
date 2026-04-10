import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import random
import os
# ------------------------------------------------------------
# FAST COLOR TRANSFORM HELPERS (NO FRAME LOOPS)
# ------------------------------------------------------------

def apply_color_matrix(frame, matrix):
    h, w = frame.shape[:2]
    reshaped = frame.reshape(-1, 3).astype(np.float32)
    out = reshaped @ matrix.T
    out = np.clip(out, 0, 255)
    return out.reshape(h, w, 3).astype(np.uint8)

def apply_lut(frame, lut):
    return cv2.LUT(frame, lut)

# ------------------------------------------------------------
# LUT BUILDERS (fast, instant)
# ------------------------------------------------------------

def lut_teal(strength):
    x = np.arange(256, dtype=np.float32)
    teal = x + strength * 60
    teal = np.clip(teal, 0, 255)
    return teal.astype(np.uint8)

def lut_orange(strength):
    x = np.arange(256, dtype=np.float32)
    orange = x + strength * 40
    orange = np.clip(orange, 0, 255)
    return orange.astype(np.uint8)

def lut_fade(strength):
    x = np.arange(256, dtype=np.float32)
    fade = x * (1 - strength)
    fade = np.clip(fade, 0, 255)
    return fade.astype(np.uint8)

def lut_crush(strength):
    x = np.arange(256, dtype=np.float32)
    crush = x ** (1 + strength)
    crush = crush / crush.max() * 255
    return crush.astype(np.uint8)

def lut_high_contrast(strength):
    x = np.arange(256, dtype=np.float32)
    hc = x * (1 + strength)
    hc = np.clip(hc, 0, 255)
    return hc.astype(np.uint8)

# ------------------------------------------------------------
# 20 CINEMATIC FILTER PRESETS
# ------------------------------------------------------------

FAST_20_FILTERS = {
    # 1
    "cinematic_teal_orange": {
        "matrix": np.array([[1.12, -0.05, 0.02],
                            [-0.01, 1.10, -0.03],
                            [0.02, -0.02, 1.18]]),
        "lut": lut_teal(0.22)
    },

    # 2
    "warm_gold": {
        "matrix": np.array([[1.05, 0, 0],
                            [0, 1.03, 0],
                            [0, 0, 1.20]]),
        "lut": lut_orange(0.20)
    },

    # 3
    "cool_blue": {
        "matrix": np.array([[1.15, 0, 0],
                            [0, 1.05, 0],
                            [0, 0, 0.85]]),
        "lut": lut_teal(0.18)
    },

    # 4
    "vibrant_pop": {
        "matrix": np.array([[1.18, 0, 0],
                            [0, 1.15, 0],
                            [0, 0, 1.10]]),
        "lut": lut_high_contrast(0.15)
    },

    # 5
    "soft_fade": {
        "matrix": np.eye(3),
        "lut": lut_fade(0.25)
    },

    # 6
    "high_contrast": {
        "matrix": np.array([[1.22, 0, 0],
                            [0, 1.22, 0],
                            [0, 0, 1.22]]),
        "lut": lut_high_contrast(0.22)
    },

    # 7
    "crushed_blacks": {
        "matrix": np.eye(3),
        "lut": lut_crush(0.15)
    },

    # 8
    "matte_low_contrast": {
        "matrix": np.array([[0.92, 0, 0],
                            [0, 0.92, 0],
                            [0, 0, 0.92]]),
        "lut": lut_fade(0.20)
    },

    # 9
    "film_orange_boost": {
        "matrix": np.array([[1.10, 0.02, 0],
                            [-0.01, 1.05, -0.02],
                            [0, -0.03, 1.02]]),
        "lut": lut_orange(0.30)
    },

    # 10
    "purple_shadow": {
        "matrix": np.array([[1.00, 0, 0.05],
                            [0, 1.00, 0],
                            [0.05, 0, 1.00]]),
        "lut": lut_teal(0.10)
    }
}

# Auto-add remaining filters (11–20)
while len(FAST_20_FILTERS) < 20:
    strength = random.uniform(0.05, 0.25)
    FAST_20_FILTERS[f"look_{len(FAST_20_FILTERS)+1}"] = {
        "matrix": np.eye(3) + np.random.uniform(-0.08, 0.08, (3, 3)),
        "lut": lut_high_contrast(strength)
    }

# ------------------------------------------------------------
# AUTO-SELECT FILTER
# ------------------------------------------------------------

def fast_auto_choose(clip):
    frame = clip.get_frame(clip.duration / 2)
    bright = frame.mean()

    if bright < 80:
        return "cinematic_teal_orange"
    elif bright > 180:
        return "cool_blue"
    else:
        return random.choice(list(FAST_20_FILTERS.keys()))

# ------------------------------------------------------------
# APPLY FILTER FAST
# ------------------------------------------------------------
from moviepy.editor import VideoFileClip

def fast_apply_filter(video_input, out_path=None, filter_name=None):
    """
    Apply a fast color filter to a clip, safe for both file paths and VideoClip objects.

    video_input: str (file path) or VideoClip object
    out_path: str (file path to save filtered video, optional)
    filter_name: key in FAST_20_FILTERS
    """
    # Load clip if input is a path
    if isinstance(video_input, str):
        if not os.path.exists(video_input):
            raise FileNotFoundError(f"Input video file not found: {video_input}")
        clip = VideoFileClip(video_input)
    else:
        clip = video_input  # already a VideoClip object

    # Check filter
    if filter_name not in FAST_20_FILTERS:
        raise ValueError(f"Filter '{filter_name}' not found in FAST_20_FILTERS")
    preset = FAST_20_FILTERS[filter_name]
    matrix = preset.get("matrix", None)
    lut = preset.get("lut", None)

    # Apply color grading
    def grade(frame):
        if matrix is not None:
            frame = apply_color_matrix(frame, matrix)
        if lut is not None:
            frame = apply_lut(frame, lut)
        return frame

    filtered_clip = clip.fl_image(grade)

    # Save to disk if requested
    if out_path:
        filtered_clip.write_videofile(out_path, codec="libx264", fps=clip.fps)
        return out_path, filter_name
    else:
        return filtered_clip, filter_name

