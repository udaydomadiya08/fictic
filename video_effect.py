 # universal_noncolor_effects_updated.py
"""
Updated version of your non-color effects pack.
Fixes applied:
- Robust frame normalization: force uint8 [0,255] frames to avoid OpenCV "unsupported depth" errors (CV_32S/CV_32F).
- All remap map arrays cast to np.float32.
- Safeguards when frames are empty or shapes mismatch.
- Ensured kernel sizes are valid (odd, >=1) for blur ops.
- Fixed the final invert_motion function and removed trailing stray markers.
- Added utility: apply_safe_effect wrapper to simplify use in pipelines.

Notes:
- MoviePy frames sometimes come as float32 in [0,1] or uint8; normalize_frame handles both.
- Many previous errors were caused by passing non-uint8 / wrong-depth frames to OpenCV.

"""

import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, vfx, VideoClip

# -------------------------
# Utilities
# -------------------------

def normalize_frame(frame):
    """Return frame as uint8 RGB in shape (H,W,3).
    Handles float frames in [0,1], float frames >1, and integer types.
    """
    if frame is None:
        return None
    # if frame has alpha channel, drop it
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame[..., :3]

    # convert floats in [0,1] -> [0,255]
    if np.issubdtype(frame.dtype, np.floating):
        # clamp then scale
        f = np.nan_to_num(frame)
        # if values appear normalized to [0,1]
        if f.max() <= 1.0:
            f = (f * 255.0).round()
        else:
            f = f.round()
        out = np.clip(f, 0, 255).astype(np.uint8)
        return out

    # integers: just clip to [0,255] and cast
    out = np.clip(frame, 0, 255).astype(np.uint8)
    return out


def safe_remap(frame, map_x, map_y, interp=cv2.INTER_LINEAR):
    """Call cv2.remap with float32 maps and return uint8 frame."""
    try:
        map_x_f = map_x.astype(np.float32)
        map_y_f = map_y.astype(np.float32)
        remapped = cv2.remap(frame, map_x_f, map_y_f, interp)
        return normalize_frame(remapped)
    except Exception:
        return normalize_frame(frame)


def safe_frame(clip, t=0):
    """Return a safe frame or black frame if fails."""
    try:
        frame = clip.get_frame(min(max(0, t), clip.duration - 1e-3))
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame")
        return normalize_frame(frame)
    except Exception:
        h, w = getattr(clip, "h", 720), getattr(clip, "w", 1280)
        return np.zeros((h, w, 3), dtype=np.uint8)


# -------------------------
# Visual Score & Beat Data
# -------------------------

def compute_visual_scores(clips, frame_time=0.1):
    visual_scores = []
    for i in range(len(clips) - 1):
        prev_clip = clips[i]
        next_clip = clips[i + 1]
        prev_frame = safe_frame(prev_clip, max(0, prev_clip.duration - frame_time))
        next_frame = safe_frame(next_clip, min(frame_time, next_clip.duration))

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)

        if prev_gray.shape != next_gray.shape:
            next_gray = cv2.resize(next_gray, (prev_gray.shape[1], prev_gray.shape[0]))

        diff = cv2.absdiff(prev_gray, next_gray)
        score = float(diff.mean())
        visual_scores.append(score)
    return visual_scores


def compute_beat_data(clips, beat_times, total_duration, tolerance=0.2):
    beat_data = []
    cumulative_time = 0.0
    for i in range(len(clips) - 1):
        cumulative_time += clips[i].duration
        close_to_beat = any(abs(cumulative_time - bt) <= tolerance for bt in beat_times)
        beat_data.append(1.0 if close_to_beat else 0.0)
    return beat_data


def load_clips_from_folder(folder_path):
    clips = []
    if os.path.exists(folder_path):
        files = sorted(os.listdir(folder_path))
        for f in files:
            if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                path = os.path.join(folder_path, f)
                clips.append(VideoFileClip(path))
    else:
        print(f"⚠️ Folder not found: {folder_path}")
    return clips


# -------------------------
# Effect helpers
# -------------------------

def _ensure_odd(v):
    v = int(max(1, v))
    return v if (v % 2 == 1) else (v + 1)


def apply_safe_effect(clip, fn):
    """Wrapper: ensures the function receives normalized frames and returns normalized frames."""
    def wrapper(frame):
        frame_n = normalize_frame(frame)
        out = fn(frame_n)
        return normalize_frame(out)
    return clip.fl_image(wrapper)


# -------------------------
# CLEAN — Non-Color Effects
# -------------------------

def subtle_zoom(clip, scale=1.05):
    return clip.fx(vfx.resize, lambda t: 1 + (scale - 1) * np.sin(t * 2 * np.pi / max(clip.duration, 0.0001)))


def shake_effect(clip, intensity=5):
    def shake(frame):
        frame = normalize_frame(frame)
        dx, dy = np.random.randint(-intensity, intensity + 1), np.random.randint(-intensity, intensity + 1)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))
    return clip.fl_image(shake)


def motion_blur(clip, ksize=15):
    # Ensure odd kernel size and valid
    ksize = _ensure_odd(ksize)
    def blur(frame):
        frame = normalize_frame(frame)
        try:
            return cv2.GaussianBlur(frame, (ksize, ksize), 0)
        except Exception:
            return frame
    return clip.fl_image(blur)





def pixelate(clip, pixel_size=10):
    def pix(frame):
        frame = normalize_frame(frame)
        h, w = frame.shape[:2]
        small_w = max(1, w // max(1, pixel_size))
        small_h = max(1, h // max(1, pixel_size))
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return clip.fl_image(pix)


def glitch_effect(clip, offset=5):
    def glitch(frame):
        frame = normalize_frame(frame)
        h, w, c = frame.shape
        new_frame = np.copy(frame)
        # channel shifts
        if offset < w:
            new_frame[:, offset:, 0] = frame[:, :-offset, 0]
            new_frame[:, :-offset, 1] = frame[:, offset:, 1]
        return new_frame
    return clip.fl_image(glitch)


def mirror_effect(clip):
    def mirror(frame):
        frame = normalize_frame(frame)
        left = frame[:, :frame.shape[1] // 2]
        mirrored = np.concatenate([left, np.flip(left, axis=1)], axis=1)
        # If width changed because of odd widths, resize back
        if mirrored.shape[1] != frame.shape[1]:
            mirrored = cv2.resize(mirrored, (frame.shape[1], frame.shape[0]))
        return mirrored
    return clip.fl_image(mirror)


def fisheye_effect(clip):
    def fisheye(frame):
        frame = normalize_frame(frame)
        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        r = np.sqrt((map_x - cx) ** 2 + (map_y - cy) ** 2)
        # avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            r_new = np.power(r, 0.8)
            map_x_f = cx + ((map_x - cx) * r_new / (r + 1e-6))
            map_y_f = cy + ((map_y - cy) * r_new / (r + 1e-6))
        return safe_remap(frame, map_x_f, map_y_f, interp=cv2.INTER_LINEAR)
    return clip.fl_image(fisheye)


def film_grain(clip, intensity=15):
    def grain(frame):
        frame = normalize_frame(frame)
        noise = np.random.randint(-intensity, intensity + 1, frame.shape, dtype=np.int16)
        out = frame.astype(np.int16) + noise
        return np.clip(out, 0, 255).astype(np.uint8)
    return clip.fl_image(grain)


def soft_focus(clip, ksize=5):
    ksize = _ensure_odd(ksize)
    def focus(frame):
        frame = normalize_frame(frame)
        try:
            return cv2.GaussianBlur(frame, (ksize, ksize), 0)
        except Exception:
            return frame
    return clip.fl_image(focus)


def spotlight(clip):
    def spot(frame):
        frame = normalize_frame(frame)
        h, w = frame.shape[:2]
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2.0, h / 2.0
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask = np.clip(1 - r / (0.6 * np.max(r + 1e-6)), 0, 1)
        return (frame * mask[..., None]).astype(np.uint8)
    return clip.fl_image(spot)


def wave_distort(clip, amplitude=5, frequency=20):
    def wave(frame):
        frame = normalize_frame(frame)
        h, w = frame.shape[:2]
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x.astype(np.float32) + amplitude * np.sin(2 * np.pi * map_y / max(1.0, frequency)))
        return safe_remap(frame, map_x, map_y)
    return clip.fl_image(wave)



def radial_blur(clip, strength=5):
    strength = _ensure_odd(max(1, strength))
    def rblur(frame):
        frame = normalize_frame(frame)
        try:
            return cv2.GaussianBlur(frame, (strength, strength), 0)
        except Exception:
            return frame
    return clip.fl_image(rblur)





def ripple_effect(clip, amplitude=5, frequency=15):
    def ripple(frame):
        frame = normalize_frame(frame)
        h, w = frame.shape[:2]
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x + amplitude * np.sin(2 * np.pi * map_y / max(1.0, frequency))).astype(np.float32)
        return safe_remap(frame, map_x, map_y)
    return clip.fl_image(ripple)


def perspective_warp(clip):
    def warp(frame):
        frame = normalize_frame(frame)
        h, w = frame.shape[:2]
        try:
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            pts2 = np.float32([[w * 0.05, h * 0.05], [w * 0.95, 0], [0, h], [w, h]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            return cv2.warpPerspective(frame, M, (w, h))
        except Exception:
            return frame
    return clip.fl_image(warp)

# -------------------------
# Subtle / cinematic versions
# -------------------------

def edge_streak(clip, colors=None, thickness=2, base_intensity=0.3, pulse_speed=1.5, glow_ksize=7):
    """
    Subtle neon edges that blend gently with the original clip.
    """
    import numpy as np
    import cv2

    if colors is None:
        colors = [
            (0, 255, 255),
            (255, 0, 255),
            (0, 255, 0),
            (255, 255, 0),
            (0, 128, 255)
        ]
    num_colors = len(colors)
    glow_ksize = glow_ksize if glow_ksize % 2 == 1 else glow_ksize + 1

    def streak(frame, t=0):
        frame = normalize_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Thicken edges
        kernel = np.ones((thickness, thickness), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)

        # Pulse intensity
        pulse_intensity = base_intensity + (0.15 * np.sin(2 * np.pi * pulse_speed * t))

        # Color edges
        edges_col = np.zeros_like(frame)
        y_idx, x_idx = np.where(edges_dilated > 0)
        for y, x in zip(y_idx, x_idx):
            color_idx = int((x + y + t * 50) % num_colors)
            color = colors[color_idx]
            edges_col[y, x] = [int(c * pulse_intensity) for c in color]

        # Soft glow
        edges_glow = cv2.GaussianBlur(edges_col, (glow_ksize, glow_ksize), 0)

        # Blend with original frame subtly
        out = cv2.addWeighted(frame, 0.85, edges_glow.astype(np.uint8), 0.15, 0)
        return normalize_frame(out)

    return clip.fl_image(streak)




def zoom_blur_effect(clip, strength=0.01, blend=0.2):
    """
    Soft zoom blur that keeps original clip visible.
    """
    def zoom(frame):
        frame = normalize_frame(frame)
        h, w = frame.shape[:2]
        zx, zy = int(round(w * (1 + strength))), int(round(h * (1 + strength)))
        zx = max(zx, w + 1)
        zy = max(zy, h + 1)
        try:
            zoomed = cv2.resize(frame, (zx, zy), interpolation=cv2.INTER_LINEAR)
            x1, y1 = (zx - w) // 2, (zy - h) // 2
            zoomed_crop = zoomed[y1:y1 + h, x1:x1 + w]
            return cv2.addWeighted(frame, 1.0 - blend, zoomed_crop, blend, 0)
        except Exception:
            return frame
    return clip.fl_image(zoom)


def rotate_xzoom(clip, max_angle=7, xzoom_factor=1.05, blend=0.2):
    """
    Gentle rotation and horizontal zoom that keeps footage readable.
    """
    def fx(frame, t):
        frame = normalize_frame(frame)
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # Rotation
        angle = max_angle * np.sin(2 * np.pi * t / clip.duration)
        M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(frame, M_rot, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Horizontal zoom
        zx = 1 + (xzoom_factor - 1) * np.sin(2 * np.pi * t / clip.duration)
        x_scaled = cv2.resize(rotated, (int(w*zx), h), interpolation=cv2.INTER_LINEAR)

        # Crop/pad
        if zx >= 1:
            start = (x_scaled.shape[1] - w) // 2
            x_scaled = x_scaled[:, start:start+w]
        else:
            pad = (w - x_scaled.shape[1]) // 2
            x_scaled = cv2.copyMakeBorder(x_scaled, 0, 0, pad, w - x_scaled.shape[1] - pad, cv2.BORDER_REFLECT)

        # Blend with original
        return cv2.addWeighted(frame, 1.0 - blend, x_scaled, blend, 0)

    return clip.fl(fx, apply_to=['mask', 'video'])



def invert_motion(clip):
    # Fixed: use vfx.time_mirror and return clip
    return clip.fx(vfx.time_mirror)


# Example: how to compose safely
# myclip = VideoFileClip('in.mp4')
# myclip = subtle_zoom(myclip, 1.03)
# myclip = film_grain(myclip, intensity=12)
# myclip.write_videofile('out.mp4')


# End of file



# -------------------------
# Effect Pack Dictionary
# -------------------------
EFFECT_PACK = {
    "subtle_zoom": subtle_zoom,
    "shake_effect": shake_effect,
    "motion_blur": motion_blur,
 
    "pixelate": pixelate,
    "glitch_effect": glitch_effect,
    # "mirror_effect": mirror_effect,
    # "fisheye_effect": fisheye_effect,
    "film_grain": film_grain,
    "soft_focus": soft_focus,
    # "spotlight": spotlight,
    "wave_distort": wave_distort,
    "zoom_blur_effect": zoom_blur_effect,
    "radial_blur": radial_blur,
    "edge_streak": edge_streak,
    "ripple_effect": ripple_effect,
    "rotate_xzoom": rotate_xzoom,
    # "perspective_warp": perspective_warp,
    # "invert_motion": invert_motion,
    # "stretch_horizontal": stretch_horizontal,
    # "stretch_vertical": stretch_vertical
}

#import random

# Example categorization of effects
HIGH_ENERGY = {
    "glitch_effect": 0.3,
    "ripple_effect": 0.3,
    "wave_distort": 0.3,
    # "invert_motion": 0.3
}

MEDIUM_ENERGY = {
    "motion_blur": 0.4,
    "soft_focus": 0.4,
    "zoom_blur_effect": 0.4,
    # "stretch_horizontal": 0.4,
    # "stretch_vertical": 0.4
}

SUBTLE_ENERGY = {

    "edge_streak": 0.5,
    # "mirror_effect": 0.5,
    # "fisheye_effect": 0.5,
    # "perspective_warp": 0.5,
    # "spotlight": 0.5,
    "radial_blur": 0.5,
    "pixelate": 0.5,
    "film_grain": 0.5,
    "subtle_zoom": 0.5,
    "rotate_xzoom": 0.5
}

def apply_best_effect(clip, visual_score=0, beat_strength=0, history=None, max_history=5):
    """
    Smartly apply one of the 20 universal non-color effects based on energy tiers.
    - High energy → strong motion/glitch effects
    - Medium energy → motion/blur
    - Subtle → stylized/soft effects
    - Avoid recent repeats
    """
    if history is None:
        history = []

    import random

    # Determine energy tier
    if beat_strength > 0.7 and visual_score > 20:
        category = list(HIGH_ENERGY.keys())
        duration = 0.3
    elif beat_strength > 0.4 or visual_score > 15:
        category = list(MEDIUM_ENERGY.keys())
        duration = 0.4
    else:
        category = list(SUBTLE_ENERGY.keys())
        duration = 0.5

    # Avoid repeating recent effects
    available = [e for e in category if e not in history[-max_history:]]
    if not available:
        available = category  # fallback if all recently used

    # Try up to 3 attempts to apply a working effect
    for attempt in range(3):
        chosen_name = random.choice(available)
        chosen_effect = EFFECT_PACK[chosen_name]
        print(f"🎨 Applying effect: {chosen_name}")

        try:
            clip_with_effect = chosen_effect(clip)
            # Simple sanity check
            if clip_with_effect is None or clip_with_effect.duration == 0:
                raise ValueError("Effect returned invalid clip")
            # Update history
            history.append(chosen_name)
            if len(history) > max_history:
                history.pop(0)
            return clip_with_effect, history, duration
        except Exception as e:
            print(f"⚠ Effect {chosen_name} failed: {e}")
            available.remove(chosen_name)
            if not available:
                break

    # Fallback: return original clip
    print("⚠ No effect could be applied safely → using original clip")
    return clip, history, duration
