# transit4.py  (UPDATED)
import os
import cv2
import random
import numpy as np
import subprocess
from moviepy.editor import (
    VideoClip, VideoFileClip, ImageClip,
    CompositeVideoClip, ColorClip, vfx
)
from transit import YoloSamSeg
segmenter = YoloSamSeg(
    yolo_model="yolov8n.pt",
    sam_checkpoint="sam_b.pth"
)

# ========================================================
# UTILITY: build RGBA from mask + bbox
# Accepts bbox as either (x0,y0,x1,y1) OR (x,y,w,h)
# ========================================================
def build_object_rgba(rgb_frame, mask_255, bbox):
    H, W = rgb_frame.shape[:2]

    # normalize mask: allow bool mask or 0/255 mask
    if mask_255 is None:
        mask = np.zeros((H, W), dtype=np.uint8)
    else:
        mask = np.array(mask_255)
        if mask.dtype == bool:
            mask = (mask.astype(np.uint8) * 255)
        elif mask.max() <= 1:
            mask = (mask.astype(np.uint8) * 255)
        else:
            mask = mask.astype(np.uint8)

    # bbox normalization
    if bbox is None:
        x0, y0, x1, y1 = 0, 0, W - 1, H - 1
    else:
        if len(bbox) != 4:
            raise ValueError("bbox must be 4-length sequence")
        a, b, c, d = bbox
        # If bbox looks like x0,y0,x1,y1 (x1>x0 and y1>y0)
        if (c >= 0 and d >= 0) and (c > a and d > b):
            x0, y0, x1, y1 = int(a), int(b), int(c), int(d)
        else:
            # Otherwise treat as x,y,w,h
            x0, y0 = int(a), int(b)
            w = int(c) if int(c) > 0 else 1
            h = int(d) if int(d) > 0 else 1
            x1, y1 = x0 + w - 1, y0 + h - 1

    # clamp bbox inside frame
    x0 = max(0, min(x0, W - 1))
    y0 = max(0, min(y0, H - 1))
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))

    # ensure non-empty
    if x1 < x0: x1 = x0
    if y1 < y0: y1 = y0

    # crop rgb and mask
    crop_rgb = rgb_frame[y0:y1+1, x0:x1+1].copy()
    crop_mask = mask[y0:y1+1, x0:x1+1].copy()

    # if mask single channel, expand to alpha (0..255)
    if crop_mask.ndim == 3:
        crop_mask = cv2.cvtColor(crop_mask, cv2.COLOR_BGR2GRAY)

    # ensure alpha shape matches crop_rgb
    if crop_mask.dtype != np.uint8:
        crop_mask = crop_mask.astype(np.uint8)

    # create 4-channel rgba
    alpha = crop_mask
    # If mask values are 0/1, scale up
    if alpha.max() <= 1:
        alpha = (alpha * 255).astype(np.uint8)

    rgba = np.dstack([crop_rgb, alpha])
    return np.clip(rgba, 0, 255).astype("uint8")


# ========================================================
# PERSON STATIC TRANSITION (uses ALL bg_clips as moving bg)
# ========================================================

def best_person_frame(c2, segmenter, num_samples=10):
    best_frame = None
    best_bbox_area = 0

    for t in np.linspace(0, c2.duration, num_samples):
        try:
            frame = c2.get_frame(t)
            mask, bbox = segmenter.segment_first_frame(frame)
            if bbox is not None:
                x, y, w, h = bbox
                area = w * h
                if area > best_bbox_area:  # bigger bbox → more fully visible
                    best_bbox_area = area
                    best_frame = frame
        except Exception:
            continue

    if best_frame is None:
        # fallback
        midtime = c2.duration / 2
        best_frame = c2.get_frame(midtime)

    return best_frame


def transition_person_static(
    c1, c2, bg_clips, segmenter,
    person_effects=True,
    bg_effects=True,
    bg_blur_k=25,
    speed_multiplier=11
):
    """
    c1: previous clip (duration/fps used)
    c2: next clip (person segmentation comes from c2)
    bg_clips: list of clips used as background (fast-scan)
    segmenter: object providing segment_first_frame(frame) -> (mask, bbox)
    """

    duration = max(0.01, float(getattr(c1, "duration", 0.01)))
    fps = c1.fps or 30

    # ------------------------------
    # SAFE SEGMENTATION (from c2)
    # ------------------------------
    try:
        frame2 = best_person_frame(c2, segmenter, num_samples=15)
        mask, bbox = segmenter.segment_first_frame(frame2)
    except Exception as e:
        print("❌ PERSON SEGMENTATION FAILED:", e)
        return c2


    # If bbox format uncertain, normalize to x,y,w,h
    if bbox is None:
        print("⚠ segmenter returned no bbox")
        return c2

    # Accept both bbox forms: (x0,y0,x1,y1) or (x,y,w,h)
    bx0, by0, bx1, by1 = None, None, None, None
    if len(bbox) == 4:
        a, b, c, d = bbox
        if (c > a) and (d > b):
            # x0,y0,x1,y1
            bx0, by0, bx1, by1 = int(a), int(b), int(c), int(d)
            x, y, w, h = bx0, by0, bx1 - bx0 + 1, by1 - by0 + 1
        else:
            # x,y,w,h
            x, y, w, h = int(a), int(b), int(c if c>0 else 1), int(d if d>0 else 1)
            bx0, by0, bx1, by1 = x, y, x + w - 1, y + h - 1
    else:
        print("⚠ unexpected bbox format, fallback")
        return c2

    # sanity check mask
    mask_arr = np.array(mask) if mask is not None else None
    if mask_arr is None or mask_arr.size == 0 or mask_arr.sum() < 30:
        print("⚠ Bad mask → fallback to raw c2")
        return c2

    # clamp bbox to frame
    H, W = frame2.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    bbox_xywh = (x, y, w, h)
    bbox_x0y0x1y1 = (x, y, x + w - 1, y + h - 1)

    # Build RGBA safely
    try:
        person_rgba = build_object_rgba(frame2, mask_arr, bbox_x0y0x1y1)
        person_clip = ImageClip(person_rgba, duration=duration)
    except Exception as e:
        print("⚠ person RGBA failed:", e)
        return c2

    # ------------------------------
    # PERSON EFFECTS
    # ------------------------------
    # SAFE person effects application
    if person_effects:
        try:
            effect = random.choice(["zoom", "fade", "wave"])
            if effect == "zoom":
                person_clip = person_clip.resize(1.05)
                person_clip = person_clip.resize(lambda t: 1.05 - 0.05 * np.sin(2*np.pi*t/duration))
            elif effect == "fade":
                def fade_fx(img):
                    alpha = np.clip(0.7 + 0.3 * np.sin(2*np.pi*np.linspace(0,1,img.shape[0]))[:, None],0,1)
                    alpha = np.repeat(alpha, img.shape[1], axis=1)[:,:,None]
                    img[:,:,:3] = (img[:,:,:3] * alpha).astype(np.uint8)
                    return img
                person_clip = person_clip.fl_image(fade_fx)
            elif effect == "wave":
                def wave_fx(img):
                    hh, ww, _ = img.shape
                    shift = (5 * np.sin(np.linspace(0, 2*np.pi, hh))).astype(np.int32)
                    out = img.copy()
                    for i in range(hh):
                        out[i] = np.roll(img[i], shift[i], axis=0)
                    return out
                person_clip = person_clip.fl_image(wave_fx)
        except Exception as e:
            print("⚠ person effect failed:", e)


    person_clip = person_clip.set_position((x, y))

    # ------------------------------
    # BACKGROUND FX (scan through all bg_clips)
    # ------------------------------
    if not isinstance(bg_clips, (list, tuple)) or len(bg_clips) == 0:
        print("⚠ bg_clips empty or invalid → fallback c2")
        return c2

    n = len(bg_clips)
    bg_effect = random.choice(["zoom", "fade", "wave"]) if bg_effects else None

    def bg_fx(t):
        try:
            # choose which bg clip by time (fast-scan)
            if not bg_clips or len(bg_clips) == 0:
                raise RuntimeError("bg_clips empty")

            idx = int(t * speed_multiplier * 10) % len(bg_clips)
            bg = bg_clips[idx]

            # compute time inside chosen bg
            bg_dur = max(0.01, getattr(bg, "duration", 0.01))
            t_c = (t / duration) * bg_dur * speed_multiplier
            t_c = t_c % bg_dur

            frame = bg.get_frame(t_c)
            if frame is None or frame.size == 0:
                raise RuntimeError("bg.get_frame returned None or empty frame")

        except Exception as e:
            # return a safe black frame with correct size
            h_out, w_out = getattr(c2, "size", (0, 0))[1], getattr(c2, "size", (0, 0))[0]
            if h_out == 0 or w_out == 0:
                h_out, w_out = 720, 1280  # fallback resolution
            print("⚠ bg frame error:", e)
            return np.zeros((h_out, w_out, 3), dtype=np.uint8)

        # blur
        try:
            k = bg_blur_k if (bg_blur_k % 2 == 1) else max(1, bg_blur_k-1)
            if frame.shape[0] > 1 and frame.shape[1] > 1:
                frame = cv2.GaussianBlur(frame, (k, k), 0)
        except Exception:
            pass

        # apply bg effects
        try:
            hh, ww = frame.shape[:2]

            if bg_effect == "zoom":
                scale = 1.05 - 0.05 * np.sin(2*np.pi*t/duration)
                M = cv2.getRotationMatrix2D((ww//2, hh//2), 0, scale)
                frame = cv2.warpAffine(frame, M, (ww, hh))

            elif bg_effect == "fade":
                alpha = np.clip(0.7 + 0.3 * np.sin(2*np.pi*t/duration), 0, 1)
                frame = np.clip(frame * alpha, 0, 255).astype(np.uint8)

            elif bg_effect == "wave":
                shift = (5 * np.sin(2*np.pi*t/duration + np.linspace(0, 2*np.pi, hh))).astype(np.int32)
                out = frame.copy()
                for i in range(hh):
                    out[i] = np.roll(frame[i], shift[i], axis=0)
                frame = out

        except Exception as e:
            print("⚠ bg effect failed:", e)

        return frame


    try:
        bg_clip = VideoClip(bg_fx, duration=duration).set_fps(fps)
    except Exception as e:
        print("⚠ bg generation failed:", e)
        return c2

    # ------------------------------
    # FINAL COMPOSITE
    # ------------------------------
    try:
        final = CompositeVideoClip([bg_clip, person_clip], size=c2.size)
        final.fps = fps
        final = final.set_duration(duration)
        return final
    except Exception as e:
        print("❌ composite failed:", e)
        return c2


# ========================================================
# RE-ENCODE BROKEN FILE
# ========================================================
def reencode_clip_safe(path, temp_folder="temp_reencoded"):
    if not os.path.isfile(path):
        print(f"❌ File missing: {path}")
        return None

    os.makedirs(temp_folder, exist_ok=True)
    base = os.path.basename(path)
    out = os.path.join(temp_folder, base)

    cmd = [
        "ffmpeg", "-y", "-i", path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k", out
    ]

    try:
        subprocess.run(cmd, check=True)
        return out
    except Exception as e:
        print(f"⚠ reencode failed for {path} -> {e}")
        return None


# ========================================================
# SAFE REVERSE CLIP
# ========================================================
from moviepy.editor import VideoFileClip, vfx

def replace_with_reverse_safe(c1, c2, min_duration=0.8, should_reverse=False):
    # ------------------------------
    # Helpers (same as before)
    # ------------------------------
    def try_read(clip, tag):
        try:
            clip.get_frame(0)
            return True
        except Exception as e:
            print(f"⚠ {tag} cannot read frame 0 -> {e}")
            return False

    def force_reload(path):
        from transit4 import reencode_clip_safe
        new = reencode_clip_safe(path)
        if not new:
            print(f"❌ reencode failed for {path}")
            return None
        try:
            fixed = VideoFileClip(new)
            fixed.get_frame(0)
            return fixed
        except Exception as e:
            print(f"❌ reencoded file still unreadable: {new} -> {e}")
            return None

    # ------------------------------
    # Validate c1
    # ------------------------------
    if not try_read(c1, "c1"):
        fixed = force_reload(getattr(c1, "filename", None))
        if not fixed:
            print("⛔ cannot use c1 → using c2 directly")
            return c2
        c1 = fixed

    # ------------------------------
    # Validate c2
    # ------------------------------
    if not try_read(c2, "c2"):
        fixed = force_reload(getattr(c2, "filename", None))
        if not fixed:
            print("⛔ cannot use c2 → using forward c1")
            return c1
        c2 = fixed

    # ------------------------------
    # Validate durations
    # ------------------------------
    if not c1.duration or c1.duration < min_duration:
        print("⚠ c1 too short → fallback")
        return c2

    if not c2.duration or c2.duration < min_duration:
        print("⚠ c2 too short → fallback")
        return c1

    # ------------------------------
    # Reverse if requested
    # ------------------------------
    if should_reverse:
        try:
            rev = c1.fx(vfx.time_mirror)
        except Exception as e:
            print("⚠ reverse failed:", e)
            fixed = force_reload(getattr(c1, "filename", None))
            if not fixed:
                print("❌ cannot reverse → fallback")
                return c2
            try:
                rev = fixed.fx(vfx.time_mirror)
            except Exception as e:
                print("❌ reverse still failing → fallback:", e)
                return c2

        try:
            rev = rev.set_duration(c1.duration)
            if hasattr(c1, "fps") and c1.fps:
                rev = rev.set_fps(c1.fps)
        except:
            pass

        try:
            rev.get_frame(0)
        except Exception:
            print("❌ reversed clip corrupt → fallback")
            return c2

        return rev

    # ------------------------------
    # Default: return c2 if no reverse
    # ------------------------------
    return c2

