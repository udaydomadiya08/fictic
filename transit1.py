import random
import numpy as np
import cv2
from moviepy.editor import VideoClip, concatenate_videoclips, vfx, ImageClip, VideoFileClip
import shutil
import os

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


FPS = 30

OUTW, OUTH = 1080, 1920  # Example output resolution
MIN_CLIP_DUR = 0.6 
import random
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


"""
fictic_mixed_transitions.py
Mixed Fictic + Reels-style transition pack (10 transitions).
Requires: moviepy, numpy, opencv-python
"""

import numpy as np
import cv2
from moviepy.editor import VideoClip, ColorClip, concatenate_videoclips

# Default output size (vertical Reels/TikTok). Change if you want landscape.
OUT_W, OUT_H = 1080, 1920

# -----------------------
# Utilities
# -----------------------
def _safe_segment(clip, start, end):
    seg = clip.subclip(max(0, start), max(0.001, end))
    return seg.resize((OUT_W, OUT_H)).without_audio()

def _make_frame_from_source(src_clip, src_t_offset=0.0):
    """Return a callable f(t) that retrieves frame at src_t_offset + t from src_clip"""
    def f(t):
        return src_clip.get_frame(src_t_offset + t)
    return f

# -----------------------
# 1) Cross-Zoom (zoom out current → zoom in next with soft crossfade)
# -----------------------
def crosszoom_transition(c1, c2, dur=0.18, zoom_strength=0.28):
    a = _safe_segment(c1, c1.duration - dur, c1.duration)
    b = _safe_segment(c2, 0, dur)

    def frame_a(frame, prog):
        # frame already given; produce zoomed crop
        h, w = frame.shape[:2]
        scale = 1.0 + zoom_strength * prog
        resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        cy, cx = resized.shape[0] // 2, resized.shape[1] // 2
        y1 = cy - h//2; x1 = cx - w//2
        return resized[y1:y1+h, x1:x1+w]

    def make_frame(t):
        prog = t / dur
        fa = a.get_frame(t)
        fb = b.get_frame(t)
        za = frame_a(fa, 1 - prog)  # outgoing zoom-out
        zb = frame_a(fb, prog)      # incoming zoom-in
        alpha = prog
        return (za * (1 - alpha) + zb * alpha).astype(np.uint8)

    return VideoClip(make_frame=make_frame, duration=dur)

# -----------------------
# 2) Whip Pan Blur (fast directional pan + motion blur)
# -----------------------
def whip_pan_blur(c1, c2, dur=0.16, direction='left'):
    a = _safe_segment(c1, c1.duration - dur, c1.duration)
    b = _safe_segment(c2, 0, dur)

    def make_frame(t):
        prog = t / dur
        offset = int(OUT_W * prog)
        if direction == 'left':
            fa = a.get_frame(t)
            fb = b.get_frame(t)
            # create motion-blurred strips
            out = np.zeros_like(fa)
            # outgoing moves left
            if offset < OUT_W:
                out[:, :OUT_W-offset] = fa[:, offset:]
            # incoming slides from right
            out[:, OUT_W-offset:] = fb[:, :offset] if offset>0 else out[:, OUT_W-offset:]
            # add slight blur to simulate whip
            return cv2.GaussianBlur(out, (9,9), sigmaX=6)
        else:
            # right direction mirror
            fa = a.get_frame(t)
            fb = b.get_frame(t)
            out = np.zeros_like(fa)
            if offset < OUT_W:
                out[:, offset:] = fa[:, :OUT_W-offset]
            out[:, :offset] = fb[:, OUT_W-offset:] if offset>0 else out[:, :offset]
            return cv2.GaussianBlur(out, (9,9), sigmaX=6)

    return VideoClip(make_frame=make_frame, duration=dur)

# -----------------------
# 3) Flash Punch (short bright flash + quick zoom punch into next clip)
# -----------------------
def flash_punch(c1, c2, dur=0.12, flash_frac=0.25, punch=0.22):
    flash_d = dur * flash_frac
    a = _safe_segment(c1, c1.duration - (dur - flash_d), c1.duration)
    b = _safe_segment(c2, 0, dur)

    def make_frame(t):
        if t < flash_d:
            # bright white flash fading
            alpha = 1 - (t/flash_d)
            return (255 * np.ones((OUT_H, OUT_W, 3), dtype=np.uint8) * alpha).astype(np.uint8)
        else:
            tt = t - flash_d
            prog = tt / (dur - flash_d)
            # punch zoom into b
            frame_b = b.get_frame(tt)
            scale = 1.0 + punch * (1 - np.cos(prog * np.pi))  # snappy ease
            resized = cv2.resize(frame_b, None, fx=scale, fy=scale)
            ch, cw = resized.shape[:2]
            y = (ch - OUT_H)//2
            x = (cw - OUT_W)//2
            return resized[y:y+OUT_H, x:x+OUT_W]

    return VideoClip(make_frame=make_frame, duration=dur)

# -----------------------
# 4) Shake-Glitch (random jitter + color channel split)
# -----------------------
def shake_glitch(c1, c2, dur=0.18, intensity=18, chroma_shift=8):
    a = _safe_segment(c1, c1.duration - dur, c1.duration)
    b = _safe_segment(c2, 0, dur)

    rng = np.random.RandomState(42)

    def make_frame(t):
        prog = t/dur
        if prog < 0.6:
            frame = a.get_frame(t)
        else:
            frame = b.get_frame(t)
        # jitter
        dx = int(intensity * (1 - abs(0.5 - prog)*2) * (rng.rand() - 0.5))
        dy = int(intensity * (rng.rand() - 0.5))
        M = np.float32([[1,0,dx],[0,1,dy]])
        jittered = cv2.warpAffine(frame, M, (OUT_W, OUT_H), borderMode=cv2.BORDER_REPLICATE)
        # chroma split
        bch, gch, rch = cv2.split(jittered)
        bch = np.roll(bch, chroma_shift, axis=1)
        rch = np.roll(rch, -chroma_shift, axis=1)
        merged = cv2.merge([bch, gch, rch])
        # mild scanline artifact
        rows = merged.copy()
        rows[::4, :, :] = (rows[::4, :, :] * 0.9).astype(np.uint8)
        return rows

    return VideoClip(make_frame=make_frame, duration=dur)

# -----------------------
# 5) Speed-Ramp Freeze → Next (fast ramp then freeze frame into next clip)
# -----------------------
def speed_ramp_to_next(c1, c2, dur=0.22, ramp_factor=5.0):
    # ramp compress final portion of c1 then quick jump into c2
    seg_d = min(dur, 0.4)
    a_src = _safe_segment(c1, c1.duration - seg_d, c1.duration)
    b = _safe_segment(c2, 0, dur)

    def make_frame(t):
        prog = t/dur
        if prog < 0.6:
            # play compressed (speed up) portion from a_src
            tt = min(a_src.duration, (t / 0.6) * a_src.duration * ramp_factor)
            tt = min(a_src.duration - 1e-6, tt)
            frame = a_src.get_frame(tt)
        else:
            # freeze last frame of a then mix into b quickly
            freeze_frame = a_src.get_frame(a_src.duration - 1e-6)
            fb = b.get_frame(t)
            alpha = (prog - 0.6) / 0.4
            frame = (freeze_frame * (1 - alpha) + fb * alpha).astype(np.uint8)
        return frame

    return VideoClip(make_frame=make_frame, duration=dur)

# -----------------------
# 6) Blur-Wipe (gaussian blur wipe from top to bottom)
# -----------------------
def blur_wipe(c1, c2, dur=0.2):
    a = _safe_segment(c1, c1.duration - dur, c1.duration)
    b = _safe_segment(c2, 0, dur)

    def make_frame(t):
        prog = t/dur
        cut = int(OUT_H * prog)
        fa = a.get_frame(t)
        fb = b.get_frame(t)
        out = fa.copy()
        # top part replaced by b, with blur blend area
        if cut > 0:
            out[:cut] = fb[:cut]
            # soft blend 40px band
            band = 40
            lo = max(0, cut - band)
            for i in range(lo, min(cut, lo+band)):
                w = (i - lo) / band
                out[i] = (1 - w) * fa[i] + w * fb[i]
        # slight global blur depending on prog
        k = 1 + int(6 * (1 - prog))
        if k % 2 == 0: k += 1
        return cv2.GaussianBlur(out.astype(np.uint8), (k, k), sigmaX=0)

    return VideoClip(make_frame=make_frame, duration=dur)

# -----------------------
# 7) Spin-Merge (rotation outgoing + fade into incoming)
# -----------------------
def spin_merge(c1, c2, dur=0.24, spins=1.0):
    a = _safe_segment(c1, c1.duration - dur, c1.duration)
    b = _safe_segment(c2, 0, dur)

    def make_frame(t):
        prog = t/dur
        fa = a.get_frame(t)
        fb = b.get_frame(t)
        angle = 360 * spins * prog
        M = cv2.getRotationMatrix2D((OUT_W/2, OUT_H/2), angle, 1.0)
        ra = cv2.warpAffine(fa, M, (OUT_W, OUT_H), borderMode=cv2.BORDER_REPLICATE)
        alpha = prog
        merged = (ra * (1 - alpha) + fb * alpha).astype(np.uint8)
        return merged

    return VideoClip(make_frame=make_frame, duration=dur)

# -----------------------
# 8) Echo-Trail (short trailing ghost frames into the next)
# -----------------------
def echo_trail(c1, c2, dur=0.2, echoes=4):
    a = _safe_segment(c1, c1.duration - dur, c1.duration)
    b = _safe_segment(c2, 0, dur)

    def make_frame(t):
        prog = t/dur
        base = b.get_frame(t)
        frame = base.astype(np.float32) * 0.0
        # sample trailing frames from a before transition (evenly spaced)
        for i in range(echoes):
            frac = (i + 1) / (echoes + 1)
            ta = max(0, a.duration * (1 - prog) - frac * (a.duration / (echoes + 1)))
            try:
                fa = a.get_frame(min(a.duration - 1e-6, ta))
            except Exception:
                fa = a.get_frame(0)
            weight = (1.0 - frac) * (1 - prog) * 0.6 / (i + 1)
            frame += fa.astype(np.float32) * weight
        frame = np.clip(frame + base.astype(np.float32), 0, 255).astype(np.uint8)
        return frame

    return VideoClip(make_frame=make_frame, duration=dur)

# -----------------------
# 9) Lens Burn (bright radial burn then cut to next)
# -----------------------
def lens_burn(c1, c2, dur=0.16, intensity=1.6):
    a = _safe_segment(c1, c1.duration - dur, c1.duration)
    b = _safe_segment(c2, 0, dur)
    center = (OUT_W // 2, OUT_H // 2)
    maxrad = np.hypot(center[0], center[1])

    def make_frame(t):
        prog = t/dur
        fa = a.get_frame(t)
        fb = b.get_frame(t)
        # radial vignette that grows and brightens
        Y, X = np.ogrid[:OUT_H, :OUT_W]
        dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = np.clip(1 - (dist / (maxrad * (0.5 + prog))), 0, 1)
        mask = mask ** (1.0 - 0.7 * prog)
        burn = (fa.astype(np.float32) * (1 - prog) + 255 * prog * mask[..., None] * intensity)
        # mix to next quickly after burn peak
        alpha = min(1.0, prog * 2.0)
        out = ((1 - alpha) * burn + alpha * fb.astype(np.float32))
        return np.clip(out, 0, 255).astype(np.uint8)

    return VideoClip(make_frame=make_frame, duration=dur)

# -----------------------
# 10) Slide-Stack (multi-layer stacked vertical slide)
# -----------------------
def slide_stack(c1, c2, dur=0.2, layers=3):
    a = _safe_segment(c1, c1.duration - dur, c1.duration)
    b = _safe_segment(c2, 0, dur)

    def make_frame(t):
        prog = t/dur
        out = np.zeros((OUT_H, OUT_W, 3), dtype=np.float32)
        # bottom-most is next clip sliding up
        for i in range(layers):
            frac = (i + 1) / (layers + 1)
            offset = int(OUT_H * (1 - prog) * (frac))
            src = b.get_frame(t) if i % 2 == 0 else a.get_frame(t)
            y0 = max(0, -offset)
            y1 = min(OUT_H, OUT_H - offset)
            # place the source at offset (clipped)
            if offset >= 0:
                out[offset:OUT_H, :, :] += src[:OUT_H - offset, :, :].astype(np.float32) * (0.6 / (i + 1))
        # normalize and clip
        out = np.clip(out, 0, 255)
        # add slight composite with current b frame to keep clarity
        return np.clip(0.6 * b.get_frame(t).astype(np.float32) + 0.4 * out, 0, 255).astype(np.uint8)

    return VideoClip(make_frame=make_frame, duration=dur)

# -----------------------
# Example usage:
# final = concatenate_videoclips([clip1, crosszoom_transition(clip1, clip2), clip2], method='compose')
# -----------------------


"""
Fictic Transitions Pack — No Mask Version
Best for Reels (9:16)
Requires: moviepy, numpy, opencv-python
"""

import numpy as np, cv2, math
from moviepy.editor import VideoClip

# -----------------------------------------------------
# ✅ Helper — ensure consistent output size (safety)
# -----------------------------------------------------


# =====================================================
# 11) SHOCKWAVE ZOOM — Beat punch/drop
# =====================================================
def shockwave_zoom(c1, c2, dur=0.20, zoom_strength=1.4):
    c1, c2 = ensure_size(c1), ensure_size(c2)
    def fx(t):
        p = t/dur
        frame = c1.get_frame(t) if p < 0.5 else c2.get_frame(t)
        h,w,_ = frame.shape
        scale = 1 + zoom_strength * math.sin(p * math.pi)
        M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
        return cv2.warpAffine(frame, M, (w,h))
    return VideoClip(fx, duration=dur)

# =====================================================
# 12) CHROMATIC WARP — EDM glitch vocals
# =====================================================
def chromatic_warp(c1, c2, dur=0.18, shift_px=10):
    c1, c2 = ensure_size(c1), ensure_size(c2)
    def fx(t):
        p = t/dur
        f = c1.get_frame(t) if p < 0.5 else c2.get_frame(t)
        shift = int(shift_px * (1-p))
        r = np.roll(f[:,:,0], shift, axis=1)
        g = f[:,:,1]
        b = np.roll(f[:,:,2], -shift, axis=0)
        return np.stack([r,g,b], axis=2)
    return VideoClip(fx, duration=dur)

# =====================================================
# 13) STROBE CUT — Fast beats
# =====================================================
def strobe_cut(c1, c2, dur=0.15, flashes=6):
    c1, c2 = ensure_size(c1), ensure_size(c2)
    def fx(t):
        idx = int((t/dur)*flashes)
        return (c2 if idx%2 else c1).get_frame(t)
    return VideoClip(fx, duration=dur)

# =====================================================
# 14) RADIAL WARP TWIST — Build-up swirl
# =====================================================
def radial_warp_twist(c1, c2, dur=0.22, angle=45):
    c1, c2 = ensure_size(c1), ensure_size(c2)
    def fx(t):
        p = t/dur
        frame = c1.get_frame(t) if p < 0.5 else c2.get_frame(t)
        h,w,_ = frame.shape
        ang = angle * math.sin(p * math.pi)
        M = cv2.getRotationMatrix2D((w/2,h/2), ang, 1.0+p*0.4)
        return cv2.warpAffine(frame, M, (w,h))
    return VideoClip(fx, duration=dur)

# =====================================================
# 15) CAMERA SHAKE BUMP — Heavy bass hit
# =====================================================
def camera_shake_bump(c1, c2, dur=0.20, amp=15):
    c1, c2 = ensure_size(c1), ensure_size(c2)
    def fx(t):
        p = t/dur
        dx = int(math.sin(p*20)*amp*(1-p))
        dy = int(math.cos(p*20)*amp*(1-p))
        frame = c2.get_frame(t) if p>=0.5 else c1.get_frame(t)
        return np.roll(np.roll(frame, dx, axis=1), dy, axis=0)
    return VideoClip(fx, duration=dur)

# =====================================================
# 16) PARALLAX SLICE SHIFT — Vocals + mid beats
# =====================================================
def parallax_slice_shift(c1, c2, dur=0.25, slices=8, shift=40):
    c1, c2 = ensure_size(c1), ensure_size(c2)
    def fx(t):
        p = t/dur
        base = c1.get_frame(t).copy()
        h,w,_ = base.shape
        slice_h = h//slices
        for i in range(slices):
            y1, y2 = i*slice_h, (i+1)*slice_h
            offset = int(shift*(p if i%2 else -p))
            base[y1:y2] = np.roll(base[y1:y2], offset, axis=1)
        return base if p<0.5 else c2.get_frame(t)
    return VideoClip(fx, duration=dur)

# =====================================================
# 17) RGB SLICE EXPLODE — High pitch FX
# =====================================================
def rgb_slice_explode(c1, c2, dur=0.18):
    c1, c2 = ensure_size(c1), ensure_size(c2)
    def fx(t):
        p = t/dur
        f = c1.get_frame(t) if p < 0.5 else c2.get_frame(t)
        h,w,_ = f.shape
        mid = w//2 + int((p-0.5)*w*0.4)
        left, right = f[:, :mid], f[:, mid:]
        left[:,:,0] = np.roll(left[:,:,0], -10, axis=1)
        right[:,:,2] = np.roll(right[:,:,2], 10, axis=1)
        return f
    return VideoClip(fx, duration=dur)

# =====================================================
# 18) NOISE TEAR DISTORTION — Aggressive glitch drop
# =====================================================
def noise_tear_distortion(c1, c2, dur=0.16):
    c1, c2 = ensure_size(c1), ensure_size(c2)
    def fx(t):
        p = t/dur
        base = c2.get_frame(t) if p>0.5 else c1.get_frame(t)
        noise = (np.random.rand(*base.shape)*60*(1-p)).astype(np.uint8)
        return cv2.add(base, noise)
    return VideoClip(fx, duration=dur)

# =====================================================
# 19) SWIPE MASK REVEAL — Smooth storytelling
# =====================================================
def swipe_mask_reveal(c1, c2, dur=0.30):
    c1, c2 = ensure_size(c1), ensure_size(c2)
    def fx(t):
        p = t/dur
        f1 = c1.get_frame(t)
        f2 = c2.get_frame(t)
        w = int(f2.shape[1]*p)
        frame = f2.copy()
        frame[:, w:] = f1[:, w:]
        return frame
    return VideoClip(fx, duration=dur)

# =====================================================
# 20) FILM SHUTTER CUT — Hard percussion hits
# =====================================================
def film_shutter_cut(c1, c2, dur=0.14):
    c1, c2 = ensure_size(c1), ensure_size(c2)
    def fx(t):
        p = t/dur
        return (c1.get_frame(t) if p < 0.33 else
                c2.get_frame(t) if p > 0.66 else
                np.zeros_like(c1.get_frame(t)))
    return VideoClip(fx, duration=dur)

# -----------------------------------------------------
# Dictionary to help auto-selection by music analysis
# -----------------------------------------------------

"""
Fictic Object-Aware Transitions Pack 🎬
(Requires YOLOv8-seg + MoviePy + NumPy + CV2)
"""

import numpy as np, cv2, math
from moviepy.editor import VideoClip
from ultralytics import YOLO

# Load seg model once
seg_model = YOLO("yolov8n-seg.pt")

FPS = 30
OUT_W, OUT_H = 1080, 1920  # Final vertical size

# def resize_frame_and_mask(frame, mask, width=OUT_W, height=OUT_H):
#     # Resize frame
#     frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    
#     if mask is not None:
#         # Mask may be single channel or boolean, resize and threshold again as uint8
#         mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
#         mask_resized = (mask_resized > 0.5).astype(np.uint8)
#     else:
#         mask_resized = None
    
#     return frame_resized, mask_resized


def get_mask(frame, conf=0.25):
    """Returns largest object mask or None"""
    r = seg_model.predict(frame, conf=conf, verbose=False)[0]
    m = getattr(r, "masks", None)
    if m is None: return None
    data = m.data.cpu().numpy()
    if len(data) == 0: return None
    return data[np.argmax(data.sum(axis=(1,2)))].astype(np.uint8)

def composite(bg, obj, mask):
    """Overlay obj on bg with mask"""
    mask = mask[...,None]
    return (bg*(1-mask) + obj*mask).astype(np.uint8)

import numpy as np
import cv2
import math
from moviepy.editor import VideoClip

FPS = 30
OUT_W, OUT_H = 1080, 1920  # Final vertical size

def resize_frame_and_mask(frame, mask, width=OUT_W, height=OUT_H):
    frame_resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        mask_resized = (mask_resized > 0.5).astype(np.uint8)
    else:
        mask_resized = None
    return frame_resized, mask_resized


def obj_punch_in(c1, c2, dur=0.35, zoom_max=1.35):
    f2, f1 = c2.get_frame(0), c1.get_frame(0)
    m2 = get_mask(f2)
    if m2 is None: return c2  # fallback

    f2, m2 = resize_frame_and_mask(f2, m2)
    f1 = cv2.resize(f1, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)

    def fn(t):
        p = t/dur
        scale = 1 + zoom_max * math.sin(p*math.pi)
        h,w,_ = f2.shape
        M = cv2.getRotationMatrix2D((w/2,h/2), 0, scale)
        obj = cv2.warpAffine(f2 * m2[..., None], M, (w,h))
        return composite(f1, obj, m2)
    return VideoClip(fn, duration=dur).set_fps(FPS)


def obj_slide_warp(c1, c2, dur=0.45):
    f1, f2 = c1.get_frame(0), c2.get_frame(0)
    m2 = get_mask(f2)
    if m2 is None: return c2

    f2, m2 = resize_frame_and_mask(f2, m2)
    f1 = cv2.resize(f1, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)

    def fn(t):
        p = t/dur
        shift = int(OUT_W*(1-p))
        shifted = np.zeros_like(f2)
        shifted[:, max(0,shift):] = f2[:, :OUT_W-shift]
        return composite(f1, shifted, m2)
    return VideoClip(fn, duration=dur).set_fps(FPS)



def obj_clone_explode(c1, c2, dur=0.35):
    f1,f2 = c1.get_frame(0), c2.get_frame(0)
    m = get_mask(f2)

    f2 = cv2.resize(f2, (OUT_W, OUT_H))
    f1 = cv2.resize(f1, (OUT_W, OUT_H))

    def fn(t):
        p = t/dur
        scale = 1+(p*2)
        obj = cv2.resize(f2, None, fx=scale, fy=scale)
        h,w,_=obj.shape
        cx,cy = OUT_W//2, OUT_H//2
        clones = np.copy(f1)
        for dx in [-80,80]:
            for dy in [-80,80]:
                xs,xe = cx+w//2+dx, cx+w//2+dx+w
                ys,ye = cy+h//2+dy, cy+h//2+dy+h
                try:
                    clones[ys:ye, xs:xe] = obj
                except: pass
        return clones
    return VideoClip(fn,duration=dur).set_fps(FPS)


def obj_rgb_glitch(c1, c2, dur=0.20):
    f1,f2=c1.get_frame(0), c2.get_frame(0)
    m=get_mask(f2)

    f2, m = resize_frame_and_mask(f2, m)
    f1 = cv2.resize(f1, (OUT_W, OUT_H))

    def fn(t):
        p=t/dur
        base=f2
        R=np.roll(base[:,:,0],int(5*(1-p)),axis=1)
        G=base[:,:,1]
        B=np.roll(base[:,:,2],int(-5*(1-p)),axis=0)
        obj=np.stack([R,G,B],axis=2)
        return composite(f1,obj,m)
    return VideoClip(fn,dur).set_fps(FPS)


def obj_twist_in(c1,c2,dur=0.4,max_angle=40):
    f1,f2=c1.get_frame(0),c2.get_frame(0)
    m=get_mask(f2)

    f2, m = resize_frame_and_mask(f2, m)
    f1 = cv2.resize(f1, (OUT_W, OUT_H))

    def fn(t):
        p=t/dur
        ang=max_angle*(1-p)
        h,w,_=f2.shape
        M=cv2.getRotationMatrix2D((w/2,h/2), ang,1+p*0.3)
        obj=cv2.warpAffine(f2,M,(w,h))
        return composite(f1,obj,m)
    return VideoClip(fn,dur).set_fps(FPS)


def obj_motion_blur(c1, c2, dur=0.30, blur_amt=25):
    f1, f2 = c1.get_frame(0), c2.get_frame(0)
    m2 = get_mask(f2)
    if m2 is None:
        return c2  # fallback for no mask

    f2, m2 = resize_frame_and_mask(f2, m2)
    f1 = cv2.resize(f1, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)

    def fn(t):
        p = t/dur
        dx = int(OUT_W*(1-p))
        shifted = np.roll(f2, dx, axis=1)
        # Ensure kernel size is odd and positive for GaussianBlur
        ksize = max(1, int(blur_amt * p))
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(shifted, (ksize, ksize), 0)
        return composite(f1, blurred, m2)
    return VideoClip(fn, duration=dur).set_fps(FPS)


def obj_metal_wipe(c1,c2,dur=0.50):
    f1,f2=c1.get_frame(0),c2.get_frame(0)
    m=get_mask(f2)
    if m is None:
        return c2  # fallback if no mask

    f2, m = resize_frame_and_mask(f2,m)
    f1 = cv2.resize(f1, (OUT_W, OUT_H))

    def fn(t):
        p=t/dur
        x=int(OUT_W*p)
        obj=np.copy(f2)
        bg=np.copy(f1)
        if x > 0:
            obj[:, :x] = 255
            # Ensure region to blur is valid
            if obj[:, :x].size > 0:
                obj[:, :x] = cv2.GaussianBlur(obj[:, :x], (19,19), 0)
        return composite(bg, obj, m)
    return VideoClip(fn, dur).set_fps(FPS)



def obj_warp_pop(c1,c2,dur=0.28):
    f1,f2=c1.get_frame(0),c2.get_frame(0)
    m=get_mask(f2)

    f2, m = resize_frame_and_mask(f2,m)
    f1 = cv2.resize(f1,(OUT_W, OUT_H))

    def fn(t):
        p=t/dur
        scale=1.4*math.sin(p*math.pi)
        h,w,_=f2.shape
        M=cv2.getRotationMatrix2D((w/2,h/2),0,scale)
        obj=cv2.warpAffine(f2,M,(w,h))
        return composite(f1,obj,m)
    return VideoClip(fn,dur).set_fps(FPS)

def obj_blur_reveal(c1,c2,dur=0.45):
    f1,f2=c1.get_frame(0),c2.get_frame(0)
    m=get_mask(f2)

    f2, m = resize_frame_and_mask(f2,m)
    f1 = cv2.resize(f1, (OUT_W, OUT_H))

    def fn(t):
        p=t/dur
        blur=cv2.GaussianBlur(f2,(0,0),30*(1-p))
        return composite(f1,blur,m)
    return VideoClip(fn,dur).set_fps(FPS)


def obj_error_break(c1,c2,dur=0.25):
    f1,f2=c1.get_frame(0),c2.get_frame(0)
    m=get_mask(f2)

    f2, m = resize_frame_and_mask(f2,m)
    f1 = cv2.resize(f1, (OUT_W, OUT_H))

    def fn(t):
        p=t/dur
        noise=(np.random.rand(*f2.shape)*80*p).astype(np.uint8)
        obj=cv2.add(f2,noise)
        return composite(f1,obj,m)
    return VideoClip(fn,dur).set_fps(FPS)


 

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

def safe_forward_reverse_transition(c1, c2, dur=0.22, rev_frac=0.45, seg_max=0.25):
    """
    Play a short reversed tail of c1 (with a subtle punch/zoom) then crossfade into c2.
    Signature matches other transitions: (c1, c2, dur=...).
    """
    c1 = ensure_size(safe_clip(c1))
    c2 = ensure_size(safe_clip(c2))

    seg = min(seg_max, max(0.05, c1.duration))
    a = c1.subclip(max(0, c1.duration - seg), c1.duration).set_duration(seg)
    b = c2.subclip(0, min(c2.duration, dur)).set_duration(dur)

    def frame_fn(t):
        # clamp t
        t = max(0.0, min(dur, t))
        if t < dur * rev_frac:
            # reversed tail with slight punch zoom
            local = t / (dur * rev_frac)  # 0..1
            tt = seg * (1.0 - local)      # play reversed from end -> start
            f = safe_frame_get(a, tt).astype(np.float32)

            # subtle zoom-out -> zoom-in punch
            zoom = 1.0 + 0.08 * (1.0 - local)
            hh, ww = OUT_H, OUT_W
            nw = max(1, int(ww / zoom)); nh = max(1, int(hh / zoom))
            cx, cy = ww // 2, hh // 2
            x1 = max(0, cx - nw // 2); y1 = max(0, cy - nh // 2)
            crop = f[y1:y1+nh, x1:x1+nw]
            try:
                out = cv2.resize(crop, (ww, hh), interpolation=cv2.INTER_LINEAR)
            except Exception:
                out = cv2.resize(f, (ww, hh), interpolation=cv2.INTER_LINEAR)
            return np.clip(out, 0, 255).astype(np.uint8)
        else:
            # crossfade into next clip
            local = (t - dur * rev_frac) / (dur * (1.0 - rev_frac) + 1e-9)
            # take last reversed frame as base (smooth) and blend with next
            f1 = safe_frame_get(a, max(0.0, seg * (1.0 - min(1.0, local)))).astype(np.float32)
            f2 = safe_frame_get(b, min(b.duration - 1e-6, t)).astype(np.float32)
            comp = (1.0 - local) * f1 + local * f2
            return np.clip(comp, 0, 255).astype(np.uint8)

    return VideoClip(lambda tt: frame_fn(tt), duration=dur).set_fps(FPS).resize((OUT_W, OUT_H))



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

all_transitions = [
    impact_zoom_transition,
    whip_transition,
    flash_transition,
    crosszoom_transition,
    whip_pan_blur,
    flash_punch,
    shake_glitch,
    speed_ramp_to_next,
    blur_wipe,
    spin_merge,
    echo_trail,
    lens_burn,
    slide_stack,
    shockwave_zoom,
    chromatic_warp,
    strobe_cut,
    radial_warp_twist,
    camera_shake_bump,
    parallax_slice_shift,
    rgb_slice_explode,
    noise_tear_distortion,
    swipe_mask_reveal,
    film_shutter_cut,
    obj_punch_in,
    obj_slide_warp,
    obj_motion_blur,
    obj_clone_explode,
    obj_rgb_glitch,
    obj_twist_in,
    obj_metal_wipe,
    obj_warp_pop,
    obj_blur_reveal,
    obj_error_break,
    masked_object_transition,
    simple_forward_slide_object_transition,
    choose_transition_with_state
]

# Assuming all transition functions are imported or defined, e.g.,
# impactzoomtransition, whiptransition, flashtransition, crosszoomtransition, ...

# List of all transition functions (add all function names from the script here)


from moviepy.editor import VideoFileClip, concatenate_videoclips
from itertools import cycle, islice

# Assuming all_transitions list and apply_transition function exist

def split_video_into_clips(video_path, clip_duration=3):
    video = VideoFileClip(video_path)
    clips = []
    start = 0
    while start < video.duration:
        end = min(start + clip_duration, video.duration)
        clips.append(video.subclip(start, end))
        start = end
    return clips

def ensure_size(clip, width=1920, height=1080):
    return clip.resize((width, height))

# Example modification at transition start

# also ensure masks or extracted frames are resized similarly

def apply_all_transitions_once_with_clip_reuse(clips, dur=0.4):
    transitions_applied = []
    n_transitions = len(all_transitions)

    # Cycle clips so we can reuse them if clip count < transitions count
    clip_pairs_iter = cycle(zip(clips, clips[1:] + [clips[0]]))  # next of last clip is first clip for reuse

    # Take exactly one instance of each transition
    for i in range(n_transitions):
        
        c1, c2 = next(clip_pairs_iter)
        c1 = ensure_size(c1)
        c2 = ensure_size(c2)
        trans_func = all_transitions[i]
        try:
            clip = trans_func(c1, c2, dur)
        except Exception as e:
            print(f"Error applying {trans_func.__name__}: {e}")
            clip = c1.crossfadeout(dur).set_duration(dur).set_start(0)
        transitions_applied.append(clip)
    
    return transitions_applied

def main():
    input_path = "/Users/uday/Downloads/fictic/input_vid/SSYouTube.online_IRON MAN 4K Scene Pack 🔥  Tony Stark Edit Clips  No Copyright_1080p.mp4"
    clips = split_video_into_clips(input_path, clip_duration=3)
    transitioned_clips = apply_all_transitions_once_with_clip_reuse(clips, dur=0.4)
    final_video = concatenate_videoclips(transitioned_clips, method="compose")
    final_video.write_videofile("output_once_each_transition.mp4", fps=30)

if __name__ == "__main__":
    main()





def choose_transition_advanced(prev, nxt, beat_strength, visual_score):
    """
    Choose transition based on audio beat strength and visual content analysis.
    
    Logic example:
      - High beat & strong visuals: high energy transitions
      - Medium beat or visuals: moderate transitions
      - Low beat & visuals: subtle transitions
    """
    if prev.duration < 1.0 or nxt.duration < 1.0:
        return None  # skip too short clips

    # High energy powerful transitions
    high_energy = [
        whip_transition,
        shockwave_zoom,
        whip_pan_blur,
        speed_ramp_to_next,
        camera_shake_bump,
        shake_glitch,
        strobe_cut,
        lens_burn,
        rgb_slice_explode,
        noise_tear_distortion,
        film_shutter_cut,
    ]
  
    # high_energy = [
    #     whip_transition,
    #     whip_pan_blur,
    #     speed_ramp_to_next,
    #     film_shutter_cut,
    # ]

    # medium_energy = [
    #     impact_zoom_transition,
    #     flash_punch,
    #     blur_wipe,
    #     spin_merge,
    #     parallax_slice_shift,
    #     swipe_mask_reveal,
    #     slide_stack,
    # ]

    # subtle_energy = [
    #     flash_transition,
    #     masked_object_transition,
    #     obj_slide_warp,
    #     obj_punch_in,
    #     obj_motion_blur,
    #     obj_clone_explode,
    #     obj_warp_pop,
    #     obj_blur_reveal,
    #     simple_forward_slide_object_transition,
    # ]

    # # High energy, moving and impactful - but no jitter
    # high_energy = [
    #     whip_transition,
    #     # impact_zoom_transition,
    #     whip_pan_blur,
    #     speed_ramp_to_next,
    #     # camera_shake_bump,
    #     film_shutter_cut,
    # ]

    # # Moderate energy, visually appealing, avoiding jitter
    # medium_energy = [
    #     # impact_zoom_transition,
    #     flash_punch,
    #     radial_warp_twist,
    #     blur_wipe,
    #     spin_merge,
    #     parallax_slice_shift,
    #     swipe_mask_reveal,
    #     slide_stack,
    #     chromatic_warp,
    # ]

    # # Subtle or object-based transitions, avoid rapid twists or glitches
    # subtle_energy = [
    #     # safe_forward_reverse_transition,
    #     flash_transition,
    #     masked_object_transition,
    #     obj_slide_warp,
    #     obj_punch_in,
    #     obj_motion_blur,
    #     obj_clone_explode,
    #     obj_warp_pop,
    #     obj_blur_reveal,
    #     # simple_forward_slide_object_transition,
    # ]

    # High energy (moving and impactful, but no jitter or stutter)
    # High energy (moving and impactful, but no jitter or stutter)
    # high_energy = [
    #     speed_ramp_to_next,
    #     film_shutter_cut,
    # ]

    # # Moderate energy (visually appealing, no jitter)
    # medium_energy = [
    #     flash_punch,
    #     radial_warp_twist,
    #     blur_wipe,
    #     spin_merge,
    #     parallax_slice_shift,
    #     swipe_mask_reveal,
    #     slide_stack,
    #     chromatic_warp,
    # ]

    # # Subtle or object-based transitions (avoid rapid twists or glitches causing stutter)
    # subtle_energy = [
    #     flash_transition,
    #     # masked_object_transition,
    #     obj_slide_warp,
    #     obj_punch_in,
    #     obj_motion_blur,
    #     obj_clone_explode,
    #     obj_warp_pop,
    #     obj_blur_reveal,
    # ]



    # Moderate energy transitions
    medium_energy = [
        impact_zoom_transition,
        flash_punch,
        radial_warp_twist,
        blur_wipe,
        spin_merge,
        echo_trail,
        parallax_slice_shift,
        swipe_mask_reveal,
        slide_stack,
        chromatic_warp,
        slide_stack,
    ]

    # Subtle or object-based transitions
    subtle_energy = [
        flash_transition,
        masked_object_transition,
        obj_slide_warp,
        obj_punch_in,
        obj_motion_blur,
        obj_clone_explode,
        obj_rgb_glitch,
        obj_twist_in,
        obj_metal_wipe,
        obj_warp_pop,
        obj_blur_reveal,
        obj_error_break,
        simple_forward_slide_object_transition,
    ]

    # if beat_strength > 0.7 and visual_score > 20:
    #     trans_func = random.choice(high_energy)
    #     # Call with some example defaults, tune as desired
    #     return trans_func(prev, nxt, dur=0.3)
    # elif beat_strength > 0.4 or visual_score > 15:
    #     trans_func = random.choice(medium_energy)
    #     return trans_func(prev, nxt, dur=0.25)
    # elif visual_score > 10:
    #     trans_func = random.choice(subtle_energy)
    #     return trans_func(prev, nxt, dur=0.2)
    # else:
    #     # fallback minimal flash
    #     return flash_transition(prev, nxt, dur=0.18)

#     if beat_strength > 0.7 and visual_score > 20:
#         trans_func = random.choice(high_energy)
#         return trans_func(prev, nxt, dur=0.4)  # slightly longer for impact
#     elif beat_strength > 0.4 or visual_score > 15:
#         trans_func = random.choice(medium_energy)
#         return trans_func(prev, nxt, dur=0.6)  # medium smooth transition
#     elif visual_score > 10:
#         trans_func = random.choice(subtle_energy)
#         return trans_func(prev, nxt, dur=0.8)  # gentle and natural
#     else:
#         return flash_transition(prev, nxt, dur=0.2)  # quick subtle flash fallback



# def stitching_with_audio_visual_cues(clips, beat_data, visual_scores):
#     seq = [clips[0]]
#     for i in range(1, len(clips)):
#         prev = seq[-1]
#         nxt = clips[i]
#         beat_strength = beat_data[i-1] if i-1 < len(beat_data) else 0
#         visual_score = visual_scores[i-1] if i-1 < len(visual_scores) else 0
#         trans = choose_transition_advanced(prev, nxt, beat_strength, visual_score)
#         if trans:
#             seq.append(trans)
#         seq.append(nxt)
#     clean = [safe_clip(s) for s in seq if s is not None]
#     return concatenate_videoclips(clean, method="compose")




