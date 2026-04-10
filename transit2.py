import random
import numpy as np
import cv2
from moviepy.editor import (
    VideoFileClip, concatenate_videoclips, CompositeVideoClip, ColorClip
)
import moviepy.video.fx.all as vfx

# -----------------------------
#  Custom Slide Effects
# -----------------------------
def slide_out(clip, duration=0.5, direction="left"):
    w, h = clip.size
    def pos(t):
        if direction=="left":
            return (-w * t/duration, 0)
        elif direction=="right":
            return (w * t/duration, 0)
        elif direction=="up":
            return (0, -h*t/duration)
        else:  # down
            return (0, h*t/duration)
    return clip.set_position(pos).set_duration(duration)

def slide_in(clip, duration=0.5, direction="left"):
    w, h = clip.size
    def pos(t):
        if direction=="left":
            return (w - w*t/duration, 0)
        elif direction=="right":
            return (-w + w*t/duration, 0)
        elif direction=="up":
            return (0, h - h*t/duration)
        else:  # down
            return (0, -h + h*t/duration)
    return clip.set_position(pos).set_duration(duration)

# -----------------------------
#  1️⃣ HIGH ENERGY TRANSITIONS
# -----------------------------
def make_whip_pan(clip1, clip2, duration=0.5, direction="left"):
    t1 = slide_out(clip1, duration, direction)
    t2 = slide_in(clip2, duration, direction)
    return concatenate_videoclips([t1.crossfadeout(duration), t2.crossfadein(duration)], method="compose")

def make_zoom_burst(clip1, clip2, duration=0.4, scale=1.2):
    zoom_out = clip1.fx(vfx.resize, lambda t: 1 + (scale-1)*(t/duration))
    zoom_in  = clip2.fx(vfx.resize, lambda t: scale - (scale-1)*(t/duration))
    return concatenate_videoclips([zoom_out.crossfadeout(duration), zoom_in.crossfadein(duration)], method="compose")

def make_glitch(clip1, clip2, duration=0.4, intensity=5):
    def shift_frame(get_frame, t):
        f = get_frame(t)
        f = np.roll(f, int(np.sin(20*t)*intensity), axis=1)
        return f
    g1 = clip1.fl(shift_frame)
    g2 = clip2.fl(shift_frame)
    return concatenate_videoclips([g1.crossfadeout(duration), g2.crossfadein(duration)], method="compose")

def make_spin_blur(clip1, clip2, duration=0.6, direction="cw"):
    angle_func = lambda t: 720*t/duration if direction=="cw" else -720*t/duration
    c1 = clip1.fx(vfx.rotate, lambda t: angle_func(t))
    c2 = clip2.fx(vfx.rotate, lambda t: angle_func(t))
    return concatenate_videoclips([c1.crossfadeout(duration), c2.crossfadein(duration)], method="compose")

def make_particle_blast(clip1, clip2, duration=0.6, color=(255,200,100)):
    overlay = ColorClip(clip1.size, color, duration=duration).fadein(0.1).fadeout(0.5)
    blended = CompositeVideoClip([clip1.crossfadeout(duration), overlay.set_opacity(0.3)])
    return concatenate_videoclips([blended, clip2.crossfadein(duration)], method="compose")

def make_impact_shake(clip1, clip2, duration=0.3, intensity=5):
    def shake(get_frame, t):
        f = get_frame(t)
        dx = int(intensity * np.sin(50*t))
        dy = int(intensity * np.cos(50*t))
        return np.roll(np.roll(f, dx, axis=1), dy, axis=0)
    c1 = clip1.fl(shake)
    return concatenate_videoclips([c1.crossfadeout(duration), clip2.crossfadein(duration)], method="compose")

# -----------------------------
#  2️⃣ MEDIUM ENERGY TRANSITIONS
# -----------------------------
def make_camera_slide(clip1, clip2, duration=0.8, direction="right"):
    t1 = slide_out(clip1, duration, direction)
    t2 = slide_in(clip2, duration, direction)
    return concatenate_videoclips([t1.crossfadeout(duration), t2.crossfadein(duration)], method="compose")

def make_light_leak(clip1, clip2, duration=1.0, color=(255,200,150)):
    leak = ColorClip(clip1.size, color, duration=duration).fadein(0.1).fadeout(0.9)
    blend = CompositeVideoClip([clip1.crossfadeout(duration), leak.set_opacity(0.3)])
    return concatenate_videoclips([blend, clip2.crossfadein(duration)], method="compose")

def make_mask_swipe(clip1, clip2, duration=0.8):
    mask = ColorClip(clip1.size, (255,255,255), duration=duration).fadein(0.1).fadeout(0.7)
    blend = CompositeVideoClip([clip1.crossfadeout(duration), mask.set_opacity(0.2)])
    return concatenate_videoclips([blend, clip2.crossfadein(duration)], method="compose")

def make_parallax_push(clip1, clip2, duration=1.0, depth=0.2):
    t1 = clip1.fx(vfx.resize, lambda t: 1 - depth*t/duration)
    t2 = clip2.fx(vfx.resize, lambda t: 1 + depth*(1-t/duration))
    return concatenate_videoclips([t1.crossfadeout(duration), t2.crossfadein(duration)], method="compose")

def make_match_cut(clip1, clip2, duration=0.3):
    return concatenate_videoclips([clip1.crossfadeout(duration), clip2.crossfadein(duration)], method="compose")

def make_lens_flare(clip1, clip2, duration=0.8, color=(255,255,200)):
    flare = ColorClip(clip1.size, color, duration=duration).fadein(0.1).fadeout(0.6)
    blend = CompositeVideoClip([clip1.crossfadeout(duration), flare.set_opacity(0.3)])
    return concatenate_videoclips([blend, clip2.crossfadein(duration)], method="compose")

def make_zoom_cross(clip1, clip2, duration=0.8):
    c1 = clip1.fx(vfx.resize, lambda t: 1 + 0.3*(t/duration))
    c2 = clip2.fx(vfx.resize, lambda t: 1.3 - 0.3*(t/duration))
    return concatenate_videoclips([c1.crossfadeout(duration), c2.crossfadein(duration)], method="compose")

# -----------------------------
#  3️⃣ SUBTLE ENERGY TRANSITIONS
# -----------------------------
def make_cross_dissolve(clip1, clip2, duration=1.0):
    return concatenate_videoclips([clip1.crossfadeout(duration), clip2.crossfadein(duration)], method="compose")

def make_fade_to_color(clip1, clip2, duration=1.2, color=(0,0,0)):
    fade = ColorClip(clip1.size, color, duration=duration)
    return concatenate_videoclips([clip1.crossfadeout(duration), fade.crossfadeout(duration), clip2.crossfadein(duration)], method="compose")

def make_blur_fade(clip1, clip2, duration=1.0, blur_amount=5):
    c1 = clip1.fx(vfx.blur, blur_amount)
    c2 = clip2.fx(vfx.blur, blur_amount)
    return concatenate_videoclips([c1.crossfadeout(duration), c2.crossfadein(duration)], method="compose")

def make_film_roll(clip1, clip2, duration=0.8):
    return concatenate_videoclips([clip1.crossfadeout(duration), clip2.crossfadein(duration)], method="compose")

def make_zoom_dissolve(clip1, clip2, duration=1.0):
    c1 = clip1.fx(vfx.resize, lambda t: 1.0 + 0.2*(t/duration))
    c2 = clip2.fx(vfx.resize, lambda t: 1.2 - 0.2*(t/duration))
    return concatenate_videoclips([c1.crossfadeout(duration), c2.crossfadein(duration)], method="compose")

def make_wipe_fade(clip1, clip2, duration=1.0, direction="up"):
    t1 = slide_out(clip1, duration, direction)
    t2 = slide_in(clip2, duration, direction)
    return concatenate_videoclips([t1.crossfadeout(duration), t2.crossfadein(duration)], method="compose")

def make_gradient_blend(clip1, clip2, duration=1.0):
    gradient = ColorClip(clip1.size, (255,255,255), duration=duration).fadein(0.1).fadeout(0.8)
    blend = CompositeVideoClip([clip1.crossfadeout(duration), gradient.set_opacity(0.2)])
    return concatenate_videoclips([blend, clip2.crossfadein(duration)], method="compose")

# -----------------------------
#  Transition Map
# -----------------------------
TRANSITION_MAP = {
    "whipPan": make_whip_pan,
    "zoomBurst": make_zoom_burst,
    "glitch": make_glitch,
    "spinBlur": make_spin_blur,
    "particleBlast": make_particle_blast,
    "impactShake": make_impact_shake,

    "cameraSlide": make_camera_slide,
    "lightLeak": make_light_leak,
    "maskSwipe": make_mask_swipe,
    "parallaxPush": make_parallax_push,
    "matchCut": make_match_cut,
    "lensFlare": make_lens_flare,
    "zoomCross": make_zoom_cross,

    "crossDissolve": make_cross_dissolve,
    "fadeToColor": make_fade_to_color,
    "blurFade": make_blur_fade,
    "filmRoll": make_film_roll,
    "zoomDissolve": make_zoom_dissolve,
    "wipeFade": make_wipe_fade,
    "gradientBlend": make_gradient_blend
}

def choose_transition_advanced(clip1, clip2, beat_strength, visual_score, prev_energy=None):
    """
    Chooses transition type based on energy, but clamps duration between 0.2s and 0.3s.
    """
    # Compute energy: mix of visual score and beat strength
    energy = min(1.0, (visual_score / 40.0) + 0.5 * beat_strength)

    # Avoid consecutive high-energy transitions
    if prev_energy and prev_energy > 0.75 and energy > 0.75:
        energy = 0.6  # force medium energy

    # Select transition type based on energy
    if energy > 0.75:
        transition_type = random.choice([
            "whipPan", "zoomBurst", "glitch", "spinBlur", "particleBlast", "impactShake"
        ])
    elif energy > 0.45:
        transition_type = random.choice([
            "cameraSlide", "lightLeak", "maskSwipe", "parallaxPush",
            "matchCut", "lensFlare", "zoomCross"
        ])
    else:
        transition_type = random.choice([
            "crossDissolve", "fadeToColor", "blurFade", "filmRoll",
            "zoomDissolve", "wipeFade", "gradientBlend"
        ])

    # Clamp duration to 0.2 – 0.3 seconds for all transitions
    duration = max(0.2, min(0.3, 0.2 + 0.1 * energy))  

    # Apply transition
    transition_clip = TRANSITION_MAP[transition_type](clip1, clip2, duration)

    return transition_clip, energy



# -----------------------------
# 5️⃣  SAFE COMPOSITION
# -----------------------------

def safe_clip(c):
    try:
        if c is not None and c.duration > 0:
            return c
    except Exception:
        pass
    return None

# -----------------------------
# 6️⃣  FINAL STITCHING FUNCTION
# -----------------------------

def stitching_with_audio_visual_cues(clips, beat_data, visual_scores):
    if not clips or len(clips) < 2:
        raise ValueError("Need at least two clips")

    prev_energy = None
    seq = [clips[0]]
    
    for i in range(1, len(clips)):
        prev_clip = seq[-1]
        next_clip = clips[i]
        beat_strength = beat_data[i-1] if i-1 < len(beat_data) else 0
        visual_score  = visual_scores[i-1] if i-1 < len(visual_scores) else 0

        # Get transition clip and energy
        tclip, energy = choose_transition_advanced(prev_clip, next_clip, beat_strength, visual_score, prev_energy)
        prev_energy = energy  # update for next iteration

        # Use transition clip if valid
        seq[-1] = safe_clip(tclip) or prev_clip
        seq.append(next_clip)

    # Final cleaning to remove None clips
    clean = [safe_clip(s) for s in seq if s is not None]

    # Concatenate safely
    if not clean:
        raise ValueError("No valid clips to concatenate")
    return concatenate_videoclips(clean, method="compose")

