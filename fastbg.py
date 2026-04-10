from moviepy.editor import VideoClip, vfx, ImageClip, CompositeVideoClip, VideoFileClip
import numpy as np
import cv2
import random
from moviepy.editor import (
    VideoFileClip, AudioFileClip, concatenate_videoclips,
    CompositeVideoClip, TextClip, vfx, CompositeAudioClip
)

# -----------------------------
# SAM segmenter and build_object_rgba should be imported/defined above
# from your previous code
# -----------------------------
def build_object_rgba(rgb_frame, mask_255, bbox):
    x0, y0, x1, y1 = bbox
    crop_rgb = rgb_frame[y0:y1+1, x0:x1+1]
    crop_mask = mask_255[y0:y1+1, x0:x1+1]
    rgba = np.dstack([crop_rgb, crop_mask])
    return np.clip(rgba, 0, 255).astype("uint8")
def apply_random_effects(clip):
    """
    Apply a random effect to a VideoClip (zoom, wave, fade in/out)
    """
    effect = random.choice(["zoom", "wave", "fade"])
    
    if effect == "zoom":
        def zoom_fx(get_frame, t):
            frame = get_frame(t)
            h, w = frame.shape[:2]
            factor = 1 + 0.05*np.sin(2*np.pi*t/clip.duration)
            M = cv2.getRotationMatrix2D((w//2,h//2),0,factor)
            frame = cv2.warpAffine(frame, M, (w,h))
            return frame
        return clip.fl(zoom_fx)
    
    elif effect == "wave":
        def wave_fx(get_frame, t):
            frame = get_frame(t)
            h, w = frame.shape[:2]
            y_shift = (5 * np.sin(2*np.pi*np.linspace(0,1,h) + t*10)).astype(np.int32)
            frame_out = np.zeros_like(frame)
            for i in range(h):
                frame_out[i] = np.roll(frame[i], y_shift[i], axis=0)
            return frame_out
        return clip.fl(wave_fx)
    
    elif effect == "fade":
        def fade_fx(get_frame, t):
            alpha = 0.7 + 0.3*np.sin(2*np.pi*t/clip.duration)
            frame = get_frame(t).astype(np.float32)
            frame = (frame * alpha).astype(np.uint8)
            return frame
        return clip.fl(fade_fx)

# -----------------------------
# Main transition function
# -----------------------------
from moviepy.editor import VideoClip, CompositeVideoClip, ImageClip
import numpy as np
import cv2
import random

def transition_person_static(c1, c2, segmenter, person_effects=True, bg_effects=True, bg_blur_k=25, speed_multiplier=11):
    """
    Transition where person from next clip overlays previous clip
    with dynamic, fast-moving blurred background.

    Zooms can now go in and out safely without showing black edges.
    """

    duration = c1.duration
    fps = c1.fps or 30  # fallback

    # --- 1. Extract best person frame from c2 ---
    frame2 = c2.get_frame(c2.duration / 2)
    mask, bbox = segmenter.segment_first_frame(frame2)
    person_rgba = build_object_rgba(frame2, mask, bbox)
    person_clip = ImageClip(person_rgba, duration=duration)

    # --- 2. Random effect on person ---
    if person_effects:
        effect = random.choice(["zoom", "fade", "wave"])
        if effect == "zoom":
            # Pre-zoom slightly so zoom-out doesn't show edges
            prezoom = 1.05
            person_clip = person_clip.resize(prezoom)
            person_clip = person_clip.resize(lambda t: prezoom - 0.05 * np.sin(2*np.pi*t/duration))
        elif effect == "fade":
            def fade_fx(img):
                H, W, C = img.shape
                alpha = (0.7 + 0.3 * np.sin(np.linspace(0, np.pi*2, H)))[:, None]
                alpha = np.repeat(alpha, W, axis=1)
                alpha = np.stack([alpha]*C, axis=2)
                return (img * alpha).astype(np.uint8)
            person_clip = person_clip.fl_image(fade_fx)
        elif effect == "wave":
            def wave_fx(img):
                h, w, c = img.shape
                shift = (5*np.sin(np.linspace(0, 2*np.pi, h))).astype(np.int32)
                for i in range(h):
                    img[i] = np.roll(img[i], shift[i], axis=0)
                return img
            person_clip = person_clip.fl_image(wave_fx)

    person_clip = person_clip.set_position((bbox[0], bbox[1]))

    # --- 3. Background function ---
    bg_effect = random.choice(["zoom", "fade", "wave"]) if bg_effects else None

    def bg_fx(t):
        t_c2 = (t / duration) * c2.duration * speed_multiplier
        t_c2 = t_c2 % c2.duration
        frame = c2.get_frame(t_c2)

        # Strong blur
        frame_blur = cv2.GaussianBlur(frame, (bg_blur_k, bg_blur_k), 0)

        # Random effect
        if bg_effect == "zoom":
            prezoom = 1.05  # pre-zoom to allow safe zoom-out
            h, w = frame_blur.shape[:2]
            M_pre = cv2.getRotationMatrix2D((w//2, h//2), 0, prezoom)
            frame_blur = cv2.warpAffine(frame_blur, M_pre, (w, h))
            scale = prezoom - 0.05 * np.sin(2*np.pi*t/duration)  # in/out
            M = cv2.getRotationMatrix2D((w//2, h//2), 0, scale)
            frame_blur = cv2.warpAffine(frame_blur, M, (w, h))
        elif bg_effect == "fade":
            alpha = 0.7 + 0.3 * np.sin(2*np.pi*t/duration)
            frame_blur = np.clip(frame_blur * alpha, 0, 255).astype(np.uint8)
        elif bg_effect == "wave":
            h, w, c = frame_blur.shape
            shift = (5*np.sin(2*np.pi*t/duration + np.linspace(0, 2*np.pi, h))).astype(np.int32)
            for i in range(h):
                frame_blur[i] = np.roll(frame_blur[i], shift[i], axis=0)

        return frame_blur

    bg_clip = VideoClip(bg_fx, duration=duration)
    bg_clip.fps = fps

    # --- 4. Composite ---
    transition_clip = CompositeVideoClip([bg_clip, person_clip], size=c2.size)
    transition_clip.fps = fps

    return transition_clip




from moviepy.editor import VideoFileClip, ColorClip
# -----------------------------
# Test independently
# -----------------------------
from moviepy.editor import VideoFileClip, ColorClip
from transit import YoloSamSeg  # replace 'transit' with your module name

if __name__ == "__main__":
    # --- 1. Load the clip where the person comes in ---
    clip_path = "/Users/uday/Downloads/fictic/firstclip/scene_08_187240_189480.mp4"  # replace with your clip path
    clip = VideoFileClip(clip_path)

    # --- 2. Create a dummy "previous clip" just for duration ---
    dummy_clip = ColorClip(size=clip.size, color=(0, 0, 0), duration=clip.duration)
    dummy_clip = dummy_clip.set_fps(clip.fps)  # set fps separately

    # --- 3. Initialize the YOLO+SAM segmenter ---
    segmenter = YoloSamSeg(
        yolo_model="yolov8n.pt",
        sam_checkpoint="sam_b.pth"
    )

    # --- 4. Apply the transition effect ---
    # Only apply if clip is 1 second or longer
    if clip.duration >= 1.0:
        result_clip = transition_person_static(dummy_clip, clip, segmenter, bg_blur_k=85)
    else:
        result_clip = clip  # skip transition for short clips

    # --- 5. Export the result ---
    result_clip.write_videofile(
        "person_static_transition.mp4",
        codec="libx264",
        audio_codec="aac",
        fps=clip.fps
    )


