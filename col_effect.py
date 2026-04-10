# import cv2
# import numpy as np
# from moviepy.editor import VideoFileClip
# from transit import YoloSamSeg

# # ------------------------------
# # Input / Output
# # ------------------------------
# input_video = "firstclip/scene_01_94320_96320.mp4"
# output_video = "output_neon_dynamic_dominant.mp4"

# # ------------------------------
# # Load YOLO + SAM segmenter
# # ------------------------------
# segmenter = YoloSamSeg(
#     yolo_model="yolov8n.pt",
#     sam_checkpoint="sam_b.pth"
# )

# # ------------------------------
# # Load video
# # ------------------------------
# clip = VideoFileClip(input_video)

# # ------------------------------
# # Step 1: Pick best frame (largest segmented area)
# # ------------------------------
# n_samples = min(5, int(clip.fps * clip.duration))
# sample_times = np.linspace(0, clip.duration, n_samples)

# best_mask = None
# best_frame = None
# max_area = 0

# print("Finding best frame for segmentation…")
# for t in sample_times:
#     frame = clip.get_frame(t)
#     try:
#         mask, bbox = segmenter.segment_first_frame(frame)
#     except RuntimeError:
#         continue

#     area = mask.sum()
#     if area > max_area:
#         max_area = area
#         best_mask = mask
#         best_frame = frame

# if best_mask is None:
#     raise RuntimeError("No person detected in video!")

# # ------------------------------
# # Step 2: Compute dominant color in segmented area
# # ------------------------------
# seg_pixels = best_frame[best_mask > 0]
# dominant_color = np.round(np.mean(seg_pixels, axis=0)).astype(np.uint8)
# print("Dominant color:", dominant_color)

# # ------------------------------
# # Step 3: Function to dynamically change dominant color
# # ------------------------------
# def dynamic_dominant_color(frame, mask, base_color, t):
#     frame = frame.copy()
#     import math

#     # ---------------------------------------------------
#     # Generate Neon + Contrast color sequence
#     # ---------------------------------------------------
#     # Base dominant color
#     R, G, B = base_color.astype(np.float32)

#     # Neon variations
#     neon1 = np.array([min(255, R * 1.6), min(255, G * 1.6), min(255, B * 1.6)])
#     neon2 = np.array([min(255, R * 2.0), min(255, G * 1.2), min(255, B * 1.4)])

#     # Contrast (inverting base)
#     contrast1 = np.array([255 - R, 255 - G, 255 - B])
#     contrast2 = np.array([255, 255 - G, B])

#     # Rotation: smoothly cycles through 4 neon colors
#     color_cycle = [neon1, neon2, contrast1, contrast2]
#     idx = int((t * 2) % len(color_cycle))
#     next_idx = (idx + 1) % len(color_cycle)

#     # Interpolate between colors for smoothness
#     alpha = (t * 2) % 1.0
#     flicker_color = ((1 - alpha) * color_cycle[idx] + alpha * color_cycle[next_idx]).astype(np.uint8)

#     # ---------------------------------------------------
#     # Expand dominant range (wider effect area)
#     # ---------------------------------------------------
#     RANGE = 17  # bigger range than before

#     seg = frame[mask > 0]

#     r_ok = (seg[:, 0] >= base_color[0] - RANGE) & (seg[:, 0] <= base_color[0] + RANGE)
#     g_ok = (seg[:, 1] >= base_color[1] - RANGE) & (seg[:, 1] <= base_color[1] + RANGE)
#     b_ok = (seg[:, 2] >= base_color[2] - RANGE) & (seg[:, 2] <= base_color[2] + RANGE)

#     strict_mask = r_ok & g_ok & b_ok

#     seg_idx = np.where(mask > 0)
#     target_idx = (seg_idx[0][strict_mask], seg_idx[1][strict_mask])

#     # Apply smooth blending
#     intensity = 0.7 + 0.3 * math.sin(2 * math.pi * 2 * t)
#     final_color = np.clip(flicker_color * intensity, 0, 255)

#     frame[target_idx] = final_color

#     return frame



# def process_frame(get_frame, t):
#     return dynamic_dominant_color(best_frame, best_mask, dominant_color, t)


# # ------------------------------
# # Step 4: Render final video
# # ------------------------------
# print("Processing video…")
# final_clip = clip.fl(process_frame, apply_to=["video"])
# final_clip.write_videofile(output_video, codec="libx264", audio_codec="aac", fps=clip.fps)

# print("✅ DONE — Saved:", output_video)

"""
yoloworld_sam_neon.py

Full pipeline:
- YOLO-World (tiny) or Ultralytics YOLO -> detect clothing/accessories
- SAM2 (or SAM) -> segment each detected box
- Pick best frame (largest total mask area) OR use first frame (configurable)
- Compute dominant color per mask (mean color inside mask)
- For each frame (or for static best frame repeated), flicker ONLY pixels
  inside mask that are within +/- RANGE of dominant color (per-channel).
- Flicker uses a smooth interpolation through neon/contrast colors.

Requirements (suggested):
- moviepy
- ultralytics (or YOLOWorld)
- sam2 or segment_anything
- opencv-python
- torch (appropriate version)
"""

import os
import math
import time
import argparse
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, VideoClip
from typing import List, Tuple
import torch
import torch

# ---------------------
# YOLO-World + SAM2 (with optional sam_b fallback) only
# ---------------------

import os, math, time, numpy as np, cv2
from moviepy.editor import VideoFileClip, VideoClip
from typing import List, Tuple
import torch



try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_FALLBACK_AVAILABLE = True
except Exception:
    SAM_FALLBACK_AVAILABLE = False

# ---------------------
# CONFIG
# ---------------------
INPUT_VIDEO = "firstclip/scene_01_94320_96320.mp4"
OUTPUT_VIDEO = "output_neon_yoloworld.mp4"
YOLOWORLD = "/Users/uday/Downloads/yolov8s-worldv2.pt"
YOLOWORLD_MODEL_PATH = "/Users/uday/Downloads/yolov8s-worldv2.pt"
SAM2_CHECKPOINT = "/Users/uday/Downloads/sam2.1_hiera_tiny.pt"
SAM_FALLBACK_CHECKPOINT = "sam_b.pth"  # fallback only if SAM2 fails

USE_STATIC_BEST_FRAME = True
SAMPLE_FRAMES = 10
DOM_COLOR_RANGE = 18
NEON_VARIATION = 1.3
FREQUENCY_HZ = 1.8
FPS = None


# ---------------------
# Detector + Segmenter class
# ---------------------
import numpy as np
import torch
from typing import List, Tuple

# ---------------------
# Config paths (update as needed)
# ---------------------
YOLOWORLD_MODEL_PATH = "/Users/uday/Downloads/yolov8s-worldv2.pt"   # YOLO-World model
SAM2_CHECKPOINT = "/Users/uday/Downloads/sam2.1_hiera_tiny.pt"      # SAM2
SAM_FALLBACK_CHECKPOINT = "sam_b.pth"                                # fallback vit_b

CLOTHES = [
    "t-shirt","shirt","hoodie","jacket","coat","sweater","dress","skirt","jeans","pants","trousers",
    "shorts","saree","lehenga","top","sling","vest","shoes","boots","sandals","slippers",
    "scarf","tie","belt","gloves","socks","hat","cap","beanie","sunglasses",
    "watch","bracelet","necklace","earrings","handbag","backpack","bag"
]
import numpy as np
from typing import Tuple, List

class DetectorAndSegmenter:
    def __init__(self):
        # YOLOv8 instead of YOLO-World
        self.yw = None
        # SAM
        self.sam2_predictor = None
        self.sam_predictor = None
        self.sam_type = None

    # -----------------
    # Load YOLOv8 (standard)
    # -----------------
    def load_yolo(self):
        try:
            from ultralytics import YOLO
            import torch
            import ultralytics.nn.tasks  # Make sure YOLO-World classes are available

            # Add the WorldModel to safe globals
            torch.serialization.add_safe_globals([ultralytics.nn.tasks.WorldModel])

            # Load the YOLO-World model
            self.yw = YOLO(YOLOWORLD_MODEL_PATH, task="world")
            print("✅ YOLO-World loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO-World model: {e}") from e

    # -----------------
    # Load SAM2 or fallback
    # -----------------
    def load_sam(self):
        SAM2_AVAILABLE = False
        SAM_FALLBACK_AVAILABLE = False

        # -----------------
        # SAM2 (optional, commented if unavailable)
        # -----------------
        # try:
        #     from sam2.build_sam import build_sam2
        #     from sam2.sam2_image_predictor import SAM2ImagePredictor
        #     sam2_model = build_sam2(SAM2_CHECKPOINT)
        #     self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        #     self.sam_type = "sam2"
        #     SAM2_AVAILABLE = True
        #     print("✅ Using SAM2 predictor.")
        #     return
        # except Exception as e:
        #     print("⚠️ SAM2 not available:", e)

        # Fallback SAM (segment_anything vit_b)
        try:
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry["vit_b"](checkpoint=SAM_FALLBACK_CHECKPOINT)
            sam.to("cpu")
            self.sam_predictor = SamPredictor(sam)
            self.sam_type = "sam"
            SAM_FALLBACK_AVAILABLE = True
            print("✅ Using fallback SamPredictor (vit_b).")
            return
        except Exception as e:
            print("⚠️ Fallback SAM vit_b not available:", e)

        raise RuntimeError("No SAM predictor available. Install SAM2 or segment_anything.")

    # -----------------
    # Detect objects in a frame
    # -----------------
    def detect(self, frame: np.ndarray):
        if self.yw is not None:
            res = self.yw.predict(frame, classes=CLOTHES, conf=0.25)[0]
            return res
        raise RuntimeError("YOLO model not loaded.")

    # -----------------
    # Segment a detected bounding box using SAM
    # -----------------
    def segment_box(self, frame: np.ndarray, xyxy: Tuple[int,int,int,int]) -> List[np.ndarray]:
        x1, y1, x2, y2 = map(int, xyxy)

        if self.sam_type == "sam2" and self.sam2_predictor is not None:
            self.sam2_predictor.set_image(frame)
            masks, _, _ = self.sam2_predictor.predict(
                box=np.array([x1, y1, x2, y2]),
                multimask_output=True
            )
            return [(m>0.5).astype(np.uint8) for m in masks]

        elif self.sam_type == "sam" and self.sam_predictor is not None:
            self.sam_predictor.set_image(frame)
            box_np = np.array([x1, y1, x2, y2], dtype=np.float32)
            masks, _, _ = self.sam_predictor.predict(box=box_np[None,:], multimask_output=True)
            return [(m>0.5).astype(np.uint8) for m in masks]

        else:
            raise RuntimeError("SAM predictor not initialized.")



# ---------------------
# Initialize models
# ---------------------
detseg = DetectorAndSegmenter()
detseg.load_yolo()
detseg.load_sam()

# ---------------------
# Utilities
# ---------------------
def masks_union(masks: List[np.ndarray], h:int, w:int) -> np.ndarray:
    total = np.zeros((h,w), dtype=np.uint8)
    for m in masks:
        if m.shape != total.shape:
            m = cv2.resize(m.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)
        total = np.logical_or(total, m).astype(np.uint8)
    return total

def mask_area(mask: np.ndarray) -> int:
    return int(np.sum(mask > 0))

# compute dominant color (mean) in RGB
def dominant_color_of_mask(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = frame_rgb[mask > 0]
    if len(pixels) == 0:
        return np.array([200,200,200], dtype=np.uint8)
    return np.round(np.mean(pixels, axis=0)).astype(np.uint8)


# ---------------------
# Step A: choose best frame (static) by sampling
# ---------------------
# ---------------------
# Step A: choose best frame (static) by sampling
# ---------------------
print("Opening input video:", INPUT_VIDEO)
clip = VideoFileClip(INPUT_VIDEO)
FPS = FPS or clip.fps
duration = clip.duration
print(f"Video duration {duration:.2f}s, fps={FPS}, sampling {SAMPLE_FRAMES} frames (static={USE_STATIC_BEST_FRAME})")

best_frame = None
best_masks = []      # list of boolean masks (H x W)
best_mask_union = None
best_area = 0

# Ensure CLOTHES is a list of integers (class IDs YOLO-World uses for clothing)
CLOTHES = [0, 1, 2, 3, 4]  # <-- adjust according to your YOLO-World model

if USE_STATIC_BEST_FRAME:
    sample_n = max(1, int(SAMPLE_FRAMES))
    times = np.linspace(0, duration, sample_n)
    for t in times:
        frame = clip.get_frame(t)  # RGB float32 usually
        # Convert to uint8
        if frame.dtype != np.uint8:
            frame_uint8 = np.clip(frame * 255, 0, 255).astype(np.uint8)
        else:
            frame_uint8 = frame

        try:
            res = detseg.detect(frame_uint8)
        except Exception as e:
            print(f"Detection failed at time {t:.3f}s ->", e)
            continue

        masks_here = []
        try:
            # If YOLO returns masks directly
            if hasattr(res, "boxes") and hasattr(res, "masks") and res.masks is not None:
                for m in res.masks.data:
                    m_np = m.cpu().numpy().astype(np.uint8)
                    masks_here.append(m_np)
            else:
                # Iterate boxes and segment with SAM
                boxes = []
                if hasattr(res, "boxes"):
                    for b in res.boxes:
                        xyxy = b.xyxy[0].cpu().numpy().astype(int)
                        boxes.append(xyxy)
                H, W = frame_uint8.shape[:2]
                for xyxy in boxes:
                    ms = detseg.segment_box(frame_uint8, xyxy)
                    for m in ms:
                        if m.shape != (H,W):
                            m = cv2.resize(m.astype(np.uint8), (W,H), interpolation=cv2.INTER_NEAREST)
                        masks_here.append(m.astype(np.uint8))
        except Exception as e:
            print("Failed parsing detection results:", e)
            masks_here = []

        if len(masks_here) == 0:
            continue

        H, W = frame_uint8.shape[:2]
        union = masks_union(masks_here, H, W)
        a = mask_area(union)
        print(f"sample t={t:.2f}s found masks total area={a}")
        if a > best_area:
            best_area = a
            best_frame = frame_uint8.copy()
            best_masks = masks_here.copy()
            best_mask_union = union.copy()

    if best_frame is None:
        raise RuntimeError("No detection found across samples — try SAMPLE_FRAMES larger or check models.")
    print("Best frame selected with mask area:", best_area)

else:
    # Non-static: detect on first frame
    frame = clip.get_frame(0)
    if frame.dtype != np.uint8:
        frame_uint8 = np.clip(frame * 255, 0, 255).astype(np.uint8)
    else:
        frame_uint8 = frame

    best_frame = frame_uint8
    res = detseg.detect(best_frame)
    masks_here = []
    H, W = best_frame.shape[:2]
    try:
        boxes = []
        if hasattr(res, "boxes"):
            for b in res.boxes:
                xyxy = b.xyxy[0].cpu().numpy().astype(int)
                boxes.append(xyxy)
        for xyxy in boxes:
            ms = detseg.segment_box(best_frame, xyxy)
            for m in ms:
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W,H), interpolation=cv2.INTER_NEAREST)
                masks_here.append(m.astype(np.uint8))
    except Exception as e:
        print("Fallback parsing error:", e)

    best_masks = masks_here
    best_mask_union = masks_union(best_masks, H, W)
    best_area = mask_area(best_mask_union)
    print("First-frame masks total area:", best_area)


# ---------------------
# Step B: compute dominant color per mask
# ---------------------
mask_objects = []

H, W = best_frame.shape[:2]

for i, m in enumerate(best_masks):
    m_bool = (m > 0).astype(np.uint8)
    dom = dominant_color_of_mask(best_frame, m_bool)

    # approx mask: pixels within DOM_COLOR_RANGE per-channel
    seg_pixels = best_frame.copy()
    seg_pixels[m_bool == 0] = 0

    r_ok = (seg_pixels[:,:,0] >= (dom[0]-DOM_COLOR_RANGE)) & (seg_pixels[:,:,0] <= (dom[0]+DOM_COLOR_RANGE))
    g_ok = (seg_pixels[:,:,1] >= (dom[1]-DOM_COLOR_RANGE)) & (seg_pixels[:,:,1] <= (dom[1]+DOM_COLOR_RANGE))
    b_ok = (seg_pixels[:,:,2] >= (dom[2]-DOM_COLOR_RANGE)) & (seg_pixels[:,:,2] <= (dom[2]+DOM_COLOR_RANGE))

    approx = (r_ok & g_ok & b_ok & (m_bool>0)).astype(np.uint8)

    if approx.sum() > 0:
        approx = cv2.dilate(approx, np.ones((3,3), np.uint8), iterations=1)

    mask_objects.append({
        "mask": m_bool,
        "dominant_color": dom,
        "approx_mask": approx
    })
    print(f"Obj {i}: dom_color={dom.tolist()} pixels_total={m_bool.sum()} approx_pixels={approx.sum()}")

# ---------------------
# Color cycle generator (returns an RGB uint8 color for time t)
# ---------------------
def color_cycle_for_base(base_color: np.ndarray, t: float):
    """
    Create a smooth variation/sequence of neon + contrast colors for a given base_color.
    Interpolates across a small palette. Returns uint8 RGB.
    """
    R,G,B = base_color.astype(np.float32)

    # build candidate colors (neon variants & contrasting)
    neon1 = np.clip(np.array([R*NEON_VARIATION, G*1.1, B*1.2]), 0, 255)
    neon2 = np.clip(np.array([R*1.2, G*NEON_VARIATION, B*1.4]), 0, 255)
    contrast1 = np.clip(np.array([255-R, 255-G, 255-B]), 0, 255)
    contrast2 = np.clip(np.array([min(255, R*1.8), max(0, G*0.6), min(255, B*1.5)]), 0, 255)

    palette = np.stack([neon1, neon2, contrast1, contrast2], axis=0)  # (4,3)
    # index & interpolation param
    cycle_speed = FREQUENCY_HZ  # cycles per second across palette
    total = len(palette)
    pos = (t * cycle_speed) % total
    idx = int(pos)
    next_idx = (idx + 1) % total
    frac = pos - idx
    col = ((1-frac)*palette[idx] + frac*palette[next_idx]).astype(np.uint8)
    return col


# ---------------------
# Frame generator for final video (static best_frame repeated)
# ---------------------
def make_frame(t: float):
    """
    For time t (seconds), return RGB frame (uint8).
    We use best_frame as static background; apply neon changes on approx_mask pixels only.
    """
    bg = best_frame.copy()  # RGB uint8
    out = bg.copy()

    # for each object mask, compute replacement color at time t and apply to approx_mask pixels only
    for obj in mask_objects:
        approx = obj["approx_mask"]  # bool 0/1
        if approx.sum() == 0:
            continue
        base = obj["dominant_color"]
        # compute flicker color for this object at time t
        new_col = color_cycle_for_base(base, t)  # uint8 RGB
        # optional intensity modulation
        intensity = 0.8 + 0.4 * math.sin(2*math.pi*FREQUENCY_HZ*t + (hash(str(base.tobytes())) % 10)/10.0)
        final_col = np.clip(new_col.astype(np.float32) * intensity, 0, 255).astype(np.uint8)

        # apply only to approx pixels
        coords = np.where(approx > 0)
        if len(coords[0]) == 0:
            continue
        out[coords] = final_col

    return out


# ---------------------
# Render final video using VideoClip(make_frame)
# ---------------------
print("Rendering final video to:", OUTPUT_VIDEO)
video_clip = VideoClip(make_frame=make_frame, duration=duration)

# ensure fps
if FPS is None:
    FPS = clip.fps
print("Writing file with fps =", FPS)
video_clip.write_videofile(OUTPUT_VIDEO, codec="libx264", audio=False, fps=FPS)

print("Done. Saved:", OUTPUT_VIDEO)
