# """
# obj_transitions_yolo_plus.py
# Object-based transitions using YOLOv8-seg (Ultralytics) for segmentation (Mac M1/M2/M3 friendly).
# Features added:
#  - Selects the detection with HIGHEST confidence (falls back to largest area).
#  - Smooth audio crossfade between composed tail and Clip B (no ducking).
#  - Batch preview: outputs one file per transition style for quick comparison.

# Install:
# pip install torch torchvision torchaudio ultralytics moviepy opencv-python-headless pillow numpy tqdm

# Usage:
#     from obj_transitions_yolo_plus import make_object_transition, make_batch_preview
#     make_object_transition("clipA.mp4", "clipB.mp4", transition_type="random", out_path="result.mp4")
#     make_batch_preview("clipA.mp4", "clipB.mp4", out_folder="previews")
# """

# import os
# import random
# import tempfile
# from typing import Optional, List

# import cv2
# import numpy as np
# from PIL import Image
# from moviepy.editor import (
#     VideoFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips, vfx, AudioFileClip
# )

# # ultralytics (YOLOv8)
# from ultralytics import YOLO

# # -----------------------
# # Device helper (MPS on Apple Silicon)
# # -----------------------
# def pick_device():
#     import torch
#     if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#         return "mps"
#     return "cpu"

# # -----------------------
# # Utilities
# # -----------------------
# def read_frame_from_video(video_path, frame_idx=0):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open {video_path}")
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         raise RuntimeError("Failed to read frame from " + video_path)
#     # convert BGR -> RGB
#     return frame[:, :, ::-1].copy()

# def rgba_to_imageclip(rgba_arr, duration=1.2, fps=30):
#     tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
#     Image.fromarray(rgba_arr).save(tmp.name, "PNG")
#     clip = ImageClip(tmp.name).set_duration(duration).set_fps(fps)
#     return clip

# def ensure_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path, exist_ok=True)

# # -----------------------
# # YOLOv8 segmentation extraction (choose highest confidence)
# # -----------------------
# class YoloSegmenter:
#     def __init__(self, model_name="yolov8n-seg.pt", device: Optional[str]=None, conf:float=0.25):
#         self.device = device or pick_device()
#         # initialize model (weights auto-download if not present)
#         self.model = YOLO(model_name)
#         self.conf = conf

#     def segment_frame(self, rgb_frame: np.ndarray, retina_masks: bool = True):
#         """
#         Run YOLOv8 segmentation on an RGB numpy frame (H,W,3 uint8).
#         Returns: RGBA numpy uint8 cropped to chosen mask bbox.
#         Selection heuristic:
#             1) Prefer detection with highest confidence (boxes.conf)
#             2) If confidences not accessible, fall back to largest mask area
#         """
#         results = self.model.predict(source=rgb_frame,
#                                      device=self.device,
#                                      imgsz=640,
#                                      conf=self.conf,
#                                      retina_masks=retina_masks,
#                                      verbose=False)
#         if len(results) == 0:
#             raise RuntimeError("YOLO returned no results")

#         r = results[0]  # Results object
#         masks = getattr(r, "masks", None)
#         boxes = getattr(r, "boxes", None)

#         if masks is None:
#             raise RuntimeError("No masks found in YOLO result. Use a '-seg' model like yolov8n-seg.pt")

#         # Extract mask tensor/array
#         try:
#             mask_data = masks.data
#             if hasattr(mask_data, "cpu"):
#                 mask_arr = mask_data.cpu().numpy()
#             else:
#                 mask_arr = np.array(mask_data)
#         except Exception as e:
#             raise RuntimeError(f"Failed to extract mask array from YOLO results: {e}")

#         if mask_arr.ndim == 2:
#             mask_arr = mask_arr[np.newaxis, ...]

#         # Try to get confidences associated with each detection
#         confs = None
#         try:
#             # ultralytics boxes.conf may be a tensor accessible as r.boxes.conf
#             if boxes is not None and hasattr(boxes, "conf"):
#                 bconf = boxes.conf
#                 if hasattr(bconf, "cpu"):
#                     confs = bconf.cpu().numpy()
#                 else:
#                     confs = np.array(bconf)
#             else:
#                 # other fallback: r.boxes.data[...,4] sometimes
#                 bdata = getattr(boxes, "data", None)
#                 if bdata is not None:
#                     if hasattr(bdata, "cpu"):
#                         bdata = bdata.cpu().numpy()
#                     # bdata columns often [x1,y1,x2,y2,conf,class]
#                     if bdata.shape[1] >= 5:
#                         confs = bdata[:, 4]
#         except Exception:
#             confs = None

#         # choose index: highest confidence if available else largest area
#         idx = None
#         if confs is not None and len(confs) == mask_arr.shape[0]:
#             idx = int(np.argmax(confs))
#         else:
#             # fallback to largest area
#             mask_bool = mask_arr > 0.5
#             areas = mask_bool.reshape(mask_bool.shape[0], -1).sum(axis=1)
#             if areas.sum() == 0:
#                 raise RuntimeError("No positive pixels in masks — segmentation failed.")
#             idx = int(np.argmax(areas))

#         chosen_mask = (mask_arr[idx] > 0.5).astype(np.uint8)  # H x W

#         # bbox from chosen mask
#         ys, xs = np.where(chosen_mask)
#         if len(xs) == 0:
#             raise RuntimeError("Chosen mask empty")
#         x0, x1 = xs.min(), xs.max()
#         y0, y1 = ys.min(), ys.max()

#         # crop original rgb frame
#         crop_rgb = rgb_frame[y0:y1+1, x0:x1+1].copy()
#         crop_mask = chosen_mask[y0:y1+1, x0:x1+1].copy()
#         alpha = (crop_mask * 255).astype(np.uint8)
#         pil = Image.fromarray(crop_rgb)
#         pil.putalpha(Image.fromarray(alpha))
#         rgba = np.array(pil)  # HxWx4 uint8
#         return rgba

# # -----------------------
# # Transition functions
# # -----------------------
# def slide_in(object_clip, video_w, video_h, duration=1.2, direction="left"):
#     ow, oh = object_clip.size
#     start_pos = {
#         "left": (-ow, (video_h - oh)//2),
#         "right": (video_w, (video_h - oh)//2),
#         "top": ((video_w - ow)//2, -oh),
#         "bottom": ((video_w - ow)//2, video_h),
#         "top-left": (-ow, -oh),
#         "bottom-right": (video_w, video_h)
#     }.get(direction, (-ow, (video_h - oh)//2))
#     end_pos = ((video_w - ow)//2, (video_h - oh)//2)
#     def make_pos(t):
#         progress = min(1, max(0, t / duration))
#         sx, sy = start_pos
#         ex, ey = end_pos
#         x = sx + (ex - sx) * (1 - (1-progress)**2)
#         y = sy + (ey - sy) * (1 - (1-progress)**2)
#         return (int(x), int(y))
#     return object_clip.set_position(make_pos).set_duration(duration)

# def zoom_in(object_clip, video_w, video_h, duration=1.2, start_scale=0.2):
#     ow, oh = object_clip.size
#     center = ((video_w - ow)//2, (video_h - oh)//2)
#     def size_at(t):
#         progress = min(1, max(0, t / duration))
#         scale = start_scale + (1 - start_scale) * (1 - (1-progress)**2)
#         return scale
#     return (object_clip.fx(vfx.resize, lambda t: size_at(t)).set_position(center).set_duration(duration))

# def pop_in(object_clip, video_w, video_h, duration=1.2):
#     ow, oh = object_clip.size
#     center = ((video_w - ow)//2, (video_h - oh)//2)
#     def size_at(t):
#         p = min(1, max(0, t / duration))
#         if p < 0.7:
#             s = 0.2 + 1.4 * (p/0.7)
#         else:
#             s = 1.0 - 0.2 * ((p-0.7)/0.3)
#         return max(0.01, s)
#     return (object_clip.fx(vfx.resize, lambda t: size_at(t)).set_position(center).set_duration(duration))

# def spin_in(object_clip, video_w, video_h, duration=1.2, spins=1):
#     ow, oh = object_clip.size
#     center = ((video_w - ow)//2, (video_h - oh)//2)
#     return (object_clip.fx(vfx.rotate, lambda t: 360*spins*(1 - (1 - min(1, t/duration))**2)).set_position(center).set_duration(duration))

# def fade_in(object_clip, video_w, video_h, duration=1.2):
#     ow, oh = object_clip.size
#     center = ((video_w - ow)//2, (video_h - oh)//2)
#     return object_clip.set_position(center).fadein(duration).set_duration(duration)

# def burst_in(object_clip, video_w, video_h, duration=1.2):
#     ow, oh = object_clip.size
#     center = ((video_w - ow)//2, (video_h - oh)//2)
#     def size_at(t):
#         p = min(1, max(0, t/duration))
#         if p < 0.3:
#             return 0.2 + 3 * p
#         else:
#             return max(1.0, 1.0 + 0.15*(1-p))
#     return (object_clip.fx(vfx.resize, lambda t: size_at(t)).set_position(center).set_duration(duration))

# # add or tweak transitions as desired
# TRANSITION_MAP = {
#     "slide": slide_in,
#     "zoom": zoom_in,
#     "pop": pop_in,
#     "spin": spin_in,
#     "fade": fade_in,
#     "burst": burst_in
# }

# # -----------------------
# # Core function (single run), now with audio crossfade and highest-confidence selection
# # -----------------------
# def make_object_transition(clipA_path: str,
#                            clipB_path: str,
#                            transition_type: str = "random",
#                            duration: float = 1.2,
#                            out_path: str = "final_transition.mp4",
#                            yolo_model: str = "yolov8n-seg.pt",
#                            device: Optional[str] = None,
#                            fps: int = 30,
#                            conf: float = 0.25,
#                            crossfade: float = 0.06,
#                            verbose: bool = True):
#     """
#     Create final transition video using YOLOv8-seg for segmentation (picks highest-confidence detection).
#     - crossfade: seconds of audio+video crossfade between composed tail and clipB (smooth, no ducking)
#     - If transition_type == "random", picks one at random.
#     """
#     if verbose: print("Loading clips...")
#     clipA = VideoFileClip(clipA_path)
#     clipB = VideoFileClip(clipB_path)
#     video_w, video_h = clipA.w, clipA.h

#     if verbose: print("Extracting first frame from clipB...")
#     frame = read_frame_from_video(clipB_path, frame_idx=0)  # RGB

#     if verbose: print("Running YOLOv8-seg for segmentation (device detection)...")
#     seg_device = device or pick_device()
#     if verbose: print("Segmentation device:", seg_device)
#     seg = YoloSegmenter(model_name=yolo_model, device=seg_device, conf=conf)
#     rgba = seg.segment_frame(frame, retina_masks=True)  # HxWx4

#     if verbose: print("Creating object image clip...")
#     obj_clip = rgba_to_imageclip(rgba, duration=duration, fps=fps)

#     # pick transition
#     if transition_type == "random":
#         transition_type = random.choice(list(TRANSITION_MAP.keys()))
#         if verbose: print("Randomly chosen transition:", transition_type)
#     trans_fn = TRANSITION_MAP.get(transition_type)
#     if trans_fn is None:
#         raise ValueError("Unknown transition_type: " + str(transition_type))

#     animated_obj = trans_fn(obj_clip, video_w, video_h, duration=duration)

#     # Compose onto tail of clipA
#     if verbose: print("Compositing object onto tail of clipA...")
#     pre_segment_start = max(0, clipA.duration - duration)
#     head = clipA.subclip(0, pre_segment_start) if pre_segment_start > 0.01 else None
#     tail = clipA.subclip(pre_segment_start, clipA.duration)
#     animated_obj = animated_obj.set_start(0)
#     composed_tail = CompositeVideoClip([tail, animated_obj]).set_duration(tail.duration)

#     # Audio crossfade: we will crossfade composed_tail audio into clipB audio when concatenating
#     # concatenate_videoclips with padding=-crossfade achieves video crossfade + audio overlap smoothing
#     parts = []
#     if head: parts.append(head)
#     parts.append(composed_tail)

#     try:
#         final = concatenate_videoclips(parts + [clipB], method="compose", padding=-crossfade)
#     except Exception:
#         # fallback simple concat (no crossfade)
#         final = concatenate_videoclips(parts + [clipB], method="compose")

#     if verbose: print(f"Writing output (crossfade={crossfade}s) to {out_path} ...")
#     final.write_videofile(out_path, fps=fps, codec="libx264", threads=4, preset="medium")
#     if verbose: print("Done.")

# # -----------------------
# # Batch preview function (outputs one file per transition)
# # -----------------------
# def make_batch_preview(clipA_path: str,
#                        clipB_path: str,
#                        out_folder: str = "previews",
#                        transitions: Optional[List[str]] = None,
#                        duration: float = 1.2,
#                        yolo_model: str = "yolov8n-seg.pt",
#                        device: Optional[str] = None,
#                        fps: int = 30,
#                        conf: float = 0.25,
#                        crossfade: float = 0.06,
#                        verbose: bool = True):
#     """
#     Generate preview videos for every transition style in TRANSITION_MAP (or given list).
#     Files are written to out_folder with names: <basename>_<transition>.mp4
#     """
#     ensure_dir(out_folder)
#     base = os.path.splitext(os.path.basename(clipB_path))[0]
#     transitions = transitions or list(TRANSITION_MAP.keys())

#     # Reuse clips and segmentation once for speed
#     if verbose: print("Loading clips (batch mode)...")
#     clipA = VideoFileClip(clipA_path)
#     clipB = VideoFileClip(clipB_path)
#     video_w, video_h = clipA.w, clipA.h

#     if verbose: print("Extracting and segmenting the first frame of clipB (once)...")
#     frame = read_frame_from_video(clipB_path, frame_idx=0)
#     seg_device = device or pick_device()
#     seg = YoloSegmenter(model_name=yolo_model, device=seg_device, conf=conf)
#     rgba = seg.segment_frame(frame, retina_masks=True)
#     obj_clip_base = rgba_to_imageclip(rgba, duration=duration, fps=fps)

#     pre_segment_start = max(0, clipA.duration - duration)
#     head = clipA.subclip(0, pre_segment_start) if pre_segment_start > 0.01 else None
#     tail = clipA.subclip(pre_segment_start, clipA.duration)

#     for tname in transitions:
#         if tname not in TRANSITION_MAP:
#             if verbose: print("Skipping unknown transition:", tname)
#             continue
#         if verbose: print("Building preview for:", tname)
#         trans_fn = TRANSITION_MAP[tname]
#         # create a fresh object clip instance for this transition
#         obj_clip = rgba_to_imageclip(rgba, duration=duration, fps=fps)
#         animated_obj = trans_fn(obj_clip, video_w, video_h, duration=duration)
#         composed_tail = CompositeVideoClip([tail, animated_obj.set_start(0)]).set_duration(tail.duration)
#         parts = []
#         if head: parts.append(head)
#         parts.append(composed_tail)
#         try:
#             final = concatenate_videoclips(parts + [clipB], method="compose", padding=-crossfade)
#         except Exception:
#             final = concatenate_videoclips(parts + [clipB], method="compose")
#         out_name = os.path.join(out_folder, f"{base}_{tname}.mp4")
#         if verbose: print("Writing:", out_name)
#         final.write_videofile(out_name, fps=fps, codec="libx264", threads=4, preset="medium")
#         # free memory for this iteration
#         final.close()
#     if verbose: print("Batch preview complete. Files in:", out_folder)

# # -----------------------
# # CLI support
# # -----------------------
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("clipA")
#     parser.add_argument("clipB")
#     parser.add_argument("--type", default="random", help="transition type or 'random'")
#     parser.add_argument("--out", default="final_transition.mp4")
#     parser.add_argument("--duration", type=float, default=1.2)
#     parser.add_argument("--model", default="yolov8n-seg.pt")
#     parser.add_argument("--fps", type=int, default=30)
#     parser.add_argument("--crossfade", type=float, default=0.06)
#     parser.add_argument("--batch", action="store_true", help="produce batch previews for all transitions")
#     parser.add_argument("--out_folder", default="previews")
#     args = parser.parse_args()

#     if args.batch:
#         make_batch_preview(args.clipA, args.clipB, out_folder=args.out_folder, duration=args.duration,
#                            yolo_model=args.model, fps=args.fps, crossfade=args.crossfade)
#     else:
#         make_object_transition(args.clipA, args.clipB, transition_type=args.type, duration=args.duration,
#                                out_path=args.out, yolo_model=args.model, fps=args.fps, crossfade=args.crossfade)
#!/usr/bin/env python3

#opt2
"""
objtrans_yolo_batch.py
Object-based transitions (batch preview) using YOLOv8-seg only (no rembg).
Designed for macOS (M1/M2/M3) or CPU.
Usage (batch):
    python3 objtrans_yolo_batch.py clipA.mp4 clipB.mp4 --batch --out_folder previews
Single:
    python3 objtrans_yolo_batch.py clipA.mp4 clipB.mp4 --type slide --out final.mp4
"""
import os, random, tempfile
from typing import Optional, List, Tuple
import argparse

import cv2
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips, vfx

from ultralytics import YOLO
from tqdm import tqdm

# ---------------------------
# Helpers
# ---------------------------

from moviepy.editor import VideoFileClip

# Prevent MoviePy from auto-closing readers inside nested transitions
VideoFileClip.close = lambda self: None

def reopen_clip(path):
    """Always return a fresh VideoFileClip to avoid closed-reader errors."""
    return VideoFileClip(path, audio=False)


def pick_device():
    import torch
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def read_frame_rgb(path: str, frame_idx: int = 0, downscale_to: Optional[Tuple[int,int]] = None) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Can't open {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read frame")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if downscale_to:
        frame = cv2.resize(frame, downscale_to, interpolation=cv2.INTER_AREA)
    return frame

def save_rgba_temp_image(rgba: np.ndarray) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(rgba).save(tmp.name, "PNG")
    return tmp.name

def clean_mask(mask: np.ndarray, k: int = 5) -> np.ndarray:
    """Binary mask (0/255) -> morphological open/close to smooth edges"""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    # slight blur to soften edge
    m = cv2.GaussianBlur(m, (5,5), 0)
    # normalize 0..255
    m = np.clip(m, 0, 255).astype(np.uint8)
    return m

# ---------------------------
# YOLO segmentation (highest-confidence selection)
# ---------------------------
# class YoloSeg:
#     def __init__(self, model_name: str = "yolov8n-seg.pt", device: Optional[str] = None, conf: float = 0.25, imgsz:int=640):
#         self.device = device or pick_device()
#         self.model = YOLO(model_name)
#         self.conf = conf
#         self.imgsz = imgsz

#     def segment_first_frame(self, rgb_frame: np.ndarray, retina_masks: bool = True):
#         # run predict on numpy rgb image
#         results = self.model.predict(source=rgb_frame, device=self.device, imgsz=self.imgsz, conf=self.conf, retina_masks=retina_masks, verbose=False)
#         if len(results) == 0:
#             raise RuntimeError("YOLO returned no results")
#         r = results[0]
#         masks = getattr(r, "masks", None)
#         boxes = getattr(r, "boxes", None)
#         if masks is None:
#             raise RuntimeError("No masks — ensure you used a '-seg' model (yolov8n-seg.pt)")
#         # masks.data: shape (N,H,W) or tensor
#         mask_data = masks.data
#         if hasattr(mask_data, "cpu"):
#             mask_arr = mask_data.cpu().numpy()
#         else:
#             mask_arr = np.array(mask_data)
#         if mask_arr.ndim == 2:
#             mask_arr = mask_arr[None, ...]
#         # pick highest-confidence detection if available
#         idx = None
#         try:
#             if boxes is not None and hasattr(boxes, "conf"):
#                 confs = boxes.conf
#                 confs = confs.cpu().numpy() if hasattr(confs, "cpu") else np.array(confs)
#                 if len(confs) == mask_arr.shape[0]:
#                     idx = int(np.argmax(confs))
#         except Exception:
#             idx = None
#         if idx is None:
#             # fallback choose largest area
#             areas = (mask_arr > 0.5).reshape(mask_arr.shape[0], -1).sum(axis=1)
#             idx = int(np.argmax(areas))
#         mask = (mask_arr[idx] > 0.5).astype(np.uint8)  # 0/1
#         # compute bbox in original frame coords (mask already aligned to frame)
#         ys, xs = mask.nonzero()
#         if xs.size == 0:
#             raise RuntimeError("Selected mask empty")
#         x0, x1 = int(xs.min()), int(xs.max())
#         y0, y1 = int(ys.min()), int(ys.max())
#         return mask * 255, (x0, y0, x1, y1)

# ================================================
#  PERFECT OBJECT CUT SCRIPT (YOLO + SAM)
#  Replaces YOLOv8-seg with:
#      - YOLO (bounding box)
#      - SAM (perfect mask)
# ================================================

# from segment_anything import sam_model_registry, SamPredictor
# import torch

# class YoloSamSeg:
#     def __init__(self,
#                  yolo_model="yolov8n.pt",
#                  sam_checkpoint="sam_b.pth",
#                  device=None,
#                  conf=0.25):

#         from ultralytics import YOLO

#         self.device = device or pick_device()

#         # YOLO (for bounding box only)
#         self.yolo = YOLO(yolo_model)

#         # SAM for PERFECT CUT
#         print("Loading SAM model …")
#         self.sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
#         self.sam.to(self.device)
#         self.predictor = SamPredictor(self.sam)

#         self.conf = conf

#     def segment_first_frame(self, rgb_frame):

#         # ----------------------------------------
#         # 1. YOLO → Bounding Box
#         # ----------------------------------------
#         results = self.yolo.predict(
#             source=rgb_frame,
#             device=self.device,
#             conf=self.conf,
#             verbose=False
#         )

#         r = results[0]
#         boxes = r.boxes

#         if boxes is None or len(boxes) == 0:
#             raise RuntimeError("YOLO found no objects")

#         confs = boxes.conf.cpu().numpy()
#         idx = int(confs.argmax())

#         b = boxes.xyxy[idx].cpu().numpy().astype(int)
#         x0, y0, x1, y1 = b

#         # ----------------------------------------
#         # 2. SAM → PERFECT MASK for that box
#         # ----------------------------------------
#         self.predictor.set_image(rgb_frame)
#         # box_tensor = torch.tensor([x0, y0, x1, y1]).to(self.device)

#         # masks, scores, logits = self.predictor.predict(
#         #     box=box_tensor,
#         #     multimask_output=True
#         # )
#         import numpy as np

#         box_np = np.array([x0, y0, x1, y1], dtype=np.float32)

#         masks, scores, logits = self.predictor.predict(
#             box=box_np[None, :],   # SAM expects 2D box array
#             multimask_output=True
#         )

#         # pick best mask
#         best_mask = masks[scores.argmax()]
#         mask_255 = (best_mask * 255).astype("uint8")

#         return mask_255, (x0, y0, x1, y1)


# # ---------------------------
# # Build RGBA object image (crop + alpha)
# # ---------------------------
# def build_object_rgba(rgb_frame: np.ndarray, mask_255: np.ndarray, bbox: Tuple[int,int,int,int]) -> np.ndarray:
#     x0,y0,x1,y1 = bbox
#     crop_rgb = rgb_frame[y0:y1+1, x0:x1+1].copy()
#     crop_mask = mask_255[y0:y1+1, x0:x1+1].copy()
#     clean = clean_mask(crop_mask, k=5)
#     alpha = clean
#     if alpha.ndim==2:
#         pass
#     rgba = np.dstack([crop_rgb, alpha])
#     rgba = np.clip(rgba, 0, 255).astype(np.uint8)
#     return rgba

from segment_anything import sam_model_registry, SamPredictor
import torch
import numpy as np
import cv2

# -----------------------------------------------------
# Utility: pick device
# -----------------------------------------------------
def pick_device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------
# Utility: expand bounding box safely
# -----------------------------------------------------


def expand_box(x0, y0, x1, y1, pad, W, H):
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W - 1, x1 + pad)
    y1 = min(H - 1, y1 + pad)
    return x0, y0, x1, y1

# -----------------------------------------------------
# Utility: dilate mask edges
# -----------------------------------------------------

def refine_mask(mask, dilate_k=25, erode_k=20):
    kernel_d = np.ones((dilate_k, dilate_k), np.uint8)
    kernel_e = np.ones((erode_k, erode_k), np.uint8)

    # 1. protect real edges
    mask = cv2.dilate(mask, kernel_d)

    # 2. remove unwanted outer junk
    mask = cv2.erode(mask, kernel_e)

    return mask

def dilate_mask(mask, k=20):
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask, kernel)

# -----------------------------------------------------
# YOLO + SAM Segmenter (UPDATED)
# -----------------------------------------------------
class YoloSamSeg:
    def __init__(self,
                 yolo_model="yolov8n.pt",
                 sam_checkpoint="sam_b.pth",
                 device=None,
                 conf=0.25):

        from ultralytics import YOLO

        self.device = device or pick_device()
        self.yolo = YOLO(yolo_model)

        print("Loading SAM model …")
        self.sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)

        self.conf = conf

    def segment_first_frame(self, rgb_frame):

        H, W = rgb_frame.shape[:2]

        # --------------------------
        # 1. YOLO → bounding box
        # --------------------------
        results = self.yolo.predict(
            source=rgb_frame,
            device=self.device,
            conf=self.conf,
            verbose=False
        )

        r = results[0]
        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            raise RuntimeError("YOLO found no objects")

        idx = int(boxes.conf.cpu().numpy().argmax())
        b = boxes.xyxy[idx].cpu().numpy().astype(int)
        x0, y0, x1, y1 = b

        # --------------------------
        # 2. SAM → perfect mask
        # --------------------------
        self.predictor.set_image(rgb_frame)

        box_np = np.array([x0, y0, x1, y1], dtype=np.float32)

        masks, scores, _ = self.predictor.predict(
            box=box_np[None, :],
            multimask_output=True
        )

        best_mask = masks[scores.argmax()]
        mask_255 = (best_mask * 255).astype("uint8")

        # --------------------------
        # 3. TRUE MASK BOUND BOX
        # --------------------------
        ys, xs = np.where(best_mask > 0)

        y0_m, y1_m = ys.min(), ys.max()
        x0_m, x1_m = xs.min(), xs.max()

        # --------------------------
        # 4. EXPAND BOX (no cut)
        # --------------------------
        PAD = 40     # Increase if still cutting
        x0_f, y0_f, x1_f, y1_f = expand_box(x0_m, y0_m, x1_m, y1_m, PAD, W, H)

        # --------------------------
        # 5. DILATE MASK (safer edges)
        # --------------------------
        mask_refined = refine_mask(mask_255, dilate_k=25, erode_k=20)


        # return strong mask + big box
        return mask_refined, (x0_f, y0_f, x1_f, y1_f)


# -----------------------------------------------------
# Build clean RGBA object crop
# -----------------------------------------------------
def build_object_rgba(rgb_frame, mask_255, bbox):

    x0, y0, x1, y1 = bbox

    crop_rgb = rgb_frame[y0:y1+1, x0:x1+1]
    crop_mask = mask_255[y0:y1+1, x0:x1+1]

    alpha = crop_mask

    rgba = np.dstack([crop_rgb, alpha])
    rgba = np.clip(rgba, 0, 255).astype("uint8")

    return rgba


# ---------------------------
# Transition templates (landing at target_pos in pixels)
# target_pos = (x, y) -> top-left position where object should land in the final frame
# object size is preserved by cropping earlier
# ===============================================================
#  ULTIMATE CINEMATIC TRANSITION PACK 
#  All transitions (your originals + new premium cinematic ones)
# ===============================================================

import math, random
from moviepy.editor import ImageClip, vfx


# ===============================================================
# ORIGINAL TRANSITIONS (you already had)
# ===============================================================

def slide_in(obj_clip, target_pos, video_w, video_h, duration=1.2, direction=None):
    ow, oh = obj_clip.size
    end_x, end_y = target_pos
    if direction is None:
        direction = random.choice(["left","right","top","bottom"])
    start_map = {
        "left": (-ow, end_y),
        "right": (video_w, end_y),
        "top": (end_x, -oh),
        "bottom": (end_x, video_h)
    }
    sx, sy = start_map.get(direction, (-ow, end_y))
    def pos(t):
        p = min(1, max(0, t/duration))
        p_ease = 1 - (1 - p)**2
        x = int(sx + (end_x - sx) * p_ease)
        y = int(sy + (end_y - sy) * p_ease)
        return (x, y)
    return obj_clip.set_position(pos).set_duration(duration)

def zoom_in(obj_clip, target_pos, video_w, video_h, duration=1.2, start_scale=0.15):
    ow, oh = obj_clip.size
    end_x, end_y = target_pos
    def scale(t):
        p = min(1, max(0, t/duration))
        return start_scale + (1-start_scale)*(1-(1-p)**2)
    return obj_clip.fx(vfx.resize, lambda t: scale(t)).set_position((end_x, end_y)).set_duration(duration)

def cinematic_zoom(obj_clip, target_pos, video_w, video_h, duration=1.3):
    end_x, end_y = target_pos
    def scale(t):
        p = min(1, max(0, t/duration))
        return 0.55 + (1 - 0.55) * p
    return obj_clip.fx(vfx.resize, lambda t: scale(t)).set_position((end_x, end_y)).set_duration(duration)

def cinematic_slide(obj_clip, target_pos, video_w, video_h, duration=1.2):
    ow, oh = obj_clip.size
    end_x, end_y = target_pos
    start_x = int(end_x + video_w * 0.18)
    def pos(t):
        p = min(1, max(0, t/duration))
        ease = 1 - (1 - p)**3
        x = int(start_x + (end_x - start_x) * ease)
        return (x, end_y)
    return obj_clip.set_position(pos).set_duration(duration)

def cinematic_in(obj_clip, target_pos, video_w, video_h, duration=1.4):
    ow, oh = obj_clip.size
    end_x, end_y = target_pos
    def scale(t):
        p = min(1, max(0, t/duration))
        return 0.78 + (1.0 - 0.78) * (1 - (1 - p)**3)
    drift_x = end_x - 18
    drift_y = end_y + 10
    def pos(t):
        p = min(1, max(0, t/duration))
        x = int(drift_x + (end_x - drift_x) * p)
        y = int(drift_y + (end_y - drift_y) * p)
        return (x, y)
    return obj_clip.fx(vfx.resize, lambda t: scale(t)).set_position(pos).set_duration(duration)

def pop_in(obj_clip, target_pos, video_w, video_h, duration=1.2):
    end_x, end_y = target_pos
    def scale(t):
        p = min(1, max(0, t/duration))
        if p < 0.7:
            s = 0.2 + 1.4*(p/0.7)
        else:
            s = 1.0 - 0.2*((p-0.7)/0.3)
        return max(0.01, s)
    return obj_clip.fx(vfx.resize, lambda t: scale(t)).set_position((end_x,end_y)).set_duration(duration)

def spin_in(obj_clip, target_pos, video_w, video_h, duration=1.2, spins=1.0):
    end_x, end_y = target_pos
    return obj_clip.fx(vfx.rotate, lambda t: 360*spins*(1-(1-min(1,t/duration))**2)).set_position((end_x,end_y)).set_duration(duration)

def fade_in(obj_clip, target_pos, video_w, video_h, duration=1.2):
    end_x, end_y = target_pos
    return obj_clip.set_position((end_x,end_y)).fadein(duration).set_duration(duration)

def parallax_in(obj_clip, target_pos, video_w, video_h, duration=1.2, start_scale=0.7):
    ow, oh = obj_clip.size
    x,y = target_pos
    def s(t):
        p = min(1, max(0, t/duration))
        return start_scale + (1-start_scale)*p
    def pos(t):
        scale = s(t)
        sw, sh = int(ow*scale), int(oh*scale)
        return (int(x + (ow-sw)/2), int(y + (oh-sh)/2))
    return obj_clip.fx(vfx.resize, lambda t: s(t)).set_position(pos).set_duration(duration)

def burst_in(obj_clip, target_pos, video_w, video_h, duration=1.2):
    x,y = target_pos
    def s(t):
        p = min(1, max(0, t/duration))
        if p < 0.25:
            return 0.2 + 4*p
        else:
            return 1.0 - 0.1*(p-0.25)
    return obj_clip.fx(vfx.resize, lambda t: s(t)).set_position((x,y)).set_duration(duration)

def perspective_in(obj_clip, target_pos, video_w, video_h, duration=1.2):
    x,y = target_pos
    def s(t):
        p = min(1, max(0, t/duration))
        return 0.6 + 0.4*p
    def r(t):
        p = min(1, max(0, t/duration))
        return (1-p)*25
    return obj_clip.fx(vfx.resize, lambda t: s(t)).fx(vfx.rotate, lambda t: r(t)).set_position((x,y)).set_duration(duration)

def motion_match(obj_clip, target_pos, video_w, video_h, duration=1.2, vec=(200, -30)):
    ow, oh = obj_clip.size
    end_x, end_y = target_pos
    sx, sy = int(end_x - vec[0]), int(end_y - vec[1])
    def pos(t):
        p = min(1, max(0, t/duration))
        return (int(sx + (end_x-sx)*p), int(sy + (end_y-sy)*p))
    return obj_clip.set_position(pos).set_duration(duration)

def wipe_in(obj_clip, target_pos, video_w, video_h, duration=1.2, direction="left"):
    x,y = target_pos
    def sx(t):
        p = min(1, max(0, t/duration))
        return max(0.01, p)
    return obj_clip.fx(vfx.resize, lambda t: sx(t)).set_position((x,y)).set_duration(duration)



# ============================================================================
# 🔥🔥 NEW PREMIUM NEXT-LEVEL CINEMATIC TRANSITIONS (15 BEST)
# ============================================================================

# 1. Velocity Ramp Slide
def velocity_ramp_slide(obj_clip, target_pos, video_w, video_h, duration=1.2):
    ow, oh = obj_clip.size
    end_x, end_y = target_pos
    start_x = -ow * 1.4
    def pos(t):
        p = t/duration
        if p < 0.4:
            e = 2.5 * (p**2)
        else:
            e = 1 - (1 - p)**3
        x = int(start_x + (end_x - start_x) * e)
        return (x, end_y)
    return obj_clip.set_position(pos).set_duration(duration)

# 2. Lens Distortion zoom (smooth)
def lens_distort_in(obj_clip, target_pos, video_w, video_h, duration=1.3):
    end_x, end_y = target_pos
    def scale(t):
        p = t/duration
        return 0.65 + (1 - 0.65) * (p**1.8)
    return obj_clip.fx(vfx.resize, lambda t: scale(t)).set_position((end_x,end_y)).set_duration(duration)

# 3. Depth Drop In
def depth_drop_in(obj_clip, target_pos, video_w, video_h, duration=1.1):
    end_x, end_y = target_pos
    start_y = -video_h * 0.65
    def pos(t):
        p = t/duration
        ease = 1 - (1 - p)**4
        y = int(start_y + (end_y - start_y) * ease)
        return (end_x, y)
    def scale(t):
        p = t/duration
        return 0.55 + 0.45*p
    return obj_clip.fx(vfx.resize, lambda t: scale(t)).set_position(pos).set_duration(duration)

# 4. Whip move
def whip_move(obj_clip, target_pos, video_w, video_h, duration=0.9):
    ow, oh = obj_clip.size
    end_x, end_y = target_pos
    sx = int(end_x + video_w * 0.8)
    def pos(t):
        p = t/duration
        e = 1 - (1 - p)**5
        x = sx + (end_x - sx) * e
        return (x, end_y)
    return obj_clip.set_position(pos).set_duration(duration)

# 5. Elastic Bounce In
def elastic_bounce_in(obj_clip, target_pos, video_w, video_h, duration=1.5):
    end_x, end_y = target_pos
    def scale(t):
        p = t/duration
        return 1 + 0.22 * math.sin(12*p) * (1 - p)
    return obj_clip.fx(vfx.resize, lambda t: scale(t)).set_position((end_x,end_y)).set_duration(duration)

# 6. Overshoot slide
def overshoot_slide(obj_clip, target_pos, video_w, video_h, duration=1.0):
    end_x, end_y = target_pos
    sx = -obj_clip.w * 1.2
    def pos(t):
        p = t/duration
        o = 1 - (1 - p)**2
        x = sx + (end_x - sx) * o + 12 * math.sin(p * 3.14)
        return (x, end_y)
    return obj_clip.set_position(pos).set_duration(duration)

# 7. Drift fade
def drift_fade(obj_clip, target_pos, video_w, video_h, duration=1.2):
    end_x, end_y = target_pos
    drift_x = end_x + 25
    def pos(t):
        p = t/duration
        ease = 1 - (1 - p)**3
        x = drift_x + (end_x - drift_x) * ease
        return (x, end_y)
    return obj_clip.set_position(pos).fadein(duration).set_duration(duration)

# 8. Kinetic pop
def kinetic_pop(obj_clip, target_pos, video_w, video_h, duration=0.8):
    end_x, end_y = target_pos
    def scale(t):
        p = t/duration
        if p < 0.5:
            return 0.2 + p*2
        return 1 + 0.08 * math.sin((p-0.5)*6.3)
    return obj_clip.fx(vfx.resize, lambda t: scale(t)).set_position((end_x,end_y)).set_duration(duration)

# 9. Blur slide
import numpy as np
from PIL import Image, ImageFilter

import numpy as np
from PIL import Image, ImageFilter

def blur_slide(obj_clip, target_pos, video_w, video_h, duration=1.2):
    end_x, end_y = target_pos
    sx = int(end_x - video_w * 0.5)

    def pos(t):
        p = t/duration
        return (int(sx + (end_x - sx) * (p**1.6)), end_y)

    # --------------------------
    # FRAME-BY-FRAME BLUR (RGBA SAFE)
    # --------------------------
    def blur_frame(get_frame, t):
        frame = get_frame(t)
        p = t / duration
        blur_amount = int(8 * (1 - p))
        if blur_amount <= 0:
            return frame

        img = Image.fromarray(frame)

        # If RGBA → blur only RGB
        if img.mode == "RGBA":
            r, g, b, a = img.split()

            rgb = Image.merge("RGB", (r, g, b))
            rgb = rgb.filter(ImageFilter.GaussianBlur(radius=blur_amount))

            r2, g2, b2 = rgb.split()
            img = Image.merge("RGBA", (r2, g2, b2, a))

        else:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_amount))

        return np.array(img)

    blurred_clip = obj_clip.fl(blur_frame)

    return blurred_clip.set_position(pos).set_duration(duration)



# 10. Anti-gravity
def anti_gravity_in(obj_clip, target_pos, video_w, video_h, duration=1.4):
    end_x, end_y = target_pos
    start_y = end_y + 140
    def pos(t):
        p = t/duration
        ease = p**0.7
        y = int(start_y + (end_y-start_y)*ease)
        return (end_x, y)
    return obj_clip.set_position(pos).fadein(duration).set_duration(duration)

# 11. Zoom burst
import numpy as np
from PIL import Image, ImageFilter

def zoom_burst(obj_clip, target_pos, video_w, video_h, duration=1.2):
    end_x, end_y = target_pos

    # --------------------------
    # ZOOM SCALE FUNCTION
    # --------------------------
    def scale(t):
        p = t / duration
        return 1 + 0.7 * (1 - p)  # big zoom at start → normal at end

    # --------------------------
    # RGBA-SAFE BLUR
    # --------------------------
    def rgba_blur(get_frame, t):
        frame = get_frame(t)
        p = t / duration
        blur_amount = int(18 * (1 - p))
        if blur_amount <= 0:
            return frame

        img = Image.fromarray(frame)

        # If RGBA → only blur RGB
        if img.mode == "RGBA":
            r, g, b, a = img.split()
            rgb = Image.merge("RGB", (r, g, b))
            rgb = rgb.filter(ImageFilter.GaussianBlur(radius=blur_amount))
            r2, g2, b2 = rgb.split()
            img = Image.merge("RGBA", (r2, g2, b2, a))
        else:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_amount))

        return np.array(img)

    # --------------------------
    # APPLY EFFECTS
    # --------------------------
    zoomed = obj_clip.fl_time(lambda t: t).fl(lambda gf, t: rgba_blur(gf, t))

    final = zoomed.fx(
        vfx.resize, lambda t: scale(t)
    ).set_position(
        (end_x, end_y)
    ).set_duration(duration)

    return final


# 12. Wobble in
def wobble_in(obj_clip, target_pos, video_w, video_h, duration=1.0):
    end_x, end_y = target_pos
    def pos(t):
        p = t/duration
        wob = math.sin(p*15) * 6 * (1 - p)
        return (end_x + wob, end_y)
    return obj_clip.set_position(pos).set_duration(duration)

# 13. Matrix drop
def matrix_drop(obj_clip, target_pos, video_w, video_h, duration=1.1):
    end_x, end_y = target_pos
    sy = -video_h * 0.7
    def pos(t):
        p = t/duration
        e = 1 - (1-p)**5
        y = sy + (end_y - sy)*e
        return (end_x, y)
    return obj_clip.set_position(pos).fadein(duration*0.8).set_duration(duration)

# 14. Hyper smooth in
def hypersmooth_in(obj_clip, target_pos, video_w, video_h, duration=1.0):
    end_x, end_y = target_pos
    def scale(t):
        p = t/duration
        return 0.85 + 0.15*(1-(1-p)**4)
    return obj_clip.fx(vfx.resize, lambda t: scale(t)).set_position((end_x,end_y)).set_duration(duration)

# 15. Shockwave in
def shockwave_in(obj_clip, target_pos, video_w, video_h, duration=0.9):
    end_x, end_y = target_pos
    def scale(t):
        p = t/duration
        return 0.1 + (1-p)*0 + p + 0.12*math.sin(p*9)*(1-p)
    return obj_clip.fx(vfx.resize, lambda t: scale(t)).set_position((end_x,end_y)).set_duration(duration)


# ===============================================================
# END OF PACK
# ===============================================================


TRANSITION_MAP = {
    # Your original transitions
    "slide": slide_in,
    "zoom": zoom_in,
    "pop": pop_in,
    "spin": spin_in,
    "fade": fade_in,
    "parallax": parallax_in,
    "burst": burst_in,
    "perspective": perspective_in,
    "motion": motion_match,
    "wipe": wipe_in,

    # Cinematic signature set
    "cinein": cinematic_in,
    "cinesl": cinematic_slide,
    "cinezo": cinematic_zoom,

    # 15 new transitions
    "veloramp": velocity_ramp_slide,
    "lensin": lens_distort_in,
    "depthdrop": depth_drop_in,
    "whip": whip_move,
    "elastic": elastic_bounce_in,
    "overshoot": overshoot_slide,
    "driftfade": drift_fade,
    "kpop": kinetic_pop,
    "blurslide": blur_slide,
    "antigrav": anti_gravity_in,
    "zoomburst": zoom_burst,
    "wobble": wobble_in,
    "matrixdrop": matrix_drop,
    "hypersmooth": hypersmooth_in,
    "shockwave": shockwave_in
}
# video_transitions.py

import os
import random
from typing import Optional, Tuple
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips

# Assuming these functions/classes exist and are imported or defined elsewhere
# from your_segmentation_module import YoloSamSeg, build_object_rgba, save_rgba_temp_image, read_frame_rgb
# from your_transitions_module import TRANSITION_MAP
# from your_utils_module import pick_device

# ---------------------------
# Compose + concat with small crossfade (audio+video)
# ---------------------------
def compose_and_concat(
    clipA: VideoFileClip,
    clipB: VideoFileClip,
    animated_obj_clip: ImageClip,
    duration: float,
    crossfade: float
) -> VideoFileClip:
    pre_start = max(0, clipA.duration - duration)
    head = clipA.subclip(0, pre_start) if pre_start > 0.01 else None
    tail = clipA.subclip(pre_start, clipA.duration)
    composed_tail = CompositeVideoClip([tail, animated_obj_clip.set_start(0)]).set_duration(tail.duration)
    
    parts = []
    if head: 
        parts.append(head)
    parts.append(composed_tail)
    
    try:
        final = concatenate_videoclips(parts + [clipB], method="compose", padding=-crossfade)
    except Exception:
        final = concatenate_videoclips(parts + [clipB], method="compose")
    return final

# ---------------------------
# Main pipeline (single or batch)
# ---------------------------
from typing import Optional, Tuple, List
import os, random
from moviepy.editor import VideoFileClip, ImageClip

from transit import flash_transition
def make_batch_or_single(
    clipA_path: str,
    clipB_path: str,
    transition_type: str = "random",
    duration: float = 1.2,
    out_path: str = "final.mp4",
    model_name: str = "yolov8n-seg.pt",
    device: Optional[str] = None,
    fps: int = 30,
    crossfade: float = 0.06,
    batch: bool = False,
    out_folder: str = "previews",
    downscale_frame: Optional[Tuple[int,int]] = None,
    verbose: bool = True
) -> List[str]:
    
    device = device or pick_device()
    if verbose: 
        print("Device:", device)

    # Load clips (will reload inside loop)
    clipA = reopen_clip(clipA_path)
    clipB = reopen_clip(clipB_path)

    video_w, video_h = clipA.w, clipA.h

    # Read first frame from clipB
    if verbose:
        print("Reading frame from clipB...")

    frame = read_frame_rgb(clipB_path, frame_idx=0, downscale_to=downscale_frame)

    # Run YOLO+SAM segmentation
    if verbose:
        print("Running YOLO-SAM segmentation...")

    seg = YoloSamSeg(yolo_model=model_name, sam_checkpoint="sam_b.pth", device=device)

    try:
        mask_255, bbox = seg.segment_first_frame(frame)
        object_found = True
    except RuntimeError:
        if verbose:
            print("⚠️ YOLO found no objects. Using fallback full-frame transition.")
        mask_255, bbox = None, None
        object_found = False

    # Build object clip if found
    tmpfile = None
    if object_found:
        if verbose:
            print("Building RGBA object (crop + alpha)...")

        rgba = build_object_rgba(frame, mask_255, bbox)
        tmpfile = save_rgba_temp_image(rgba)

        base_obj_clip = ImageClip(tmpfile).set_duration(duration).set_fps(fps)

        x0, y0, _, _ = bbox
        target_pos = (x0, y0)

    # Build transition list
    tlist = (
        list(TRANSITION_MAP.keys()) if batch
        else ([transition_type] if transition_type != "random" else [random.choice(list(TRANSITION_MAP.keys()))])
    )

    os.makedirs(out_folder, exist_ok=True)
    outputs = []

    for tname in tlist:

        if tname not in TRANSITION_MAP:
            print("Skipping unknown transition:", tname)
            continue

        if verbose:
            print("Rendering transition:", tname)

        # Reload clips every loop (MoviePy safety)
        clipA = reopen_clip(clipA_path)
        clipB = reopen_clip(clipB_path)

        # Build animated transition
        if object_found:
            objclip = ImageClip(tmpfile).set_duration(duration).set_fps(fps)
            animated = TRANSITION_MAP[tname](
                objclip, target_pos, video_w, video_h, duration=duration
            )
        else:
            animated = flash_transition(clipA, clipB, dur=duration)

        final = compose_and_concat(
            clipA, clipB, animated, duration=duration, crossfade=crossfade
        )

        # output name
        outname = (
            out_path
            if not batch
            else os.path.join(
                out_folder,
                f"{os.path.splitext(os.path.basename(clipB_path))[0]}_{tname}.mp4"
            )
        )

        final.write_videofile(
            outname,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            verbose=verbose,
            logger=None
        )

        final.close()
        clipA.close()
        clipB.close()

        outputs.append(outname)

    # Cleanup temporary RGBA file
    if tmpfile and os.path.exists(tmpfile):
        try:
            os.remove(tmpfile)
        except:
            pass

    return outputs

# Utility: reopen clip

# ---------------------------
# Optional CLI
# ---------------------------
def cli():
    import argparse
    p = argparse.ArgumentParser(description="Video Transition Composer")
    p.add_argument("clipA")
    p.add_argument("clipB")
    p.add_argument("--type", default="random")
    p.add_argument("--out", default="final.mp4")
    p.add_argument("--duration", type=float, default=0.6)
    p.add_argument("--model", default="yolov8n-seg.pt")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--crossfade", type=float, default=0.06)
    p.add_argument("--batch", action="store_true")
    p.add_argument("--out_folder", default="previews")
    args = p.parse_args()
    make_batch_or_single(
        args.clipA, args.clipB,
        transition_type=args.type,
        duration=args.duration,
        out_path=args.out,
        model_name=args.model,
        fps=args.fps,
        crossfade=args.crossfade,
        batch=args.batch,
        out_folder=args.out_folder
    )

if __name__ == "__main__":
    cli()
