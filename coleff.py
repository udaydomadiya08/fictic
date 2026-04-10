# universal_ai_colorfx_20.py
"""
Universal AI Object Color FX (20-effect pack)

Features:
- Detects main object using YOLOv8-seg if available (falls back to bbox if not)
- Creates smooth feathered masks
- Applies 20 object-only color effects (mixed cinematic + artistic pack)
- Performance options: reuse masks every N frames, temp low-res processing
- M1-friendly: auto-selects device 'mps' when available
- MoviePy-friendly wrappers for in-memory clips and file-based processing

Dependencies:
- numpy
- opencv-python
- moviepy
- ultralytics (optional; if missing we fallback to bbox)
- torch (for device detection; ultralytics brings torch)

Note: If you don't have YOLO seg model, the module will still run using a center-box fallback mask.
"""

import cv2
import numpy as np
import os
from moviepy.editor import VideoClip, VideoFileClip
from typing import Optional

# Try to import ultralytics YOLO (optional)
_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except Exception:
    _YOLO_AVAILABLE = False

# Try to detect MPS (Apple) / CUDA
_device = "cpu"
try:
    import torch
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = "mps"
    elif torch.cuda.is_available():
        _device = "cuda"
    else:
        _device = "cpu"
except Exception:
    _device = "cpu"


# ---------------------------
# Utilities: mask refinement
# ---------------------------
def _feather_mask(mask: np.ndarray, ksize: int = 31, sigma: float = 0) -> np.ndarray:
    """
    mask: float or uint8 mask (0..1 or 0..255) single channel
    returns: float mask 0..1 with feathered edges
    """
    if mask.dtype != np.uint8:
        mask_u8 = (np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        mask_u8 = mask
    # morphological clean
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    # gaussian blur for feather
    k = ksize if ksize % 2 == 1 else ksize + 1
    blurred = cv2.GaussianBlur(mask_u8, (k, k), sigmaX=sigma)
    out = blurred.astype(np.float32) / 255.0
    out = np.expand_dims(out, axis=2)  # H,W,1
    return out


def _largest_mask_from_yolo(result) -> Optional[np.ndarray]:
    """
    result: single YOLO result object (ultralytics). Return mask as binary HxW float (0..1)
    """
    try:
        masks = result.masks
        if masks is None:
            return None
        # masks.data may be (n, Hmask, Wmask) or provide xy coordinates
        mdata = masks.data.cpu().numpy()  # (N, Mh, Mw)
        # pick largest
        areas = mdata.reshape(mdata.shape[0], -1).sum(axis=1)
        idx = int(np.argmax(areas))
        mask = mdata[idx]
        # Note: mask might be smaller (mask coords) but ultralytics usually returns full-size
        return (mask > 0.5).astype(np.uint8)
    except Exception:
        return None


def _bbox_to_mask(frame_shape, bbox):
    """
    bbox: (x1, y1, x2, y2) ints
    frame_shape: (h,w,...)
    returns binary mask HxW uint8
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return mask
    mask[y1:y2, x1:x2] = 255
    return mask


# ---------------------------
# Color effect helpers
# ---------------------------
def _hsv_shift(img_bgr, h_shift=0, s_mul=1.0, v_mul=1.0):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    h = (h + h_shift) % 180
    s = np.clip(s * s_mul, 0, 255)
    v = np.clip(v * v_mul, 0, 255)
    hsv2 = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)


def _bgr_add(img, add=(0, 0, 0)):
    b, g, r = cv2.split(img.astype(np.int32))
    b = np.clip(b + add[0], 0, 255).astype(np.uint8)
    g = np.clip(g + add[1], 0, 255).astype(np.uint8)
    r = np.clip(r + add[2], 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])


def _bgr_mul(img, mul=(1.0, 1.0, 1.0)):
    b, g, r = cv2.split(img.astype(np.float32))
    b = np.clip(b * mul[0], 0, 255).astype(np.uint8)
    g = np.clip(g * mul[1], 0, 255).astype(np.uint8)
    r = np.clip(r * mul[2], 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r])


def _soft_glow(img_bgr, radius=15, strength=0.08):
    img = img_bgr.astype(np.float32) / 255.0
    bright = np.clip(img - 0.7, 0, 1)
    blur = cv2.GaussianBlur((bright * 255).astype(np.uint8), (0, 0), radius).astype(np.float32) / 255.0
    out = np.clip(img + blur * strength, 0, 1)
    return (out * 255).astype(np.uint8)


def _cartoonify(img_bgr):
    img = img_bgr.copy()
    # bilateral filter then edge detection
    color = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=9, C=2)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(color, edges_color)


# ---------------------------
# 20 Effects (name -> function(frame_region, intensity) => region_out)
# region_in/out are BGR uint8 crops matching object's bounding box or full frame when used; functions must be fast.
# ---------------------------
def effect_teal_orange(region, intensity=1.0):
    # subtle teal shadows + warm highlights
    out = _hsv_shift(region, h_shift=-6 * intensity, s_mul=1.05 + 0.05 * intensity, v_mul=1.02)
    out = _bgr_add(out, add=(0, 0, int(8 * intensity)))
    return out


def effect_neon_punch(region, intensity=1.0):
    out = _bgr_add(region, add=(30 * intensity, 10 * intensity, 30 * intensity))
    out = _soft_glow(out, radius=9, strength=0.12 * intensity)
    return out


def effect_warm_gold(region, intensity=1.0):
    out = _hsv_shift(region, h_shift=-12 * intensity, s_mul=1.08, v_mul=1.06)
    out = _bgr_add(out, add=(0, 6 * intensity, 18 * intensity))
    return out


def effect_ice_blue(region, intensity=1.0):
    out = _bgr_mul(region, mul=(1.05, 1.0, 0.85))
    out = _bgr_add(out, add=(18 * intensity, 0, -8 * intensity))
    return out


def effect_cyber_purple(region, intensity=1.0):
    r = region.copy()
    b, g, rch = cv2.split(r)
    rch = np.clip(rch + 10 * intensity, 0, 255)
    b = np.clip(b + 20 * intensity, 0, 255)
    out = cv2.merge([b.astype(np.uint8), g, rch.astype(np.uint8)])
    out = _soft_glow(out, radius=7, strength=0.06 * intensity)
    return out


def effect_lime_green_glow(region, intensity=1.0):
    out = _bgr_add(region, add=(0, 40 * intensity, 0))
    out = _soft_glow(out, radius=11, strength=0.08 * intensity)
    return out


def effect_rgb_split(region, intensity=1.0):
    # lightweight RGB offset + blend
    h, w = region.shape[:2]
    b, g, r = cv2.split(region)
    shift = int(3 * intensity)
    empty = np.zeros_like(b)
    b2 = np.roll(b, shift, axis=1)
    r2 = np.roll(r, -shift, axis=1)
    merged = cv2.merge([b2, g, r2])
    return cv2.addWeighted(region, 1.0 - 0.25 * intensity, merged, 0.25 * intensity, 0)


def effect_chrome_silver(region, intensity=1.0):
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    l = np.clip(l * (1.0 + 0.12 * intensity), 0, 255)
    lab = cv2.merge([l, a, b]).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def effect_soft_pastel(region, intensity=1.0):
    out = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
    out[..., 1] *= (1 - 0.35 * intensity)
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
    out = _soft_glow(out, radius=13, strength=0.05 * intensity)
    return out


def effect_high_contrast_film(region, intensity=1.0):
    lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    lab2 = cv2.merge([l, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    out = _bgr_mul(out, mul=(1.05 + 0.02 * intensity, 1.05, 1.05))
    return out


def effect_anime_pop(region, intensity=1.0):
    return _cartoonify(region)


def effect_dream_glow(region, intensity=1.0):
    out = _soft_glow(region, radius=21, strength=0.12 * intensity)
    out = _bgr_add(out, add=(8 * intensity, 6 * intensity, 12 * intensity))
    return out


def effect_red_inferno(region, intensity=1.0):
    out = _bgr_add(region, add=(0, 0, 40 * intensity))
    out = _bgr_mul(out, mul=(1.0, 0.9, 1.05 + 0.02 * intensity))
    return out


def effect_deepsea_blue(region, intensity=1.0):
    out = _bgr_mul(region, mul=(0.9, 1.0, 1.1))
    out = _bgr_add(out, add=(12 * intensity, 6 * intensity, -8 * intensity))
    return out


def effect_vintage_warm(region, intensity=1.0):
    out = cv2.cvtColor(region, cv2.COLOR_BGR2HSV).astype(np.float32)
    out[..., 1] *= (1 - 0.15 * intensity)
    out[..., 2] *= (1 - 0.02 * intensity)
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_HSV2BGR)
    out = _bgr_add(out, add=(0, 6 * intensity, 14 * intensity))
    return out


def effect_toxic_sci_fi(region, intensity=1.0):
    out = _bgr_add(region, add=(0, 60 * intensity, 0))
    out = _hsv_shift(out, h_shift=20 * intensity, s_mul=1.15, v_mul=1.02)
    return out


def effect_hdr_boost(region, intensity=1.0):
    # mild local contrast via unsharp + vibrance
    blur = cv2.GaussianBlur(region, (0, 0), 3)
    out = cv2.addWeighted(region, 1.2 + 0.02 * intensity, blur, -0.2 * intensity, 0)
    return out


def effect_purple_aura(region, intensity=1.0):
    out = _bgr_add(region, add=(12 * intensity, 0, 18 * intensity))
    out = _soft_glow(out, radius=9, strength=0.06 * intensity)
    return out


def effect_soft_beauty(region, intensity=1.0):
    # denoise + slight warmth + soft glow
    den = cv2.bilateralFilter(region, d=9, sigmaColor=75, sigmaSpace=75)
    den = _bgr_add(den, add=(0, 6 * intensity, 10 * intensity))
    out = _soft_glow(den, radius=11, strength=0.04 * intensity)
    return out


# registry
_EFFECT_REGISTRY = {
    "teal_orange": effect_teal_orange,
    "neon_punch": effect_neon_punch,
    "warm_gold": effect_warm_gold,
    "ice_blue": effect_ice_blue,
    "cyber_purple": effect_cyber_purple,
    "lime_green_glow": effect_lime_green_glow,
    "rgb_split": effect_rgb_split,
    "chrome_silver": effect_chrome_silver,
    "soft_pastel": effect_soft_pastel,
    "high_contrast_film": effect_high_contrast_film,
    "anime_pop": effect_anime_pop,
    "dream_glow": effect_dream_glow,
    "red_inferno": effect_red_inferno,
    "deepsea_blue": effect_deepsea_blue,
    "vintage_warm": effect_vintage_warm,
    "toxic_sci_fi": effect_toxic_sci_fi,
    "hdr_boost": effect_hdr_boost,
    "purple_aura": effect_purple_aura,
    "soft_beauty": effect_soft_beauty,
    "cinematic_blue": effect_ice_blue,   # alias
}


# ---------------------------
# UniversalAIColorFX20 class
# ---------------------------
class UniversalAIColorFX20:
    def __init__(self, model_path: str = "yolov8n-seg.pt", device: Optional[str] = None):
        """
        model_path: path to a YOLOv8-seg model (optional). If not available, module falls back to bbox center mask.
        device: 'mps', 'cuda', or 'cpu'. Default auto-detect.
        """
        self.model = None
        self.device = device or _device
        self.yolo_enabled = False
        if _YOLO_AVAILABLE:
            try:
                # try to load model path if exists else default pretrained
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                else:
                    # try a small segmentation-capable weights name (user may have installed yolov8n-seg)
                    try:
                        self.model = YOLO("yolov8n-seg.pt")
                    except Exception:
                        # fall back to generic yolov8n (may not have masks)
                        self.model = YOLO("yolov8n.pt")
                # move model device (ultralytics uses .to behind the scenes)
                self.yolo_enabled = True
            except Exception:
                self.model = None
                self.yolo_enabled = False
        else:
            self.yolo_enabled = False

    def list_effects(self):
        return list(_EFFECT_REGISTRY.keys())

    def _detect_mask(self, frame_bgr: np.ndarray, prefer_segmentation: bool = True):
        """
        Returns a binary mask HxW uint8 (0 or 255).
        Uses YOLO segmentation when available. If not, will use bbox fallback.
        """
        h, w = frame_bgr.shape[:2]
        # try segmentation predict
        if self.yolo_enabled and prefer_segmentation:
            try:
                # call predict with smallest overhead; use model.predict single frame
                results = self.model.predict(frame_bgr, device=self.device, verbose=False)
                res = results[0]
                # if masks are available
                if hasattr(res, "masks") and res.masks is not None and len(res.masks.data) > 0:
                    m = _largest_mask_from_yolo(res)
                    if m is not None:
                        # ensure size: resize to frame shape if needed
                        if m.shape != (h, w):
                            m = cv2.resize((m * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                        return (m > 0).astype(np.uint8) * 255
                # else fallback to boxes
                if hasattr(res, "boxes") and len(res.boxes) > 0:
                    boxes = res.boxes.xyxy.cpu().numpy()
                    # choose largest
                    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                    idx = int(np.argmax(areas))
                    x1, y1, x2, y2 = boxes[idx].astype(int)
                    return _bbox_to_mask((h, w), (x1, y1, x2, y2))
            except Exception:
                pass

        # fallback: use simple saliency / center box heuristic
        try:
            # OpenCV saliency (FineGrainedSaliency) faster fallback if available
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, sal_map = saliency.computeSaliency(frame_bgr)
            if success:
                sal_map_u8 = (sal_map * 255).astype(np.uint8)
                _, thresh = cv2.threshold(sal_map_u8, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
                # choose largest blob
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    return mask
        except Exception:
            pass

        # final fallback: center box
        cx1, cy1 = int(w * 0.2), int(h * 0.15)
        cx2, cy2 = int(w * 0.8), int(h * 0.85)
        return _bbox_to_mask((h, w), (cx1, cy1, cx2, cy2))

    def _apply_effect_on_frame(self, frame_bgr: np.ndarray, effect_name: str, intensity: float,
                               mask_binary: Optional[np.ndarray] = None, feather_ksize=31):
        """
        frame_bgr: full frame (H,W,3) uint8 BGR
        mask_binary: optional binary mask HxW uint8 (0/255)
        returns: frame with effect applied only inside mask (smoothly blended)
        """
        h, w = frame_bgr.shape[:2]
        if effect_name not in _EFFECT_REGISTRY:
            # unknown effect -> return original
            return frame_bgr

        if mask_binary is None:
            mask = self._detect_mask(frame_bgr)
        else:
            mask = mask_binary

        # refine & feather mask
        maskf = _feather_mask(mask, ksize=feather_ksize)
        mask3 = np.dstack([maskf, maskf, maskf])  # H,W,3 float

        # compute bounding rect to speed up processing
        ys, xs = np.where((mask > 0))
        if len(xs) == 0 or len(ys) == 0:
            return frame_bgr
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        # expand a bit for safety
        pad = 8
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)

        region = frame_bgr[y1:y2, x1:x2]
        # effect on region
        eff_fn = _EFFECT_REGISTRY[effect_name]
        try:
            region_out = eff_fn(region.copy(), intensity=float(intensity))
        except Exception:
            # fallback: small brightness boost
            region_out = _bgr_add(region.copy(), add=(10 * intensity, 10 * intensity, 10 * intensity))

        # assemble final image via mask blend (use maskf)
        mask_region = maskf[y1:y2, x1:x2]  # H_region,W_region,1
        mask_region_3 = np.dstack([mask_region, mask_region, mask_region])
        region_f = region.astype(np.float32)
        region_out_f = region_out.astype(np.float32)
        blended = (region_out_f * mask_region_3) + (region_f * (1.0 - mask_region_3))
        out = frame_bgr.copy().astype(np.float32)
        out[y1:y2, x1:x2] = blended
        out = np.clip(out, 0, 255).astype(np.uint8)
        return out

    # -------------------------
    # Public API: apply to MoviePy clip (in-memory)
    # -------------------------
    def apply_to_moviepy_clip(self, clip, effect: str = "teal_orange", intensity: float = 0.9,
                              sample_mask_every_n_frames: int = 6, feather_ksize: int = 31,
                              temp_lowres: bool = True, target_width: Optional[int] = 1280):
        """
        clip: MoviePy VideoFileClip or VideoClip
        sample_mask_every_n_frames: compute mask every N frames, reuse for intermediate frames (performance)
        temp_lowres: if True, process region at half-res internally (speed)
        target_width: if set, frames wider than this will be resized for processing then upscaled back
        returns a MoviePy VideoClip that processes frames via fl_image
        """
        fps = clip.fps or 30
        frame_cache = {}
        last_mask_info = {"mask": None, "frame_idx": -9999}

        def process_frame_rgb(frame_rgb, t=None):
            # MoviePy supplies RGB frames; convert to BGR
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return _process_frame_bgr(bgr, t)

        def _process_frame_bgr(bgr, t):
            nonlocal last_mask_info
            # optional resizing for speed
            orig_h, orig_w = bgr.shape[:2]
            proc = bgr
            if target_width is not None and orig_w > target_width:
                scale = target_width / orig_w
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                proc = cv2.resize(proc, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # sample mask strategy
            frame_idx = int(np.round((t or 0) * fps))
            if (frame_idx - last_mask_info["frame_idx"]) >= sample_mask_every_n_frames or last_mask_info["mask"] is None:
                # compute mask on this frame
                mask = self._detect_mask(proc)
                last_mask_info["mask"] = mask
                last_mask_info["frame_idx"] = frame_idx
            else:
                mask = last_mask_info["mask"]

            # optionally operate at half resolution for region processing speed
            if temp_lowres:
                small_proc = cv2.resize(proc, (proc.shape[1]//2, proc.shape[0]//2), interpolation=cv2.INTER_AREA)
                small_mask = cv2.resize(mask, (small_proc.shape[1], small_proc.shape[0]), interpolation=cv2.INTER_NEAREST)
                out_small = self._apply_effect_on_frame(small_proc, effect, intensity, mask_binary=small_mask,
                                                        feather_ksize=max(7, feather_ksize//2))
                proc_out = cv2.resize(out_small, (proc.shape[1], proc.shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                proc_out = self._apply_effect_on_frame(proc, effect, intensity, mask_binary=mask,
                                                      feather_ksize=feather_ksize)

            # upscale back if needed
            if target_width is not None and orig_w > target_width:
                proc_out = cv2.resize(proc_out, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            # return RGB for MoviePy
            rgb_out = cv2.cvtColor(proc_out, cv2.COLOR_BGR2RGB)
            return rgb_out

        # wrapper for MoviePy's fl_image (expects function(frame) or function(get_frame,t))
        return clip.fl_image(process_frame_rgb)

    # -------------------------
    # Public API: apply to file path (reads, processes, writes)
    # -------------------------
    def apply_to_file(self, input_path: str, output_path: str, effect: str = "teal_orange", intensity: float = 0.9,
                      sample_mask_every_n_frames: int = 6, temp_lowres: bool = True, target_width: Optional[int] = 1280,
                      codec: str = "libx264", fps_out: Optional[int] = None):
        """
        Read input_path and write output_path with object-only color effect.
        Useful when you want an output file directly.
        """
        clip = VideoFileClip(input_path)
        fps = fps_out or clip.fps
        processed_clip = self.apply_to_moviepy_clip(clip, effect=effect, intensity=intensity,
                                                    sample_mask_every_n_frames=sample_mask_every_n_frames,
                                                    temp_lowres=temp_lowres, target_width=target_width)
        processed_clip.write_videofile(output_path, codec=codec, fps=fps,
                                       temp_audiofile="temp_audio_m4a.m4a", remove_temp=True,
                                       audio_codec="aac", threads=4,
                                       ffmpeg_params=["-movflags", "+faststart", "-crf", "18", "-pix_fmt", "yuv420p"])
        clip.close()
        processed_clip.close()
        return output_path


# ---------------------------
# Convenience factory
# ---------------------------
def make_universal_fx(model_path: Optional[str] = None, device: Optional[str] = None):
    return UniversalAIColorFX20(model_path=model_path or "yolov8n-seg.pt", device=device or _device)


## ---------------------------
# Public callable API
# ---------------------------

def load_fx(model_path: str = "yolov8n-seg.pt", device: Optional[str] = None):
    """
    Returns an initialized UniversalAIColorFX20 object.
    Example:
        fx = load_fx()
    """
    return UniversalAIColorFX20(model_path=model_path, device=device)


import numpy as np
import cv2
import cv2
import numpy as np

def auto_pick_effect_from_frame(frame):
    """Decides best effect for a single frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_mean = hsv[:, :, 0].mean()
    s_mean = hsv[:, :, 1].mean()
    v_mean = hsv[:, :, 2].mean()

    temp = frame[:, :, 2].mean() - frame[:, :, 0].mean()

    if v_mean < 50:
        return "neon_punch"
    if v_mean > 180:
        return "cinematic_soft"
    if s_mean < 40:
        return "vintage_fade"
    if s_mean > 160:
        return "deep_warm"
    if temp > 20:
        return "cool_blue"
    if temp < -20:
        return "deep_warm"

    return "teal_orange"


def auto_pick_effect(video_path):
    """Samples video and returns best effect."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "teal_orange"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, 6).astype(int)

    effects = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (320, 180))
        effects.append(auto_pick_effect_from_frame(frame))

    cap.release()

    if not effects:
        return "teal_orange"

    # return most common pick
    return max(set(effects), key=effects.count)

from coleff import auto_pick_effect, UniversalAIColorFX20

def auto_apply_filter_to_clip_dynamic(clip, video_path, strength=0.9):
    """
    Automatically selects the best effect from a video and applies it
    using the UniversalAIColorFX20 engine on a MoviePy clip.
    
    clip: MoviePy VideoFileClip or VideoClip
    video_path: path to the video file to sample for auto effect selection
    intensity: float, 0-1, strength of the effect
    """
    # Initialize the FX engine
    fx = UniversalAIColorFX20()

    # Pick the best effect from video
    chosen_effect = auto_pick_effect(video_path)
    print(f"🎨 AutoFX → Selected: {chosen_effect}")

    # Apply effect directly on the MoviePy clip
    return fx.apply_to_moviepy_clip(clip, effect=chosen_effect, intensity=strength)



