
# import cv2
# import numpy as np
# from moviepy.editor import VideoFileClip, ImageSequenceClip

# # ---------------- ULTRA-SMOOTH SETTINGS ----------------
# OUT_W = 1080
# OUT_H = 1920

# ALPHA = 0.03           # very strong smoothing (NO JITTER)
# MIN_MOVE = 2           # ignore tiny micro-movements
# HISTORY_LEN = 15       # long moving average = buttery smooth
# MAX_DELTA = 8          # max movement allowed per frame

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
#                                      'haarcascade_frontalface_default.xml')

# # ---------------- HELPER FUNCTIONS ----------------
# def clamp(prev, curr, max_delta=MAX_DELTA):
#     if prev is None:
#         return curr
#     px1, py1, px2, py2 = prev
#     cx1, cy1, cx2, cy2 = curr
#     return (
#         int(px1 + np.clip(cx1 - px1, -max_delta, max_delta)),
#         int(py1 + np.clip(cy1 - py1, -max_delta, max_delta)),
#         int(px2 + np.clip(cx2 - px2, -max_delta, max_delta)),
#         int(py2 + np.clip(cy2 - py2, -max_delta, max_delta))
#     )

# def smooth_box(prev, current, alpha=ALPHA, min_move=MIN_MOVE):
#     if prev is None:
#         return current

#     dx = abs(prev[0] - current[0])
#     dy = abs(prev[1] - current[1])
#     dw = abs(prev[2] - current[2])
#     dh = abs(prev[3] - current[3])

#     if dx < min_move and dy < min_move and dw < min_move and dh < min_move:
#         return prev

#     x1 = int(prev[0]*(1-alpha) + current[0]*alpha)
#     y1 = int(prev[1]*(1-alpha) + current[1]*alpha)
#     x2 = int(prev[2]*(1-alpha) + current[2]*alpha)
#     y2 = int(prev[3]*(1-alpha) + current[3]*alpha)

#     return (x1, y1, x2, y2)

# def smooth_box_history(current, history, max_len=HISTORY_LEN):
#     history.append(current)
#     if len(history) > max_len:
#         history.pop(0)
#     x1 = int(np.mean([b[0] for b in history]))
#     y1 = int(np.mean([b[1] for b in history]))
#     x2 = int(np.mean([b[2] for b in history]))
#     y2 = int(np.mean([b[3] for b in history]))
#     return (x1, y1, x2, y2)

# def smart_full_crop(frame, prev_box=None, history=None):
#     h, w, _ = frame.shape
#     target_ratio = OUT_W / OUT_H

#     if w/h > target_ratio:
#         crop_h = h
#         crop_w = int(h * target_ratio)
#         x1 = (w - crop_w)//2
#         y1 = 0
#     else:
#         crop_w = w
#         crop_h = int(w / target_ratio)
#         x1 = 0
#         y1 = (h - crop_h)//2

#     # detect face
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.1, 5)

#     if len(faces) > 0:
#         x, y, fw, fh = max(faces, key=lambda b: b[2]*b[3])
#         face_cx = x + fw//2
#         face_cy = y + fh//2
#         x1 = max(0, min(w - crop_w, face_cx - crop_w//2))
#         y1 = max(0, min(h - crop_h, face_cy - crop_h//2))

#     box = (x1, y1, x1 + crop_w, y1 + crop_h)

#     # 1. exponential smoothing
#     box = smooth_box(prev_box, box)

#     # 2. clamp sudden movements
#     box = clamp(prev_box, box)

#     # 3. long moving average smoothing
#     if history is not None:
#         box = smooth_box_history(box, history)

#     x1, y1, x2, y2 = box
#     crop = frame[y1:y2, x1:x2]
#     crop_resized = cv2.resize(crop, (OUT_W, OUT_H), cv2.INTER_CUBIC)

#     return crop_resized, box

# # ---------------- MAIN FUNCTION ----------------
# def smart_crop_video(input_path, output_path):
#     cap = cv2.VideoCapture(input_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frames = []
#     prev_box = None
#     history = []

#     print("Processing frames...")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         crop_frame, prev_box = smart_full_crop(frame, prev_box, history)
#         frames.append(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB))

#     cap.release()
#     print(f"Processed {len(frames)} frames")

#     original_clip = VideoFileClip(input_path)
#     clip = ImageSequenceClip(frames, fps=fps)
#     clip = clip.set_audio(original_clip.audio)

#     print("Exporting ultra-smooth high-quality video...")
#     clip.write_videofile(
#         output_path,
#         codec="libx264",
#         audio_codec="aac",
#         preset="slow",
#         ffmpeg_params=["-crf", "16", "-pix_fmt", "yuv420p"]
#     )

#     print(f"Ultra-smooth smart crop saved → {output_path}")


import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip
import argparse
import math

# ===============================
# OUTPUT SETTINGS
# ===============================
OUT_W = 1080
OUT_H = 1920

# ===============================
# STABILIZATION TUNING
# ===============================
ALPHA = 0.03           # smoothing inside smooth_box (used for box interpolation)
MIN_MOVE = 2
HISTORY_LEN = 15
MAX_DELTA = 12         # max pixels a box side can move per frame (clamp)
FACE_WEIGHT = 0.75     # how strongly we follow face vs flow/center
DEADZONE_FLOW = 1.0    # ignore extremely tiny flow values

# face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ===============================
# OPTICAL FLOW (Lucas-Kanade) BASED BOX SHIFT
# ===============================
def flow_smooth(prev_frame, curr_frame, prev_box, step=20, lk_win=(15, 15), lk_max_level=2):
    """
    Estimate average motion inside prev_box using sparse LK and shift the box by that motion.
    Returns a new_box (x1,y1,x2,y2). If not enough tracks, returns prev_box unchanged.
    """
    if prev_frame is None or prev_box is None:
        return prev_box

    x1, y1, x2, y2 = map(int, prev_box)
    h_box = max(2, y2 - y1)
    w_box = max(2, x2 - x1)

    # Create a grid of points inside the previous box
    pts = []
    for yy in range(y1, y2, step):
        for xx in range(x1, x2, step):
            pts.append([xx, yy])
    if len(pts) == 0:
        return prev_box

    p0 = np.float32(pts).reshape(-1, 1, 2)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # calcOpticalFlowPyrLK
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, p0, None,
        winSize=lk_win, maxLevel=lk_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    )

    if p1 is None or st is None:
        return prev_box

    st = st.reshape(-1)
    valid_old = p0.reshape(-1, 2)[st == 1]
    valid_new = p1.reshape(-1, 2)[st == 1]

    if len(valid_old) < max(3, len(p0) // 6):  # require a few good points
        return prev_box

    shifts = valid_new - valid_old
    mean_shift = np.mean(shifts, axis=0)
    dx, dy = float(mean_shift[0]), float(mean_shift[1])

    # ignore tiny flow
    if math.hypot(dx, dy) < DEADZONE_FLOW:
        return prev_box

    # shift the box
    nx1 = int(round(x1 + dx))
    ny1 = int(round(y1 + dy))
    nx2 = int(round(x2 + dx))
    ny2 = int(round(y2 + dy))

    return (nx1, ny1, nx2, ny2)

# ===============================
# BOX SMOOTHING / HISTORY / CLAMP
# ===============================
def clamp(prev, curr, max_delta=MAX_DELTA):
    """
    Clamp each coordinate movement relative to prev by max_delta.
    prev may be None.
    """
    if prev is None:
        return curr
    px1, py1, px2, py2 = prev
    cx1, cy1, cx2, cy2 = curr
    return (
        int(px1 + np.clip(cx1 - px1, -max_delta, max_delta)),
        int(py1 + np.clip(cy1 - py1, -max_delta, max_delta)),
        int(px2 + np.clip(cx2 - px2, -max_delta, max_delta)),
        int(py2 + np.clip(cy2 - py2, -max_delta, max_delta)),
    )

def smooth_box(prev, current, alpha=ALPHA, min_move=MIN_MOVE):
    """
    Exponential smoothing of all four box coordinates.
    """
    if prev is None:
        return current

    dx = abs(prev[0] - current[0])
    dy = abs(prev[1] - current[1])
    dw = abs(prev[2] - current[2])
    dh = abs(prev[3] - current[3])

    if dx < min_move and dy < min_move and dw < min_move and dh < min_move:
        return prev

    x1 = int(prev[0] * (1 - alpha) + current[0] * alpha)
    y1 = int(prev[1] * (1 - alpha) + current[1] * alpha)
    x2 = int(prev[2] * (1 - alpha) + current[2] * alpha)
    y2 = int(prev[3] * (1 - alpha) + current[3] * alpha)
    return (x1, y1, x2, y2)

def smooth_box_history(current, history, max_len=HISTORY_LEN):
    """
    Moving average over previous boxes (history contains past boxes).
    """
    history.append(current)
    if len(history) > max_len:
        history.pop(0)
    x1 = int(np.mean([b[0] for b in history]))
    y1 = int(np.mean([b[1] for b in history]))
    x2 = int(np.mean([b[2] for b in history]))
    y2 = int(np.mean([b[3] for b in history]))
    return (x1, y1, x2, y2)

# ===============================
# FACE DETECTION
# ===============================
def detect_face_center(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 6)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return (x + w // 2, y + h // 2)

# ===============================
# SMART FULL CROP (integrates flow + face + smoothing)
# ===============================
def smart_full_crop(frame, prev_box=None, history=None, prev_frame_for_flow=None):
    """
    frame: current BGR frame (numpy)
    prev_box: previous smoothed box tuple or None
    history: list for history smoothing
    prev_frame_for_flow: previous BGR frame for optical flow estimation
    returns: (crop_resized, new_box, curr_frame)  -> curr_frame to be used next iteration as prev_frame_for_flow
    """

    h, w = frame.shape[:2]
    target_ratio = OUT_W / OUT_H

    # --- base center crop (aspect-correct)
    if w / h > target_ratio:
        crop_h = h
        crop_w = int(h * target_ratio)
        base_x1 = (w - crop_w) // 2
        base_y1 = 0
    else:
        crop_w = w
        crop_h = int(w / target_ratio)
        base_x1 = 0
        base_y1 = (h - crop_h) // 2

    # choose face-centered candidate
    face_center = detect_face_center(frame)
    if face_center:
        face_cx, face_cy = face_center
        face_x1 = int(face_cx - crop_w // 2)
        face_y1 = int(face_cy - crop_h // 2)
        face_x1 = max(0, min(w - crop_w, face_x1))
        face_y1 = max(0, min(h - crop_h, face_y1))
        face_box = (face_x1, face_y1, face_x1 + crop_w, face_y1 + crop_h)
    else:
        face_box = None

    # base box (center)
    base_box = (base_x1, base_y1, base_x1 + crop_w, base_y1 + crop_h)
    box = base_box

    # If previous box exists, use optical flow to nudge it
    if prev_box is not None and prev_frame_for_flow is not None:
        try:
            flow_box = flow_smooth(prev_frame_for_flow, frame, prev_box)
            if flow_box is not None:
                box = flow_box
        except Exception:
            # fallback to prev_box if flow fails
            box = prev_box

    # If face exists, blend face_box and current box by FACE_WEIGHT
    if face_box is not None:
        bx1, by1, bx2, by2 = box
        fx1, fy1, fx2, fy2 = face_box
        blended = (
            int(FACE_WEIGHT * fx1 + (1 - FACE_WEIGHT) * bx1),
            int(FACE_WEIGHT * fy1 + (1 - FACE_WEIGHT) * by1),
            int(FACE_WEIGHT * fx2 + (1 - FACE_WEIGHT) * bx2),
            int(FACE_WEIGHT * fy2 + (1 - FACE_WEIGHT) * by2),
        )
        box = blended

    # Smooth the box coordinates (exponential)
    box = smooth_box(prev_box, box)

    # Clamp movement relative to prev_box
    box = clamp(prev_box, box)

    # Apply long-history smoothing
    if history is not None:
        box = smooth_box_history(box, history)

    # Ensure box inside frame
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 2, min(w, x2))
    y2 = max(y1 + 2, min(h, y2))
    box = (x1, y1, x2, y2)

    # final crop + resize to exact output
    crop = frame[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)

    return crop_resized, box, frame

# ===============================
# MAIN PIPELINE
# ===============================
def smart_crop_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("❌ Couldn't open input:", input_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    prev_box = None
    history = []
    prev_frame_for_flow = None

    print("Processing frames...")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # smart crop using prev_frame_for_flow for flow calculations
        crop_frame, prev_box, prev_frame_for_flow = smart_full_crop(
            frame, prev_box=prev_box, history=history, prev_frame_for_flow=prev_frame_for_flow
        )

        frames.append(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB))
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  processed {frame_count} frames...")

    cap.release()
    print(f"✔ Processed {len(frames)} frames")

    # export with original audio
    original_clip = VideoFileClip(input_path)
    clip = ImageSequenceClip(frames, fps=fps)
    clip = clip.set_audio(original_clip.audio)

    print("Exporting ultra-smooth high-quality video...")
    clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        preset="slow",
        ffmpeg_params=["-crf", "14", "-pix_fmt", "yuv420p"],
    )

    print(f"🔥 Ultra-smooth smart crop saved → {output_path}")

# ===============================
# CLI
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultra-smooth smart crop (face + flow + smoothing)")
    parser.add_argument("--i", "--input", dest="input", required=True, help="Input video path")
    parser.add_argument("--o", "--output", dest="output", required=True, help="Output mp4 path")
    args = parser.parse_args()

    smart_crop_video(args.input, args.output)






