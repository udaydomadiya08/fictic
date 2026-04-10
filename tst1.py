import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ---- CONFIG ----
input_video = "/Users/uday/Downloads/fictic/out/final_edit_9x16_20251109_160931.mp4"
output_video = "output_watermark_fixed.mp4"
text = "UdayEditz"
font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
font_size = 120
opacity = 0.8
alpha = 0.9  # smoothing

# ---- YOLO MODEL ----
model = YOLO("yolov8s-seg.pt")

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (W, H))

prev_center = None
print("Processing watermark...")

# Pre-render text once (bright and visible)
txt_img = Image.new("RGBA", (600, 200), (0, 0, 0, 0))
d = ImageDraw.Draw(txt_img)
font = ImageFont.truetype(font_path, font_size)
d.text((0, 0), text, font=font, fill=(255, 255, 255, int(255 * opacity)))
txt_np = np.array(txt_img)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        x1, y1, x2, y2 = map(int, boxes[np.argmax(areas)])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    else:
        cx, cy = W // 2, H // 2

    # Smooth motion
    if prev_center is not None:
        cx = int(alpha * prev_center[0] + (1 - alpha) * cx)
        cy = int(alpha * prev_center[1] + (1 - alpha) * cy)
    prev_center = (cx, cy)

    # Overlay text image at the smoothed position
    overlay = frame.copy()
    h_t, w_t = txt_np.shape[:2]
    x1, y1 = max(0, cx - w_t // 2), max(0, cy - h_t // 2)
    x2, y2 = min(W, x1 + w_t), min(H, y1 + h_t)

    text_crop = txt_np[0:y2 - y1, 0:x2 - x1]
    if text_crop.shape[0] > 0 and text_crop.shape[1] > 0:
        roi = overlay[y1:y2, x1:x2]
        alpha_mask = text_crop[:, :, 3:] / 255.0
        overlay[y1:y2, x1:x2] = (1 - alpha_mask) * roi + alpha_mask * text_crop[:, :, :3]

    frame = cv2.addWeighted(frame, 0.8, overlay, 0.8, 0)
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Watermark visibly and smoothly follows main object.")
