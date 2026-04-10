# import cv2
# import torch
# import numpy as np
# from ultralytics import YOLO

# # -----------------------------
# # Load YOLO model (person detection)
# # -----------------------------
# yolo_model = YOLO('yolov8n.pt')  # or your trained YOLO weights

# # -----------------------------
# # Load Face Landmark Model
# # -----------------------------
# class Network(torch.nn.Module):
#     def __init__(self,num_classes=136):
#         super().__init__()
#         self.model_name='resnet18'
#         self.model=torch.hub.load('pytorch/vision:v0.14.0', 'resnet18', pretrained=False)
#         self.model.conv1=torch.nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False)
#         self.model.fc=torch.nn.Linear(self.model.fc.in_features,num_classes)
#     def forward(self,x):
#         return self.model(x)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# landmark_model = Network().to(device)
# landmark_model.load_state_dict(torch.load('face_landmarks_best.pth', map_location=device))
# landmark_model.eval()

# # -----------------------------
# # Helper Functions
# # -----------------------------
# def get_face_landmarks(image):
#     """Returns 68 landmarks in (x, y) format scaled to image size"""
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     img_resized = cv2.resize(img, (224, 224))
#     img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)/255.0
#     img_tensor = (img_tensor - 0.5)/0.5
#     img_tensor = img_tensor.to(device)
    
#     with torch.no_grad():
#         pred = landmark_model(img_tensor).cpu().numpy().reshape(68,2)
#     # Scale back to original image size
#     h, w = image.shape[:2]
#     pred[:,0] = pred[:,0] * w
#     pred[:,1] = pred[:,1] * h
#     return pred.astype(np.int32)

# def crop_face_from_landmarks(image, landmarks):
#     x_min, y_min = np.min(landmarks, axis=0)
#     x_max, y_max = np.max(landmarks, axis=0)
#     face_crop = image[y_min:y_max, x_min:x_max]
#     face_landmarks = landmarks - [x_min, y_min]  # adjust landmarks relative to crop
#     return face_crop, face_landmarks, (x_min, y_min, x_max, y_max)

# # -----------------------------
# # Face Swap Function
# # -----------------------------
# def swap_face(src_img, dst_img):
#     # 1️⃣ Detect persons in destination image
#     results = yolo_model(dst_img)[0]
#     boxes = results.boxes.xyxy.cpu().numpy()
#     scores = results.boxes.conf.cpu().numpy()
#     if len(boxes) == 0:
#         print("❌ No person detected in destination image")
#         return dst_img
#     # Pick the person with highest confidence
#     idx = np.argmax(scores)
#     x1, y1, x2, y2 = boxes[idx].astype(int)
#     person_crop = dst_img[y1:y2, x1:x2]
    
#     # 2️⃣ Detect landmarks in destination face
#     dst_landmarks = get_face_landmarks(person_crop)
#     dst_face_crop, dst_face_landmarks, (fx1, fy1, fx2, fy2) = crop_face_from_landmarks(person_crop, dst_landmarks)
    
#     # 3️⃣ Detect landmarks in source face
#     src_landmarks = get_face_landmarks(src_img)
#     src_face_crop, src_face_landmarks, _ = crop_face_from_landmarks(src_img, src_landmarks)
    
#     # Resize source face to match destination face
#     dst_h, dst_w = dst_face_crop.shape[:2]
#     src_face_resized = cv2.resize(src_face_crop, (dst_w, dst_h))
    
#     # 4️⃣ Create mask from destination landmarks
#     mask = np.zeros((dst_h, dst_w), dtype=np.uint8)
#     points = cv2.convexHull(dst_face_landmarks)
#     cv2.fillConvexPoly(mask, points, 255)
    
#     # 5️⃣ Paste source face on destination
#     center = (fx1 + dst_w//2, fy1 + dst_h//2)
#     dst_img_copy = dst_img.copy()
#     dst_img_copy[fy1:fy2, fx1:fx2] = dst_img[fy1:fy2, fx1:fx2]  # ensure background intact
#     output = cv2.seamlessClone(src_face_resized, dst_img, mask, center, cv2.NORMAL_CLONE)
#     return output

# # -----------------------------
# # Example Usage
# # -----------------------------
# src_image = cv2.imread('deadpool.jpg')   # source face
# dst_image = cv2.imread('spiderman.jpeg')   # destination person image

# output = swap_face(src_image, dst_image)
# cv2.imwrite('face_swapped.jpg', output)
# cv2.imshow('Face Swapped', output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import torch

# ------------------------------
# DEVICE
# ------------------------------
def pick_device():
    return "mps" if torch.backends.mps.is_available() else \
           "cuda" if torch.cuda.is_available() else "cpu"

device = pick_device()

# ------------------------------
# YOLO Head Detector
# ------------------------------
yolo_pose = YOLO("yolo11x-pose.pt")

# ------------------------------
# Load SAM
# ------------------------------
sam = sam_model_registry["vit_b"](checkpoint="sam_b.pth")
sam.to(device)
predictor = SamPredictor(sam)

# ------------------------------
# KEEP ONLY THE LARGEST OBJECT
# ------------------------------
def keep_largest(mask_255):
    num, cc, stats, _ = cv2.connectedComponentsWithStats(mask_255, connectivity=8)

    if num <= 1:
        return mask_255

    sizes = stats[1:, cv2.CC_STAT_AREA]     # ignore background
    biggest_idx = 1 + np.argmax(sizes)      # +1 because background is index 0

    clean_mask = np.where(cc == biggest_idx, 255, 0).astype("uint8")
    return clean_mask

# ------------------------------
# REFINEMENT (optional)
# ------------------------------
def refine(mask):
    k = np.ones((15,15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

# ------------------------------
# PROCESS IMAGE
# ------------------------------
img = cv2.imread("deadpool.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H, W = rgb.shape[:2]

# ------------------------------
# YOLO → head keypoint
# ------------------------------
res = yolo_pose(rgb)[0]
x_head, y_head = res.keypoints.xy[0][0].cpu().numpy().astype(int)

# ------------------------------
# CROP HEAD REGION
# ------------------------------
BOX_SIZE = 260
UP = 40
DOWN = 40

x1 = max(x_head - BOX_SIZE//2, 0)
x2 = min(x_head + BOX_SIZE//2, W)

y1 = max(y_head - BOX_SIZE//2 - UP, 0)
y2 = min(y_head + BOX_SIZE//2 - UP + DOWN, H)

crop_rgb = rgb[y1:y2, x1:x2]

# ------------------------------
# SAM segmentation
# ------------------------------
predictor.set_image(crop_rgb)
h, w = crop_rgb.shape[:2]

box = np.array([0, 0, w, h], dtype=np.float32)

masks, scores, _ = predictor.predict(
    box=box[None, :],
    multimask_output=True
)

mask = (masks[scores.argmax()] * 255).astype("uint8")

# ------------------------------
# REMOVE BACKGROUND COMPLETELY
# ------------------------------
mask = keep_largest(mask)
mask = refine(mask)

# ------------------------------
# BUILD FINAL PNG WITH ALPHA
# ------------------------------
final_rgba = np.dstack([
    crop_rgb[:,:,::-1],   # RGB → BGR
    mask
])

cv2.imwrite("segmented_head.png", final_rgba)
print("Saved clean segmentation → segmented_head.png")

cv2.imshow("CLEAN HEAD", final_rgba)
cv2.waitKey(0)
cv2.destroyAllWindows()
