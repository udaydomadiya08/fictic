import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ------------------------------------------------
# LOAD SAM MODEL
# ------------------------------------------------
sam_checkpoint = "sam_vit_h.pth"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device="cuda" if torch.cuda.is_available() else "cpu")

# Create SAM mask generator
mask_generator = SamAutomaticMaskGenerator(sam)

# ------------------------------------------------
# INPUT IMAGES
# ------------------------------------------------
original_path = "/Users/uday/Downloads/fictic/deadpool.jpg"
new_path = "/Users/uday/Downloads/fictic/spiderman.jpeg"

original = cv2.imread(original_path)
new_person = cv2.imread(new_path)

print("Original loaded:", original is not None)
print("New person loaded:", new_person is not None)

if original is None or new_person is None:
    raise RuntimeError("❌ Image paths wrong! Check the paths or file extensions.")

# ------------------------------------------------
# SEGMENT PERSON IN BOTH IMAGES
# ------------------------------------------------
def get_primary_mask(image):
    masks = mask_generator.generate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if len(masks) == 0:
        return None
    # Take the largest mask (assuming that's the person)
    largest_mask = max(masks, key=lambda x: x["segmentation"].sum())
    return (largest_mask["segmentation"] * 255).astype(np.uint8)

old_mask = get_primary_mask(original)
new_mask = get_primary_mask(new_person)

# ------------------------------------------------
# CHECK IF MASKS ARE EMPTY
# ------------------------------------------------
if old_mask is None or old_mask.sum() == 0:
    raise RuntimeError("❌ Old person mask is empty! Cannot replace.")
if new_mask is None or new_mask.sum() == 0:
    raise RuntimeError("❌ New person mask is empty! Cannot replace.")

# ------------------------------------------------
# CUT NEW PERSON CLEANLY
# ------------------------------------------------
ys, xs = np.where(new_mask == 255)
y1, y2 = ys.min(), ys.max()
x1, x2 = xs.min(), xs.max()

new_cut = cv2.bitwise_and(new_person, new_person, mask=new_mask)
new_cut = new_cut[y1:y2, x1:x2]

# ------------------------------------------------
# REMOVE OLD PERSON FROM ORIGINAL
# ------------------------------------------------
ys_old, xs_old = np.where(old_mask == 255)
oy1, oy2 = ys_old.min(), ys_old.max()
ox1, ox2 = xs_old.min(), xs_old.max()

original_clean = original.copy()
original_clean[old_mask == 255] = 0

old_h = oy2 - oy1
old_w = ox2 - ox1

# ------------------------------------------------
# RESIZE NEW PERSON TO MATCH OLD SIZE
# ------------------------------------------------
new_resized = cv2.resize(new_cut, (old_w, old_h), interpolation=cv2.INTER_AREA)

# ------------------------------------------------
# PASTE NEW PERSON
# ------------------------------------------------
roi = original_clean[oy1:oy2, ox1:ox2]

mask_gray = cv2.cvtColor(new_resized, cv2.COLOR_BGR2GRAY)
_, mask_bin = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask_bin)

bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
fg = cv2.bitwise_and(new_resized, new_resized, mask=mask_bin)

final = cv2.add(bg, fg)
original_clean[oy1:oy2, ox1:ox2] = final

# ------------------------------------------------
# SAVE RESULT
# ------------------------------------------------
output_path = "/Users/uday/Downloads/full_body_replaced.jpg"
cv2.imwrite(output_path, original_clean)
print(f"✅ Full body replaced successfully → {output_path}")