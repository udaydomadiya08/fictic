import os
import random
from moviepy.editor import VideoFileClip, ColorClip, concatenate_videoclips

# import transition functions
from transit4 import replace_with_reverse_safe, transition_person_static
from transit import YoloSamSeg

# ------------------------
# Helpers
# ------------------------
def safe_load(p):
    return VideoFileClip(p, audio=False)

def is_valid(path):
    try:
        c = VideoFileClip(path, audio=False)
        c.get_frame(0)
        c.close()
        return True
    except:
        return False

# ------------------------
# Gather files
# ------------------------
folder = "firstclip"
paths = sorted([
    os.path.join(folder, f)
    for f in os.listdir(folder)
    if f.lower().endswith((".mp4", ".mov", ".mkv"))
])

valid = []
for p in paths:
    print("Checking:", p)
    if is_valid(p):
        print("   ✔ GOOD")
        valid.append(p)
    else:
        print("   ❌ BAD — removed")

if len(valid) < 2:
    raise ValueError("Need at least 2 usable videos")

# ------------------------
# Preload metadata
# ------------------------
clips_meta = []
for p in valid:
    c = safe_load(p)
    clips_meta.append((p, c.size, c.fps, max(0.01, c.duration - 0.02)))
    c.close()

print(f"\n🎥 Valid clips: {len(clips_meta)}")

# ------------------------
# Init segmenter
# ------------------------
segmenter = YoloSamSeg(
    yolo_model="yolov8n.pt",
    sam_checkpoint="sam_b.pth"
)

# ------------------------
# Preload background clips
# ------------------------
print("Loading bg clips for transitions...")
bg_clips = [
    safe_load(p).subclip(0, clips_meta[i][3])
    for i, p in enumerate(valid)
]
print(f"Loaded {len(bg_clips)} bg clips")

# ------------------------
# Build stitched sequence
# ------------------------
stitched = []

for i in range(len(clips_meta) - 1):
    p1, size1, fps1, d1 = clips_meta[i]
    p2, size2, fps2, d2 = clips_meta[i + 1]

    print(f"\nProcessing {i} → {i+1}")

    # fresh isolated handles
    c1 = safe_load(p1).subclip(0, d1)
    c2 = safe_load(p2).subclip(0, d2)

    try:
        # 80% chance reverse, 20% person-static
        if random.random() < 0.8:
            print("  🔁 Reverse transition")

            out = replace_with_reverse_safe(
                safe_load(p1).subclip(0, d1),
                safe_load(p2).subclip(0, d2)
            )

            if out is c2:
                print("   ➡ reverse fallback (no transition)")
            else:
                print("   ✔ Reverse transition applied")

        else:
            print("  🧍 Person-static transition")

            out = transition_person_static(
                c1,        # previous clip basis
                c2,        # clip to segment
                bg_clips,  # background motion clips
                segmenter,
                person_effects=True,
                bg_effects=True,
                bg_blur_k=60,
                speed_multiplier=11
            )

            if out is c2:
                print("   ➡ person-static fallback")
            else:
                print("   ✔ Person-static transition applied")

    except Exception as e:
        print("⛔ Transition failed:", e)
        out = safe_load(p2).subclip(0, d2)

    stitched.append(out)

    c1.close()
    c2.close()

# ------------------------
# Validate stitched clips
# ------------------------
final_clips = []
print("\n🔍 Final validation:")
for idx, clip in enumerate(stitched):
    try:
        clip.get_frame(0)
        final_clips.append(clip)
        print(f"  ✔ clip {idx} OK")
    except:
        print(f"  ❌ clip {idx} dropped")

# add the first video
first = safe_load(valid[0]).subclip(0, clips_meta[0][3])
timeline = [first] + final_clips

# ------------------------
# Export final video
# ------------------------
final = concatenate_videoclips(timeline, method="compose")
final.write_videofile("test_output.mp4", fps=30, codec="libx264", audio=False)

print("\n✅ DONE — FULL UPDATED SCRIPT READY")
