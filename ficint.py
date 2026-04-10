import json
from moviepy.editor import *
import whisper
from gemini_resp import generate_gemini_response

# -----------------------------
# 1. PATHS
# -----------------------------
INPUT_VIDEO = "/Users/uday/Downloads/fictic/best_scene.mp4"
OUTPUT_VIDEO = "fictic_final.mp4"

# -----------------------------
# 2. LOAD WHISPER MODEL
# -----------------------------
print("Loading Whisper (medium)…")
model = whisper.load_model("medium")

# -----------------------------
# 3. TRANSCRIBE VIDEO
# -----------------------------
print("Transcribing video with Whisper…")
raw = model.transcribe(INPUT_VIDEO, word_timestamps=True)
segments = raw["segments"]  # exact Whisper segments

print(f"Found {len(segments)} Whisper segments.\n")

# -----------------------------
# 4. BUILD FULL CONTEXT FOR GEMINI
# -----------------------------
full_context = "\n".join([seg["text"].strip() for seg in segments if seg.get("text")])

# -----------------------------
# 5. GEMINI: LINE-BY-LINE CORRECTION
# -----------------------------
def correct_line_by_line(text, original_lines):
    """
    Sends full context to Gemini but returns a list of corrected lines 
    corresponding to original Whisper segments.
    """
    prompt = f"""
You are an ASR (Automatic Speech Recognition) correction model.

TASK:
- Correct the entire transcript below using full context.
- Preserve **exact spoken style** and meaning.
- DO NOT merge or split lines.
- DO NOT rewrite, summarize, or make cinematic.
- Keep the same number of lines as original.
- Each line must correspond **exactly** to the line in the input.

Full Whisper transcript (lines separated by newlines):
{text}

Return ONLY the corrected transcript, line by line, in the SAME order as input.
"""
    response = generate_gemini_response(prompt).text.strip()
    # Split Gemini output by line
    corrected_lines = response.split("\n")
    # Clean empty lines
    corrected_lines = [line.strip() for line in corrected_lines if line.strip()]
    
    # If Gemini returned fewer lines, pad with original
    if len(corrected_lines) < len(original_lines):
        corrected_lines += original_lines[len(corrected_lines):]
    # If Gemini returned more lines, truncate
    elif len(corrected_lines) > len(original_lines):
        corrected_lines = corrected_lines[:len(original_lines)]
    
    return corrected_lines

# -----------------------------
# 6. GET CORRECTED LINES
# -----------------------------
original_lines = [seg["text"].strip() for seg in segments if seg.get("text")]
print("Sending full transcript to Gemini for line-by-line correction…")
corrected_lines = correct_line_by_line(full_context, original_lines)

# -----------------------------
# 7. BUILD FINAL SEGMENTS
# -----------------------------
final_segments = []
for i, seg in enumerate(segments):
    final_segments.append({
        "start": seg["start"],
        "end": seg["end"],
        "text": corrected_lines[i] if i < len(corrected_lines) else seg["text"]
    })

# -----------------------------
# 8. FICTIC TEXT ANIMATION
# -----------------------------
from moviepy.video.VideoClip import TextClip, VideoClip
from moviepy.editor import VideoClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap

from moviepy.editor import VideoClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def fictic_text_clip(text, start, end, video_w, video_h, font_path="trajan-pro/TrajanPro-Regular.ttf", font_size=70):


    duration = end - start
    font = ImageFont.truetype(font_path, font_size)

    # Pre-render full text
    txt_img = Image.new("RGBA", (video_w, video_h), (0,0,0,0))
    draw = ImageDraw.Draw(txt_img)

    # Compute text size using textbbox
    bbox = draw.textbbox((0,0), text, font=font, stroke_width=2)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x = (video_w - w)//2
    y = int(video_h*0.65)  # bottom-ish
    draw.text((x, y), text, font=font, fill=(255,255,0,255), stroke_width=2, stroke_fill=(255,255,255,255))

    txt_frame = np.array(txt_img)

    def make_frame(t):
        progress = t / duration
        fade_fraction = 0.25
        if progress < fade_fraction:
            reveal_w = int(video_w * (progress/fade_fraction))
        else:
            reveal_w = video_w

        frame = np.zeros_like(txt_frame)
        if reveal_w > 0:
            frame[:, :reveal_w, :] = txt_frame[:, :reveal_w, :]
        
        # Convert RGBA → RGB for MoviePy
        frame_rgb = frame[..., :3]
        return frame_rgb


    return VideoClip(make_frame, duration=duration).set_start(start)


# -----------------------------
# 9. BUILD FINAL VIDEO
# -----------------------------
print("Building final video…")
base = VideoFileClip(INPUT_VIDEO)
video_w, video_h = base.size
overlays = []

for seg in final_segments:
    if seg["text"].strip():
        overlays.append(fictic_text_clip(seg["text"], seg["start"], seg["end"], video_w, video_h))

final_video = CompositeVideoClip([base] + overlays)
final_video.write_videofile(OUTPUT_VIDEO, fps=24, codec="libx264")


print("\nDONE! 🎬 Corrected dialogue video ready.")
