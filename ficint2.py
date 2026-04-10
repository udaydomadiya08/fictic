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
def draw_animated_text_on_frame_animated(frame, text, t, start, end,
                                         font_path="trajan-pro/TrajanPro-Regular.ttf",
                                         font_size=30,
                                         bottom_margin_ratio=0.05,
                                         pos_ratio_x=0.5):
    """
    Draw animated text on a video frame with:
    - Fixed font size
    - Horizontal reveal animation (grows from left)
    - Bottom-aligned
    - Yellow fill with white stroke
    - No background box
    """
    duration = end - start
    progress = (t - start) / duration
    progress = max(0, min(progress, 1))
    fade_fraction = 0.25  # first 25% of segment = reveal

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    video_w, video_h = img.size
    max_w = int(video_w * 0.8)

    # Wrap text to fit width
    words = text.split()
    lines = []
    line = ""
    for word in words:
        test_line = f"{line} {word}".strip()
        bbox = draw.textbbox((0,0), test_line, font=ImageFont.truetype(font_path, font_size))
        w = bbox[2]-bbox[0]
        if w > max_w:
            lines.append(line)
            line = word
        else:
            line = test_line
    lines.append(line)

    # Draw from bottom
    total_h = len(lines) * font_size
    y_offset = video_h - int(video_h*bottom_margin_ratio) - total_h

    for line in lines:
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0,0), line, font=font)
        w = bbox[2]-bbox[0]
        x = int(video_w*pos_ratio_x - w/2)

        # Horizontal reveal animation
        if progress < fade_fraction:
            reveal_w = int(w * (progress / fade_fraction))
        else:
            reveal_w = w

        if reveal_w > 0:
            # Draw only the revealed portion
            txt_img = Image.new("RGBA", (w, font_size), (0,0,0,0))
            txt_draw = ImageDraw.Draw(txt_img)
            txt_draw.text((0,0), line, font=font,
                          fill=(255,255,0,255),
                          stroke_width=2,
                          stroke_fill=(0,0,0,0))
            txt_array = np.array(txt_img)
            frame_array = np.array(img)
            frame_array[y_offset:y_offset+font_size, x:x+reveal_w, :3] = txt_array[:,:reveal_w,:3]
            img = Image.fromarray(frame_array)

        y_offset += font_size

    return np.array(img)



# 9. BUILD FINAL VIDEO
# -----------------------------
print("Building final video…")
from moviepy.editor import VideoFileClip, VideoClip

# Load original video
base = VideoFileClip(INPUT_VIDEO)
original_audio = base.audio  # keep the audio

# Function to generate frames with animated text
def make_frame(t):
    frame = base.get_frame(t)
    for seg in final_segments:
        if seg["start"] <= t <= seg["end"]:
            frame = draw_animated_text_on_frame_animated(frame, seg["text"], t, seg["start"], seg["end"])
            break
    return frame

# Create animated clip
animated_clip = VideoClip(make_frame, duration=base.duration)

# Add original audio back
animated_clip = animated_clip.set_audio(original_audio)

# Write final video
animated_clip.write_videofile(OUTPUT_VIDEO, fps=24, codec="libx264")



print("\nDONE! 🎬 Corrected dialogue video ready.")
