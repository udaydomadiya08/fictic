# fictic_editor.py

import json
import gc
import tempfile
import os

import whisper
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, VideoClip

from gemini_resp import generate_gemini_response


# Configurable constants; you can parameterize these later if you want
INPUT_VIDEO = "/Users/uday/Downloads/fictic/best_scene.mp4"
OUTPUT_VIDEO = "fictic_final.mp4"
FONT_PATH = "trajan-pro/TrajanPro-Regular.ttf"
FONT_SIZE = 80


def load_whisper_model():
    print("Loading Whisper (tiny)…")
    # Keep tiny to minimize RAM usage
    model = whisper.load_model("tiny")
    return model


def transcribe_video(model, input_path):
    """
    Runs Whisper and returns segments.
    SILENT if no speech is detected.
    """
    raw = model.transcribe(input_path, word_timestamps=True)
    segments = raw.get("segments", [])

    # 🔇 If Whisper detects no speech, return empty silently
    if not segments:
        return []

    return segments



def build_full_context(segments):
    return "\n".join([seg["text"].strip() for seg in segments if seg.get("text")])


def correct_line_by_line(text, original_lines):
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
    corrected_lines = response.split("\n")
    corrected_lines = [line.strip() for line in corrected_lines if line.strip()]

    if len(corrected_lines) < len(original_lines):
        corrected_lines += original_lines[len(corrected_lines):]
    elif len(corrected_lines) > len(original_lines):
        corrected_lines = corrected_lines[:len(original_lines)]

    return corrected_lines


def build_final_segments(segments, corrected_lines):
    final_segments = []
    for i, seg in enumerate(segments):
        final_segments.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": corrected_lines[i] if i < len(corrected_lines) else seg["text"],
            }
        )
    return final_segments


def draw_animated_text_on_frame_animated(
    frame,
    text,
    t,
    start,
    end,
    font_path=FONT_PATH,
    font_size=FONT_SIZE,
    bottom_margin_ratio=0.05,
    pos_ratio_x=0.5,
):
    duration = end - start
    progress = (t - start) / duration
    progress = max(0, min(progress, 1))
    fade_fraction = 0.25  # first 25% of segment = reveal

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    video_w, video_h = img.size
    max_w = int(video_w * 0.8)

    words = text.split()
    lines = []
    line = ""
    for word in words:
        test_line = f"{line} {word}".strip()
        bbox = draw.textbbox(
            (0, 0), test_line, font=ImageFont.truetype(font_path, font_size)
        )
        w = bbox[2] - bbox[0]
        if w > max_w:
            lines.append(line)
            line = word
        else:
            line = test_line
    lines.append(line)

    total_h = len(lines) * font_size
    y_offset = video_h - int(video_h * bottom_margin_ratio) - total_h

    for line in lines:
        font = ImageFont.truetype(font_path, font_size)
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        x = int(video_w * pos_ratio_x - w / 2)

        if progress < fade_fraction:
            reveal_w = int(w * (progress / fade_fraction))
        else:
            reveal_w = w

        if reveal_w > 0:
            txt_img = Image.new("RGBA", (w, font_size), (0, 0, 0, 0))
            txt_draw = ImageDraw.Draw(txt_img)
            txt_draw.text(
                (0, 0),
                line,
                font=font,
                fill=(255, 255, 0, 255),
                stroke_width=2,
                stroke_fill=(0, 0, 0, 0),
            )
            txt_array = np.array(txt_img)
            frame_array = np.array(img)
            frame_array[
                y_offset : y_offset + font_size, x : x + reveal_w, :3
            ] = txt_array[:, :reveal_w, :3]
            img = Image.fromarray(frame_array)

        y_offset += font_size

    return np.array(img)


def build_final_video_clip(input_video_path, final_segments):
    """
    Build, write temp video, reopen as VideoFileClip.
    Required for later MoviePy filters to work.
    """
    print("Building final video clip object…")

    base = VideoFileClip(input_video_path)
    original_audio = base.audio

    def make_frame(t):
        frame = base.get_frame(t)
        for seg in final_segments:
            if seg["start"] <= t <= seg["end"]:
                frame = draw_animated_text_on_frame_animated(
                    frame, seg["text"], t, seg["start"], seg["end"]
                )
                break
        return frame

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    animated_clip = VideoClip(make_frame, duration=base.duration)
    animated_clip = animated_clip.set_audio(original_audio)

    animated_clip.write_videofile(
        temp_out,
        codec="libx264",
        audio_codec="aac",
        fps=base.fps,
        verbose=False,
        logger=None,
    )

    # Release animated/base resources
    try:
        if animated_clip.audio:
            animated_clip.audio.close()
    except Exception:
        pass
    try:
        animated_clip.close()
    except Exception:
        pass
    try:
        if original_audio:
            original_audio.close()
    except Exception:
        pass
    try:
        base.close()
    except Exception:
        pass

    final_clip = VideoFileClip(temp_out)
    return final_clip

def process_video_return_clip(input_source):
    """
    Accepts either:
      - a file path (string)
      - a MoviePy VideoFileClip object

    Returns:
      - Final MoviePy VideoClip with animated subtitles (if speech exists)

    Whisper is COMPLETELY SILENT when:
      - no audio
      - audio but no speech
      - Whisper fails
    """

    from moviepy.editor import VideoFileClip
    import tempfile
    import gc
    import os

    # 1️⃣ Handle input type
    owned_base = False
    if isinstance(input_source, str):
        base_clip = VideoFileClip(input_source)
        owned_base = True
    else:
        base_clip = input_source

    # 2️⃣ Ensure FPS exists
    fps = getattr(base_clip, "fps", None)
    if fps is None:
        fps = 30
        base_clip = base_clip.set_fps(fps)

    # 3️⃣ Export safe temp video
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video_path = tmp.name
    tmp.close()

    audio_codec = "aac" if base_clip.audio else None

    base_clip.write_videofile(
        temp_video_path,
        codec="libx264",
        audio_codec=audio_codec,
        fps=fps,
        temp_audiofile=temp_video_path.replace(".mp4", "_audio.m4a"),
        remove_temp=True,
        threads=4,
        preset="medium",
        logger=None,
    )

    # Release base clip if owned
    if owned_base:
        try:
            if base_clip.audio:
                base_clip.audio.close()
        except Exception:
            pass
        try:
            base_clip.close()
        except Exception:
            pass

    # 4️⃣ Whisper transcription (SILENT MODE)
    final_segments = []
    model = None

    if audio_codec is not None:
        try:
            model = load_whisper_model()
            segments = transcribe_video(model, temp_video_path)

            # 🔇 No speech → silently skip subtitles
            if not segments:
                final_segments = []
                raise StopIteration

            full_context = build_full_context(segments)
            original_lines = [
                seg["text"].strip() for seg in segments if seg.get("text")
            ]

            try:
                corrected_lines = correct_line_by_line(
                    full_context, original_lines
                )
            except Exception:
                # Silent fallback to Whisper text
                corrected_lines = original_lines

            final_segments = build_final_segments(
                segments, corrected_lines
            )

        except StopIteration:
            pass  # intentional silent exit
        except Exception:
            final_segments = []
        finally:
            if model is not None:
                del model
            gc.collect()
    else:
        # 🔇 No audio track → silent skip
        final_segments = []

    # 5️⃣ Build final video clip
    try:
        final_clip = build_final_video_clip(
            temp_video_path, final_segments
        )
    except Exception:
        # Safe fallback reader
        safe_clip = VideoFileClip(
            temp_video_path, fps_source="fps"
        )
        final_clip = build_final_video_clip(
            temp_video_path, final_segments
        )
        try:
            safe_clip.close()
        except Exception:
            pass

    # 6️⃣ Cleanup temp files
    try:
        os.remove(temp_video_path)
    except Exception:
        pass

    return final_clip



# Optional: to allow this script to be run standalone
if __name__ == "__main__":
    clip = process_video_return_clip(INPUT_VIDEO)
    # Example: save and then release
    clip.write_videofile(
        OUTPUT_VIDEO,
        codec="libx264",
        audio_codec="aac",
        fps=getattr(clip, "fps", 30),
    )
    try:
        if clip.audio:
            clip.audio.close()
    except Exception:
        pass
    clip.close()
    del clip
    gc.collect()
