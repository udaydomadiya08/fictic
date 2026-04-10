from moviepy.editor import VideoFileClip, VideoClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ------------ SETTINGS ------------
VIDEO_PATH = "/Users/uday/Downloads/fictic/best_scene.mp4"
OUTPUT_PATH = "font_test.mp4"

TEXT = "This is a FONT SIZE TEST.\nCheck visibility."
FONT_PATH = "trajan-pro/TrajanPro-Regular.ttf"
FONT_SIZE = 80               # ← CHANGE THIS TO TEST
TEXT_COLOR = (255, 255, 0)   # yellow
STROKE_COLOR = (0, 0, 0)     # black
BOTTOM_MARGIN = 0.05         # 5% above bottom
# ---------------------------------

def draw_text(frame, t):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    W, H = img.size
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # Split lines
    lines = TEXT.split("\n")
    total_h = len(lines) * FONT_SIZE

    # Starting y (bottom aligned)
    y = int(H * (1 - BOTTOM_MARGIN) - total_h)

    for line in lines:
        w = draw.textlength(line, font=font)
        x = (W - w) // 2

        draw.text(
            (x, y),
            line,
            font=font,
            fill=TEXT_COLOR,
            stroke_width=2,
            stroke_fill=STROKE_COLOR
        )
        y += FONT_SIZE

    return np.array(img)


def main():
    base = VideoFileClip(VIDEO_PATH)

    final = base.fl_image(lambda f: draw_text(f, 0))

    final.write_videofile(
        OUTPUT_PATH,
        codec="libx264",
        audio_codec="aac",
        fps=base.fps
    )

    print("🎉 Font test video saved:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
