from moviepy.editor import AudioFileClip
import os

# --------------------------
# CHANGE THIS
# --------------------------
MUSIC_PATH = "/Users/uday/Downloads/fictic/music/ESSE CARA! - Sayfalse.mp3"
OUT_DIR = "/Users/uday/Downloads/fictic/audio_volume_test/"
# --------------------------

os.makedirs(OUT_DIR, exist_ok=True)

audio = AudioFileClip(MUSIC_PATH)

volumes = [0.8,0.02,0.2,0.03,0.08,0.06,0.05]

for v in volumes:
    print(f"Creating audio at volume: {v}")
    a = audio.volumex(v)

    out_file = os.path.join(OUT_DIR, f"audio_vol_{int(v*100)}.mp3")

    a.write_audiofile(out_file, fps=44100)

print("\nDONE! Your test files:")
for v in volumes:
    print(f"audio_vol_{int(v*100)}.mp3")
