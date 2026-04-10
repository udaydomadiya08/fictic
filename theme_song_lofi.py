import shutil
import os
from moviepy.editor import VideoFileClip, CompositeVideoClip, TextClip, concatenate_videoclips
from pydub.generators import WhiteNoise
def safe_clear_pycache(start_path='.'):
    for root, dirs, files in os.walk(start_path):
        for d in dirs:
            if d == '__pycache__':
                full_path = os.path.join(root, d)
                try:
                    shutil.rmtree(full_path)
                    print(f"Deleted {full_path}")
                except PermissionError as e:
                    print(f"Permission denied to delete {full_path}: {e}")

safe_clear_pycache()


import shutil
from pathlib import Path
def delete_path(path):
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            path.unlink()
        print(f"✅ Deleted: {path}")
    except Exception as e:
        print(f"❌ Error deleting {path}: {e}")

# Cache directories
cache_dirs = [
    Path.home() / "Library" / "Caches",
    Path("/Library/Caches"),
]

# Log directories
log_dirs = [
    Path.home() / "Library" / "Logs",
    Path("/Library/Logs")
]





dirs_to_clean = cache_dirs + log_dirs 
print("🧹 Cleaning up...")

for d in dirs_to_clean:
    if d.exists():
        for item in d.iterdir():
            delete_path(item)
    else:
        print(f"⚠️ Directory not found: {d}")

print("✅ Done cleaning up!")

import os
import subprocess
import sys
import random
import requests
from io import BytesIO

import numpy as np
from pydub import AudioSegment
from PIL import Image
from moviepy.editor import ImageClip, AudioFileClip
from moviepy.video.fx import resize as vfx_resize



# ---------------- FIX PILLOW >= 10 ----------------
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

# ---------------- CONFIG ----------------
INPUT_MP3 = "/Users/uday/Downloads/30 seconds of labon ko🫀 #song #lyrics #hindisong - Its_anshika_78.mp3"
OUTPUT_MP3 = "instrumental_only.mp3"
from datetime import datetime

# Get current date and time
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")  # e.g., 20251214_141530


FINAL_VIDEO = f"youtube_ready_video_{timestamp}.mp4"

DEMUCS_MODEL = "htdemucs"
TEMP_DIR = "separated"
IMAGE_OUTPUT_DIR = "temp/images"

VIDEO_SIZE = (1080, 1920)   # 9:16 vertical
FPS = 30
SEGMENT_DURATION_MS = 58000  # ~57–58 sec

PEXELS_API_KEY = "DGhCtAB83klpCIv5yq5kMIb2zun7q67IvHJysvW4lInb0WVXaQF2xLMu"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

# =================================================
# 🎤 DEMUCS VOCAL REMOVAL
# =================================================
def separate_with_demucs(audio_path):
    print("🎤 Running Demucs (vocal removal)...")
    subprocess.run(
        [sys.executable, "-m", "demucs", "-n", DEMUCS_MODEL, audio_path],
        check=True
    )

    song_name = os.path.splitext(os.path.basename(audio_path))[0]
    base_path = os.path.join(TEMP_DIR, DEMUCS_MODEL, song_name)

    return {
        "drums": os.path.join(base_path, "drums.wav"),
        "bass": os.path.join(base_path, "bass.wav"),
        "other": os.path.join(base_path, "other.wav"),
    }

def merge_instrumental(stems, output_mp3):
    print("🎶 Merging instrumental tracks...")
    drums = AudioSegment.from_wav(stems["drums"])
    bass = AudioSegment.from_wav(stems["bass"])
    other = AudioSegment.from_wav(stems["other"])

    instrumental = drums.overlay(bass).overlay(other)
    instrumental.export(output_mp3, format="mp3", bitrate="320k")
    print("✅ Instrumental saved:", output_mp3)

# =================================================
# 🎯 BEST MUSIC SEGMENT DETECTION
# =================================================
def get_best_segment(input_mp3, duration_ms):
    audio = AudioSegment.from_mp3(input_mp3)

    best_start = 0
    best_energy = 0

    for start in range(0, len(audio) - duration_ms, 1000):
        segment = audio[start:start + duration_ms]
        energy = segment.rms
        if energy > best_energy:
            best_energy = energy
            best_start = start

    best = audio[best_start:best_start + duration_ms]
    out_path = input_mp3.replace(".mp3", "_best_segment.mp3")
    best.export(out_path, format="mp3", bitrate="320k")
    print("✅ Best segment extracted:", out_path)
    return out_path

from pydub.effects import low_pass_filter, high_pass_filter, normalize
from pydub.generators import WhiteNoise

def convert_to_lofi(input_mp3, output_mp3):
    print("🎧 Creating Lo-Fi version...")
    audio = AudioSegment.from_mp3(input_mp3)

    # 1️⃣ Slight slow-down
    audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * 0.92)
    }).set_frame_rate(audio.frame_rate)

    # 2️⃣ Filters (Lo-Fi tone)
    audio = low_pass_filter(audio, 3800)
    audio = high_pass_filter(audio, 90)

    # 3️⃣ Warmth
    audio += 2

    # 4️⃣ Vinyl noise (very subtle)
    noise = WhiteNoise().to_audio_segment(duration=len(audio), volume=-42)
    audio = audio.overlay(noise)

    # 5️⃣ Normalize
    audio = normalize(audio)

    audio.export(output_mp3, format="mp3", bitrate="320k")
    print("✅ Lo-Fi audio saved:", output_mp3)
    return output_mp3




# =================================================
# 🎨 SMART IMAGE SELECTION (NO RANDOM SHIT)
# =================================================
MOOD_KEYWORDS = {
    "low": [
        "sad portrait",
        "lonely street",
        "moody night",
        "dark alley",
        "foggy lamp",
        "broken sign",
        "blue shadows",
        "melancholic city",
        "slow rain",
        "empty street"
    ],

    "mid": [
        "dreamy landscape",
        "soft neon",
        "floating lights",
        "calm alley",
        "city twilight",
        "pastel motion",
        "moody skyline",
        "soft focus",
        "indie night",
        "cinematic glow"
    ],

    "high": [
        "cyberpunk neon",
        "neon street",
        "futuristic city",
        "light trails",
        "electric colors",
        "city rush",
        "abstract energy",
        "night speed",
        "sci fi",
        "dramatic lighting"
    ]
}



def score_image(img):
    arr = np.array(img)
    brightness = arr.mean()
    contrast = arr.std()

    score = 0
    if 50 < brightness < 190:
        score += 1
    if contrast > 45:
        score += 2
    if brightness < 140:   # cinematic preference
        score += 1

    return score

def crop_to_9_16(img):
    w, h = img.size
    target_ratio = 9 / 16

    if w / h > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    return img
def fetch_best_music_image(mood_keywords_dict):
    """
    Fetches a suitable image for the video from Pexels:
    - Only uses top 10 images from page 1
    - Tries keywords in order
    - Crops to 9:16 and resizes to VIDEO_SIZE
    """
    headers = {"Authorization": PEXELS_API_KEY}

    # Flatten the dict into a single list preserving order
    keywords_list = []
    for mood in ["high", "mid", "low"]:  # prioritize high energy first
        keywords_list.extend(mood_keywords_dict[mood])

    for keyword in keywords_list:
        print("🔍 Searching Pexels for:", keyword)
        params = {
            "query": keyword,
            "orientation": "portrait",
            "per_page": 10,
            "page": 1
        }

        try:
            res = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params, timeout=10)
            if res.status_code != 200:
                print("❌ Pexels error:", res.status_code, res.text)
                continue

            photos = res.json().get("photos", [])[:10]
            if not photos:
                continue

            best_img = None
            best_score = -1

            for p in photos:
                try:
                    img_data = requests.get(p["src"]["original"], timeout=8).content
                    img = Image.open(BytesIO(img_data)).convert("RGB")
                    img = crop_to_9_16(img)
                    img = img.resize(VIDEO_SIZE, Image.LANCZOS)

                    score = score_image(img)
                    if score > best_score:
                        best_score = score
                        best_img = img
                except:
                    continue

            if best_img:
                out_path = os.path.join(IMAGE_OUTPUT_DIR, f"background_{keyword.replace(' ','_')}.jpg")
                best_img.save(out_path, quality=95)
                print(f"✅ Best image selected for '{keyword}' (score {best_score})")
                return out_path

        except Exception as e:
            print("❌ Request failed:", e)
            continue

    print("❌ No suitable images found from top 10 of first page for any keyword")
    return None

# =================================================
# 🎬 CINEMATIC VIDEO CREATION
# =================================================
# =================================================
# 🎬 STATIC VIDEO CREATION (NO EFFECTS)
# =================================================
def create_video(image_path, audio_path, output_path):
    audio = AudioFileClip(audio_path)
    duration = audio.duration

    clip = (
        ImageClip(image_path)
        .set_duration(duration)
        .set_audio(audio)
        .resize(VIDEO_SIZE)   # ONLY resize, no animation
    )

    print("🎬 Rendering static image video...")
    clip.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        preset="slow",
        threads=4
    )

    print("✅ FINAL STATIC VIDEO READY:", output_path)


# =================================================
# 🚀 MAIN PIPELINE
# =================================================
if __name__ == "__main__":
    stems = separate_with_demucs(INPUT_MP3)
    merge_instrumental(stems, OUTPUT_MP3)

    LOFI_MP3 = "instrumental_lofi.mp3"
    convert_to_lofi(OUTPUT_MP3, LOFI_MP3)

    best_audio = get_best_segment(LOFI_MP3, SEGMENT_DURATION_MS)


    # fetch image from top 10 of first page using two-word keywords
    image = fetch_best_music_image(MOOD_KEYWORDS)
    if image:
        create_video(image, best_audio, FINAL_VIDEO)
    else:
        print("❌ Image fetch failed")

