#!/usr/bin/env python3
import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
import subprocess
from io import BytesIO

import numpy as np
import requests
from pydub import AudioSegment
from pydub.effects import low_pass_filter, high_pass_filter, normalize
from pydub.generators import WhiteNoise
from PIL import Image
from moviepy.editor import ImageClip, AudioFileClip
from moviepy.video.fx import resize as vfx_resize  # not heavily used but kept

# ---------------- FIX PILLOW >= 10 ----------------
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

# =================================================
# 🧹 ONE-TIME CACHE / LOG CLEAN
# =================================================
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

def delete_path(path: Path):
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            path.unlink()
        print(f"✅ Deleted: {path}")
    except Exception as e:
        print(f"❌ Error deleting {path}: {e}")

def clean_system_caches():
    safe_clear_pycache()

    cache_dirs = [
        Path.home() / "Library" / "Caches",
        Path("/Library/Caches"),
    ]
    log_dirs = [
        Path.home() / "Library" / "Logs",
        Path("/Library/Logs"),
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

# =================================================
# 🔧 GLOBAL CONFIG
# =================================================
DEMUCS_MODEL = "htdemucs"
TEMP_DIR = "separated"
IMAGE_OUTPUT_DIR = "temp/images"
FPS = 30
SEGMENT_DURATION_MS = 58000  # ~57–58 sec

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

MOOD_KEYWORDS = {
    "low": [
        "sad portrait", "lonely street", "moody night", "dark alley",
        "foggy lamp", "broken sign", "blue shadows", "melancholic city",
        "slow rain", "empty street"
    ],
    "mid": [
        "dreamy landscape", "soft neon", "floating lights", "calm alley",
        "city twilight", "pastel motion", "moody skyline", "soft focus",
        "indie night", "cinematic glow"
    ],
    "high": [
        "cyberpunk neon", "neon street", "futuristic city", "light trails",
        "electric colors", "city rush", "abstract energy", "night speed",
        "sci fi", "dramatic lighting"
    ]
}

# =================================================
# 🎵 SHARED AUDIO HELPERS
# =================================================
def separate_with_demucs(audio_path, temp_dir=TEMP_DIR, model=DEMUCS_MODEL):
    print("🎤 Running Demucs (vocal removal)...")
    subprocess.run(
        [sys.executable, "-m", "demucs", "-n", model, audio_path],
        check=True
    )
    song_name = os.path.splitext(os.path.basename(audio_path))[0]
    base_path = os.path.join(temp_dir, model, song_name)
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
    return output_mp3

def convert_to_lofi(input_mp3, output_mp3):
    print("🎧 Creating Lo-Fi version...")
    audio = AudioSegment.from_mp3(input_mp3)
    audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * 0.92)
    }).set_frame_rate(audio.frame_rate)
    audio = low_pass_filter(audio, 3800)
    audio = high_pass_filter(audio, 90)
    audio += 2
    noise = WhiteNoise().to_audio_segment(duration=len(audio), volume=-42)
    audio = audio.overlay(noise)
    audio = normalize(audio)
    audio.export(output_mp3, format="mp3", bitrate="320k")
    print("✅ Lo-Fi audio saved:", output_mp3)
    return output_mp3

def create_slowed_reverb(input_mp3, output_path, speed=0.85, reverb_amount=0.03):
    print("🎧 Creating Slowed + Subtle Reverb...")
    audio = AudioSegment.from_mp3(input_mp3)
    slowed_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * speed)
    }).set_frame_rate(audio.frame_rate)
    delay_ms = 30
    attenuated = slowed_audio - 6
    delayed = AudioSegment.silent(duration=delay_ms) + attenuated
    reverb_audio = slowed_audio.overlay(delayed, gain_during_overlay=-reverb_amount * 6)
    final_audio = normalize(reverb_audio)
    final_audio.export(output_path, format="mp3", bitrate="320k")
    print(f"✅ Slowed + Subtle Reverb audio saved: {output_path}")
    return output_path

def get_best_segment(input_mp3, duration_ms):
    audio = AudioSegment.from_mp3(input_mp3)
    best_start = 0
    best_energy = 0
    for start in range(0, max(0, len(audio) - duration_ms), 1000):
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

def trim_to_max_duration(mp3_path, duration_ms):
    audio = AudioSegment.from_mp3(mp3_path)
    if len(audio) > duration_ms:
        audio = audio[:duration_ms]
        audio.export(mp3_path, format="mp3", bitrate="320k")
        print(f"⏱ Audio trimmed to {duration_ms/1000:.2f} sec -> {mp3_path}")
    return mp3_path

# =================================================
# 🖼 IMAGE HELPERS
# =================================================
def score_image(img):
    arr = np.array(img)
    brightness = arr.mean()
    contrast = arr.std()
    score = 0
    if 50 < brightness < 190:
        score += 1
    if contrast > 45:
        score += 2
    if brightness < 140:
        score += 1
    return score

def crop_to_16_9(img):
    w, h = img.size
    target_ratio = 16 / 9
    if w / h > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    return img

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

def fetch_best_music_image(mood_keywords_dict, pexels_api_key, video_size, vertical=False):
    """
    vertical=False -> landscape (16:9) search
    vertical=True -> portrait (9:16) search
    """
    headers = {"Authorization": pexels_api_key}
    keywords_list = []
    for mood in ["high", "mid", "low"]:
        keywords_list.extend(mood_keywords_dict[mood])

    orientation = "portrait" if vertical else "landscape"

    for keyword in keywords_list:
        print("🔍 Searching Pexels for:", keyword)
        params = {
            "query": keyword,
            "orientation": orientation,
            "per_page": 10,
            "page": 1
        }
        try:
            res = requests.get("https://api.pexels.com/v1/search",
                               headers=headers, params=params, timeout=10)
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
                    if vertical:
                        img = crop_to_9_16(img)
                    else:
                        img = crop_to_16_9(img)
                    img = img.resize(video_size, Image.LANCZOS)
                    score = score_image(img)
                    if score > best_score:
                        best_score = score
                        best_img = img
                except Exception:
                    continue

            if best_img:
                safe_kw = keyword.replace(" ", "_")
                out_path = os.path.join(IMAGE_OUTPUT_DIR, f"background_{safe_kw}.jpg")
                best_img.save(out_path, quality=95)
                print(f"✅ Best image selected for '{keyword}' (score {best_score})")
                return out_path
        except Exception as e:
            print("❌ Request failed:", e)
            continue

    print("❌ No suitable images found from top 10 of first page for any keyword")
    return None

# =================================================
# 🎬 VIDEO HELPER
# =================================================
def create_video(image_path, audio_path, output_path, video_size):
    audio = AudioFileClip(audio_path)
    duration = audio.duration
    clip = (
        ImageClip(image_path)
        .set_duration(duration)
        .set_audio(audio)
        .resize(video_size)
    )
    print(f"🎬 Rendering static image video -> {output_path} ...")
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
# 8 PROGRAM WRAPPERS
# =================================================

# 1) Lo-Fi, instrumental, full, landscape (1920x1080)
def prog1_lofi_landscape_instr_full(input_mp3, pexels_key, out_path):
    stems = separate_with_demucs(input_mp3)
    instr_mp3 = merge_instrumental(stems, "p1_instrumental_only.mp3")
    lofi_mp3 = convert_to_lofi(instr_mp3, "p1_instrumental_lofi.mp3")
    img = fetch_best_music_image(MOOD_KEYWORDS, pexels_key, (1920, 1080), vertical=False)
    if not img:
        print("❌ Program 1 image fetch failed")
        return
    create_video(img, lofi_mp3, out_path, (1920, 1080))

# 2) Lo-Fi, instrumental, best 58s, vertical (1080x1920)
def prog2_lofi_vertical_instr_best(input_mp3, pexels_key, out_path):
    stems = separate_with_demucs(input_mp3)
    instr_mp3 = merge_instrumental(stems, "p2_instrumental_only.mp3")
    lofi_mp3 = convert_to_lofi(instr_mp3, "p2_instrumental_lofi.mp3")
    best_audio = get_best_segment(lofi_mp3, SEGMENT_DURATION_MS)
    img = fetch_best_music_image(MOOD_KEYWORDS, pexels_key, (1080, 1920), vertical=True)
    if not img:
        print("❌ Program 2 image fetch failed")
        return
    create_video(img, best_audio, out_path, (1080, 1920))

# 3) Slowed+reverb, instrumental full, landscape (1920x1080)
def prog3_slowrev_landscape_instr_full(input_mp3, pexels_key, out_path):
    stems = separate_with_demucs(input_mp3)
    instr_mp3 = merge_instrumental(stems, "p3_instrumental_only.mp3")
    slowed_reverb = f"p3_slowed_reverb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    create_slowed_reverb(instr_mp3, slowed_reverb)
    img = fetch_best_music_image(MOOD_KEYWORDS, pexels_key, (1920, 1080), vertical=False)
    if not img:
        print("❌ Program 3 image fetch failed")
        return
    create_video(img, slowed_reverb, out_path, (1920, 1080))

# 4) Slowed+reverb, instrumental best 58s, vertical (1080x1920)
def prog4_slowrev_vertical_instr_best(input_mp3, pexels_key, out_path):
    stems = separate_with_demucs(input_mp3)
    instr_mp3 = merge_instrumental(stems, "p4_instrumental_only.mp3")
    best_segment = get_best_segment(instr_mp3, SEGMENT_DURATION_MS)
    slowed_reverb = f"p4_slowed_reverb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    create_slowed_reverb(best_segment, slowed_reverb)
    trim_to_max_duration(slowed_reverb, SEGMENT_DURATION_MS)
    img = fetch_best_music_image(MOOD_KEYWORDS, pexels_key, (1080, 1920), vertical=True)
    if not img:
        print("❌ Program 4 image fetch failed")
        return
    create_video(img, slowed_reverb, out_path, (1080, 1920))

# 5) Lo-Fi from original FULL, landscape (1920x1080)
def prog5_lofi_landscape_original_full(input_mp3, pexels_key, out_path):
    lofi_mp3 = convert_to_lofi(input_mp3, "p5_lofi_original.mp3")
    img = fetch_best_music_image(MOOD_KEYWORDS, pexels_key, (1920, 1080), vertical=False)
    if not img:
        print("❌ Program 5 image fetch failed")
        return
    create_video(img, lofi_mp3, out_path, (1920, 1080))


# 6) Lo-Fi from original BEST 58s, VERTICAL (1080x1920)
def prog6_lofi_vertical_original_best(input_mp3, pexels_key, out_path):
    # make lofi from original
    lofi_mp3 = convert_to_lofi(input_mp3, "p6_lofi_original.mp3")
    # take best ~58s segment of lo-fi
    best_audio = get_best_segment(lofi_mp3, SEGMENT_DURATION_MS)
    # vertical image
    img = fetch_best_music_image(MOOD_KEYWORDS, pexels_key, (1080, 1920), vertical=True)
    if not img:
        print("❌ Program 6 image fetch failed")
        return
    # vertical 9:16 output
    create_video(img, best_audio, out_path, (1080, 1920))


# 7) Slowed+reverb from original full, landscape (1920x1080)
def prog7_slowrev_landscape_original_full(input_mp3, pexels_key, out_path):
    slowed_reverb = f"p7_slowed_reverb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    create_slowed_reverb(input_mp3, slowed_reverb)
    img = fetch_best_music_image(MOOD_KEYWORDS, pexels_key, (1920, 1080), vertical=False)
    if not img:
        print("❌ Program 7 image fetch failed")
        return
    create_video(img, slowed_reverb, out_path, (1920, 1080))

# 8) Slowed+reverb on best 58s of original, vertical (1080x1920)
def prog8_slowrev_vertical_original_best(input_mp3, pexels_key, out_path):
    best_segment = get_best_segment(input_mp3, SEGMENT_DURATION_MS)
    slowed_reverb = f"p8_slowed_reverb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    create_slowed_reverb(best_segment, slowed_reverb)
    trim_to_max_duration(slowed_reverb, SEGMENT_DURATION_MS)
    img = fetch_best_music_image(MOOD_KEYWORDS, pexels_key, (1080, 1920), vertical=True)
    if not img:
        print("❌ Program 8 image fetch failed")
        return
    create_video(img, slowed_reverb, out_path, (1080, 1920))

# =================================================
# MAIN ORCHESTRATOR
# =================================================
import os
import sys

def main():
    # Hardcode your inputs here
    input_mp3 = "/Users/uday/Downloads/ATLXS - PASSO BEM SOLTO (SLOWED) - phonk.mp3"
    pexels_key = "DGhCtAB83klpCIv5yq5kMIb2zun7q67IvHJysvW4lInb0WVXaQF2xLMu"

    # Output directory
    output_dir = "output_songs"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(input_mp3):
        print("❌ Input MP3 not found:", input_mp3)
        sys.exit(1)

    # First clean (optional)
    clean_system_caches()

    prog1_lofi_landscape_instr_full(input_mp3, pexels_key, os.path.join(output_dir, "out1.mp4"))
    clean_system_caches()

    prog2_lofi_vertical_instr_best(input_mp3, pexels_key, os.path.join(output_dir, "out2.mp4"))
    clean_system_caches()

    prog3_slowrev_landscape_instr_full(input_mp3, pexels_key, os.path.join(output_dir, "out3.mp4"))
    clean_system_caches()

    prog4_slowrev_vertical_instr_best(input_mp3, pexels_key, os.path.join(output_dir, "out4.mp4"))
    clean_system_caches()

    prog5_lofi_landscape_original_full(input_mp3, pexels_key, os.path.join(output_dir, "out5.mp4"))
    clean_system_caches()
    
    prog6_lofi_vertical_original_best(input_mp3, pexels_key, os.path.join(output_dir, "out6.mp4"))
    clean_system_caches()

    prog7_slowrev_landscape_original_full(input_mp3, pexels_key, os.path.join(output_dir, "out7.mp4"))
    clean_system_caches()

    prog8_slowrev_vertical_original_best(input_mp3, pexels_key, os.path.join(output_dir, "out8.mp4"))
    clean_system_caches()

    print("✅ All 8 videos rendered inside the 'output/' folder.")


if __name__ == "__main__":
    main()



