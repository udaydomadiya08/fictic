# """
# Full integrated script:
# - Sequential folder order
# - Smart crop each clip (frame-by-frame using smtcro.smart_full_crop)
# - Merge until total duration < 60s
# - Subtitles via ficint1.process_video_return_clip
# - Run Demucs once on merged clip audio (remove BG music)
# - Add single background music (looped) and export
# """

# import os
# import cv2
# import uuid
# import subprocess
# import shutil
# from moviepy.editor import VideoFileClip, ImageSequenceClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip
# from moviepy.audio.fx.all import audio_loop
# # import any moviepy fx if you plan to use them (not required here)

# # ---- Replace these with your modules/paths ----
# from smtcro import smart_full_crop      # your smart cropping function
# from ficint1 import process_video_return_clip  # your subtitle processor
# # -----------------------------------------------

# # ---------------- USER SETTINGS ----------------
# INPUT_FOLDER = "/Users/uday/Downloads/fictic/firstclip"      # folder with source clips
# OUTPUT_FOLDER = "/Users/uday/Downloads/fictic/out"
# MUSIC_PATH = "bg_music.mp3"                    # single BG music for entire final video
# MAX_DURATION = 60.0                            # strict upper bound (in seconds)
# NUM_VIDEOS = 5                                 # number of final videos to produce
# EXTENSIONS = (".mp4", ".mov", ".mkv")
# # ------------------------------------------------

# os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# # ---------------- smart crop (frame-by-frame) ----------------
# def smart_crop_clip(moviepy_clip):
#     """
#     Use your smart_full_crop on every frame and return a new MoviePy clip.
#     Note: This can be slow (frame-by-frame processing).
#     """
#     frames = []
#     prev_box = None
#     history = []

#     # iterate frames at the source clip fps
#     for frame in moviepy_clip.iter_frames(fps=moviepy_clip.fps, dtype="uint8"):
#         # frame is RGB; convert to BGR for OpenCV-based smart_full_crop
#         frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#         # apply your smart_full_crop -> returns cropped_frame (BGR) and updated prev_box
#         cropped_bgr, prev_box = smart_full_crop(frame_bgr, prev_box, history)

#         # convert back to RGB for MoviePy
#         cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
#         frames.append(cropped_rgb)

#     # Build a new clip from processed frames
#     new_clip = ImageSequenceClip(frames, fps=moviepy_clip.fps)
#     # If you want a fixed size (e.g., 1080x1920), you can add .resize((W,H)) here
#     return new_clip

# def pick_music_file():
#     music_dir = "music"
#     files = [f for f in os.listdir(music_dir) if f.lower().endswith((".mp3", ".wav", ".m4a"))]

#     if not files:
#         raise Exception("❌ No music files found in 'music' folder.")

#     print("\n🎵 Available Music Files:")
#     for i, f in enumerate(files):
#         print(f"{i+1}. {f}")

#     choice = int(input("\nSelect music file number: ")) - 1
#     return os.path.join(music_dir, files[choice])

# # ---------------- audio helper: extract raw wav from a video file ----------------
# def extract_audio_from_video_file(video_filepath, out_wav_path):
#     cmd = [
#         "ffmpeg", "-y",
#         "-i", video_filepath,
#         "-vn",
#         "-ac", "2",
#         "-ar", "44100",
#         out_wav_path
#     ]
#     subprocess.run(cmd, check=True)
#     return out_wav_path


# # ---------------- run Demucs on a WAV path -> return vocals path ----------------
# def run_demucs_on_wav(wav_path):
#     """
#     Run demucs on the wav and return the vocals file path that Demucs produces.
#     Expects Demucs to create: separated/htdemucs/<basename>/vocals.wav
#     """
#     print("🎤 Running Demucs on merged audio (this may take a while)...")
#     # call demucs (ensure demucs is installed & in PATH)
#     subprocess.run(["demucs", wav_path], check=True)

#     base = os.path.basename(wav_path).replace(".wav", "")
#     # demucs output path structure (htdemucs backend)
#     vocals_path = os.path.join("separated", "htdemucs", base, "vocals.wav")
#     if not os.path.exists(vocals_path):
#         raise FileNotFoundError(f"Demucs output vocals not found at: {vocals_path}")
#     return vocals_path


# # ---------------- attach vocals.wav to a MoviePy clip ----------------
# def replace_audio_with_vocals(movie_clip, vocals_wavpath, boost=2.5):
#     vocal_audio = AudioFileClip(vocals_wavpath)
#     # optional: trim/loop vocals to clip duration
#     if vocal_audio.duration < movie_clip.duration:
#         vocal_audio = audio_loop(vocal_audio, duration=movie_clip.duration)
#     else:
#         vocal_audio = vocal_audio.subclip(0, movie_clip.duration)
#     # boost speech
#     vocal_audio = vocal_audio.volumex(boost)
#     return movie_clip.set_audio(vocal_audio)


# # ---------------- add one BG music (looped) to final clip (mix with existing audio) ----------------
# def add_single_background_music(final_clip, music_path, music_volume=0.10):
#     video_duration = final_clip.duration
#     bg = AudioFileClip(music_path)
#     # loop music to match duration
#     bg_looped = audio_loop(bg, duration=video_duration)
#     bg_looped = bg_looped.volumex(music_volume)
#     # combine original (vocals) with bg
#     mixed = CompositeAudioClip([final_clip.audio, bg_looped])
#     return final_clip.set_audio(mixed)


# # ---------------- build one merged clip by consuming sequential files ----------------
# # ---------------- build one merged clip by consuming sequential files ----------------
# def build_merged_clip_from_folder(input_folder, used_set, max_duration=MAX_DURATION):
#     """
#     Iterate files in folder (sorted), skip used_set, apply smart crop for each clip,
#     append until < max_duration. Returns concatenated MoviePy clip or None if no clips left.
#     NOTE: This function does NOT run Demucs; Demucs is run once on merged clip later.
#     """
#     files = sorted(os.listdir(input_folder))
#     final_parts = []
#     accumulated = 0.0

#     # If everything in folder is already used → stop
#     usable_files = [f for f in files if f.lower().endswith(EXTENSIONS)]
#     if len(used_set) >= len(usable_files):
#         print("[STOP] All clips in firstclip folder are used.")
#         return None

#     for fname in files:

#         # Skip non-video files
#         if not fname.lower().endswith(EXTENSIONS):
#             continue

#         path = os.path.join(input_folder, fname)

#         # Skip if already used
#         if path in used_set:
#             continue

#         # STOP if all files are consumed mid-loop
#         if len(used_set) >= len(usable_files):
#             print("[STOP] Folder exhausted — no more clips left.")
#             break

#         print(f"[INFO] Loading source clip: {path}")
#         src = VideoFileClip(path)

#         cropped = smart_crop_clip(src)
#         dur = cropped.duration

#         # If adding this clip crosses limit → stop, do NOT add partial
#         if accumulated + dur >= max_duration:
#             print(f"[STOP] Reached max duration at {accumulated:.2f}s (cannot add more).")
#             break

#         # Append full clip
#         final_parts.append(cropped)
#         used_set.add(path)
#         accumulated += dur

#     if not final_parts:
#         return None

#     merged = concatenate_videoclips(final_parts, method="compose")
#     return merged

# from fictic14 import find_continuous_segment
# def apply_smart_music(final_video, music_path):
#     video_duration = final_video.duration

#     # Find best continuous segment inside music
#     seg_start, _ = find_continuous_segment(
#         music_path,
#         video_duration,
#         0   # no fillers now
#     )

#     # Load music from smart start
#     music_full = AudioFileClip(music_path).subclip(seg_start)

#     # Loop music to match full video duration
#     music_looped = audio_loop(music_full, duration=video_duration)

#     # Attach final audio
#     final = final_video.set_audio(music_looped.volumex(0.8))

#     return final


# # ---------------- main pipeline to create N final videos ----------------
# def create_n_videos(input_folder, output_folder, music_path, n_videos=NUM_VIDEOS):
#     os.makedirs(output_folder, exist_ok=True)
#     used_files = set()

#     for idx in range(1, n_videos + 1):
#         print(f"\n====== Building final video #{idx} ======")

#         merged_clip = build_merged_clip_from_folder(input_folder, used_files, MAX_DURATION)
#         dur=None if merged_clip.duration <=10  else merged_clip.duration
#         if dur is None:
#             print("[INFO] No more usable clips. Stopping.")
#             break

#         # 1) Add subtitles (your module)
#         print("[INFO] Applying subtitles (process_video_return_clip)...")
#         try:
#             merged_with_subs = process_video_return_clip(merged_clip)
#         except Exception as e:
#             print("[ERROR] process_video_return_clip failed:", e)
#             print("[INFO] Proceeding without subtitles for this clip.")
#             merged_with_subs = merged_clip

#         # Save temporary merged video to run Demucs on its audio
#         temp_id = uuid.uuid4().hex[:8]
#         temp_video_file = f"temp_merged_{temp_id}.mp4"
#         print(f"[INFO] Writing temporary merged file for audio extraction: {temp_video_file}")
#         merged_with_subs.write_videofile(
#             temp_video_file,
#             codec="libx264",
#             audio_codec="aac",
#             fps=60,
#             preset="fast",
#             threads=4,
#             verbose=False,
#             logger=None,
#             ffmpeg_params=["-movflags","+faststart","-crf","18","-pix_fmt","yuv420p"]
#         )

#         # 2) Extract raw audio
#         temp_wav = f"temp_audio_{temp_id}.wav"
#         extract_audio_from_video_file(temp_video_file, temp_wav)

#         # 3) Run Demucs to remove BG music -> get vocals.wav
#         vocals_wav = run_demucs_on_wav(temp_wav)

#         # 4) Replace merged clip audio with isolated vocals (boosted)
#         print("[INFO] Replacing audio with isolated vocals...")
#         final_vocals_clip = replace_audio_with_vocals(merged_with_subs, vocals_wav, boost=2.0)

#         # 5) Add ONE background music track (looped, constant volume)
#         print("[INFO] Adding single background music track...")
#         final_with_bg = apply_smart_music(final_vocals_clip, music_path)

#         # 6) Export final
#         out_path = os.path.join(output_folder, f"final_video_{idx}.mp4")
#         print(f"[INFO] Exporting final video to: {out_path}")
#         final_with_bg.write_videofile(
#             out_path,
#             codec="libx264",
#             audio_codec="aac",
#             fps=60,
#             preset="fast",
#             threads=4,
#             ffmpeg_params=["-movflags","+faststart","-crf","18","-pix_fmt","yuv420p"]
#         )

#         # cleanup temporary files (optional)
#         try:
#             os.remove(temp_video_file)
#             os.remove(temp_wav)
#             # Demucs output folder for this wav remains at separated/htdemucs/<basename>
#             # if you want, you may remove that folder to reclaim disk space:
#             demucs_out_folder = os.path.join("separated", "htdemucs", os.path.basename(temp_wav).replace(".wav", ""))
#             if os.path.isdir(demucs_out_folder):
#                 shutil.rmtree(demucs_out_folder)
#         except Exception:
#             pass

#     print("\nAll done.")


# # ---------------- run if executed as script ----------------
# if __name__ == "__main__":
#     music_file = pick_music_file()
#     num_vid = int(input(f"Enter number of final videos to create "))
#     create_n_videos(INPUT_FOLDER, OUTPUT_FOLDER, music_file, num_vid)

# """
# Option 2 (RECOMMENDED):
# - Smart crop using MoviePy fx (NOT frame-by-frame Python loops)
# - Sequential pick from folder (not random)
# - Merge until < 60 sec
# - Subtitles added
# - Write temp video
# - Run DEMUCS only ONCE per final merged clip
# - Replace audio with vocals
# - Smart-loop background music using find_continuous_segment
# - Output final video
# """

# import os
# import uuid
# import shutil
# import subprocess
# from moviepy.editor import *
# from moviepy.audio.fx.all import audio_loop

# # --------------------------------------------
# # Import your modules
# # --------------------------------------------
# from smtcro import smart_full_crop         # your smart crop
# from ficint1 import process_video_return_clip
# from fictic14 import find_continuous_segment

# # --------------------------------------------
# # USER SETTINGS
# # --------------------------------------------
# INPUT_FOLDER = "/Users/uday/Downloads/fictic/firstclip"
# OUTPUT_FOLDER = "/Users/uday/Downloads/fictic/out"
# EXTENSIONS = (".mp4", ".mov", ".mkv")
# MAX_DURATION = 60.0

# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # --------------------------------------------
# # Faster smart crop (NOT frame by frame)
# # --------------------------------------------
# def apply_smart_crop(clip):
#     """Runs smart_full_crop per-frame internally but without creating ImageSequenceClip."""
#     return clip.fl(lambda gf, t: smart_full_crop(gf(t))[0])


# # --------------------------------------------
# # Pick music from folder
# # --------------------------------------------
# def pick_music_file():
#     music_dir = "music"
#     files = [f for f in os.listdir(music_dir)
#              if f.lower().endswith((".mp3", ".wav", ".m4a"))]

#     print("\n🎵 Available music files:")
#     for i, f in enumerate(files):
#         print(f"{i+1}. {f}")

#     choice = int(input("\nChoose music number: ")) - 1
#     return os.path.join(music_dir, files[choice])


# # --------------------------------------------
# # Build merged clip (<60 sec limit)
# # --------------------------------------------
# def build_merged_clip(input_folder, used_files):

#     files = sorted(os.listdir(input_folder))
#     final_parts = []
#     accumulated = 0.0

#     for f in files:
#         if not f.lower().endswith(EXTENSIONS):
#             continue

#         path = os.path.join(input_folder, f)
#         if path in used_files:
#             continue

#         print(f"[INFO] Using: {f}")
#         clip = VideoFileClip(path)

#         # FAST smart crop
        

#         if accumulated + clip.duration >= MAX_DURATION:
#             break
#         clip = apply_smart_crop(clip)

#         final_parts.append(clip)
#         used_files.add(path)
#         accumulated += clip.duration

#     if not final_parts:
#         return None

#     return concatenate_videoclips(final_parts, method="compose")


# # --------------------------------------------
# # Extract WAV for Demucs
# # --------------------------------------------
# def extract_audio(video_path, wav_path):
#     subprocess.run([
#         "ffmpeg", "-y",
#         "-i", video_path,
#         "-vn",
#         "-ac", "2",
#         "-ar", "44100",
#         wav_path
#     ], check=True)


# # --------------------------------------------
# # Run Demucs ONCE on final merged video
# # --------------------------------------------
# def run_demucs(wav_path):
#     print("🎤 Running Demucs…")
#     subprocess.run(["demucs", wav_path], check=True)

#     base = os.path.basename(wav_path).replace(".wav", "")

#     vocals = os.path.join("separated", "htdemucs", base, "vocals.wav")
#     if not os.path.exists(vocals):
#         raise Exception("❌ Demucs vocals not found!")

#     return vocals


# # --------------------------------------------
# # Replace audio with vocals only
# # --------------------------------------------
# def replace_audio_with_vocals(final_clip, vocals_path):
#     vocals = AudioFileClip(vocals_path)
#     vocals = audio_loop(vocals, duration=final_clip.duration)
#     vocals = vocals.volumex(2.0)
#     return final_clip.set_audio(vocals)


# # --------------------------------------------
# # Smart background music (find_continuous_segment)
# # --------------------------------------------
# def add_smart_music(final_clip, music_path):

#     video_dur = final_clip.duration

#     seg_start, _ = find_continuous_segment(
#         music_path,
#         video_dur,
#         0  # no fillers
#     )

#     bg = AudioFileClip(music_path).subclip(seg_start)
#     bg = audio_loop(bg, duration=video_dur).volumex(0.25)

#     return final_clip.set_audio(
#         CompositeAudioClip([final_clip.audio, bg])
#     )


# # --------------------------------------------
# # Main function to create N videos
# # --------------------------------------------
# def create_videos(n, input_folder, output_folder, music_path):

#     used = set()

#     for i in range(1, n + 1):

#         print(f"\n========== Making Video #{i} ==========")

#         merged = build_merged_clip(input_folder, used)
#         if merged is None:
#             print("No more clips.")
#             break

#         # 1) Subtitles
#         print("[INFO] Adding subtitles…")
#         try:
#             sub_clip = process_video_return_clip(merged)
#         except:
#             sub_clip = merged

#         # 2) Save TEMP file for Demucs
#         temp_id = uuid.uuid4().hex[:8]
#         temp_video = f"temp_{temp_id}.mp4"

#         sub_clip.write_videofile(
#             temp_video,
#             codec="libx264",
#             audio_codec="aac",
#             fps=24,
#             preset="fast",
#             threads=4,
#             logger=None
#         )

#         # 3) RUN DEMUCS ONLY ON TEMP FILE
#         temp_wav = f"audio_{temp_id}.wav"
#         extract_audio(temp_video, temp_wav)

#         vocals_path = run_demucs(temp_wav)

#         # 4) Replace audio with vocals only
#         clean_audio_clip = replace_audio_with_vocals(sub_clip, vocals_path)

#         # 5) Add smart-looped background music
#         final_clip = add_smart_music(clean_audio_clip, music_path)

#         # 6) Export final
#         out_file = os.path.join(output_folder, f"final_{i}.mp4")
#         print(f"[INFO] Saving → {out_file}")

#         final_clip.write_videofile(
#             out_file,
#             codec="libx264",
#             audio_codec="aac",
#             fps=24,
#             preset="fast",
#             threads=4,
#             logger=None
#         )

#         # Cleanup
#         try:
#             os.remove(temp_video)
#             os.remove(temp_wav)
#             shutil.rmtree(os.path.join("separated", "htdemucs", temp_id), ignore_errors=True)
#         except:
#             pass


# # --------------------------------------------
# # RUN SCRIPT
# # --------------------------------------------
# if __name__ == "__main__":
#     music_file = pick_music_file()
#     num = int(input("Enter number of videos to create: "))
#     create_videos(num, INPUT_FOLDER, OUTPUT_FOLDER, music_file)


#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Final integrated pipeline (full):
- Sequential folder order (no shuffle)
- Stateful smart crop per clip (streaming via fl_image; preserves audio)
- Merge until total duration < MAX_DURATION (strict)
- Subtitles via ficint1.process_video_return_clip (safe fallback)
- Write temp merged video, extract audio
- Run Demucs ONCE on merged audio (vocals extraction)
- Replace merged audio with vocals (loop/trim + boost)
- Add ONE background music track: find_continuous_segment start -> loop to video end
- Export final video
- Safe cleanup (temp files, Demucs output)
Notes: make sure `demucs`, `ffmpeg` are installed & in PATH.
"""

import os
import uuid
import shutil
import subprocess
import gc
import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip
from moviepy.audio.fx.all import audio_loop

# ---------------------- USER CONFIG ----------------------
INPUT_FOLDER = "/Users/uday/Downloads/fictic/firstclip"   # your clips folder
OUTPUT_FOLDER = "/Users/uday/Downloads/fictic/out"        # output folder
MUSIC_FOLDER = "music"                                    # folder containing BG music files
EXTENSIONS = (".mp4", ".mov", ".mkv")
MAX_DURATION = 14.0
DEFAULT_FPS = 24
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# ---------------------------------------------------------

# ---------------- external module imports -----------------
# These must be available on your PYTHONPATH
from smtcro import smart_full_crop           # function: (frame_bgr, prev_box, history) -> (cropped_bgr, new_prev_box)
from ficint1 import process_video_return_clip
from fictic14 import find_continuous_segment
# ---------------------------------------------------------


# -------------------- Helper: pick music -----------------
def pick_music_file():
    if not os.path.isdir(MUSIC_FOLDER):
        raise RuntimeError(f"Music folder not found: {MUSIC_FOLDER}")
    files = [f for f in os.listdir(MUSIC_FOLDER) if f.lower().endswith((".mp3", ".wav", ".m4a"))]
    if not files:
        raise RuntimeError("No music files found in music folder.")
    print("\n🎵 Available music files:")
    for i, f in enumerate(files, 1):
        print(f"{i}. {f}")
    idx = int(input("\nChoose music number: ")) - 1
    chosen = os.path.join(MUSIC_FOLDER, files[idx])
    print(f"[INFO] Selected music: {chosen}")
    return chosen


# ----------- Stateful streaming smart crop (preserve audio) -----------
def apply_smart_crop(clip):
    """
    Apply smart_full_crop to each frame while preserving per-clip prev_box & history state.
    Returns a clip with same audio attached.
    """
    prev_box = None
    history = []

    def transform_frame(frame_rgb):
        nonlocal prev_box, history
        # Convert RGB -> BGR for smart_full_crop
        try:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cropped_bgr, prev_box = smart_full_crop(frame_bgr, prev_box, history)
            cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            return cropped_rgb
        except Exception as e:
            # if smart crop fails for any frame, fallback to original frame
            # (keeps processing stable)
            return frame_rgb

    # Use fl_image (streaming) to avoid building frames list in RAM
    cropped_clip = clip.fl_image(transform_frame)
    # IMPORTANT: keep original audio to avoid subbed.audio = None bug
    try:
        return cropped_clip.set_audio(clip.audio)
    except Exception:
        return cropped_clip


# ---------------- build merged clip (< MAX_DURATION) ----------------
def build_merged_clip(input_folder, used_files):
    files = sorted(os.listdir(input_folder))
    parts = []
    accumulated = 0.0

    # quick check: no more usable files
    usable = [f for f in files if f.lower().endswith(EXTENSIONS)]
    if len(used_files) >= len(usable):
        return None

    for fname in files:
        if not fname.lower().endswith(EXTENSIONS):
            continue
        path = os.path.join(input_folder, fname)
        if path in used_files:
            continue

        print(f"[INFO] Loading source clip: {fname}")
        src = VideoFileClip(path)

        # apply streaming smart crop (keeps audio)
        cropped = apply_smart_crop(src)

        # strict rule: do NOT add a clip that would make duration >= MAX_DURATION
        if accumulated + cropped.duration >= MAX_DURATION:
            print(f"[INFO] would exceed {MAX_DURATION}s, skipping {fname}")


        parts.append(cropped)
        used_files.add(path)
        accumulated += cropped.duration

    

        if accumulated >= MAX_DURATION:
            break

    if not parts:
        return None

    merged = concatenate_videoclips(parts, method="compose")
    return merged


# ---------------- ffmpeg audio extraction helper ----------------
def extract_audio_from_video(video_path, wav_out):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "2",
        "-ar", "44100",
        wav_out
    ]
    subprocess.run(cmd, check=True)


# ---------------- run demucs once and return vocals + folder -----------
def run_demucs_and_get_vocals(wav_path):
    print("🎤 Running Demucs on:", wav_path)
    subprocess.run(["demucs", wav_path], check=True)  # may raise if demucs fails
    base = os.path.basename(wav_path).replace(".wav", "")
    demucs_out = os.path.join("separated", "htdemucs", base)
    vocals_path = os.path.join(demucs_out, "vocals.wav")
    if not os.path.exists(vocals_path):
        raise FileNotFoundError(f"Demucs output missing: {vocals_path}")
    return vocals_path, demucs_out


# ------------ replace clip audio with isolated vocals (loop/trim + boost) ------------
def replace_audio_with_vocals(clip, vocals_wav, boost=2.0):
    vocals = AudioFileClip(vocals_wav)
    if vocals.duration < clip.duration:
        vocals = audio_loop(vocals, duration=clip.duration)
    else:
        vocals = vocals.subclip(0, clip.duration)
    vocals = vocals.volumex(boost)
    return clip.set_audio(vocals)

def mix_video_and_bg(video_clip, music_path, video_boost=2.0, bg_volume=0.10):
    # --- VIDEO AUDIO BOOST ---
    base_audio = video_clip.audio.volumex(video_boost)

    # --- LOAD BACKGROUND MUSIC ---
    bg = AudioFileClip(music_path)

    # Loop/trim music to match video duration
    if bg.duration < video_clip.duration:
        bg = audio_loop(bg, duration=video_clip.duration)
    else:
        bg = bg.subclip(0, video_clip.duration)

    # Light background volume
    bg = bg.volumex(bg_volume)

    # --- MIX BOTH ---
    final_audio = CompositeAudioClip([base_audio, bg])

    return video_clip.set_audio(final_audio)


# --------------- smart continuous-start music (loop to match video) --------------
from moviepy.editor import AudioFileClip, CompositeAudioClip
from moviepy.audio.fx.all import audio_loop

def add_smart_music(final_clip, music_path, music_volume=0.25):
    video_dur = final_clip.duration

    # --- find best continuous segment ---
    seg_start, _ = find_continuous_segment(music_path, video_dur, 0)

    # --- load fresh music object ---
    music = AudioFileClip(music_path).subclip(seg_start)

    # --- loop music so it matches video duration ---
    bg_loop = audio_loop(music, duration=video_dur).volumex(music_volume)

    # --- remove original audio completely ---
    clip_no_audio = final_clip.set_audio(None)

    # --- assign only background music ---
    final_with_music = clip_no_audio.set_audio(bg_loop)

    return final_with_music

def extract_audio(video_path, audio_out="temp_raw.wav"):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "2",
        "-ar", "44100",
        audio_out
    ]
    subprocess.run(cmd)
    return audio_out


def separate_vocals(audio_path):
    print("🎤 Running AI vocal separation (Demucs)…")
    cmd = ["demucs", audio_path]
    subprocess.run(cmd)

    base = audio_path.replace(".wav", "")
    vocals_path = f"separated/htdemucs/{base}/vocals.wav"
    return vocals_path


# --------------------- Main pipeline: create N videos ---------------------
def create_videos(count, input_folder, output_folder, music_path):
    used = set()

    for idx in range(1, count + 1):
        print(f"\n===== Building final video #{idx} =====")

        merged = build_merged_clip(input_folder, used)
        if merged is None:
            print("[INFO] No more clips to use. Stopping.")
            break

        # # No subtitles for now (fast)
        # subbed = merged

        # # Ensure we have audio
        # if subbed.audio is None:
        #     if merged.audio is not None:
        #         print("[INFO] Restoring merged audio onto subbed.")
        #         subbed = subbed.set_audio(merged.audio)
        #     else:
        #         print("[WARN] No audio found; continuing without Demucs.")
        
        # # -------------------------------
        # # ⭐ FAST METHOD: WAV ONLY (no temp video)
        # # -------------------------------
        # temp_id = uuid.uuid4().hex[:8]
        # temp_wav = f"temp_audio_{temp_id}.wav"

        # print(f"[INFO] Extracting audio (WAV only) for Demucs -> {temp_wav}")
        # subbed.audio.write_audiofile(temp_wav, fps=44100, logger=None)

        clip = merged

        # 2️⃣ Save TEMP video for audio separation
        temp_scene_video = "temp_scene.mp4"
        clip.write_videofile(
            temp_scene_video,
            codec="libx264",
            audio_codec="aac",
            fps=24,
            preset="fast",
            threads=4,
       
        )

        # 3️⃣ Extract original audio
        raw_audio = extract_audio(temp_scene_video)

        # 4️⃣ REMOVE MUSIC → Dialogue only
        clean_vocals = separate_vocals(raw_audio)
        audio_clean = AudioFileClip(clean_vocals)
        final_clip = clip.set_audio(audio_clean.volumex(3.0))  # Boost speech

        # Demucs
        # vocals_wav = None
        # try:
        #     vocals_wav, demucs_out_folder = run_demucs_and_get_vocals(temp_wav)
        # except Exception as e:
        #     print(f"[ERROR] Demucs failed: {e}; continuing without vocal separation.")

        # # Replace audio with vocals
        # if vocals_wav:
        #     print("[INFO] Replacing merged audio with Demucs vocals...")
        #     final_vocals = replace_audio_with_vocals(subbed, vocals_wav, boost=4.0)
        # else:
        #     print("[INFO] Using original audio (no vocals extracted).")
        #     final_vocals = subbed

        # Add background music
        print("[INFO] Adding smart continuous background music...")
        final_with_bg = mix_video_and_bg(final_clip, music_path, video_boost=3.0, bg_volume=0.1)

        # Export final video
        out_path = os.path.join(output_folder, f"final_{idx}.mp4")
        print(f"[INFO] Exporting final video -> {out_path}")

        final_with_bg.write_videofile(
            out_path,
            codec="libx264",
            audio_codec="aac",
            fps=DEFAULT_FPS,
            preset="fast",
            threads=4,
            logger=None
        )


        # # ---------------- cleanup ----------------
        # print("[INFO] Cleaning up temp files and clips...")
        # try:
        #     final_with_bg.close()
        # except:
        #     pass
        # try:
        #     final_vocals.close()
        # except:
        #     pass
        # try:
        #     subbed.close()
        # except:
        #     pass
        # try:
        #     merged.close()
        # except:
        #     pass

        # # remove temp files
        # for p in (temp_video, temp_wav):
        #     try:
        #         if os.path.exists(p):
        #             os.remove(p)
        #     except Exception:
        #         pass

        # # remove demucs output folder for this wav if exists
        # if demucs_out_folder and os.path.isdir(demucs_out_folder):
        #     try:
        #         shutil.rmtree(demucs_out_folder)
        #     except Exception:
        #         pass

        # # Final GC
        # gc.collect()

    print("\nAll done.")


# ---------------------- Run script ----------------------
if __name__ == "__main__":
    music_file = pick_music_file()
    how_many = int(input("Enter number of final videos to create: "))
    create_videos(how_many, INPUT_FOLDER, OUTPUT_FOLDER, music_file)
