import os
import cv2
import numpy as np
import subprocess
from moviepy.editor import VideoFileClip, AudioFileClip
import whisper


# -------------------------------------------------
# MODELS
# -------------------------------------------------
whisper_model = whisper.load_model("small")     # fast + good accuracy
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")


# -------------------------------------------------
# FAST FFMPEG SCENE DETECTION (1–2 seconds)
# -------------------------------------------------
def detect_scenes(video_path, threshold=0.25):
    """
    Uses ffmpeg built-in scene detection. Much faster than PySceneDetect.
    Returns list of (start, end) timestamps.
    """

    cmd = [
        "ffmpeg", "-i", video_path,
        "-filter:v", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null", "-"
    ]

    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
    lines = proc.stderr.readlines()

    cut_points = []
    for line in lines:
        if "pts_time:" in line:
            t = float(line.split("pts_time:")[1].split(" ")[0])
            cut_points.append(t)

    # Build scenes (start, end)
    scenes = []
    prev = 0
    for t in cut_points:
        scenes.append((prev, t))
        prev = t

    return scenes


# -------------------------------------------------
# UTIL FUNCTIONS
# -------------------------------------------------
def extract_audio_segment(video_path, audio_path, start, end):
    clip = VideoFileClip(video_path).subclip(start, end)
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    clip.close()


def extract_video_segment(video_path, output_path, start, end):
    clip = VideoFileClip(video_path).subclip(start, end)
    clip.write_videofile(output_path, codec="libx264",
                         audio_codec="aac", verbose=False, logger=None)
    clip.close()


def save_frame(video_path, frame_path, timestamp):
    clip = VideoFileClip(video_path)
    frame = clip.get_frame(timestamp)
    clip.close()
    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


# -------------------------------------------------
# DIALOGUE SCORE
# -------------------------------------------------
def transcribe(audio_path):
    try:
        result = whisper_model.transcribe(audio_path, fp16=False)
        return result["text"].strip()
    except:
        return ""


def dialogue_strength(text):
    if not text or len(text) < 10:
        return 0.0

    score = 0
    if text[-1] in ".!?":
        score += 0.4
    if len(text.split()) >= 4:
        score += 0.4
    if text[0].isupper():
        score += 0.2

    return score


# -------------------------------------------------
# EMOTION SCORE (audio energy)
# -------------------------------------------------
def emotion_score(audio_path):
    try:
        clip = AudioFileClip(audio_path)
        arr = clip.to_soundarray(fps=22050)
        clip.close()

        if arr.size == 0:
            return 0.0

        mono = arr.mean(axis=1)
        variability = np.std(mono)
        loudness = np.mean(np.abs(mono))

        return float(min(1.0, loudness * 3 + variability * 4))

    except:
        return 0.0


# -------------------------------------------------
# FACE INTENSITY (very fast)
# -------------------------------------------------
def face_intensity(frame_path):
    img = cv2.imread(frame_path)
    if img is None:
        return 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    count = len(faces)
    return min(1.0, count / 3)


# -------------------------------------------------
# FINAL SCENE SCORE
# -------------------------------------------------
def scene_score(dialogue, emotion, face):
    return 0.45 * dialogue + 0.45 * emotion + 0.10 * face


# -------------------------------------------------
# PICK BEST SCENE
# -------------------------------------------------
def pick_best_scene(input_path, scenes):
    best_score = -1
    best_scene = None

    for i, (start, end) in enumerate(scenes):
        dur = end - start

        # Only 3–12 sec clips for hooks
        if dur < 3 or dur > 12:
            continue

        audio_path = f"tmp_audio_{i}.wav"
        frame_path = f"tmp_frame_{i}.jpg"

        extract_audio_segment(input_path, audio_path, start, end)
        save_frame(input_path, frame_path, start + dur / 2)

        text = transcribe(audio_path)
        dialog = dialogue_strength(text)
        emo = emotion_score(audio_path)
        face = face_intensity(frame_path)

        final = scene_score(dialog, emo, face)

        print(f"Scene {i}: Score={final:.3f} | D={dialog:.2f} | E={emo:.2f} | F={face:.2f}")

        if final > best_score:
            best_score = final
            best_scene = (start, end)

        os.remove(audio_path)
        os.remove(frame_path)

    return best_scene


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def find_best_scene(video_path):
    """
    Returns (start, end) tuple for the best scene,
    or None if no suitable scene is found.
    """
    scenes = detect_scenes(video_path)
    best = pick_best_scene(video_path, scenes)
    return best



# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default="best_scene.mp4")
    args = parser.parse_args()

    best = find_best_scene(args.video)
    if best is None:
        print("❌ No suitable scene found.")
    else:
        start, end = best
        print(f"✅ Best scene: {start:.2f}s → {end:.2f}s")
        extract_video_segment(args.video, args.out, start, end)
        print(f"🎬 Saved: {args.out}")
