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
    return 0.5 * dialogue + 0.25 * emotion + 0.25 * face

# -------------------------------------------------
# PICK BEST SCENE
# -------------------------------------------------
def pick_top_scenes(input_path, scenes, top_k=5):
    """
    Score all scenes and return top_k highest scoring ones.
    """
    scored_scenes = []

    for i, (start, end) in enumerate(scenes):
        dur = end - start

        # duration sanity
        # if dur <= 3 or dur > 11:
        #     continue

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

        scored_scenes.append({
            "start": start,
            "end": end,
            "score": final
        })

        os.remove(audio_path)
        os.remove(frame_path)

    # Sort scenes by score DESC
    scored_scenes.sort(key=lambda x: x["score"], reverse=True)

    return scored_scenes[:top_k]


# -------------------------------------------------
# SAVE RANGE OF SCENES
# -------------------------------------------------
def save_scene_range(input_path, scenes, start_idx=10, end_idx=20, output_folder="firstclip"):
    os.makedirs(output_folder, exist_ok=True)
    selected_scenes = scenes[start_idx:end_idx]

    for idx, scene in enumerate(selected_scenes, start=start_idx):
        start = scene["start"]
        end = scene["end"]
        out_path = os.path.join(
            output_folder,
            f"scene_{int(start*1000)}_{int(end*1000)}.mp4"
        )

        print(
            f"Saving scene {idx+1} | Score={scene['score']:.3f} | "
            f"{start:.2f}s → {end:.2f}s → {out_path}"
        )

        extract_video_segment(input_path, out_path, start, end)


# -------------------------------------------------
# SAVE ALL TOP SCENES
# -------------------------------------------------
def save_top_scenes(input_path, scenes, output_folder="firstclip"):
    os.makedirs(output_folder, exist_ok=True)

    for idx, scene in enumerate(scenes):
        start = scene["start"]
        end = scene["end"]
        out_path = os.path.join(
            output_folder,
            f"scene_{idx+1:02d}_{int(start*1000)}_{int(end*1000)}.mp4"
        )

        print(
            f"Saving scene {idx+1} | Score={scene['score']:.3f} | "
            f"{start:.2f}s → {end:.2f}s → {out_path}"
        )

        extract_video_segment(input_path, out_path, start, end)


# -------------------------------------------------
# MAIN scene extraction helper
# -------------------------------------------------
def find_top_scenes(video_path, top_k=20, start_idx=None, end_idx=None, output_folder="firstclip"):
    """
    1) Detect scenes
    2) Pick top_k scenes
    3) Optionally save a slice of them
    """
    scenes = detect_scenes(video_path)
    top_scenes = pick_top_scenes(video_path, scenes, top_k)

    # Optional slicing of the top scenes
    if start_idx is not None and end_idx is not None:
        selected_scenes = top_scenes[start_idx:end_idx]
    else:
        selected_scenes = top_scenes

    save_top_scenes(video_path, selected_scenes, output_folder)

    return selected_scenes


# -------------------------------------------------
# CLI EXECUTION
# -------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--outfolder", default="firstclip", help="Folder to save scenes")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)

    args = parser.parse_args()

    scenes = find_top_scenes(
        video_path=args.video,
        top_k=args.topk,
        start_idx=args.start,
        end_idx=args.end,
        output_folder=args.outfolder
    )

    if not scenes:
        print("❌ No suitable scenes found.")
    else:
        print(f"✅ Saved {len(scenes)} scenes → Folder: {args.outfolder}")
