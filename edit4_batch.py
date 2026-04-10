import os
import librosa
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# -------------------------------------------------
# STEP 1: HUMAN-LIKE BEAT DETECTION
# -------------------------------------------------
def detect_beats(audio_path, min_gap=0.7, every_n=2):
    y, sr = librosa.load(audio_path)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        trim=True
    )

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_times = beat_times[::every_n]

    filtered = [beat_times[0]]
    for t in beat_times[1:]:
        if t - filtered[-1] >= min_gap:
            filtered.append(t)

    return np.array(filtered)


# -------------------------------------------------
# STEP 2: SCENE DETECTION
# -------------------------------------------------
def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27))

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scenes = scene_manager.get_scene_list()
    video_manager.release()

    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]


# -------------------------------------------------
# STEP 3: SCENE MOTION SCORE
# -------------------------------------------------
def scene_motion_score(video_path, start, end):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    prev, score, frames = None, 0, 0

    while cap.get(cv2.CAP_PROP_POS_MSEC) < end * 1000:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            score += np.mean(cv2.absdiff(prev, gray))
            frames += 1
        prev = gray

    cap.release()
    return score / frames if frames else 0


# -------------------------------------------------
# STEP 4: EDIT SINGLE VIDEO
# -------------------------------------------------
def process_video(video_path, beats, music_path, output_path):
    print(f"🎬 Processing: {os.path.basename(video_path)}")

    scenes = detect_scenes(video_path)
    video = VideoFileClip(video_path).without_audio()
    audio = AudioFileClip(music_path)

    scored_scenes = []
    for s in scenes:
        score = scene_motion_score(video_path, s[0], s[1])
        scored_scenes.append((score, s))

    scored_scenes.sort(reverse=True, key=lambda x: x[0])

    used_scenes = set()
    clips = []

    for i in range(len(beats) - 1):
        beat_len = max(beats[i + 1] - beats[i], 0.8)

        for _, scene in scored_scenes:
            if scene in used_scenes:
                continue

            s, e = scene
            if e - s >= beat_len:
                clips.append(video.subclip(s, s + beat_len))
                used_scenes.add(scene)
                break

    if not clips:
        print("⚠️ Skipped (no valid scenes)")
        return

    final_video = concatenate_videoclips(clips, method="compose")
    final_video = final_video.set_audio(audio.subclip(0, final_video.duration))

    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=video.fps,
        bitrate="9000k",
        audio_bitrate="320k",
        preset="slow",
        threads=4
    )


# -------------------------------------------------
# STEP 5: BATCH RUNNER
# -------------------------------------------------
def batch_process(input_video_dir, music_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("🎵 Detecting beats once...")
    beats = detect_beats(music_path)

    for file in os.listdir(input_video_dir):
        if not file.lower().endswith((".mp4", ".mov", ".mkv")):
            continue

        video_path = os.path.join(input_video_dir, file)
        output_path = os.path.join(
            output_dir,
            os.path.splitext(file)[0] + "_edit.mp4"
        )

        try:
            process_video(video_path, beats, music_path, output_path)
        except Exception as e:
            print(f"❌ Failed {file}: {e}")


# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    batch_process(
        input_video_dir="input_videos",
        music_path="music/track.mp3",
        output_dir="output"
    )
