import librosa
import numpy as np
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# -------------------------------------------------
# STEP 1: STRONG BEAT DETECTION (MAJOR BEATS ONLY)
# -------------------------------------------------
def detect_beats(audio_path):
    y, sr = librosa.load(audio_path)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        trim=True
    )
    return librosa.frames_to_time(beats, sr=sr)


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
# STEP 3: SCENE MOTION SCORING (BEST SCENES FIRST)
# -------------------------------------------------
def scene_motion_score(video, start, end):
    cap = cv2.VideoCapture(video.filename)
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)

    prev = None
    score = 0
    frames = 0

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
# STEP 4: ULTIMATE BEAT-SYNC EDITOR
# -------------------------------------------------
def beat_sync_edit(video_path, music_path, output_path):
    beats = detect_beats(music_path)
    scenes = detect_scenes(video_path)

    video = VideoFileClip(video_path).without_audio()
    audio = AudioFileClip(music_path)

    # Score scenes
    scored_scenes = []
    for s in scenes:
        score = scene_motion_score(video, s[0], s[1])
        scored_scenes.append((score, s))

    # Sort best scenes first
    scored_scenes.sort(reverse=True, key=lambda x: x[0])

    used_scenes = set()
    clips = []

    for i in range(len(beats) - 1):
        beat_len = beats[i + 1] - beats[i]

        for _, scene in scored_scenes:
            if scene in used_scenes:
                continue

            s, e = scene
            if e - s >= beat_len:
                clip = video.subclip(s, s + beat_len)
                clips.append(clip)
                used_scenes.add(scene)
                break

    if not clips:
        raise RuntimeError("No valid clips created — check input video.")

    final_video = concatenate_videoclips(clips, method="compose")
    final_video = final_video.set_audio(audio.subclip(0, final_video.duration))

    # -------------------------------------------------
    # HIGH QUALITY EXPORT
    # -------------------------------------------------
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
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    beat_sync_edit(
        "X SLIDE - 2KE, 808iuli  (Slowed & Reverb) x Demon Slayer - Overtaker (720p, h264).mp4",
        "udaydomadiya08_4l6yu8_Endless Rhythm Loop.mp3",
        "output_edit.mp4"
    )
