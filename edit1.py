import librosa
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

def detect_beats(audio_path):
    y, sr = librosa.load(audio_path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    return librosa.frames_to_time(beat_frames, sr=sr)

def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=30))

    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scenes = scene_manager.get_scene_list()
    video_manager.release()

    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scenes]

def beat_sync_edit(video_path, music_path, output_path):
    beats = detect_beats(music_path)
    scenes = detect_scenes(video_path)

    video = VideoFileClip(video_path).without_audio()
    audio = AudioFileClip(music_path)

    used_scenes = set()
    clips = []

    for i in range(len(beats) - 1):
        beat_duration = beats[i + 1] - beats[i]

        for scene in scenes:
            if scene not in used_scenes:
                s, e = scene
                if e - s >= beat_duration:
                    clips.append(video.subclip(s, s + beat_duration))
                    used_scenes.add(scene)
                    break

    final_video = concatenate_videoclips(clips, method="compose")
    final_video = final_video.set_audio(audio.subclip(0, final_video.duration))

    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=30,
        bitrate="8000k",
        audio_bitrate="320k",
        preset="slow",
        threads=4
    )