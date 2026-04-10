# musicrem.py

import os
import subprocess
import gc
from moviepy.editor import VideoFileClip, AudioFileClip
from pathlib import Path

VIDEO_INPUT = "/Users/uday/Downloads/fictic/best_scene_final.mp4"  # not used when importing
OUTPUT_VIDEO = "dialogue_only.mp4"                                 # not used when importing

DEMUCS_PYTHON = Path.home() / "demucs310" / "bin" / "python"


def extract_audio_simple(video_path, audio_out="temp_audio.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_out, logger=None)

    # HARD CLEANUP
    clip.audio.close()
    clip.close()
    del clip
    gc.collect()

    return audio_out


def separate_audio_with_demucs(audio_path):
    print("🎤 Running Demucs (Python 3.10 env)...")

    env = os.environ.copy()
    env["DEMUCS_AUDIO_BACKEND"] = "soundfile"

    subprocess.run(
        [
            str(DEMUCS_PYTHON),
            "-m",
            "demucs.separate",
            "-n",
            "htdemucs",
            audio_path,
        ],
        check=True,
        env=env,
        cwd=os.getcwd(),
    )

    folder = os.path.join(
        "separated",
        "htdemucs",
        os.path.splitext(os.path.basename(audio_path))[0],
    )
    vocal_path = os.path.join(folder, "vocals.wav")

    if not os.path.exists(vocal_path):
        raise FileNotFoundError("❌ Demucs output not found: vocals.wav")

    return vocal_path


def attach_dialogue_to_video(video_path, dialogue_wav, output_path):
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(dialogue_wav)

    final = video.set_audio(new_audio)
    final.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )

    # HARD CLEANUP
    new_audio.close()
    video.close()
    final.close()
    del new_audio, video, final
    gc.collect()


if __name__ == "__main__":
    print("🎬 Extracting audio...")
    audio = extract_audio_simple(VIDEO_INPUT)

    print("🎤 Separating vocals/dialogue using Demucs...")
    dialogue_track = separate_audio_with_demucs(audio)

    print("🔊 Re-attaching dialogue (music removed)...")
    attach_dialogue_to_video(VIDEO_INPUT, dialogue_track, OUTPUT_VIDEO)

    print("✅ DONE: Dialogue-only video saved as:", OUTPUT_VIDEO)
