from moviepy.editor import VideoFileClip, AudioFileClip
import numpy as np
import soundfile as sf
import librosa

# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------
INPUT_VIDEO = "best_scene.mp4"
OUTPUT_VIDEO = "audio_loud_test.mp4"

BOOST_GAIN = 3.0      # You can change 1.0 → 3.0 also
LIMIT_THRESHOLD = 0.98  # Hard limiter level
NORMALIZE_TARGET = -1.0  # dBFS target (−1 dBFS safe)
# --------------------------------------------------


def apply_limiter(audio, threshold=0.98):
    """Hard limiter: clamps peaks to avoid distortion"""
    return np.clip(audio, -threshold, threshold)


def normalize_audio(audio, target_db=-1.0):
    """Normalize audio to target peak level"""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    target_linear = 10 ** (target_db / 20)
    return audio * (target_linear / peak)


def boost_audio_with_protection(video_path, output_path):
    print("🔊 Loading video...")
    video = VideoFileClip(video_path)
    audio = video.audio

    # Extract raw samples
    print("🎧 Extracting audio samples...")
    temp_audio = "temp_audio.wav"
    audio.write_audiofile(temp_audio, verbose=False, logger=None)

    samples, sr = sf.read(temp_audio)

    print("⚡ Boosting audio gain...")
    samples = samples * BOOST_GAIN

    print("🛡 Applying limiter...")
    samples = apply_limiter(samples, LIMIT_THRESHOLD)

    print("📈 Normalizing...")
    samples = normalize_audio(samples, NORMALIZE_TARGET)

    # Save processed audio
    print("💾 Saving processed audio...")
    out_wav = "processed_audio.wav"
    sf.write(out_wav, samples, sr)

    print("🎬 Attaching louder audio to video...")
    new_audio = AudioFileClip(out_wav)
    final = video.set_audio(new_audio)

    final.write_videofile(output_path, codec="libx264", audio_codec="aac")

    print("✅ Done! Output saved:", output_path)


# ---------------- RUN TEST ----------------
boost_audio_with_protection(INPUT_VIDEO, OUTPUT_VIDEO)
