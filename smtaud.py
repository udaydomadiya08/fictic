# ===============================================================
# SMART AUDIO MODULE
# Noise clean + loudness detection + adaptive smart ducking
# ===============================================================

import librosa
import numpy as np
from moviepy.editor import AudioFileClip, CompositeAudioClip

# ---------------------------------------------------------------
# 1) Noise Removal (spectral gating)
# ---------------------------------------------------------------
import librosa
import numpy as np
import soundfile as sf  # updated: use soundfile instead of librosa.output

def remove_noise(input_audio_path, out_path="clean_dialogue.wav"):
    # 1️⃣ Load audio
    y, sr = librosa.load(input_audio_path, sr=None)

    # 2️⃣ Estimate noise profile from first 0.5 sec
    noise_sample = y[: int(0.5 * sr)]
    noise_profile = np.mean(np.abs(librosa.stft(noise_sample)), axis=1)

    # 3️⃣ Apply spectral subtraction
    S = librosa.stft(y)
    S_clean = S - noise_profile.reshape((-1, 1))
    S_clean = np.maximum(S_clean, 0)  # avoid negative values

    # 4️⃣ Inverse STFT to waveform
    cleaned = librosa.istft(S_clean)

    # 5️⃣ Save denoised audio using soundfile
    sf.write(out_path, cleaned, sr)

    return out_path


# ---------------------------------------------------------------
# 2) Speech Loudness Detection
# ---------------------------------------------------------------
def compute_speech_loudness(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    rms = librosa.feature.rms(y=y)[0]
    return float(np.mean(rms))


# ---------------------------------------------------------------
# 3) Smart Adaptive Ducking
# ---------------------------------------------------------------
def smart_duck_music(vocal_audio_clip, music_audio_path, speech_loudness, duration, music_start=0):
    """
    speech_loudness HIGH → music lower
    speech_loudness LOW → music louder
    music_start → start music from specific time position
    """
    # Map loudness → ducking level
    if speech_loudness > 0.10:   # very strong voice
        duck_volume = 0.05
    elif speech_loudness > 0.05: # normal
        duck_volume = 0.10
    else:                        # weak speech
        duck_volume = 0.15

    # Start music from music_start position
    music = AudioFileClip(music_audio_path).subclip(music_start, music_start + duration)
    music = music.volumex(duck_volume)

    return CompositeAudioClip([vocal_audio_clip, music])

