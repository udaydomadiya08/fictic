
# from pydub import AudioSegment

# def mp3_to_wav(mp3_path, wav_path):
#     sound = AudioSegment.from_mp3(mp3_path)
#     sound.export(wav_path, format="wav")
#     print("Converted MP3 → WAV:", wav_path)

# mp3_to_wav("voice.mp3", "output2.wav")

# input_wav = "output2.wav"
# ref_wav = "output.wav"
# output_wav = "audio_final.wav"
# from gtts import gTTS
# input_wav = "output2.wav"
ref_wav = "output.wav"
# output_wav = "audio_final.wav"
from TTS.api import TTS

# Initialize YourTTS multilingual model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")

# Generate TTS from text
tts.tts_to_file(
    text="Hello bitch, long time no see! can i kiss you?",
    speaker_wav=ref_wav,  # leave None for default voice
    language="en",
    file_path="Wav2Lip/output1.wav"
)

print("✅ TTS audio saved as output1.wav")

import os
import sys
import subprocess
import cv2
import imageio

# -------------------------------
# MONKEY PATCH OPENCV VIDEOWRITER
# -------------------------------
class FakeVideoWriter:
    def __init__(self, outfile, fourcc, fps, size):
        self.outfile = outfile
        self.writer = imageio.get_writer(outfile, fps=fps)

    def isOpened(self):
        return True

    def write(self, frame):
        # BGR -> RGB
        frame = frame[:, :, ::-1]
        self.writer.append_data(frame)

    def release(self):
        self.writer.close()

cv2.VideoWriter = FakeVideoWriter

# -------------------------------
# IMPORT WAV2LIP
# -------------------------------
sys.path.append("Wav2Lip")
from Wav2Lip.inference import parser, run_inference

# -------------------------------
# RUN WAV2LIP INFERENCE
# -------------------------------
def run_wav2lip_inference(
    checkpoint_path,
    face_video,
    audio_path,
    output_dir="temp_result",
    static=False,
    fps=24,
    wav2lip_batch_size=128,
    resize_factor=1,
    out_height=480
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_video = os.path.join(output_dir, "output.avi")

    args_list = [
        "--checkpoint_path", checkpoint_path,
        "--face", face_video,
        "--audio", audio_path,
        "--outfile", output_video,
        "--fps", str(fps),
        "--wav2lip_batch_size", str(wav2lip_batch_size),
        "--resize_factor", str(resize_factor),
        "--out_height", str(out_height),
    ]

    if static:
        args_list.append("--static")

    args = parser.parse_args(args_list)

    print("🎬 Starting Wav2Lip inference...")
    run_inference(args)
    print("✅ Wav2Lip inference done!")
    print("Saved temporary video (AVI):", output_video)

    return output_video

# -------------------------------
# USER CONFIG
# -------------------------------
checkpoint_path = "checkpoints/wav2lip_gan.pth"
face_video = "Wav2Lip/scene_18_208120_214800.mp4"
audio_path = "Wav2Lip/output1.wav"
output_dir = "temp"
output_final = "/Users/uday/Downloads/fictic/fic21.mp4"

# Run inference (full video, not static)
temp_video = run_wav2lip_inference(
    checkpoint_path=checkpoint_path,
    face_video=face_video,
    audio_path=audio_path,
    output_dir=output_dir,
    static=False,
    fps=24,
    wav2lip_batch_size=256,
    resize_factor=1,
    out_height=1080
)

# -------------------------------
# MERGE FINAL VIDEO WITH AUDIO
# -------------------------------
os.makedirs(os.path.dirname(output_final), exist_ok=True)

print("🎵 Merging final video with audio...")

subprocess.run([
    "ffmpeg", "-y",
    "-i", temp_video,
    "-i", audio_path,
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-c:v", "libx264",
    "-preset", "medium",
    "-c:a", "aac",
    "-b:a", "320k",
    "-ar", "48000",
    "-ac", "2",
    "-shortest",
    output_final
], check=True)

print(f"🎉 Done! Final video with audio saved at:\n{output_final}")


# text = "Hello, this is a test voice. i am jsut kidding you know, ha ha ha ha ha!!!!!!"
# tts = gTTS(text, lang='en')
# tts.save("voice.mp3")





# import subprocess



# # # === Step 1: Convert merged_audio.mp3 to WAV ===
# # mp3_path = '/Users/uday/Downloads/VIDEOYT/audio/merged_audio.mp3'
# # wav_path = '/Users/uday/Downloads/VIDEOYT/Wav2Lip/merged_audio.wav'
# # mp3_to_wav(mp3_path, wav_path)

# # === Step 2: Run OpenVoice CLI via subprocess ===


# openvoice_command = [
#     "python3", "-m", "openvoice_cli", "single",
#     "-i", input_wav,  # your converted WAV
#     "-r", ref_wav, # reference voice
#     "-o", output_wav,
#     "-d", "cpu"      # run on CPU
# ]

# try:
#     subprocess.run(openvoice_command, check=True)
#     print(f"✅ OpenVoice processed: {output_wav}")
# except subprocess.CalledProcessError as e:
#     print("❌ OpenVoice CLI failed:")
#     print(e)



# #multi batch
# output = tune_batch(
#     input_dir='path_to_input_directory',   # folder with audio files
#     ref_file='path_to_reference.wav',      # reference audio for tone
#     output_dir='path_to_output_directory', # where converted files go
#     device='cpu',                          # use CPU instead of GPU
#     output_format='.wav'                    # output format
# )


# import os
# from pydub import AudioSegment

# from pydub import AudioSegment

# wav_path = "input.wav"
# mp3_path = "output.mp3"

# audio = AudioSegment.from_wav(wav_path)
# audio.export(mp3_path, format="mp3")

# print("WAV → MP3 done!")


# def batch_mp3_to_wav(folder):
#     for file in os.listdir(folder):
#         if file.endswith(".mp3"):
#             mp3_path = os.path.join(folder, file)
#             wav_path = os.path.join(folder, file.replace(".mp3", ".wav"))
#             AudioSegment.from_mp3(mp3_path).export(wav_path, format="wav")
#             print("Converted:", file)

# batch_mp3_to_wav("folder_path")

# from moviepy.editor import VideoFileClip

# # Input video
# video_path = "firstclip/scene_18_208120_214800.mp4"

# # Extract audio
# video = VideoFileClip(video_path)
# audio = video.audio

# # Trim between exact seconds
# # start_sec = 0      # trim start
# # end_sec = video.duration       # trim end

# # trimmed_audio = audio.subclip(start_sec, end_sec)

# # Save result (WAV or MP3)
# output_path = "output.wav"   # change to .mp3 if you want mp3

# audio.write_audiofile(output_path)

# print("🎉 Audio extracted and trimmed successfully!")



