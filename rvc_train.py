import os
from pydub import AudioSegment
import subprocess

# ---------------- CONFIG ----------------
# Target voice folder with MP3s you provide
TARGET_MP3_DIR = "dataset/obama"
TARGET_WAV_DIR = "dataset/target_voice_wav"

# Source audio (to be converted)
SOURCE_AUDIO = "/Users/uday/Downloads/President Obama's best speeches - CNN.mp3"

# Trained or pretrained RVC model path
MODEL_PATH = "pretrained_models/target_model.pth"

# Output file
OUTPUT_FILE = "converted_full_output.wav"

# Temp working folder
WORKDIR = "rvc_work"
os.makedirs(WORKDIR, exist_ok=True)
os.makedirs(TARGET_WAV_DIR, exist_ok=True)

# Chunk duration in milliseconds
CHUNK_MS = 60000
DEVICE = "cpu"

# ---------------- STEP 1: Convert Target MP3s → WAV ----------------
print("Converting target MP3s → WAV (16kHz mono)...")
for fname in os.listdir(TARGET_MP3_DIR):
    if fname.lower().endswith(".mp3"):
        src_path = os.path.join(TARGET_MP3_DIR, fname)
        dst_path = os.path.join(TARGET_WAV_DIR, os.path.splitext(fname)[0] + ".wav")
        audio = AudioSegment.from_file(src_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(dst_path, format="wav")
        print(f"Converted: {fname} -> {dst_path}")

print("Target voice WAVs ready ✔")

# ---------------- STEP 2: Split Source Audio ----------------
def split_source_audio(input_path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    chunks = []
    for i in range(0, len(audio), CHUNK_MS):
        chunk_path = os.path.join(WORKDIR, f"chunk_{i//1000}.wav")
        audio[i:i + CHUNK_MS].export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

print("Splitting source audio into chunks...")
source_chunks = split_source_audio(SOURCE_AUDIO)

# ---------------- STEP 3: Run RVC inference ----------------
def rvc_infer(chunk_path, out_path):
    subprocess.run([
        "python", "-m", "rvc.wrapper.cli.cli", "infer",
        "--inputPath", chunk_path,        # corrected
        "--refPath", TARGET_WAV_DIR,      # corrected
        "--modelPath", MODEL_PATH,        # corrected
        "--outputPath", out_path,         # corrected
        "--device", DEVICE
    ], check=True)


converted_chunks = []
for i, chunk in enumerate(source_chunks):
    out_chunk = os.path.join(WORKDIR, f"converted_{i}.wav")
    print(f"Converting chunk {i+1}/{len(source_chunks)}...")
    rvc_infer(chunk, out_chunk)
    converted_chunks.append(out_chunk)

# ---------------- STEP 4: Merge Converted Chunks ----------------
print("Merging converted chunks...")
final_audio = AudioSegment.empty()
for f in converted_chunks:
    final_audio += AudioSegment.from_file(f)

final_audio.export(OUTPUT_FILE, format="wav")
print(f"✅ Conversion complete! Final output: {OUTPUT_FILE}")
