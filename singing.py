import torch
from TTS.api import TTS
from pydub import AudioSegment
import os

# Disable WAVLM to reduce load
os.environ["COQUI_TTS_DISABLE_WAVLM"] = "1"

# Split audio into equal segments (NO OVERLAP)
def split_audio(input_path, chunk_duration_ms=60000):
    audio = AudioSegment.from_file(input_path)
    chunks = []
    start = 0

    while start < len(audio):
        end = min(start + chunk_duration_ms, len(audio))
        chunk = audio[start:end]

        chunk_name = f"temp_chunk_{start}.wav"
        chunk.set_frame_rate(16000).set_channels(1).export(chunk_name, format="wav")
        chunks.append(chunk_name)

        start += chunk_duration_ms  # <-- NO overlap

    return chunks

# Convert a single source chunk into target voice
def convert_segment(src_chunk, trg_wav, tts_model, output_path):
    tts_model.voice_conversion_to_file(
        source_wav=src_chunk,
        target_wav=trg_wav,
        file_path=output_path
    )

# Merge converted segments WITHOUT crossfade
def merge_segments(segment_files, output_file):
    final_audio = AudioSegment.empty()

    for seg_file in segment_files:
        seg = AudioSegment.from_file(seg_file)
        final_audio += seg  # pure concat

    final_audio.export(output_file, format="wav")
    print(f"Merged audio saved -> {output_file}")

# Full pipeline
def run_voice_conversion(src_path, trg_path, output_path="converted_full_output.wav"):
    print("Splitting source audio into segments...")
    src_chunks = split_audio(src_path, chunk_duration_ms=60000)

    # Prepare full target voice reference
    trg_wav = "temp_target.wav"
    audio = AudioSegment.from_file(trg_path).set_frame_rate(16000).set_channels(1)
    audio.export(trg_wav, format="wav")

    print("Loading FreeVC voice conversion model...")
    tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24")

    converted_files = []
    for i, src_seg in enumerate(src_chunks, 1):
        out_seg = f"converted_segment_{i}.wav"
        print(f"Converting segment {i}/{len(src_chunks)}...")
        convert_segment(src_seg, trg_wav, tts, out_seg)
        converted_files.append(out_seg)

    print("Merging segments...")
    merge_segments(converted_files, output_path)

    print("All segments converted and merged successfully!")

# Usage
if __name__ == "__main__":
    run_voice_conversion(
        src_path="/Users/uday/Downloads/Luis Fonsi - Despacito ft. Daddy Yankee - LuisFonsiVEVO.mp3",
        trg_path="/Users/uday/Downloads/President Obama's best speeches - CNN.mp3",
        output_path="converted_full_output.wav"
    )
