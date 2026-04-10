from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cpu", compute_type="int8")

segments, info = model.transcribe(
    "/Users/uday/Downloads/edmmusic/input_voices/voiced.wav",
    beam_size=5
)

print("Language:", info.language)

for segment in segments:
    print(f"[{segment.start:.2f}s → {segment.end:.2f}s] {segment.text}")
