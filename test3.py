from moviepy.editor import VideoFileClip, CompositeAudioClip, AudioFileClip, afx

# Replace with your video and music paths
video_path = "/Users/uday/Downloads/fictic/out/final_video_0_20251116_182143.mp4"   # a small test clip
music_path = "music/COM MEDO! (Super Slowed) - Sayfalse.mp3"

# Load video
clip = VideoFileClip(video_path)
video_len = clip.duration

# Load music
music = AudioFileClip(music_path)

# Test 1: Using subclip only (no looping)
music_subclip = music.subclip(0, min(music.duration, video_len))
final1 = clip.set_audio(CompositeAudioClip([clip.audio, music_subclip]))
final1.write_videofile("test_no_loop.mp4", fps=24, audio_codec="aac")

# Test 2: Using audio_loop
from moviepy.audio.fx.all import audio_loop
music_looped = audio_loop(music, duration=video_len)
final2 = clip.set_audio(CompositeAudioClip([clip.audio, music_looped]))
final2.write_videofile("test_with_loop.mp4", fps=24, audio_codec="aac")