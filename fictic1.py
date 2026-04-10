#!/usr/bin/env python3
"""
fictic_final_v2.py
Final Fictic-style vertical edit (9:16) with Whisper subtitles (neon bounce),
beat-synced creative cutting, effects, and final audio mix (original audio + music).

Usage:
python3 fictic_final_v2.py \
  --input_videos ./input_vid \
  --music ./music_folder_or_file \
  --out ./out/final_edit_9x16.mp4 \
  [--whisper_model small]

Notes:
- Requires ffmpeg, moviepy, openai-whisper, librosa, numpy, tqdm, soundfile
- Example install:
  pip install moviepy openai-whisper librosa numpy tqdm soundfile
"""
import os
import glob
import random
import argparse
import tempfile
import shutil
import math
import numpy as np
import whisper
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from moviepy.editor import (
    VideoFileClip, AudioFileClip, concatenate_videoclips,
    CompositeVideoClip, TextClip, vfx, CompositeAudioClip
)
from moviepy.audio.fx.all import audio_loop

# ---------- Config ----------
OUT_W, OUT_H = 1080, 1920   # Vertical 9:16
FPS = 30
FONT_NAME = "BebasNeue-Regular"   # change if not installed (e.g., "Arial-Bold")
FONT_SIZE = 84
STROKE_WIDTH = 3
TEXT_COLOR = "white"
GLOW_COLORS = ["#00ffff", "#ff2d95"]  # cyan + pink glow layers
WHISPER_DEFAULT = "small"

# Effects config
GLITCH_PROB = 0.09
GLITCH_MAX_DUR = 0.12

# ---------- Utilities ----------
def find_videos_in_folder(folder):
    patterns = ["*.mp4","*.mov","*.mkv","*.avi","*.webm","*.mpg","*.mpeg"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(folder,p)))
    return sorted(files)

def find_music_file(music_arg):
    """If folder, pick the first music file. If file, return path."""
    if os.path.isdir(music_arg):
        patterns = ["*.mp3","*.wav","*.m4a","*.flac","*.aac"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(music_arg,p)))
        files = sorted(files)
        if not files:
            raise FileNotFoundError(f"No music files found in folder: {music_arg}")
        return files[0]
    elif os.path.isfile(music_arg):
        return music_arg
    else:
        raise FileNotFoundError(f"Music path not found: {music_arg}")

def detect_beats(music_path, sr=22050, hop_length=512):
    """Return beat timestamps in seconds; None if fails."""
    try:
        import librosa
        y, sr = librosa.load(music_path, sr=sr, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        if len(beats) < 2:
            return None
        return beats
    except Exception as e:
        print("Beat detection failed:", e)
        return None

def center_crop_vertical(clip, out_w=OUT_W, out_h=OUT_H):
    """Crop clip to vertical 9:16 while keeping aspect ratio."""
    scale_w = out_w / clip.w
    scale_h = out_h / clip.h
    scale = max(scale_w, scale_h)
    new_w, new_h = int(clip.w * scale), int(clip.h * scale)
    clip = clip.resize((new_w, new_h))
    x1 = (new_w - out_w) // 2
    y1 = (new_h - out_h) // 2
    return clip.crop(x1=x1, y1=y1, x2=x1+out_w, y2=y1+out_h)

def quick_glitch_overlay(clip, rng=None):
    """Random short RGB shifts / flashes layered on clip."""
    if rng is None:
        rng = np.random.RandomState(int(random.random()*1e6))
    overlays = []
    t = 0.0
    step = 0.07
    while t < clip.duration:
        if rng.rand() < GLITCH_PROB:
            g_d = float(rng.rand()*GLITCH_MAX_DUR) + 0.03
            t0 = t
            t1 = min(clip.duration, t+g_d)
            sub = clip.subclip(t0, t1).fx(vfx.colorx, 1.5).set_opacity(0.45)
            dx = int(rng.randint(-10,10))
            dy = int(rng.randint(-10,10))
            sub = sub.set_position((dx,dy)).set_start(t0)
            overlays.append(sub)
            t += g_d
        t += step
    if overlays:
        return CompositeVideoClip([clip] + overlays).set_duration(clip.duration)
    return clip

def make_variant_from_clip(src_path, seg_start, seg_dur, variant_type="normal"):
    """Create a creative variant from a source clip segment."""
    src = VideoFileClip(src_path)
    seg_end = min(src.duration, seg_start + seg_dur)
    clip = src.subclip(seg_start, seg_end).set_fps(FPS)
    clip = center_crop_vertical(clip)

    if variant_type == "speedup":
        clip = clip.fx(vfx.speedx, 1.25)
    elif variant_type == "slow":
        # stretch slightly
        clip = clip.fx(vfx.speedx, final_duration=max(0.2, seg_dur*1.12))
    elif variant_type == "reverse":
        clip = clip.fx(vfx.time_mirror)
    elif variant_type == "stutter":
        small = clip.subclip(0, min(0.12, clip.duration))
        clip = concatenate_videoclips([small]*3).set_duration(min(clip.duration, 0.36))

    # subtle zoom (Ken Burns)
    zoom_amount = 1.04 + random.uniform(0, 0.05)
    clip = clip.fx(vfx.resize, lambda t: 1.0 + (zoom_amount - 1.0)*(t / max(1e-6, clip.duration)))

    # micro ramp for punch
    ramp_dur = min(0.09, clip.duration*0.12)
    if ramp_dur > 0.02 and clip.duration > ramp_dur + 0.01:
        main_dur = clip.duration - ramp_dur
        try:
            a = clip.subclip(0, main_dur)
            b = clip.subclip(main_dur, clip.duration).fx(vfx.speedx, 1.08)
            clip = concatenate_videoclips([a,b])
        except Exception:
            pass

    clip = clip.fx(vfx.colorx, 1.05)
    clip = quick_glitch_overlay(clip, rng=np.random.RandomState(int(random.random()*1e6)))
    return clip

def build_sequence_filling_duration(video_paths, target_duration, music_beats=None):
    """Construct sequence of short creative segments totaling target_duration."""
    rng = np.random.RandomState(123456)
    src_durations = []
    for p in video_paths:
        try:
            d = VideoFileClip(p).duration
        except Exception:
            d = 0.0
        src_durations.append(max(0.0, d))

    # choose target segment gaps guided by beats if available
    if music_beats is not None and len(music_beats) > 1:
        gaps = np.diff(music_beats).tolist()
        gaps = [max(0.45, min(2.6, g)) for g in gaps]
    else:
        avg = 1.2
        n = max(8, int(target_duration / avg))
        gaps = list(np.clip(np.random.normal(loc=avg, scale=0.35, size=n), 0.5, 2.4))

    segments = []
    cur = 0.0
    gi = 0
    cnt_src = len(video_paths)
    while cur < target_duration - 0.25:
        gap = gaps[gi % len(gaps)]
        desired = min(gap * random.uniform(0.88,1.12), target_duration - cur)
        if cnt_src == 1:
            src_i = 0
        else:
            src_i = (gi + rng.randint(0, cnt_src)) % cnt_src
        src_path = video_paths[src_i]
        src_dur = src_durations[src_i] if src_durations[src_i] > 0 else 0.01
        margin = max(0.0, src_dur - desired)
        start = float(rng.uniform(0, margin)) if margin > 0 else 0.0
        vt = rng.choice(["normal","speedup","slow","reverse","stutter"], p=[0.45,0.2,0.15,0.1,0.1])
        segments.append({"src": src_path, "start": start, "dur": desired, "variant": vt})
        cur += desired
        gi += 1

    # convert segments to clips
    clip_list = []
    for s in tqdm(segments, desc="Creating segments"):
        try:
            c = make_variant_from_clip(s["src"], s["start"], s["dur"], variant_type=s["variant"])
            # ensure duration close to desired
            if abs(c.duration - s["dur"]) > 0.12:
                try:
                    c = c.fx(vfx.speedx, final_duration=s["dur"])
                except Exception:
                    c = c.subclip(0, min(c.duration, s["dur"]))
            if c.duration > 0.08:
                c = c.crossfadeout(0.06)
            clip_list.append(c)
        except Exception as e:
            print("Segment creation failed:", s["src"], e)

    if not clip_list:
        raise RuntimeError("No segments could be created from inputs.")

    final = concatenate_videoclips(clip_list, method="compose")
    # trim or pad last frame (no loop)
    if final.duration > target_duration:
        final = final.subclip(0, target_duration)
    elif final.duration < target_duration - 0.05:
        last_frame = final.to_ImageClip(t=max(0, final.duration-0.01)).set_duration(target_duration - final.duration)
        final = concatenate_videoclips([final, last_frame], method="compose")
    final = final.set_fps(FPS).set_duration(target_duration)
    return final

# ---------- Whisper transcription ----------
def transcribe_whisper(audio_path, model_name=WHISPER_DEFAULT):
    print("Loading Whisper model (this may take time)...")
    model = whisper.load_model(model_name)
    print("Transcribing audio with Whisper...")
    res = model.transcribe(audio_path, word_timestamps=False)
    captions = []
    for seg in res.get("segments", []):
        captions.append((seg["start"], seg["end"], seg["text"].strip()))
    return captions

# ---------- Neon-bounce subtitle overlay ----------
def neon_bounce_text_layer(text, start, dur):
    """Return a CompositeVideoClip layer that simulates neon glow + bounce using multiple TextClips."""
    # base white text
    base = (TextClip(text, fontsize=FONT_SIZE, font=FONT_NAME, color=TEXT_COLOR,
                     stroke_width=STROKE_WIDTH, stroke_color="black", method="label")
            .set_start(start).set_duration(dur).set_position(("center", int(OUT_H*0.68))))
    # glow layers (colored, larger, low opacity)
    glow_layers = []
    for i, col in enumerate(GLOW_COLORS):
        size_mult = 1.08 + 0.03*i
        glow = (TextClip(text, fontsize=int(FONT_SIZE*size_mult), font=FONT_NAME, color=col,
                         stroke_width=0, method="label")
                .set_start(start).set_duration(dur).set_position(("center", int(OUT_H*0.68)))
                .set_opacity(0.28 - 0.06*i))
        glow_layers.append(glow)
    # subtle bounce: scale animation via resize lambda tied to t
    def scale_fx(get_frame, t):
        return get_frame(t)
    # apply small sine-based resize on base (via fx resize)
    base = base.fx(vfx.resize, lambda t: 1 + 0.04*math.sin(6*t))
    layers = glow_layers + [base]
    return layers

def overlay_captions_on_video(video_clip, captions):
    """Overlay neon-bounce captions on the video timeline (captions list of (start,end,text))."""
    layers = [video_clip]
    for (start, end, text) in captions:
        dur = max(0.05, end - start)
        for layer in neon_bounce_text_layer(text, start, dur):
            layers.append(layer)
    composed = CompositeVideoClip(layers).set_duration(video_clip.duration)
    return composed

# ---------- MAIN ----------
def main(args):
    video_paths = find_videos_in_folder(args.input_videos)
    if not video_paths:
        print("No input videos found in:", args.input_videos)
        return

    music_path = find_music_file(args.music)
    print(f"Using {len(video_paths)} video(s). Music chosen: {music_path}")

    # music length -> target duration
    music_clip = AudioFileClip(music_path)
    target_duration = music_clip.duration
    print(f"Target final duration (music length) = {target_duration:.2f}s")

    beats = detect_beats(music_path)
    if beats is not None:
        print(f"Detected {len(beats)} beats (using for segment guidance).")
    else:
        print("No beats detected. Using adaptive timings.")

    print("Building creative visual sequence (this may take time)...")
    intermediate = build_sequence_filling_duration(video_paths, target_duration, music_beats=beats)

    # export intermediate audio for Whisper
    tempdir = tempfile.mkdtemp(prefix="fictic_tmp_")
    temp_audio_path = os.path.join(tempdir, "intermediate_audio.wav")
    if intermediate.audio is None:
        print("Intermediate has no audio track — falling back to first video audio trimmed.")
        fallback = AudioFileClip(video_paths[0])
        if fallback.duration > target_duration:
            fallback = fallback.subclip(0, target_duration)
        fallback.write_audiofile(temp_audio_path, fps=16000, verbose=False, logger=None)
    else:
        print("Writing intermediate audio for Whisper transcription...")
        intermediate.audio.write_audiofile(temp_audio_path, fps=16000, verbose=False, logger=None)

    # Whisper transcription (user requested subtitles)
    captions = transcribe_whisper(temp_audio_path, model_name=args.whisper_model)
    print(f"Whisper returned {len(captions)} segments.")

    # overlay captions (neon bounce)
    print("Overlaying neon-bounce captions...")
    video_with_captions = overlay_captions_on_video(intermediate, captions)

    # final audio mix: original audio (from intermediate) + music at lower volume
    music_final = AudioFileClip(music_path)
    if music_final.duration > video_with_captions.duration:
        music_final = music_final.subclip(0, video_with_captions.duration)
    else:
        # music shorter than video unlikely since target == music length, but safe:
        music_final = audio_loop(music_final, duration=video_with_captions.duration)

    orig_audio = video_with_captions.audio
    if orig_audio is None:
        orig_audio = AudioFileClip(music_path).volumex(0.0).subclip(0, video_with_captions.duration)

    # mix: original audio prominent, music behind at lower volume
    combined_audio = CompositeAudioClip([orig_audio.volumex(1.0), music_final.volumex(0.08)])
    final = video_with_captions.set_audio(combined_audio).resize((OUT_W, OUT_H)).set_duration(video_with_captions.duration)

    # write output
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print("Rendering final video (this may take a while)...")
    final.write_videofile(args.out, codec="libx264", audio_codec="aac", fps=FPS, preset="medium", threads=4, bitrate="5M")
    print("Cleaning temporary files...")
    try:
        shutil.rmtree(tempdir)
    except Exception:
        pass
    print("Done — output saved to:", args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fictic final vertical edit (neon subtitles).")
    parser.add_argument("--input_videos", required=True, help="Folder with input clips.")
    parser.add_argument("--music", required=True, help="Music file or folder (music length becomes target).")
    parser.add_argument("--out", default="./out/final_edit_9x16.mp4", help="Output path.")
    parser.add_argument("--whisper_model", default=WHISPER_DEFAULT, help="Whisper model (tiny/base/small/medium/large).")
    args = parser.parse_args()
    main(args)
