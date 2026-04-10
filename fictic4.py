#!/usr/bin/env python3
"""
fictic_final_full.py
Hybrid Fictic CPU-friendly editor (Punch Intro + Action Start combined).
Usage:
 python3 fictic_final_full.py --input_videos ./input_vid --music ./music/song.mp3 --out ./out/final_edit_9x16.mp4
"""
import os, sys, argparse, random, tempfile, shutil, math, warnings
from tqdm import tqdm
import numpy as np
import librosa
import cv2
from moviepy.editor import *
from moviepy.video.fx import all as vfx

warnings.filterwarnings("ignore")
FPS = 30
OUT_W, OUT_H = 1080, 1920
FONT_PATH = None  # set path to .ttf if you want custom font
FONT_SIZE = 84
MIN_CLIP_DUR = 0.6  # clips shorter than this will be looped safely

# Optional whisper (kept optional)
try:
    import whisper
except Exception:
    whisper = None

# ---------- Utilities ----------
def find_videos(folder):
    if not os.path.isdir(folder):
        return []
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
    files = []
    for f in sorted(os.listdir(folder)):
        if os.path.splitext(f)[1].lower() in exts:
            p = os.path.join(folder, f)
            if os.path.getsize(p) > 1000:
                files.append(p)
    return files

def find_music_file(path):
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for f in sorted(os.listdir(path)):
            if f.lower().endswith((".mp3", ".wav", ".m4a", ".flac", ".aac")):
                return os.path.join(path, f)
    raise FileNotFoundError("No music file found at: " + str(path))

def safe_load_audio_clip(path):
    return AudioFileClip(path)

def center_crop_vertical(clip, target_w=OUT_W, target_h=OUT_H):
    # defensive: ensure clip valid
    if clip is None:
        return None
    try:
        w, h = clip.w, clip.h
    except Exception:
        return None
    if not w or not h:
        return None
    # scale so height equals target_h, then center-crop width to target_w
    scale = float(target_h) / float(h)
    new_w = int(round(w * scale))
    resized = clip.resize(height=target_h)
    # if resized width smaller than target_w, scale by width instead
    if resized.w < target_w:
        resized = clip.resize(width=target_w)
    # now crop center
    x_center = resized.w / 2
    y_center = resized.h / 2
    x1 = int(max(0, x_center - target_w/2))
    y1 = int(max(0, y_center - target_h/2))
    x2 = x1 + target_w
    y2 = y1 + target_h
    try:
        return resized.crop(x1=x1, y1=y1, x2=x2, y2=y2)
    except Exception:
        # fallback: resize to exact and return (may distort)
        return resized.resize((target_w, target_h))

def detect_beats(music_path, sr=22050, hop_length=512):
    try:
        y, sr = librosa.load(music_path, sr=sr, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        return beats
    except Exception:
        return None

# ---------- Safe clip helper ----------
def safe_clip(clip, min_dur=MIN_CLIP_DUR):
    if clip is None:
        return None
    if clip.duration >= min_dur - 1e-6:
        return clip
    try:
        return clip.fx(vfx.loop, duration=min_dur)
    except Exception:
        # last resort: extend by freezing last frame via ImageClip
        try:
            frm = clip.get_frame(max(0, clip.duration - 1e-3))
            imc = ImageClip(frm).set_duration(min_dur - clip.duration)
            return concatenate_videoclips([clip, imc], method="compose")
        except Exception:
            return clip.set_duration(min_dur)

# ---------- Visual scoring for best start ----------
def visual_energy(clip_path, start=0.0, dur=0.8):
    try:
        v = VideoFileClip(clip_path)
        s = max(0, start)
        e = min(v.duration, s + dur)
        f0 = v.get_frame(s)
        f1 = v.get_frame(min(e, s + min(0.06, max(0.01, e-s)) - 1e-6))
        v.reader.close(); v.audio = None
        motion = float(np.mean(np.abs(f1.astype(np.float32) - f0.astype(np.float32))))
        bright = float(np.mean(f0.astype(np.float32)))
        score = 0.6*(motion/255.0) + 0.4*(bright/255.0)
        return float(score)
    except Exception:
        return 0.0

# ---------- Optional whisper transcription ----------
def transcribe_each_video(video_paths, whisper_model="small", language=None):
    if whisper is None:
        return []
    tmpd = tempfile.mkdtemp(prefix="fictic_whisper_")
    model = whisper.load_model(whisper_model)
    out_segments = []
    for p in tqdm(video_paths, desc="Whisper (per-video)"):
        try:
            a_out = os.path.join(tmpd, os.path.basename(p) + ".wav")
            try:
                v = VideoFileClip(p)
                if v.audio:
                    v.audio.write_audiofile(a_out, fps=16000, verbose=False, logger=None)
                v.reader.close(); v.audio = None
            except Exception:
                continue
            opts = {}
            if language:
                opts['language'] = language; opts['task']='transcribe'
            res = model.transcribe(a_out, **opts)
            for seg in res.get('segments', []):
                out_segments.append({'video': p, 'start': float(seg['start']), 'end': float(seg['end']), 'text': seg.get('text',''), 'logprob': seg.get('avg_logprob', -9.0)})
        except Exception:
            continue
    return out_segments

def score_segments_whisper(segs):
    # score by length + logprob
    scored=[]
    for s in segs:
        dur = max(0.01, s['end'] - s['start'])
        score = dur + max(-5, s.get('logprob', -9.0))/10.0 + len(s.get('text','').split())*0.1
        scored.append((score, s))
    if not scored: return None
    scored.sort(reverse=True, key=lambda x:x[0])
    return scored[0][1]

# ---------- Make variant (subclip + subtle zoom/slide) ----------
def make_variant(path, start, dur):
    try:
        v = VideoFileClip(path)
        sub = v.subclip(start, min(v.duration, start+dur)).set_fps(FPS)
        sub = center_crop_vertical(sub)
        # subtle randomized zoom or drift
        if random.random() < 0.5:
            z = 1.02 + random.random()*0.06
            sub = sub.fx(vfx.resize, lambda t: 1.0 + (z-1.0)*(t/max(1e-6, sub.duration)))
        if random.random() < 0.18:
            drift = int(40*(0.5+random.random()))
            sub = sub.set_position(lambda t: (int(drift*(1.0 - t/sub.duration)), 'center'))
        v.reader.close(); v.audio = None
        return sub
    except Exception:
        return None

# ---------- Build fillers guided by beats ----------
def build_fillers(video_paths, beats, target_duration, used_keys=None):
    candidates = []
    for p in video_paths:
        try:
            v = VideoFileClip(p)
            dur = v.duration
            v.reader.close(); v.audio = None
            # some sample starts
            starts = list(np.linspace(0, max(0, dur - 0.6), max(3, min(8, int(dur//0.5)+1))))
            starts += [random.uniform(0, max(0, dur-0.6)) for _ in range(2)]
            starts = sorted(set([max(0, s) for s in starts]))
            for s in starts:
                candidates.append({'path': p, 'start': float(s), 'dur': float(min(1.2, max(0.5, dur - s)))})
        except Exception:
            continue
    random.shuffle(candidates)
    # sort by visual energy estimate approx
    candidates.sort(key=lambda c: visual_energy(c['path'], c['start'], min(0.8,c['dur'])), reverse=True)
    fillers=[]
    used=set(used_keys or [])
    t=0.0
    bi=0
    beat_gaps = None
    if beats is not None and len(beats)>1:
        gaps = np.diff(beats)
        beat_gaps = [max(0.45, min(2.4, g)) for g in gaps]
    while t < target_duration - 0.25 and candidates:
        gap = beat_gaps[bi % len(beat_gaps)] if beat_gaps else 1.0
        dur = min(gap * random.uniform(0.85,1.05), target_duration - t)
        # pick next unused candidate
        found=False
        for i,c in enumerate(candidates):
            key=(c['path'], round(c['start'],2))
            if key in used:
                continue
            segdur = min(c['dur'], max(0.4, dur))
            clip = make_variant(c['path'], c['start'], segdur)
            if clip is None:
                continue
            clip = safe_clip(clip)
            fillers.append(clip.without_audio())
            used.add(key); t+=clip.duration
            candidates.pop(i); found=True; break
        if not found:
            break
        bi += 1
    # if still short, add short random snippets
    attempts=0
    while t < target_duration - 0.22 and attempts < 80:
        p = random.choice(video_paths)
        try:
            v = VideoFileClip(p)
            if v.duration < 0.4:
                v.reader.close(); v.audio = None; attempts+=1; continue
            s = random.uniform(0, max(0, v.duration - 0.5))
            segdur = min(1.0, target_duration - t)
            clip = make_variant(p, s, segdur)
            if clip:
                fillers.append(safe_clip(clip).without_audio()); t+=clip.duration
            v.reader.close(); v.audio = None
        except Exception:
            attempts+=1
            continue
    return fillers

# ---------- Transitions (SAFE + FIXED RESOLUTION) ----------

def safe_frame_get(clip, t):
    t = np.clip(t, 0, max(0, clip.duration - 1e-3))
    f = clip.get_frame(t)
    # ✅ Force exact output size
    return cv2.resize(f, (OUT_W, OUT_H))

def ensure_size(c):
    return c.resize((OUT_W, OUT_H)).set_position(("center", "center"))

def impact_zoom_transition(c1, c2, dur=0.22, strength=0.18):
    c1 = ensure_size(safe_clip(c1))
    c2 = ensure_size(safe_clip(c2))
    a = c1.subclip(max(0, c1.duration - dur), c1.duration).set_duration(dur)
    b = c2.subclip(0, min(c2.duration, dur)).set_duration(dur)

    def frame_fn(t):
        f1 = safe_frame_get(a, t).astype(np.float32)
        f2 = safe_frame_get(b, t).astype(np.float32)

        if t < dur*0.55:
            frac = t/(dur*0.55)
            zoom = 1.0 + strength*(frac**1.7)

            hh, ww = OUT_H, OUT_W
            nw = int(ww/zoom); nh = int(hh/zoom)
            cx, cy = ww//2, hh//2
            x1 = max(0, cx - nw//2); y1 = max(0, cy - nh//2)
            crop = f1[y1:y1+nh, x1:x1+nw]
            crop = cv2.resize(crop, (ww, hh))
            return crop.astype(np.uint8)
        else:
            frac = (t - dur*0.55)/(dur*0.45 + 1e-8)
            comp = (1-frac)*f1 + frac*f2
            return np.clip(comp, 0, 255).astype(np.uint8)

    return VideoClip(lambda tt: frame_fn(tt), duration=dur).set_fps(FPS)


def whip_transition(c1, c2, dur=0.22, direction='left'):
    c1 = ensure_size(safe_clip(c1))
    c2 = ensure_size(safe_clip(c2))
    a = c1.subclip(max(0, c1.duration - dur), c1.duration).set_duration(dur)
    b = c2.subclip(0, min(c2.duration, dur)).set_duration(dur)

    def frame_fn(t):
        f1 = safe_frame_get(a, t).astype(np.float32)
        f2 = safe_frame_get(b, t).astype(np.float32)

        frac = t/dur
        shift = int((OUT_W + 200)*(frac**1.2))

        if direction == 'left':
            f1s = np.roll(f1, -shift, axis=1)
            f2s = np.roll(f2, OUT_W-shift, axis=1)
        else:
            f1s = np.roll(f1, shift, axis=1)
            f2s = np.roll(f2, -OUT_W+shift, axis=1)

        mask = np.linspace(0, 1, OUT_W) < frac
        mask = np.repeat(mask[np.newaxis, :], OUT_H, axis=0)[:,:,None]
        comp = (1-mask)*f1s + mask*f2s
        return np.clip(comp, 0,255).astype(np.uint8)

    return VideoClip(lambda tt: frame_fn(tt), duration=dur).set_fps(FPS)


def flash_transition(c1, c2, dur=0.16):
    c1 = ensure_size(safe_clip(c1))
    c2 = ensure_size(safe_clip(c2))
    a = c1.subclip(max(0, c1.duration - dur), c1.duration).set_duration(dur)
    b = c2.subclip(0, min(c2.duration, dur)).set_duration(dur)

    def frame_fn(t):
        frac = t/dur
        f1 = safe_frame_get(a, t).astype(np.float32)
        f2 = safe_frame_get(b, t).astype(np.float32)

        if frac < 0.35:
            alpha = 1.0 + 3.0*(frac/0.35)
            return np.clip(f1*alpha,0,255).astype(np.uint8)
        else:
            r = (frac-0.35)/0.65
            comp = (1-r)*f1 + r*f2
            if 0.45 < frac < 0.55:
                comp = np.clip(comp + 40*(1 - abs(frac-0.5)*4), 0,255)
            return comp.astype(np.uint8)

    return VideoClip(lambda tt: frame_fn(tt), duration=dur).set_fps(FPS)


def choose_transition(prev, nxt):
    r = random.random()
    if r < 0.46:
        return impact_zoom_transition(prev, nxt, dur=0.22, strength=0.18)
    elif r < 0.86:
        return whip_transition(prev, nxt, dur=0.22, direction=random.choice(['left','right']))
    else:
        return flash_transition(prev, nxt, dur=0.16)


# ---------- Stitch sequence with transitions ----------
def stitch_with_hybrid(first_clip, fillers):
    seq = [first_clip]
    for nxt in fillers:
        prev = seq[-1]
        try:
            trans = choose_transition(prev, nxt)
            seq.append(trans)
            seq.append(nxt)
            # occasional tiny white flash pop
            if random.random() < 0.10:
                seq.append(ColorClip((OUT_W, OUT_H), color=(255,255,255)).set_duration(0.03).fx(vfx.fadein,0.01).fx(vfx.fadeout,0.01))
        except Exception:
            seq.append(nxt)
    # concatenate
    clean = [safe_clip(s) for s in seq if s is not None]
    return concatenate_videoclips(clean, method="compose")

# ---------- Audio mixing ----------
def mix_audio_for_final(final_clip, first_clip, music_path, dialogue_duration=0.0):
    music_orig = AudioFileClip(music_path)

    # ✅ Always fit music to video duration (no looping)
    if music_orig.duration > final_clip.duration:
        music_orig = music_orig.subclip(0, final_clip.duration)
    else:
        # Trim video to music instead of looping
        final_clip = final_clip.subclip(0, music_orig.duration)

    # 🔊 Under-music during dialogue (soft volume)
    music_under = music_orig.subclip(
        0, min(dialogue_duration, music_orig.duration)
    ).volumex(0.05).audio_fadein(0.1).audio_fadeout(0.1)

    # 🔊 Normal music after dialogue
    music_after = None
    if music_orig.duration > dialogue_duration:
        music_after = music_orig.subclip(
            dialogue_duration, music_orig.duration
        ).volumex(1.0)

    layers = []

    # Dialogue audio (first clip)
    if first_clip and getattr(first_clip, "audio", None):
        speech = first_clip.audio.subclip(
            0, min(dialogue_duration, first_clip.audio.duration)
        ).volumex(1.45)
        layers.append(speech.set_start(0))

    # Soft background music under dialogue
    if music_under:
        layers.append(music_under.set_start(0))

    # Full music after dialogue ends
    if music_after:
        layers.append(music_after.set_start(dialogue_duration))

    # ✅ Combine & fade out end
    if layers:
        combined = CompositeAudioClip(layers).set_duration(final_clip.duration)
        return final_clip.set_audio(combined).audio_fadeout(0.75)

    # ✅ Fallback (no dialogue)
    return final_clip.set_audio(music_orig).audio_fadeout(0.75)


# ---------- Subtitles helper ----------
def make_subtitle_clip(text, parent_clip, fontsize=FONT_SIZE, max_width=OUT_W-120):
    # create a TextClip and place near bottom with semi-opaque background
    try:
        # try to keep it readable: wrap long lines
        # basic wrap
        words = text.strip().split()
        lines = []
        cur = []
        cur_len = 0
        for w in words:
            cur.append(w)
            cur_len += len(w) + 1
            if cur_len > 28:
                lines.append(" ".join(cur))
                cur = []
                cur_len = 0
        if cur:
            lines.append(" ".join(cur))
        txt = "\n".join(lines)
        tc = TextClip(txt, fontsize=int(fontsize*0.8), font=(FONT_PATH or "Arial"), method='label')
        tc = tc.set_duration(parent_clip.duration)
        # create background box
        try:
            bg = tc.on_color(size=(parent_clip.w, tc.h + 40), color=(0,0,0), pos=('center', 'center'), col_opacity=0.55)
            bg = bg.set_duration(parent_clip.duration)
            bg = bg.set_position(('center', parent_clip.h - bg.h - 80))
            return bg
        except Exception:
            tc = tc.set_position(('center', parent_clip.h - 160))
            return tc
    except Exception:
        return None

# ---------- Main ----------
def main(args):
    random.seed(42)
    video_paths = find_videos(args.input_videos)
    if not video_paths:
        print("No input videos found in", args.input_videos); return
    music_path = find_music_file(args.music)
    music = safe_load_audio_clip(music_path)
    target_duration = music.duration
    print(f"Music length: {target_duration:.2f}s")

    # 1) attempt transcript segments (optional)
    whisper_segs = []
    try:
        if whisper:
            whisper_segs = transcribe_each_video(video_paths, whisper_model=args.whisper_model, language=args.language)
            print("Whisper segments:", len(whisper_segs))
    except Exception:
        whisper_segs = []

    # 2) Choose best starting clip
    best = None
    dialogue_text = None
    dialogue_duration = 0.0
    if whisper_segs:
        ws = score_segments_whisper(whisper_segs)
        if ws:
            # choose the spoken segment as the punchy opening
            best = {'path': ws['video'], 'start': max(0, ws['start']-0.2), 'dur': min(2.5, ws['end'] - ws['start'] + 0.6)}
            dialogue_text = ws.get('text','').strip()
            dialogue_duration = max(0.0, ws.get('end',0.0) - ws.get('start',0.0))
    if not best:
        # visual fallback: choose candidate with highest visual energy
        bestscore=-1; best=None
        for p in video_paths:
            s = visual_energy(p, 0.0, 0.8)
            if s > bestscore:
                bestscore=s; best={'path': p, 'start': 0.0, 'dur': min(2.5, max(0.8, 0.8))}

    # build first clip
    try:
        first_src = best['path']
        src_v = VideoFileClip(first_src)
        first_clip = src_v.subclip(best['start'], min(src_v.duration, best['start']+best['dur']))
        # keep audio for dialogue
        first_clip = center_crop_vertical(first_clip)
        first_clip = safe_clip(first_clip)

        # if we have detected whisper text, overlay subtitle to make it punchy
        if dialogue_text:
            sub = make_subtitle_clip(dialogue_text, first_clip, fontsize=FONT_SIZE)
            if sub is not None:
                try:
                    # ensure sizes align
                    composite = CompositeVideoClip([first_clip, sub.set_start(0)])
                    composite = composite.set_duration(first_clip.duration)
                    first_clip = composite
                except Exception:
                    pass
    except Exception:
        # fallback to first file
        src_v = VideoFileClip(video_paths[0])
        first_clip = src_v.subclip(0, min(2.5, src_v.duration))
        first_clip = center_crop_vertical(first_clip); first_clip = safe_clip(first_clip)

    # 3) detect beats and build fillers
    beats = detect_beats(music_path)
    print("Beats detected:", 0 if beats is None else len(beats))
    remaining = max(0.0, target_duration - first_clip.duration)
    # prefer marking used_keys by path,start to avoid missing attributes
    used_key = (best['path'], round(best.get('start',0.0),2)) if best and 'path' in best else None
    fillers = build_fillers(video_paths, beats, remaining, used_keys=[used_key] if used_key else None)

    # 4) stitch sequence
    final_seq = stitch_with_hybrid(first_clip, fillers)

    # 5) ensure fill to exact music end
    if final_seq.duration < target_duration - 0.02:
        extra = build_fillers(video_paths, beats, target_duration - final_seq.duration, used_keys=None)
        if extra:
            final_seq = concatenate_videoclips([final_seq] + extra, method="compose")
    # trim to exact music length
    if final_seq.duration > target_duration:
        final_seq = final_seq.subclip(0, target_duration)
    final_seq = final_seq.set_fps(FPS).set_duration(target_duration)

    # 6) audio mixing and export
    out = mix_audio_for_final(final_seq, first_clip, music_path, dialogue_duration=(dialogue_duration if dialogue_duration>0.0 else (first_clip.duration if first_clip else 0.0)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print("Rendering final video. This may take a while...")
    out.write_videofile(args.out, codec="libx264", audio_codec="aac", fps=FPS, preset="medium", threads=4, bitrate="6M", ffmpeg_params=["-movflags","+faststart"])
    print("Done ->", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_videos", required=True)
    p.add_argument("--music", required=True)
    p.add_argument("--out", default="./out/final_edit_9x16.mp4")
    p.add_argument("--whisper_model", default="small")
    p.add_argument("--language", default=None)
    args = p.parse_args()
    main(args)
