#!/usr/bin/env python3
"""
fictic_final_hybrid_v9x16.py
Hybrid Fictic editor (Impact Zoom + Whip + Flash) — CPU-friendly, vertical 9:16 default.
- Picks a strong punch dialogue (Whisper if available) with pre-roll/post-roll.
- Places punch dialogue first with cinematic subtitle (punch-only).
- Fills rest of music with unique moving filler clips (no repeats), muted.
- Transitions: Impact Zoom, Whip Pan, Flash Beat (auto-chosen based on motion & beat).
- Preserves original colors, no black/static frames, ends exactly at music end.
"""
import os, glob, argparse, tempfile, shutil, random, math, warnings, time
from tqdm import tqdm
import numpy as np
import librosa
import cv2
from moviepy.editor import (
    VideoFileClip, AudioFileClip, concatenate_videoclips,
    CompositeVideoClip, TextClip, VideoClip, ColorClip, CompositeAudioClip
)
from moviepy.audio.fx.all import audio_loop
from moviepy.video.fx.all import fadein, fadeout, resize

# Optional whisper
try:
    import whisper
except Exception:
    whisper = None

warnings.filterwarnings("ignore")

# ---------- Config ----------
OUT_W, OUT_H = 1080, 1920   # vertical 9:16
FPS = 30
FONT = None   # set to a ttf path if you want custom font
FONT_SIZE = 86
WHISPER_DEFAULT = "small"
PREROLL = 0.45
POSTROLL = 0.30
GLITCH_FLASH_PROB = 0.12

BAD_WORDS = ["thanks for watching", "subscribe", "like", "follow", "please subscribe", "hit the bell", "youtube", "check out", "visit"]
NEON_PALETTE = ["#00FFFF", "#FF00FF", "#39FF14", "#FF3131", "#FFD700", "#00BFFF", "#9400D3", "#00FF7F"]

# ---------- Utilities ----------
def find_videos(folder):
    pats = ["*.mp4","*.mov","*.mkv","*.avi","*.webm","*.mpg","*.mpeg"]
    files = []
    for p in pats:
        files += glob.glob(os.path.join(folder, p))
    return sorted(files)

def find_music_file(arg):
    if os.path.isdir(arg):
        pats = ["*.mp3","*.wav","*.m4a","*.flac","*.aac"]
        files = []
        for p in pats:
            files += glob.glob(os.path.join(arg, p))
        files = sorted(files)
        if not files:
            raise FileNotFoundError("No music in folder: " + arg)
        return files[0]
    elif os.path.isfile(arg):
        return arg
    else:
        raise FileNotFoundError("Music path not found: " + arg)

def detect_beats(music_path, sr=22050, hop_length=512):
    try:
        y, sr = librosa.load(music_path, sr=sr, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        return beats
    except Exception:
        return None

def center_crop_vertical(clip, out_w=OUT_W, out_h=OUT_H):
    scale = max(out_w/clip.w, out_h/clip.h)
    nw, nh = int(clip.w*scale), int(clip.h*scale)
    clip = clip.resize((nw, nh))
    x1 = (nw - out_w)//2; y1 = (nh - out_h)//2
    return clip.crop(x1=x1, y1=y1, x2=x1+out_w, y2=y1+out_h)

# ---------- Whisper per-video transcription ----------
def transcribe_each_video(video_paths, whisper_model_name="small", language=None, tmpdir=None):
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(prefix="fictic_whisper_")
    if whisper is None:
        return [], tmpdir
    model = whisper.load_model(whisper_model_name)
    all_segs = []
    for p in tqdm(video_paths, desc="Whisper (per-video)"):
        try:
            a_out = os.path.join(tmpdir, os.path.basename(p) + ".wav")
            try:
                ac = AudioFileClip(p)
                ac.write_audiofile(a_out, fps=16000, verbose=False, logger=None)
                ac.close()
            except Exception:
                v = VideoFileClip(p)
                if v.audio:
                    v.audio.write_audiofile(a_out, fps=16000, verbose=False, logger=None)
                v.reader.close(); v.audio = None
            opts = {}
            if language:
                opts['language'] = language
                opts['task'] = 'transcribe'
            res = model.transcribe(a_out, **opts)
            for seg in res.get('segments', []):
                all_segs.append({'video': p, 'start': float(seg['start']), 'end': float(seg['end']), 'text': seg['text'].strip(), 'avg_logprob': seg.get('avg_logprob', -5)})
        except Exception:
            continue
    return all_segs, tmpdir

# ---------- Scoring ----------
def compute_visual_energy_from_file(path, start, dur=0.8):
    try:
        v = VideoFileClip(path)
        start = max(0, start)
        end = min(v.duration, start+dur)
        f0 = v.get_frame(start)
        f1 = v.get_frame(min(start+min(0.12, end-start-1e-3), v.duration-1e-3))
        motion = float(np.mean(np.abs(f1.astype(np.float32) - f0.astype(np.float32))))
        bright = float(np.mean(f0.astype(np.float32)))
        v.reader.close(); v.audio = None
        return 0.6 * (motion/255.0) + 0.4 * (bright/255.0)
    except Exception:
        return 0.0

def score_segments(segments, tmp_audio_dir):
    scored = []
    for seg in segments:
        text = (seg.get('text') or "").strip()
        lowtxt = text.lower()
        if any(b in lowtxt for b in BAD_WORDS):
            continue
        dur = max(0.01, seg['end'] - seg['start'])
        dlg = float(seg.get('avg_logprob', -5))
        dlg_score = max(0.0, min(1.0, (dlg + 10.0) / 10.0))
        vis = compute_visual_energy_from_file(seg['video'], seg['start'], dur=dur)
        total = 0.55 * dlg_score + 0.45 * vis + 0.02 * len(text.split())
        scored.append((total, seg))
    if not scored:
        return None
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1]

# ---------- Subtitle (punch-only) ----------
def create_punch_subtitle(text, duration):
    neon = random.choice(NEON_PALETTE)
    fontsize = int(FONT_SIZE * 1.15)
    kw = {}
    if FONT:
        kw['font'] = FONT
    txt = TextClip(text, fontsize=fontsize, color="white", stroke_color=neon, stroke_width=int(fontsize*0.03), method="label", **kw)
    txt = txt.set_position(("center", int(OUT_H*0.75) - txt.h//2)).set_duration(duration)
    txt = txt.fx(fadein, 0.06).fx(fadeout, 0.06)
    return txt

# ---------- Candidate extraction ----------
def visual_motion_estimate(path, samples=6):
    try:
        clip = VideoFileClip(path)
        dur = max(0.5, clip.duration)
        times = np.linspace(0, dur, min(samples, max(2, int(dur*2))))
        prev = None; diffs = []
        for t in times:
            f = clip.get_frame(min(t, clip.duration-1e-3))
            gray = np.mean(f.astype(np.float32), axis=2)
            if prev is not None:
                diffs.append(np.mean(np.abs(gray - prev)))
            prev = gray
        clip.reader.close(); clip.audio = None
        return float(np.mean(diffs)) if diffs else 0.0
    except Exception:
        return 0.0

def extract_candidates(video_paths):
    candidates = []
    for idx, p in enumerate(video_paths):
        try:
            v = VideoFileClip(p)
            dur = v.duration
            steps = max(3, min(8, int(dur//0.5) + 1))
            starts = list(np.linspace(0, max(0, dur-0.5), steps))
            starts += [random.uniform(0, max(0, dur-0.5)) for _ in range(2)]
            starts = sorted(set([max(0, s) for s in starts]))
            for s in starts:
                seg_dur = min(1.6, max(0.6, min(1.4, dur - s))) if dur - s > 0.2 else min(0.6, dur - s)
                try:
                    ac = AudioFileClip(p).subclip(s, min(s + seg_dur, dur))
                    arr = ac.to_soundarray(fps=16000)
                    rms = float(np.sqrt((arr**2).mean())) if arr.size else 0.0
                    ac.close()
                except Exception:
                    rms = 0.0
                motion = visual_motion_estimate(p)
                candidates.append({"path": p, "start": float(s), "dur": float(seg_dur), "rms": float(rms), "motion": float(motion), "idx": idx})
            v.reader.close(); v.audio = None
        except Exception:
            continue
    return candidates

# ---------- Variants & fillers ----------
def load_clip_uniform(path):
    clip = VideoFileClip(path)
    # Resize and center crop to 1080x1920 (vertical Fictic style)
    return clip.resize(height=1920).crop(
        x_center=clip.w / 2, 
        y_center=clip.h / 2,
        width=1080, height=1920
    )


def make_variant(src_path, start, dur, kind="normal"):
    src = VideoFileClip(src_path)
    end = min(src.duration, start + dur)
    clip = src.subclip(start, end).set_fps(FPS)
    clip = center_crop_vertical(clip)
    # subtle zoom or horizontal slide (no color changes)
    if random.random() < 0.45:
        zoom_amt = 1.02 + random.random()*0.06
        clip = clip.fx(resize, lambda t: 1.0 + (zoom_amt-1.0)*(t/max(1e-6, clip.duration)))
    if random.random() < 0.18:
        # small positional drift to emulate motion
        wmove = int(60 * (0.4 + random.random()))
        clip = clip.set_position(lambda t: (int(wmove*(1.0 - t/clip.duration)), 'center'))
    try:
        src.reader.close(); src.audio = None
    except Exception:
        pass
    return clip

def build_fillers(video_paths, candidates, target_duration, used_key, beat_gaps=None):
    pool = sorted(candidates, key=lambda x: (x.get("motion",0)*0.7 + x.get("rms",0)*50.0 + random.random()*0.01), reverse=True)
    fillers = []
    used = set([used_key])
    remaining = target_duration
    bi = 0
    while remaining > 0.35 and pool:
        gap = (beat_gaps[bi % len(beat_gaps)] if beat_gaps else 1.0)
        dur = min(gap * random.uniform(0.88,1.06), remaining)
        bi += 1
        found = False
        for i, c in enumerate(pool):
            key = (c['path'], round(c['start'],2))
            if key in used:
                continue
            take = min(c['dur'], max(0.35, dur))
            try:
                clip = make_variant(c['path'], c['start'], take, kind=random.choice(["normal","speed"]))
                clip = clip.without_audio().crossfadein(0.06)
                try:
                    frame = clip.get_frame(min(0.05, clip.duration-1e-3)).astype(np.uint8)
                    if frame.mean() < 6:
                        pool.pop(i); continue
                except Exception:
                    pass
                fillers.append((clip, key))
                used.add(key)
                remaining -= clip.duration
                pool.pop(i)
                found = True
                break
            except Exception:
                continue
        if not found:
            break
    attempts = 0
    while remaining > 0.25 and attempts < 60:
        p = random.choice(video_paths)
        try:
            v = VideoFileClip(p)
            start = random.uniform(0, max(0, v.duration - 0.6))
            dur = min(remaining, random.uniform(0.5, 1.2))
            key = (p, round(start,2))
            if key in used:
                v.reader.close(); v.audio = None; attempts += 1; continue
            add = make_variant(p, start, dur)
            add = add.without_audio().crossfadein(0.06)
            fillers.append((add, key))
            used.add(key)
            remaining -= add.duration
            v.reader.close(); v.audio = None
        except Exception:
            attempts += 1
            continue
    return [f[0] for f in fillers]

# ---------- Transitions (CPU-friendly hybrid) ----------
def impact_zoom_transition(c1, c2, dur=0.22, strength=0.18):
    # last portion of c1 zooms in rapidly then cuts to c2 with a pop
    c1r = c1.set_fps(FPS).fx(resize, lambda t: 1.0)  # ensure fps
    c2r = c2.set_fps(FPS)
    # get short trims
    a = c1r.subclip(max(0, c1r.duration - dur), c1r.duration).set_duration(dur)
    b = c2r.subclip(0, min(c2r.duration, dur)).set_duration(dur)
    def frame_fn(t):
        # zoom c1 up, then snap to b with easing
        if t < dur * 0.55:
            frac = t / (dur * 0.55)
            zoom = 1.0 + strength * (frac**1.7)
            f = a.get_frame(t).astype(np.float32)
            h,w,_ = f.shape
            # center zoom
            cx, cy = w//2, h//2
            nw = int(w/zoom); nh = int(h/zoom)
            x1 = max(0, cx - nw//2); y1 = max(0, cy - nh//2)
            crop = f[y1:y1+nh, x1:x1+nw]
            res = cv2.resize(crop.astype(np.uint8), (w,h)).astype(np.float32)
            return res.astype(np.uint8)
        else:
            # switch/overlay into next with small blend
            frac = (t - dur*0.55) / (dur*0.45 + 1e-8)
            f1 = a.get_frame(min(t, a.duration-1e-3)).astype(np.float32)
            f2 = b.get_frame(min(t, b.duration-1e-3)).astype(np.float32)
            comp = (1-frac)*f1 + frac*f2
            return np.clip(comp,0,255).astype(np.uint8)
    return VideoClip(lambda t: frame_fn(t), duration=dur).set_fps(FPS)

def whip_transition(c1, c2, dur=0.22, direction='left'):
    c1r = c1.set_fps(FPS)
    c2r = c2.set_fps(FPS)
    a = c1r.subclip(max(0, c1r.duration - dur), c1r.duration).set_duration(dur)
    b = c2r.subclip(0, min(c2r.duration, dur)).set_duration(dur)
    w,h = OUT_W, OUT_H
    def frame_fn(t):
        frac = t/dur
        # Prev moves out, Next moves in
        shift = int((w+200) * (frac**1.2))
        if direction == 'left':
            f1 = a.get_frame(min(t,a.duration-1e-3)).astype(np.float32)
            f2 = b.get_frame(min(t,b.duration-1e-3)).astype(np.float32)
            canvas = np.zeros_like(f1)
            # f1 shifted left
            x1 = max(-w, -shift)
            try:
                canvas[:, max(0,x1):max(0,x1)+w] += np.roll(f1, -shift, axis=1)
            except Exception:
                canvas = f1
            # f2 sliding in from right
            f2s = np.roll(f2, w-shift, axis=1)
            mask = (np.linspace(0,1,w) < frac).astype(np.float32)
            mask = np.repeat(mask[np.newaxis,...], h, axis=0)[:,:,None]
            comp = (1-mask)*canvas + mask*f2s
            return np.clip(comp,0,255).astype(np.uint8)
        else:
            # right direction similar
            f1 = a.get_frame(min(t,a.duration-1e-3)).astype(np.float32)
            f2 = b.get_frame(min(t,b.duration-1e-3)).astype(np.float32)
            shift = int((w+200) * (frac**1.2))
            canvas = np.zeros_like(f1)
            canvas[:, :] += np.roll(f1, shift, axis=1)
            f2s = np.roll(f2, -w+shift, axis=1)
            mask = (np.linspace(0,1,w) < frac).astype(np.float32)
            mask = np.repeat(mask[np.newaxis,...], h, axis=0)[:,:,None]
            comp = (1-mask)*canvas + mask*f2s
            return np.clip(comp,0,255).astype(np.uint8)
    return VideoClip(lambda t: frame_fn(t), duration=dur).set_fps(FPS)

def flash_transition(c1, c2, dur=0.16):
    c1r = c1.set_fps(FPS)
    c2r = c2.set_fps(FPS)
    a = c1r.subclip(max(0, c1r.duration - dur), c1r.duration).set_duration(dur)
    b = c2r.subclip(0, min(c2r.duration, dur)).set_duration(dur)
    def frame_fn(t):
        frac = t/dur
        if frac < 0.35:
            f = a.get_frame(min(t,a.duration-1e-3)).astype(np.float32)
            # tiny bright flash ramp
            alpha = 1.0 + 3.0*(frac/0.35)
            f = np.clip(f*alpha,0,255)
            return f.astype(np.uint8)
        else:
            # blend to b quickly
            f1 = a.get_frame(min(t,a.duration-1e-3)).astype(np.float32)
            f2 = b.get_frame(min(t,b.duration-1e-3)).astype(np.float32)
            r = (frac - 0.35)/0.65
            comp = (1-r)*f1 + r*f2
            # add small white flash on the exact beat moment
            if 0.45 < frac < 0.55:
                comp = np.clip(comp + 40*(1 - abs(frac-0.5)*4), 0,255)
            return comp.astype(np.uint8)
    return VideoClip(lambda t: frame_fn(t), duration=dur).set_fps(FPS)

# ---------- Stitch with hybrid transitions ----------
def choose_transition(prev, nxt):
    # choose transition by random weighted by motion/energy
    r = random.random()
    if r < 0.46:
        return impact_zoom_transition(prev, nxt, dur=0.22, strength=0.18)
    elif r < 0.86:
        dirc = random.choice(['left','right'])
        return whip_transition(prev, nxt, dur=0.22, direction=dirc)
    else:
        return flash_transition(prev, nxt, dur=0.16)

def stitch_with_hybrid(first_clip, fillers):
    sequence = [first_clip]
    for nxt in fillers:
        prev = sequence[-1]
        try:
            trans = choose_transition(prev, nxt)
            sequence.append(trans)
            sequence.append(nxt)
            # occasional tiny white flash pop
            if random.random() < GLITCH_FLASH_PROB:
                sequence.append(ColorClip((OUT_W, OUT_H), color=(255,255,255)).set_duration(0.03).fx(fadein,0.01).fx(fadeout,0.01))
        except Exception:
            # fallback crossfade
            try:
                sequence.append(prev.crossfadeout(0.06))
                sequence.append(nxt.crossfadein(0.06))
            except Exception:
                sequence.append(nxt)
    try:
        return concatenate_videoclips(sequence, method="compose")
    except Exception:
        return concatenate_videoclips([first_clip] + fillers, method="compose")

# ---------- Audio mixing ----------
def mix_audio_for_final(final_clip, first_clip, music_path, dialogue_duration):
    music_orig = AudioFileClip(music_path)
    if music_orig.duration > final_clip.duration:
        music_orig = music_orig.subclip(0, final_clip.duration)
    else:
        music_orig = audio_loop(music_orig, duration=final_clip.duration)
    music_under = music_orig.subclip(0, min(dialogue_duration, music_orig.duration)).volumex(0.06).audio_fadein(0.08).audio_fadeout(0.06)
    music_after = None
    if music_orig.duration > dialogue_duration:
        music_after = music_orig.subclip(dialogue_duration, final_clip.duration).volumex(1.0)
    first_audio = first_clip.audio if first_clip and first_clip.audio else None
    audio_layers = []
    if music_under:
        audio_layers.append(music_under.set_start(0))
    if first_audio:
        audio_layers.append(first_audio.subclip(0, min(dialogue_duration, first_audio.duration)).volumex(1.6).set_start(0))
    if music_after:
        audio_layers.append(music_after.set_start(dialogue_duration))
    combined_audio = CompositeAudioClip(audio_layers).set_duration(final_clip.duration)
    out = final_clip.set_audio(combined_audio).audio_fadeout(0.6)
    return out

# ---------- Main ----------
def main(args):
    video_paths = find_videos(args.input_videos)
    if not video_paths:
        print("No input videos found:", args.input_videos); return

    music_path = find_music_file(args.music)
    print(f"Found {len(video_paths)} video(s). Music: {music_path}")

    music_audio = AudioFileClip(music_path)
    target_duration = music_audio.duration
    print(f"Target final duration (music length) = {target_duration:.2f}s")

    # 1) transcribe (if whisper available)
    print("Transcribing input videos (Whisper)...")
    whisper_segs, tmp_wdir = transcribe_each_video(video_paths, whisper_model_name=args.whisper_model, language=args.language)
    print(f"Whisper produced {len(whisper_segs)} segments." if whisper_segs else "Whisper skipped or produced no segments.")

    # 2) pick best punch
    best_seg = score_segments(whisper_segs, tmp_wdir) if whisper_segs else None
    if best_seg:
        print("Picked punch dialogue:", best_seg.get('text','')[:140])
    else:
        print("No suitable dialogue found; using visual fallback.")

    # 3) candidates
    print("Extracting visual candidates for fillers...")
    candidates = extract_candidates(video_paths)
    if not candidates:
        print("No visual candidates found; aborting.")
        try: shutil.rmtree(tmp_wdir)
        except Exception: pass
        return

    # 4) first clip build
    if best_seg:
        src = best_seg['video']; s0 = max(0.0, best_seg['start']); s1 = best_seg['end']
        pre = max(0.0, s0 - PREROLL); post = min(VideoFileClip(src).duration, s1 + POSTROLL)
        first_clip = VideoFileClip(src).subclip(pre, post).set_fps(FPS)
        first_clip = center_crop_vertical(first_clip)
        subtitle_text = best_seg.get('text','')
        dialogue_duration = max(0.25, s1 - s0)
        sub = create_punch_subtitle(subtitle_text, duration=dialogue_duration)
        first_with_subs = CompositeVideoClip([first_clip, sub.set_start(0)]).set_duration(first_clip.duration)
    else:
        fallback = max(candidates, key=lambda c: (c['motion']*0.6 + c['rms']*20.0 + random.random()*0.01))
        src = fallback['path']; pre = max(0.0, fallback['start']-0.18)
        post = min(VideoFileClip(src).duration, fallback['start'] + min(2.0, fallback['dur']) + 0.18)
        first_clip = VideoFileClip(src).subclip(pre, post).set_fps(FPS)
        first_clip = center_crop_vertical(first_clip)
        dialogue_duration = min(first_clip.duration, 1.8)
        first_with_subs = first_clip

    # 5) beats and fillers
    beats = detect_beats(music_path)
    beat_gaps = None
    if beats is not None and len(beats) > 1:
        gaps = np.diff(beats)
        beat_gaps = [max(0.45, min(2.4, g)) for g in gaps] if len(gaps) > 0 else None
        print(f"Detected {len(beats)} beats; using for filler durations.")
    else:
        print("No beats detected; using adaptive filler durations.")

    used_key = (src, round(pre,2))
    remaining_time = max(0.0, target_duration - first_with_subs.duration)
    fillers = build_fillers(video_paths, candidates, remaining_time, used_key, beat_gaps=beat_gaps)

    # 6) stitch with hybrid transitions
    final_seq = stitch_with_hybrid(first_with_subs, fillers)

    # 7) if still short, append unique moving snippets
    if final_seq.duration < target_duration - 0.02:
        extra_needed = target_duration - final_seq.duration
        more = build_fillers(video_paths, candidates, extra_needed, used_key, beat_gaps=beat_gaps)
        if more:
            final_seq = stitch_with_hybrid(final_seq, more)

    attempts = 0
    while final_seq.duration < target_duration - 0.02 and attempts < 40:
        try:
            p = random.choice(video_paths)
            v = VideoFileClip(p)
            if v.duration > 0.7:
                start = random.uniform(0, max(0, v.duration - 0.7))
                add = make_variant(p, start, min(1.0, target_duration - final_seq.duration))
                add = add.without_audio()
                final_seq = concatenate_videoclips([final_seq, add], method="compose")
                v.reader.close(); v.audio = None
        except Exception:
            pass
        attempts += 1

    # 8) final trim if slightly longer (avoid black)
    if final_seq.duration > target_duration:
        final_seq = final_seq.subclip(0, target_duration)
    final_seq = final_seq.set_fps(FPS).set_duration(target_duration)

    # 9) mix audio and export
    print("Mixing audio and rendering final video...")
    final_out = mix_audio_for_final(final_seq, first_with_subs, music_path, dialogue_duration)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    final_out.write_videofile(args.out, codec="libx264", audio_codec="aac", fps=FPS, preset="medium", threads=4, bitrate="6M", ffmpeg_params=["-movflags","+faststart"])
    try: shutil.rmtree(tmp_wdir)
    except Exception: pass
    print("Done ->", args.out)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fictic hybrid CPU-friendly editor (9:16 default).")
    p.add_argument("--input_videos", required=True, help="Folder with input clips")
    p.add_argument("--music", required=True, help="Music file or folder (music length used)")
    p.add_argument("--out", default="./out/final_edit_9x16.mp4", help="Output path")
    p.add_argument("--whisper_model", default=WHISPER_DEFAULT, help="Whisper model (tiny/base/small/medium/large)")
    p.add_argument("--language", default=None, help="Whisper language code (optional)")
    args = p.parse_args()
    main(args)
