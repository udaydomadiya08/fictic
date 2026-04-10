"""
best_scene_extractor.py

Goal: Given an input video, find the best 5-10s "hook" mini-scene that is self-contained,
contains a complete dialogue moment and emotional punch, and export it as best_scene.mp4.

Design choices made: "Best" = high dialogue coherence + audio intensity + visible faces + balanced motion.
This script uses a hybrid pipeline combining scene detection, speech transcription, audio analysis,
face detection, motion analysis, and a sliding-window optimizer to pick the best 5-10s slice.

Instructions:
- Recommended environment: Linux / macOS / Google Colab. GPU strongly recommended for YOLO & Whisper.
- Install dependencies (example):
    pip install scenedetect[opencv] moviepy librosa soundfile numpy torch torchvision torchaudio
    pip install ultralytics faster-whisper tqdm webrtcvad

Notes on models:
- This script tries to use faster-whisper (local) for transcription. If you prefer OpenAI Whisper API
  or another model, swap the transcribe function.
- Ultralityics YOLO is used for face detection (you can use a dedicated face detector like face_recognition or mtcnn).

"""

import os
import math
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from moviepy.editor import VideoFileClip

# scene detection
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

# audio
import librosa

# transcription (faster-whisper)
try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except Exception:
    HAS_WHISPER = False

# yolo
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception:
    HAS_YOLO = False

# utilities
from tqdm import tqdm


@dataclass
class SceneInfo:
    start_s: float
    end_s: float
    duration: float
    transcription: str = ""
    dialogue_score: float = 0.0
    audio_emotion_score: float = 0.0
    face_score: float = 0.0
    motion_score: float = 0.0
    combined_score: float = 0.0


def detect_scenes(video_path: str, threshold: float = 27.0) -> List[SceneInfo]:
    """Use PySceneDetect content detector to get coarse scenes."""
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    scenes = []
    for (start, end) in scene_list:
        start_s = start.get_seconds()
        end_s = end.get_seconds()
        scenes.append(SceneInfo(start_s=start_s, end_s=end_s, duration=end_s - start_s))
    if not scenes:
        # fallback - whole video as one scene
        clip = VideoFileClip(video_path)
        scenes = [SceneInfo(0.0, clip.duration, clip.duration)]
    return scenes


def extract_audio_segment(video_path: str, start: float, end: float, sr: int = 16000) -> Tuple[np.ndarray, int]:
    clip = VideoFileClip(video_path).subclip(start, end)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    clip.audio.write_audiofile(tmp_path, fps=sr, verbose=False, logger=None)
    y, _ = librosa.load(tmp_path, sr=sr)
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return y, sr


def compute_audio_emotion_score(y: np.ndarray, sr: int) -> float:
    """Estimate loudness spikes and voice activity as proxy for emotional punch.
    Returns a 0..1 score.
    """
    if len(y) == 0:
        return 0.0
    # RMS energy and dynamic range
    hop_length = 512
    frame_rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(frame_rms + 1e-6)
    # measure dynamic: variance and peak - baseline
    var = float(np.var(rms_db))
    peak = float(np.max(rms_db) - np.median(rms_db))
    # Voice activity ratio (simple energy threshold)
    thresh = np.median(rms_db) + 3.0
    voice_ratio = float((rms_db > thresh).sum()) / max(1, len(rms_db))
    # combine
    score = (np.tanh(var / 30.0) + np.tanh(peak / 20.0) + voice_ratio) / 3.0
    return float(np.clip(score, 0.0, 1.0))


def transcribe_with_whisper(y: np.ndarray, sr: int) -> str:
    """Transcribe audio array with faster-whisper if available. If not, return empty string.
    This function loads a medium-large model by default (can change model_size param).
    """
    if not HAS_WHISPER:
        return ""
    # save to temp wav
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    librosa.output.write_wav(tmp_path, y, sr)
    model_size = 'large-v2'  # chosen for better ASR quality; change if you lack resources
    model = WhisperModel(model_size, device='cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu')
    segments, info = model.transcribe(tmp_path, beam_size=5)
    text = " ".join([seg.text.strip() for seg in segments]).strip()
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return text


def dialogue_coherence_score(text: str) -> float:
    """
    Scores how coherent a spoken segment is.
    Prefers:
      - Sentence starts with capital/interjection
      - Ends with proper punctuation (.!?)
      - Has enough words (not a fragment)
    Returns score in [0, 1].
    """
    if not text or len(text.strip()) < 3:
        return 0.0

    clean = text.strip()

    # Split into sentences by punctuation
    temp = clean.replace('?', '.').replace('!', '.')
    sentences = [s.strip() for s in temp.split('.') if s.strip()]
    if not sentences:
        return 0.0

    best = 0.0

    for s in sentences:
        words = s.split()
        wcount = len(words)

        # Sentence start check
        start_cap = 1.0 if (words and words[0][0].isupper()) else 0.2

        # Sentence end punctuation check (safe)
        last_char = clean[-1] if len(clean) > 0 else ""
        end_punct = 1.0 if last_char in ".!?" else 0.0

        # Sentence completeness based on length (8+ words = full)
        completeness = min(1.0, wcount / 8.0)

        # Combine coherence metrics
        score = 0.4 * start_cap + 0.4 * completeness + 0.2 * end_punct

        best = max(best, score)

    # More sentences → more coherent flow
    sentence_count_factor = min(1.0, len(sentences) / 3.0)

    final_score = best * sentence_count_factor
    return float(np.clip(final_score, 0.0, 1.0))



def compute_face_score(video_path: str, start: float, end: float, sample_fps: int = 2) -> float:
    """Use YOLO (ultralytics) to detect faces in sampled frames.
    Returns 0..1 based on face presence and size.
    """
    if not HAS_YOLO:
        return 0.0
    model = YOLO('yolov8n.pt')  # small detection model; requires weights (downloads automatically)
    # sample frames
    clip = VideoFileClip(video_path).subclip(start, end)
    duration = clip.duration
    if duration <= 0:
        return 0.0
    n_samples = int(max(1, math.ceil(duration * sample_fps)))
    times = np.linspace(0, duration, n_samples)
    face_presence = []
    for t in times:
        frame = clip.get_frame(t)
        results = model(frame, imgsz=320)
        # ultralytics returns boxes in results[0].boxes
        boxes = results[0].boxes if len(results) > 0 else []
        # crude face proxy: count of detected people/classify by size - we assume model detects faces/people
        count = len(boxes)
        face_presence.append(count)
    avg_faces = float(np.mean(face_presence))
    # scale: 0 faces -> 0; 1 face close-up -> 0.6; 2+ faces -> 1.0
    score = np.tanh(avg_faces / 2.0)
    return float(np.clip(score, 0.0, 1.0))


def compute_motion_score_cv(video_path: str, start: float, end: float, sample_rate: int = 5) -> float:
    """Compute average frame-to-frame motion magnitude using optical flow.
    We want balanced motion: not completely static and not chaotic. Scores around medium are best.
    Returns 0..1.
    """
    try:
        import cv2
    except Exception:
        return 0.0
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    start_frame = int(start * fps)
    end_frame = int(end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        return 0.0
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motions = []
    frame_idx = start_frame + 1
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motions.append(np.mean(mag))
        prev_gray = gray
        frame_idx += int(fps / sample_rate) if sample_rate > 0 else 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    cap.release()
    if not motions:
        return 0.0
    mean_motion = float(np.mean(motions))
    # normalized expected motion values - these are heuristics (tweak if needed)
    # small mean_motion -> boring; huge -> chaotic. Best around moderate value
    # use tanh curve to compress
    normalized = np.tanh(mean_motion / 5.0)
    # prefer medium: map normalized (0..1) so that 0.3-0.7 is best
    # score = 1 - abs(normalized - 0.5) * 2  -> gives 1 at 0.5, 0 at extremes
    score = 1.0 - min(1.0, abs(normalized - 0.5) * 2.0)
    return float(np.clip(score, 0.0, 1.0))


def evaluate_scenes(video_path: str, scenes: List[SceneInfo]):
    """Fill scene objects with computed scores."""
    for s in tqdm(scenes, desc='Evaluating scenes'):
        # transcription & dialogue score
        try:
            y, sr = extract_audio_segment(video_path, s.start_s, s.end_s)
            s.audio_emotion_score = compute_audio_emotion_score(y, sr)
            if HAS_WHISPER:
                s.transcription = transcribe_with_whisper(y, sr)
            else:
                s.transcription = ""
            s.dialogue_score = dialogue_coherence_score(s.transcription)
        except Exception as e:
            print('Audio/transcription error for scene', s, e)
            s.audio_emotion_score = 0.0
            s.dialogue_score = 0.0
        # face score
        try:
            s.face_score = compute_face_score(video_path, s.start_s, s.end_s)
        except Exception as e:
            print('Face score error', e)
            s.face_score = 0.0
        # motion score
        try:
            s.motion_score = compute_motion_score_cv(video_path, s.start_s, s.end_s)
        except Exception as e:
            print('Motion score error', e)
            s.motion_score = 0.0
        # combined
        s.combined_score = 0.4 * s.dialogue_score + 0.3 * s.audio_emotion_score + 0.2 * s.face_score + 0.1 * s.motion_score


def sliding_best_window(video_path: str, scene: SceneInfo, window_s: float = 10.0, step_s: float = 1.0, min_s: float = 5.0) -> Tuple[float, float, float]:
    """Slide a window inside the scene and score each window; return (best_start, best_end, score)
    Score uses same components but computed on the window level for audio & motion & faces.
    """
    clip = VideoFileClip(video_path)
    scene_dur = min(scene.duration, clip.duration - scene.start_s)
    window = min(window_s, scene_dur)
    step = step_s
    best_score = -1.0
    best_range = (scene.start_s, min(scene.end_s, scene.start_s + window))
    t = scene.start_s
    while t + min_s <= scene.end_s:
        end_t = min(t + window, scene.end_s)
        try:
            y, sr = extract_audio_segment(video_path, t, end_t)
            audio_score = compute_audio_emotion_score(y, sr)
            trans = transcribe_with_whisper(y, sr) if HAS_WHISPER else ""
            dialog_score = dialogue_coherence_score(trans)
        except Exception:
            audio_score = 0.0
            dialog_score = 0.0
        try:
            face_score = compute_face_score(video_path, t, end_t)
        except Exception:
            face_score = 0.0
        try:
            motion_score = compute_motion_score_cv(video_path, t, end_t)
        except Exception:
            motion_score = 0.0
        score = 0.45 * dialog_score + 0.30 * audio_score + 0.15 * face_score + 0.10 * motion_score
        if score > best_score:
            best_score = score
            best_range = (t, end_t)
        t += step
    return best_range[0], best_range[1], best_score


def export_segment(video_path: str, start: float, end: float, out_path: str = 'best_scene.mp4'):
    clip = VideoFileClip(video_path).subclip(start, end)
    # optional: add fade in/out to make it more 'hooky'
    # clip = clip.fadein(0.2).fadeout(0.2)
    clip.write_videofile(out_path, codec='libx264', audio_codec='aac')


# def main(video_path: str, out_path: str = 'best_scene.mp4'):
#     print('Detecting scenes...')
#     scenes = detect_scenes(video_path)
#     print(f'Found {len(scenes)} scenes')
#     evaluate_scenes(video_path, scenes)
#     # pick top-K scenes by combined score and then run sliding window inside each
#     scenes_sorted = sorted(scenes, key=lambda s: s.combined_score, reverse=True)
#     candidate_windows = []
#     for s in scenes_sorted[:6]:  # check top 6 scenes
#         start, end, score = sliding_best_window(video_path, s, window_s=10.0, step_s=1.0, min_s=5.0)
#         candidate_windows.append((start, end, score))
#     # pick best window overall
#     if not candidate_windows:
#         # fallback export first 8s
#         export_segment(video_path, 0.0, min(8.0, VideoFileClip(video_path).duration), out_path)
#         print('No candidate windows found - exported fallback segment')
#         return
#     best = max(candidate_windows, key=lambda x: x[2])
#     print(f'Best window: {best[0]:.2f}s -> {best[1]:.2f}s (score {best[2]:.3f})')
#     export_segment(video_path, best[0], best[1], out_path)
#     print('Export complete:', out_path)

def pick_best_complete_scene(scene_list, audio_scores):
    """
    scene_list: [(start, end), ...]
    audio_scores: list of { 'dialogue': x, 'emotion': y, 'clarity': z }
    Returns (best_start, best_end)
    """
    best_idx = None
    best_score = -1

    for i, ((start, end), a) in enumerate(zip(scene_list, audio_scores)):
        final_score = (
            0.4 * a["dialogue"] +
            0.3 * a["emotion"] +
            0.2 * a["clarity"] +
            0.1 * a["energy"]
        )
        if final_score > best_score:
            best_score = final_score
            best_idx = i

    return scene_list[best_idx]


def main(input_path, output_path):
    # 1. Detect REAL scene boundaries
    scenes = detect_scenes(input_path)  # [(start, end), ...]

    # 2. Extract audio features for each full scene
    audio_scores = []
    for (start, end) in scenes:
        clip_audio = extract_audio_segment(input_path, start, end)
        text = transcribe_sentence(clip_audio)
        audio_scores.append({
            "dialogue": dialogue_coherence_score(text),
            "emotion": emotion_score(clip_audio),
            "clarity": clarity_score(clip_audio),
            "energy": energy_score(clip_audio),
        })

    # 3. Pick best COMPLETE SCENE
    best_start, best_end = pick_best_complete_scene(scenes, audio_scores)

    # 4. Export scene AS-IS
    extract_video_segment(input_path, output_path, best_start, best_end)

    print("Best scene saved to", output_path)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract best 5-10s mini-scene as a hook from an input video')
    parser.add_argument('--video', help='Path to input video file')
    parser.add_argument('--out', default='best_scene.mp4', help='Output filename')
    args = parser.parse_args()
    main(args.video, args.out)
