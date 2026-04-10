# Fictic: AI-Powered Cinematic Video Engine

Fictic is a comprehensive suite of AI-driven tools designed for high-end video editing automation, specializing in music-driven "punch-intro" edits, dynamic subtitling, and advanced visual effects.

## 🚀 Core Features

### 🎬 Automated Punch-Intro Editing (`fictic.py`)
-   **Whisper-Powered Transcription**: Automatically transcribes input video clips to identify the most compelling dialogue segments.
-   **Beat-Synced Filler**: Uses `librosa` for beat detection to sync filler clips perfectly with the rhythm of chosen background music.
-   **Intelligent Dialogue Selection**: Automatically avoids CTAs and outros, picking the strongest punch-line for the intro.

### ✍️ Dynamic Neon Subtitles
-   Automated subtitle generation with customizable neon glow effects and animated "shine-through" strokes.
-   Rhythmic word-by-word timing for maximum viewer engagement.

### 🎭 Visual & Audio FX
-   **Face Swapping**: Includes `faceswap` and `swap.py` utilities for seamless face replacement.
-   **Audio Processing**: Tools for audio denoising, slowed+reverb effects, and high-fidelity narration generation.
-   **Visual Enhancements**: Glitch overlays, center-cropping for 9:16 (TikTok/Shorts) formats, and cinematic zoom effects.

## 🛠 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/udaydomadiya08/fictic.git
   cd fictic
   ```

2. **Install Dependencies**:
   Choose the appropriate requirements file based on your environment:
   ```bash
   pip install -r requirements_venv.txt
   # OR
   pip install -r requirements_yolovenv.txt
   ```

3. **External Requirements**:
   -   **FFmpeg**: Must be installed on your system for video processing.
   -   **Fonts**: The engine uses custom fonts (e.g., `Anton-Regular.ttf`). Ensure the paths in `fictic.py` are updated to your local setup.

## 📂 Project Structure

-   `fictic.py`: Main entry point for the punch-intro video engine.
-   `transcribe.py`: Dedicated Whisper transcription utility.
-   `swap.py` & `faceswap/`: Face-swapping modules and integration.
-   `denoise.py`: Audio enhancement and noise reduction.
-   `requirements_*.txt`: Dependency manifests for different environments.

## 📖 Usage Example

To generate a Fictic-style edit:
```bash
python3 fictic.py --input_videos ./input_vid --music ./music_folder --out ./output/final_edit.mp4
```

---
*Created with the help of Antigravity AI.*
