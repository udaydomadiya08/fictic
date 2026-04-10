import os
import base64
from io import BytesIO
from PIL import Image
import requests
from nltk.tokenize import sent_tokenize
from collections import defaultdict

from moviepy.editor import (
    AudioFileClip, ImageClip, CompositeVideoClip,
    concatenate_videoclips, concatenate_audioclips,
    TextClip
)

from moviepy.video.fx.all import resize
from moviepy.video.tools.subtitles import SubtitlesClip

import spacy
import whisperx

# from Wav2Lip.inference import parser, run_inference

import os
import time
import random
import json
from datetime import datetime
from pathlib import Path

import os
import json
import random
import time
from datetime import datetime
import google.generativeai as genai
from collections import defaultdict

class GeminiResponse:
    def __init__(self, text):
        self.text = text

FAILED_KEYS_FILE = "disabled_keys.json"
USAGE_FILE = "usage_counts.json"
DAILY_LIMIT = 500
PER_MINUTE_LIMIT = 10

# In-memory per-minute usage tracker
minute_usage_tracker = defaultdict(list)

def load_json_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}

def save_json_file(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def load_disabled_keys():
    data = load_json_file(FAILED_KEYS_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    return set(data.get(today, []))

def save_disabled_key(api_key):
    today = datetime.now().strftime("%Y-%m-%d")
    data = load_json_file(FAILED_KEYS_FILE)

    if today not in data:
        data[today] = []
    if api_key not in data[today]:
        data[today].append(api_key)

    save_json_file(FAILED_KEYS_FILE, data)

def increment_usage(api_key):
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")

    # Daily count
    usage = load_json_file(USAGE_FILE)
    if today not in usage:
        usage[today] = {}
    if api_key not in usage[today]:
        usage[today][api_key] = 0
    usage[today][api_key] += 1
    save_json_file(USAGE_FILE, usage)

    # Per-minute in memory
    minute = now.strftime("%Y-%m-%d %H:%M")
    minute_usage_tracker[api_key] = [ts for ts in minute_usage_tracker[api_key] if ts.startswith(minute)]
    minute_usage_tracker[api_key].append(now.strftime("%Y-%m-%d %H:%M:%S"))

def has_exceeded_daily_limit(api_key, limit=DAILY_LIMIT):
    today = datetime.now().strftime("%Y-%m-%d")
    usage = load_json_file(USAGE_FILE)
    return usage.get(today, {}).get(api_key, 0) >= limit


def has_exceeded_minute_limit(api_key, limit=PER_MINUTE_LIMIT):
    current_minute = datetime.now().strftime("%Y-%m-%d %H:%M")
    recent_times = [ts for ts in minute_usage_tracker[api_key] if ts.startswith(current_minute)]
    return len(recent_times) >= limit


def generate_gemini_response(prompt, model_name=None, max_retries=1111, wait_seconds=5):
    api_keys = [
        "AIzaSyCG-BB-0iP8bTiTJiT9ZgC5eJkzDftV28I",
        "AIzaSyBGpWydub8jBjKW_JM808Q57x_KSVg1Fxw",
        "AIzaSyD-7i3eVHY_tQBlLedDGUYb12tPm88F2bg",
        "AIzaSyCT-678mR3ur4beLyWJJ-QdWA8W8cHvWtM",
        "AIzaSyBnKryfOjV-XsR0tdXWdYv4MXnbvvh_QWU",
        "AIzaSyBenRCth2XXKL6BXh_gRtDAznPfbTd9t4k",
        "AIzaSyCG6iIVxuoPAwRC8FL0DMHhywAFg58vxbM",
        "AIzaSyCWyJeh999WPRt5Mf8hgAfT78hkl_oyy3I",
        "AIzaSyDQoF2-V-jPVinMIHIs4Dts8KPpXeL-5_E",
        "AIzaSyA5VLL-EFpKs2Z0iVdLLK6ir_n9-b1wtrc",
        "AIzaSyCRNFcI51fF1KoS3YbBnaGtFMLIhSqnaSs",
        "AIzaSyCkhmr6hYUKCQMVuNaVMwhUmfLIrvOMn7g",
        "AIzaSyBFZS7DX_wDWvWjln22G3zN2XjORuMJV5o",
        "AIzaSyDOFT9J2OlqyR2KhhMP9qBaE3LqeLQLaIc",
        "AIzaSyDDBqYMNprBSxs006y_Mjm-2iFsedqvyE4",
        "AIzaSyAv7KdGJul7xb5tCnx2bLqZEStXTTtY-NA",
        "AIzaSyAFn_8ws-tj-ix7R_MIvTg-REUZ-93riZo",
        "AIzaSyB9ESuSqJMbEnAdvBKxaGJsfTrkdvaobYc",
        "AIzaSyCqXHtA2dl3tUumG21cMwbhxQdVP9LzypY",
        "AIzaSyAQCRR1KSbgiF3OkXUOInZOntFw1VU4n4k",
        "AIzaSyChdyTOEFX11YlnDaLKMh7IAXA_OzxpWSg",
        "AIzaSyBcBKM39mCY2x2-90tId2LRRQbOzwefLpE",

    ]

    model_names = [
        "gemini-2.5-flash-preview-05-20"
    ]

    disabled_keys_today = load_disabled_keys()

    for attempt in range(max_retries):
        available_keys = [
            k for k in api_keys
            if k not in disabled_keys_today and not has_exceeded_daily_limit(k) and not has_exceeded_minute_limit(k)
        ]

        if not available_keys:
            raise RuntimeError("❌ All API keys are either disabled or have reached daily/minute limits.")

        key = random.choice(available_keys)
        model = model_name or random.choice(model_names)

        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            gemini = genai.GenerativeModel(model)
            print(f"✅ Using model: {model}, key ending in: {key[-6:]}")
            
            response = gemini.generate_content(prompt)
            increment_usage(key)
            return GeminiResponse(response.text.strip())

        except Exception as e:
            print(f"❌ API call failed with key {key[-6:]} (Attempt {attempt + 1}): {e}")
            save_disabled_key(key)
            time.sleep(wait_seconds)

    raise RuntimeError("❌ All Gemini API attempts failed after retries.")




import sys
import os
sys.path.append(os.path.abspath('Wav2Lip'))

def run_wav2lip_inference(
    checkpoint_path,
    face_video,
    audio_path,
    output_video,
    static=True,
    fps=24,
    wav2lip_batch_size=128,   # default value
    resize_factor=1,
    out_height=480
):
    args_list = [
        '--checkpoint_path', checkpoint_path,
        '--face', face_video,
        '--audio', audio_path,
        '--outfile', output_video,
        '--fps', str(fps),
        '--wav2lip_batch_size', str(wav2lip_batch_size),
        '--resize_factor', str(resize_factor),
        '--out_height', str(out_height),
    ]

    if static:
        args_list.append('--static')

    args = parser.parse_args(args_list)

    print("Starting Wav2Lip inference...")
    run_inference(args)
    print("Inference done!")

# === Setup ===
nlp = spacy.load("en_core_web_sm")  # Load spaCy model once

# Gemini API keys (replace with your keys)
api_keys = [
        "AIzaSyCG-BB-0iP8bTiTJiT9ZgC5eJkzDftV28I",
        "AIzaSyBGpWydub8jBjKW_JM808Q57x_KSVg1Fxw",
        "AIzaSyD-7i3eVHY_tQBlLedDGUYb12tPm88F2bg",
        "AIzaSyCT-678mR3ur4beLyWJJ-QdWA8W8cHvWtM",
        "AIzaSyBnKryfOjV-XsR0tdXWdYv4MXnbvvh_QWU",
        "AIzaSyBenRCth2XXKL6BXh_gRtDAznPfbTd9t4k",
        "AIzaSyCG6iIVxuoPAwRC8FL0DMHhywAFg58vxbM",
        "AIzaSyCWyJeh999WPRt5Mf8hgAfT78hkl_oyy3I",
        "AIzaSyDQoF2-V-jPVinMIHIs4Dts8KPpXeL-5_E",
        "AIzaSyA5VLL-EFpKs2Z0iVdLLK6ir_n9-b1wtrc",
        "AIzaSyCRNFcI51fF1KoS3YbBnaGtFMLIhSqnaSs",
        "AIzaSyCkhmr6hYUKCQMVuNaVMwhUmfLIrvOMn7g",
        "AIzaSyBFZS7DX_wDWvWjln22G3zN2XjORuMJV5o",
        "AIzaSyDOFT9J2OlqyR2KhhMP9qBaE3LqeLQLaIc",
        "AIzaSyDDBqYMNprBSxs006y_Mjm-2iFsedqvyE4",
        "AIzaSyAv7KdGJul7xb5tCnx2bLqZEStXTTtY-NA",
        "AIzaSyAFn_8ws-tj-ix7R_MIvTg-REUZ-93riZo",
        "AIzaSyB9ESuSqJMbEnAdvBKxaGJsfTrkdvaobYc",
        "AIzaSyCqXHtA2dl3tUumG21cMwbhxQdVP9LzypY",
        "AIzaSyAQCRR1KSbgiF3OkXUOInZOntFw1VU4n4k",
        "AIzaSyChdyTOEFX11YlnDaLKMh7IAXA_OzxpWSg",
        "AIzaSyBcBKM39mCY2x2-90tId2LRRQbOzwefLpE",

    ]
# Google TTS service account json path
TTS_JSON_KEY_PATH = "my-project-tts-461911-dbd39de52028.json"

# Video size & font for subtitles
VIDEO_SIZE = (1080, 1920)
FONT_PATH = "/Users/uday/Downloads/VIDEOYT/Anton-Regular.ttf"
FONT_SIZE = 110

# Create needed directories once
os.makedirs("audio", exist_ok=True)
os.makedirs("temp/images", exist_ok=True)
os.makedirs("video_created", exist_ok=True)

# === Gemini Image Generation ===
from google import genai
from google.genai import types

def resize_to_1080x1920_stretch(image: Image.Image) -> Image.Image:
    return image.resize((1080, 1920), Image.LANCZOS)

def compress_image(input_image: Image.Image, output_path: str, max_size_kb=2048):
    quality = 90
    while quality >= 20:
        input_image.save(output_path, format="JPEG", quality=quality)
        if os.path.getsize(output_path) <= max_size_kb * 1024:
            break
        quality -= 5
    return output_path

from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import base64
import time
import random
import os

def resize_to_1080x1920_stretch(image: Image.Image) -> Image.Image:
    return image.resize((1080, 1920), Image.LANCZOS)

def compress_image(input_image: Image.Image, output_path: str, max_size_kb=2048):
    quality = 90
    while quality >= 20:
        input_image.save(output_path, format="JPEG", quality=quality)
        if os.path.getsize(output_path) <= max_size_kb * 1024:
            break
        quality -= 5
    return output_path

def generate_image_for_sentence(topic: str, sentence: str, image_index: int = 0, max_retries=50, wait_seconds=5) -> str:
    prompt = (
        f"Create a high-resolution vertical image (1080x1920) that visually conveys the following concept:"
        f"\n📌 Topic: '{topic}'"
        f"\n💡 Idea: \"{sentence}\""
        "\n\nThe image should be cinematic, emotionally powerful, and professionally composed with vibrant colors, deep contrast, and sharp details."
        " It must visually represent the essence of both the topic and the idea without using any text, titles, overlays, logos, or watermarks."
        " The result should be a clean, compelling visual with no words or text — purely imagery — making it perfect for use in modern YouTube Shorts or Reels."
        " Focus on storytelling through visuals alone. Do **not** include any text or writing in the image under any circumstance."
    )

    disabled_keys_today = load_disabled_keys()
    
    for attempt in range(max_retries):
        available_keys = [
            k for k in api_keys
            if k not in disabled_keys_today
            and not has_exceeded_daily_limit(k, limit=100)
            and not has_exceeded_minute_limit(k)
        ]

        if not available_keys:
            print("❌ All Gemini API keys are either disabled or at daily/minute limit.")
            return None

        key = random.choice(available_keys)
        print(f"🔁 Trying Gemini API key ending with: {key[-6:]}")
        
        try:
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model="gemini-2.0-flash-preview-image-generation",
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
            )
            
            increment_usage(key)

            parts = response.candidates[0].content.parts
            if not parts:
                print("❌ Gemini response missing image data.")
                save_disabled_key(key)
                disabled_keys_today.add(key)
                time.sleep(wait_seconds)
                continue

            for part in parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    data = part.inline_data.data
                    image_data = base64.b64decode(data) if isinstance(data, str) else data

                    if image_data:
                        try:
                            image = Image.open(BytesIO(image_data)).convert("RGB")
                            image = resize_to_1080x1920_stretch(image)
                            os.makedirs("temp/images", exist_ok=True)
                            output_path = f"temp/images/img_{image_index}.jpg"
                            compress_image(image, output_path)
                            print(f"✅ Image saved: {output_path}")
                            return output_path
                        except Exception as e:
                            print("❌ Error processing image:", e)
                            return None

            print("❌ No valid image part found in response.")
            save_disabled_key(key)
            disabled_keys_today.add(key)
            time.sleep(wait_seconds)

        except Exception as e:
            print(f"❌ Failed with key {key[-6:]} (Attempt {attempt + 1}): {e}")
            save_disabled_key(key)
            disabled_keys_today.add(key)
            time.sleep(wait_seconds)

    print("❌ All Gemini API attempts failed.")
    return None




def generate_youtube_shorts_script(topic):
    # prompt = f"""
    # Write a YouTube Shorts script about "{topic}" in a high-energy, fast-paced, no-fluff style. create best stongest hook to keep viewers engaged in video, then scrip tmust too be very engaging and high retention one so user dont move shortt to nex tmind it that highest lvel of engagment must be done viewwer shoul dsee full video mind it writet that level of script. The script must be delivered as one single paragraph of plain text — no formatting, no headings, no bullet points, no labels — just pure spoken content. The tone should be exciting, direct, and packed with value, instantly hooking the viewer and keeping them engaged throughout. 
    # video script msut be engaging  so best high level that user tend to see full video and dont leave in betweeen, it should be educatiave informative fro user infintly valuable fro user so user can be highly benefitted. user is beyond god for us mind it. -we are making videos for audience and not robots so give script likwise we need to serve our audience mind it
    # Keep the message clear and compelling with short, punchy sentences designed for maximum impact. Avoid intros or outros — just jump straight into the content and keep the energy high. Make it sound like a rapid-fire narration by a confident creator. i want script that gtts can understand because gtts canno tunderstand short forms avoid any ytpe of short forms use only where necessary, like(example, use is not instaed of isn't) so dont use unnecessary short froms only use when it is neccessary fro specific word shortforms like (RIP etc)
    # entire script must be hooky, hook throughout video script write like that, mind it very important, user msut see full video  viewwer must not left my video unseen viewer should must see my full video short, viewer must becmoe a apart of my channel by likig sharing subscribing commenting good hings write ethat level of script i wan to earn infinte money pls help me with oyur video script i beg you pls write that level of scritp pls mind ti i want like that please request.
    # The final script must not exceed what can be spoken in 55–60 seconds using gTTS (around 200-250 words, or ~16-18 concise sentences). This will be used by an AI video generator, so keep formatting clean and the paragraph tight. Prioritize viral potential, monetization value, and viewer retention. at end tell user to like comment and subscribe to my channel tell it dont ask it if they liked then do just tell them to do all 3 at end, okay mind it,  mind it it should bes seo optimised
    # dont inlude astreiks bold text only plain format normal text mind it
    # """
    # prompt = f"""
    # Write a YouTube Shorts script about "{topic}" that lasts exactly 21 seconds of spoken content (about 60-70 words). Begin with a powerful hook in the first sentence that immediately grabs attention and forces the viewer to keep watching. Deliver fast-paced, high-energy, no-fluff content packed with massive value, curiosity, or shock — whatever keeps the viewer locked in. Every sentence must be sharp, useful, and make the viewer feel smarter, surprised, or empowered. Avoid any intros, outros, or explanations — just pure impactful content from the very first second to the last. Do not use unnecessary abbreviations or short forms — only use short forms like RIP or DIY if essential. Make the script sound like it’s being confidently spoken by a creator who knows what they’re talking about. Keep the tone exciting, engaging, and educational — something the viewer would never skip. Output a single paragraph of plain text — no formatting, no labels, no line breaks. This script should feel custom-made to go viral and retain attention till the very end.
    # """
    prompt = f"""
    Write a YouTube Shorts script about "{topic}" that lasts exactly 21 seconds of spoken content (about 60–70 words). The script must sound like the creator is *actually doing* the crazy, high-stakes challenge or action — not just narrating or explaining it. Start with an insane hook in the very first sentence that immediately pulls the viewer into the action. Make every second intense, visual, and adrenaline-packed, just like a MrBeast video. Do not use intros, outros, or explanations — only pure, high-energy, immersive storytelling from the first second to the last. Use present tense and active voice to make it feel like the viewer is right there. Every sentence must escalate the situation or tension — keep it unpredictable, emotional, and suspenseful. Avoid fluff, stats, or opinions — make it *feel real* and *impossible to look away*. Output a single paragraph of plain text — no formatting, no labels, no line breaks. This must feel like a viral short where the creator is risking, winning, surviving, or giving in real-time.
    """
    # prompt = f"""
    # Write a YouTube Shorts script about "{topic}" that lasts exactly 21 seconds of spoken content (about 60–70 words). The script must feel like a creator is *actually doing* a wild, high-stakes challenge or living through an intense, real-time moment — not just narrating or explaining. Start with an instantly insane, irresistible hook that throws the viewer into the chaos from the first second. Every word must count. Make the storytelling *ultra-dense* — compress a full-length video’s worth of action, stakes, emotion, and detail into just 21 seconds. Make every sentence escalate tension, emotion, or visual intensity. Deliver twists, danger, emotion, and visuals in rapid succession — like the viewer is watching a life-or-death situation unfold. Avoid intros, fluff, stats, or opinions — only pure, high-octane action or revelation in present tense. Viewers must leave feeling like they experienced something unforgettable, intense, and meaningful in record time — almost like an adrenaline-packed enlightenment. no bold text to bes used in script , just give the plain format normal text only script and nothing else  mind it.
    # """



    response = generate_gemini_response(prompt, model_name="gemini-2.5-flash-preview-05-20")
    return response.text.strip()

def mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 to WAV using ffmpeg"""
    if os.path.exists(wav_path):
        os.remove(wav_path)
        print(f"🗑️ Deleted existing WAV file: {wav_path}")

    try:
        ffmpeg.input(mp3_path).output(wav_path, ar=16000, ac=1).run(overwrite_output=True)
        print(f"✅ MP3 to WAV Conversion complete: {wav_path}")
    except ffmpeg.Error as e:
        print("❌ FFmpeg error during mp3_to_wav:")
        print(e.stderr.decode())



def wav_to_mp3(wav_path, mp3_path):
    """Convert WAV to MP3 using ffmpeg"""
    if os.path.exists(mp3_path):
        os.remove(mp3_path)
        print(f"🗑️ Deleted existing MP3 file: {mp3_path}")

    try:
        ffmpeg.input(wav_path).output(mp3_path, acodec='libmp3lame').run(overwrite_output=True)
        print(f"✅ WAV to MP3 Conversion complete: {mp3_path}")
    except ffmpeg.Error as e:
        print("❌ FFmpeg error during wav_to_mp3:")
        print(e.stderr.decode())


# === Google Text-to-Speech ===
from google.oauth2 import service_account
from google.auth.transport.requests import Request

def generate_tts_audio(text, filename="output.mp3", json_key_path=TTS_JSON_KEY_PATH,
                       voice_name="en-US-Studio-O", speaking_rate=1.5):
    # Set credentials env once (optional, but safe)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path

    credentials = service_account.Credentials.from_service_account_file(
        json_key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(Request())
    access_token = credentials.token

    url = "https://texttospeech.googleapis.com/v1/text:synthesize"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": {"text": text},
        "voice": {"languageCode": "en-US", "name": voice_name},
        "audioConfig": {"audioEncoding": "MP3", "speakingRate": speaking_rate}
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        audio_content = response.json()["audioContent"]
        with open(filename, "wb") as out:
            out.write(base64.b64decode(audio_content))
        print(f"✅ Audio saved to {filename}")
        return filename
    else:
        print(f"❌ TTS error {response.status_code}: {response.text}")
        return None

# === WhisperX subtitle creation using SubtitlesClip ===
from moviepy.editor import TextClip, CompositeVideoClip
from moviepy.video.fx.all import fadein, fadeout, resize

import numpy as np
import random
from moviepy.editor import VideoClip, CompositeVideoClip, TextClip
from moviepy.video.fx.all import fadein, fadeout, resize
import colorsys

def create_gradient_frame(w, h, offset, direction, left_color, right_color):
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    X, Y = np.meshgrid(x, y)

    if direction == "diagonal_tl_br":
        pos = (X + Y) / 2
        ratio = (pos + offset) % 1

    elif direction == "diagonal_br_tl":
        pos = (X + Y) / 2
        ratio = (1 - pos + offset) % 1

    else:
        ratio = np.full((h, w), offset)

    gradient = (1 - ratio[..., None]) * left_color + ratio[..., None] * right_color
    return gradient.astype(np.uint8)

def hsv_to_rgb_array(h, s, v):
    """Convert arrays of HSV to RGB arrays (values 0-1)"""
    import colorsys
    rgb = np.array([colorsys.hsv_to_rgb(h_, s, v) for h_ in h.flatten()])  # (N,3)
    return rgb.reshape(h.shape + (3,))

def random_bright_color():
    # Return bright HSV (random hue, full saturation & value)
    h = random.random()
    s = 1.0
    v = 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return np.array([r, g, b]) * 255

def create_word_gradient_clip(word, duration, font, fontsize, video_size):
    text_clip = TextClip(word, fontsize=fontsize, font=font, color="white", method="label")

    text_mask = text_clip.to_mask()
    w, h = text_mask.size

    left_color = random_bright_color()
    right_color = random_bright_color()
    direction = random.choice(["diagonal_tl_br", "diagonal_br_tl"])

    def make_frame(t):
        progress = (t / duration) % 1
        offset = progress
        gradient = create_gradient_frame(w, h, offset, direction, left_color, right_color)
        mask_frame = text_mask.get_frame(t)
        colored_frame = (gradient * mask_frame[:, :, None]).astype(np.uint8)
        return colored_frame

    return VideoClip(make_frame, duration=duration).set_position("center").set_mask(text_mask)


def create_word_by_word_subtitles(
    word_segments,
    video_size=(1080, 1920),
    font="/Users/uday/Downloads/VIDEOYT/Anton-Regular.ttf",
    fontsize=110,
):
    from moviepy.editor import CompositeVideoClip
    from moviepy.video.fx.all import fadein, fadeout

    clips = []

    for word_info in word_segments:
        word = word_info["word"]
        start = word_info["start"]
        end = word_info["end"]
        duration = end - start

        grad_clip = create_word_gradient_clip(word, duration, font, fontsize, video_size)
        grad_clip = grad_clip.set_start(start).fx(fadein, 0.05).fx(fadeout, 0.05)
        clips.append(grad_clip)

    return CompositeVideoClip(clips, size=video_size)

# Load whisperx model once, outside to avoid reloading multiple times
device = "cpu"
print(f"Using device for WhisperX: {device}")

whisper_model = whisperx.load_model("tiny.en", device=device, compute_type="float32")
align_model, align_metadata = whisperx.load_align_model(language_code="en", device=device)


def create_scene_clip(sentence, image_path, audio_path, video_size=VIDEO_SIZE):
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration

    img_clip = ImageClip(image_path).set_duration(duration).resize(video_size).set_fps(30)

    # Transcribe & align audio for subtitles
    result = whisper_model.transcribe(audio_path)
    aligned_result = whisperx.align(result["segments"], align_model, align_metadata, audio_path, device)
    word_segments = aligned_result["word_segments"]

    subtitle_clip = create_word_by_word_subtitles(word_segments, video_size=video_size)

    scene = CompositeVideoClip([img_clip, subtitle_clip.set_duration(duration)])
    scene = scene.set_audio(audio_clip)
    scene = scene.set_duration(duration)  # very important to avoid overlaps

    return scene

# === Main function to create video from script ===
from moviepy.editor import AudioFileClip, CompositeAudioClip, concatenate_videoclips, concatenate_audioclips
def create_scene_clip(sentence, image_path, audio_path, video_size=VIDEO_SIZE):
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration

    img_clip = ImageClip(image_path).set_duration(duration).resize(video_size).set_fps(30)

    # Transcribe & align audio for subtitles
    result = whisper_model.transcribe(audio_path)
    aligned_result = whisperx.align(result["segments"], align_model, align_metadata, audio_path, device)
    word_segments = aligned_result["word_segments"]

    subtitle_clip = create_word_by_word_subtitles(word_segments, video_size=video_size)

    scene = CompositeVideoClip([img_clip, subtitle_clip.set_duration(duration)])

    scene = scene.set_duration(duration).set_audio(None)
  # very important to avoid overlaps

    return scene

# === Main function to create video from script ===
from moviepy.editor import AudioFileClip, CompositeAudioClip, concatenate_videoclips, concatenate_audioclips

def create_video_from_script_with_whisperx(script, user_topic):
    import tempfile
    import subprocess
    import random
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, vfx
    from nltk.tokenize import sent_tokenize
    import os

    def save_clip_to_tempfile(clip, suffix="", pad_duration=0.08):
        clip = clip.set_fps(30)
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=f"{suffix}.mp4").name
        padded_path = tempfile.NamedTemporaryFile(delete=False, suffix=f"{suffix}_padded.mp4").name

        clip.write_videofile(temp_path, codec="libx264", audio_codec="aac", fps=30, preset="ultrafast", threads=8, logger=None)

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", temp_path,
            "-vf", f"tpad=stop_mode=clone:stop_duration={pad_duration}",
           
            "-preset", "ultrafast",
            padded_path
        ]
        subprocess.run(ffmpeg_cmd, check=True)

        os.remove(temp_path)
        return padded_path

    def ffmpeg_chain_random_transitions(clip_paths, output_path, transition_duration=0.08):
        import tempfile
        import os
        import random
        from moviepy.editor import VideoFileClip
        import subprocess

        transitions = [
            "fade", "fadeblack", "fadewhite", "radial", "circleopen",
            "circleclose", "rectcrop", "distance", "slideleft", "slideright", "slideup", "slidedown"
        ]

        # If only one clip, just copy it to the output path
        if len(clip_paths) == 1:
            os.rename(clip_paths[0], output_path)
            return

        # First clip remains unchanged
        current_input = clip_paths[0]

        for i in range(1, len(clip_paths)):
            next_input = clip_paths[i]

            # Get durations
            duration_current = VideoFileClip(current_input).duration
            offset = duration_current - transition_duration

            transition_type = random.choice(transitions)
            print(f"🎞️ Transition {i}: '{transition_type}' at offset={offset:.2f}s")

            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", current_input,
                "-i", next_input,
                "-filter_complex",
                f"[0:v][1:v]xfade=transition={transition_type}:duration={transition_duration}:offset={offset},format=yuv420p[v]",
                "-map", "[v]",
                "-map", "0:a?",  # use audio from first
                "-c:a", "aac",
                "-preset", "ultrafast",
                "-crf", "23",
                temp_output
            ]

            subprocess.run(ffmpeg_cmd, check=True)

            if current_input not in clip_paths:
                try:
                    os.remove(current_input)
                except:
                    pass

            current_input = temp_output

        os.rename(current_input, output_path)


    # --- MAIN LOGIC START ---
    sentences = sent_tokenize(script)
    scene_clips = []

    print(f"🎥 Total scenes to generate: {len(sentences)}")

    for idx, sentence in enumerate(sentences):
        print(f"\n🔹 Generating Scene {idx + 1}: {sentence}")

        image_path = generate_image_for_sentence(user_topic, sentence, idx)
        if not image_path:
            print(f"❌ Failed to generate image for scene {idx + 1}")
            continue

        audio_path = f"audio/scene_{idx}.mp3"
        audio_file = generate_tts_audio(sentence, audio_path)
        if not audio_file or not os.path.exists(audio_file):
            print(f"❌ Failed to generate audio for scene {idx + 1}")
            continue

        scene_clip = create_scene_clip(sentence, image_path, audio_path)
        scene_clips.append(scene_clip)

    if not scene_clips:
        print("❌ No scenes generated successfully.")
        return None

    # Save each clip to padded temporary file
    transition_duration = 0.08
    temp_files = [
        save_clip_to_tempfile(
            clip.resize((1080, 1920)),
            suffix=f"_scene{i}",
            pad_duration=0 if i == 0 else transition_duration  # ⬅️ No padding for first scene
        )
        for i, clip in enumerate(scene_clips)
    ]


    # Apply transitions
    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    ffmpeg_chain_random_transitions(temp_files, output_temp, transition_duration)

    # Clean up temp files
    for f in temp_files:
        try:
            os.remove(f)
        except:
            pass

    final_video = VideoFileClip(output_temp)

    # Add background music - UPDATED TO AVOID AUDIO ISSUES
    try:
        from moviepy.audio.fx.all import audio_loop

        # Load background music and reduce volume
        bg_music_raw = AudioFileClip("/Users/uday/Downloads/VIDEOYT/Cybernetic Dreams.mp3").volumex(0.08)

        # Loop background music to match video duration and set start to 0 explicitly
        bg_music_looped = audio_loop(bg_music_raw, duration=final_video.duration).set_start(0)

        if final_video.audio:
            # Mix existing audio with background music
            final_audio_with_bg = CompositeAudioClip([
                final_video.audio.set_duration(final_video.duration).set_start(0),
                bg_music_looped
            ])
        else:
            # If no original audio, just use background music
            final_audio_with_bg = bg_music_looped

        # Set the combined audio back to video
        final_video = final_video.set_audio(final_audio_with_bg)

    except Exception as e:
        print(f"⚠️ Background music could not be applied: {e}")
        # Fallback: keep original audio if exists, else None
        final_video = final_video.set_audio(final_video.audio if final_video.audio else None)

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/Users/uday/Downloads/VIDEOYT/video_created/final_video_{timestamp}.mp4"
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", audio=True, fps=30)

    print(f"\n✅ Final video created at: {output_path}")
    return output_path

from googleapiclient.http import MediaFileUpload


import os
import pickle
from googleapiclient.discovery import build

TOKEN_PATHS = [
    "/Users/uday/Downloads/VIDEOYT/token1.pickle",
    "/Users/uday/Downloads/VIDEOYT/token2.pickle",
    "/Users/uday/Downloads/VIDEOYT/token3.pickle",
    "/Users/uday/Downloads/VIDEOYT/token.pickle"
]

def build_youtube_client(token_path):
    with open(token_path, "rb") as token_file:
        credentials = pickle.load(token_file)
    return build("youtube", "v3", credentials=credentials)

def is_token_usable(youtube):
    try:
        channel_response = youtube.channels().list(part="contentDetails", mine=True).execute()
        return channel_response is not None
    except Exception as e:
        print(f"❌ Token not usable or quota exceeded: {e}")
        return False

def get_available_token():
    for path in TOKEN_PATHS:
        try:
            youtube = build_youtube_client(path)
            if is_token_usable(youtube):
                print(f"✅ Using token: {path}")
                return youtube, path
        except Exception as e:
            print(f"⚠️ Failed to load token: {path} — {e}")
    return None, None


# === Example usage ===
if __name__ == "__main__":
        
    while True:

    # list_items = ["The Abandoned Park That Resurrects Your Childhood Fears", "The Most Chilling Moment in a Deserted Factory", "Exploring the Dark Side of Productivity Hacks"

    # ]

    # for i in range(len(list_items)):
    #     my_topic = list_items[i]
    #     print(my_topic)

    


        import shutil
        from pathlib import Path
        def delete_path(path):
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                elif path.is_file():
                    path.unlink()
                print(f"✅ Deleted: {path}")
            except Exception as e:
                print(f"❌ Error deleting {path}: {e}")

        # Cache directories
        cache_dirs = [
            Path.home() / "Library" / "Caches",
            Path("/Library/Caches"),
        ]

        # Log directories
        log_dirs = [
            Path.home() / "Library" / "Logs",
            Path("/Library/Logs")
        ]





        dirs_to_clean = cache_dirs + log_dirs 
        print("🧹 Cleaning up...")

        for d in dirs_to_clean:
            if d.exists():
                for item in d.iterdir():
                    delete_path(item)
            else:
                print(f"⚠️ Directory not found: {d}")

        print("✅ Done cleaning up!")

        import os

        output_path = "final_video/myfinal.mp4"  # replace with your actual output filename

        # Delete the file if it already exists
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"Previous file '{output_path}' deleted.")


        folder_path1 = "video_creation/"
        folder_path2 = "audio/"
        folder_path3 = "video_created/"

        # Function to delete files from a folder
        def delete_files_in_folder(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                

        # Delete files from both folders
        delete_files_in_folder(folder_path1)
        delete_files_in_folder(folder_path2)
        delete_files_in_folder(folder_path3)
        youtube, token_path = get_available_token()

        # Step 1: Get your channel ID - if you already know it, skip this step
        channel_response = youtube.channels().list(
            part="contentDetails",
            mine=True  # gets your own channel details
        ).execute()

        # uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        # # Step 2: Retrieve all video titles from the uploads playlist
        # video_titles = []
        # next_page_token = None

        # while True:
        #     playlist_response = youtube.playlistItems().list(
        #         part="snippet",
        #         playlistId=uploads_playlist_id,
        #         maxResults=50,
        #         pageToken=next_page_token
        #     ).execute()

        #     for item in playlist_response['items']:
        #         video_titles.append(item['snippet']['title'])

        #     next_page_token = playlist_response.get('nextPageToken')
        #     if not next_page_token:
        #         break

        # Your custom playlist ID (e.g., Shorts playlist)
  
        custom_playlist_id = "PLy_LBg9TCEx9v0AginRCne_X_X8acGuzu" #finance
       
 
        
        # Step 2: Retrieve all video titles from the custom playlist
        video_titles = []
        next_page_token = None

        while True:
            playlist_response = youtube.playlistItems().list(
                part="snippet",
                playlistId=custom_playlist_id,  # ✅ use your own playlist ID here
                maxResults=50,
                pageToken=next_page_token
            ).execute()

            for item in playlist_response['items']:
                video_titles.append(item['snippet']['title'])

            next_page_token = playlist_response.get('nextPageToken')
            if not next_page_token:
                break

        # Optionally print titles
        for title in video_titles:
            print(f"🎬 {title}")


        # Step 3: Save video titles to a text file and CSV file
        with open("/Users/uday/Downloads/VIDEOYT/used_topics.txt", "w", encoding="utf-8") as txt_file:
            for title in video_titles:
                txt_file.write(title + "\n")



        print(f"Retrieved and saved {len(video_titles)} video titles.")

        TOPIC_LOG_FILE = "used_topics.txt"

    
        def generate_viral_wealth_topic():
            # prompt = (
            #     "Titles that create curiosity and enagagment and attractive that user tend to click highly"
            #     "1) Personal Finance & Investing 2)Health & Fitness 3) Tech Tutorials and Reviews 4) Tech and AI Innovations 5) Personal Development & Motivation 6) how to and facts are our target niches on which we will create videos\n\n"
            #     "generate a YouTube video title from above niches randomly that is:\n"
            #     "- catchy SEO-optimized with high click-through potential\n"
            #     "dont include i as title should be ggeneral but specific but not include me\n"
            #     "i want ot generate infinite wealth by using this title to create my video and post on yt to earn money"
            #     "- Strictly a single-line output (only the video title)\n\n"
            #     "dont include year, The output must be only the video title. Keep it human, and audience-focused."
            # )
            # prompt = [
            #     """
            #     Act as a infinite wealth YouTube content creator and expert viral scriptwriter in the Personal Finance and Investing niche.

            #     I want you to generate **highly engaging, viral video content ideas and titles** based on the following subcategories (ranked in descending order of profitability and audience interest):
            
            #     1. Passive Income & Wealth Building  
            #     2. Investing for Beginners  
            #     3. Stock Market & Equity Investing  
            #     4. Real Estate Investing  
            #     5. Cryptocurrency & Blockchain Investing  
            #     6. Budgeting & Saving Money  
            #     7. Tax Optimization & Smart Filing  
            #     8. Credit Score & Loan Management  
            #     9. Financial Planning & Goal Setting  
            #     10. Finance for Freelancers and Digital Creators  

            #     From  subcategories listed above, chose randomly one and give me:
            #     -title which is  such that user just clicks the video and want to see my video that much hooky catchy and attractive title, he tend to subscribe my video anyhow by thsi title create that level of bes ttitle it should immediately make viewer to watch video mind it i want that level of title. mind it it should bes seo optimised
            #     - 1 viral, curiosity-driven YouTube video title 
            #     -we are making videos for audience and not robots so give  a title likwise
            #     - dont include me in the title it should be general but specific mind it
            #     - title selected must be most trendy which is most trending on google search or whatever it must be that ehich peope are most searching for so video has highest potential of getiing viral, title must be specific not generic okay and aslo catcah attractice that user want to watch video and subscribe my channel mind it 
            #     - title msut be short not long one, max 50 characters, be precise consices and it should m,sut be engaging tiitle hooky title but short only and it must be valid title not an invalid one plas mind it, mind it
            #     Make title attention-grabbing, specific, and designed for high watch time and shares — ideal for platform YouTube, only return title plain text in one line and nothing else don tuse bold text just plain format normal  text and nothing else.
                
            #     """
            # ]

            # prompt = [
            #     """
            #     Act as a infinite wealth YouTube content creator and expert viral scriptwriter in the Tech & Gadgets niche, including AI Tools.

            #     I want you to generate **highly engaging, viral video content ideas and titles** based on the following subcategories (ranked in descending order of profitability and audience interest):

            #     1. AI Tools That Boost Productivity  
            #     2. Futuristic Gadgets & Smart Devices  
            #     3. Tech Reviews & Comparisons  
            #     4. Mobile & Laptop Innovations  
            #     5. Emerging Technologies & Trends  
            #     6. Software Tools & Chrome Extensions  
            #     7. Best Tech Under Budget (phones, gadgets etc.)  
            #     8. Automation & Workflow Tools  
            #     9. Daily Tech Life Hacks  
            #     10. Tech News & Quick Updates  

            #     From subcategories listed above, choose randomly one and give me:
            #     - title which is such that user just clicks the video and want to see my video that much hooky catchy and attractive title, he tend to subscribe my video anyhow by this title create that level of best title it should immediately make viewer to watch video mind it i want that level of title. mind it it should be seo optimised
            #     - 1 viral, curiosity-driven YouTube video title 
            #     - do not include me in the title it should be general but specific mind it
            #     - title selected must be most trendy which is most trending on google search or whatever it must be that which people are most searching for so video has highest potential of getting viral
            #     - title must be short not long one, max 50 characters, be precise concise and it should must be engaging title hooky title but short only and it must be valid title not an invalid one please mind it, mind it
            #     Make title attention-grabbing, specific, and designed for high watch time and shares — ideal for platform YouTube, only return title plain text in one line and nothing else do not use bold text just plain format normal text and nothing else.
            #     """
            # ]

            # prompt = [
            #     """
            #     Act as a infinite wealth YouTube content creator and expert viral scriptwriter in the Health & Fitness niche.

            #     I want you to generate **highly engaging, viral video content ideas and titles** based on the following subcategories (ranked in descending order of profitability and audience interest):

            #     1. Weight Loss & Fat Burning  
            #     2. Muscle Building & Home Workouts  
            #     3. Morning Routines for Energy & Focus  
            #     4. Fitness Challenges (7-day, 30-day, etc.)  
            #     5. Nutrition & Healthy Eating Hacks  
            #     6. Mental Health & Stress Relief Techniques  
            #     7. Sleep Hacks & Night Routines  
            #     8. Mobility, Flexibility & Posture Fixes  
            #     9. Biohacking & Recovery Tools  
            #     10. Fitness Tips for Busy People  

            #     From subcategories listed above, choose randomly one and give me:
            #     - title which is such that user just clicks the video and want to see my video that much hooky catchy and attractive title, he tend to subscribe my video anyhow by this title create that level of best title it should immediately make viewer to watch video mind it i want that level of title. mind it it should be seo optimised
            #     - 1 viral, curiosity-driven YouTube video title 
            #     - do not include me in the title it should be general but specific mind it
            #     - title selected must be most trendy which is most trending on google search or whatever it must be that which people are most searching for so video has highest potential of getting viral
            #     - title must be short not long one, max 50 characters, be precise concise and it should must be engaging title hooky title but short only and it must be valid title not an invalid one please mind it, mind it
            #     Make title attention-grabbing, specific, and designed for high watch time and shares — ideal for platform YouTube, only return title plain text in one line and nothing else do not use bold text just plain format normal text and nothing else.
            #     """
            # ]

            # prompt = [
            #     """
            #     Act as a infinite wealth YouTube content creator and expert viral scriptwriter in the Make Money Online & Side Hustles niche.

            #     I want you to generate **highly engaging, viral video content ideas and titles** based on the following subcategories (ranked in descending order of profitability and audience interest):

            #     1. AI-Powered Side Hustles  
            #     2. Freelancing & Remote Work Platforms  
            #     3. Dropshipping & E-commerce  
            #     4. Affiliate Marketing Strategies  
            #     5. Print-on-Demand & Digital Products  
            #     6. Passive Income with Content Creation  
            #     7. Earning from YouTube, TikTok & Reels  
            #     8. Earning with ChatGPT & AI Automation  
            #     9. Selling Courses & Info Products  
            #     10. Best Work-from-Home Hacks  

            #     From subcategories listed above, choose randomly one and give me:
            #     - title which is such that user just clicks the video and want to see my video that much hooky catchy and attractive title, he tend to subscribe my video anyhow by this title create that level of best title it should immediately make viewer to watch video mind it i want that level of title. mind it it should be seo optimised
            #     - 1 viral, curiosity-driven YouTube video title 
            #     - do not include me in the title it should be general but specific mind it
            #     - title selected must be most trendy which is most trending on google search or whatever it must be that which people are most searching for so video has highest potential of getting viral
            #     - title must be short not long one, max 50 characters, be precise concise and it should must be engaging title hooky title but short only and it must be valid title not an invalid one please mind it, mind it
            #     Make title attention-grabbing, specific, and designed for high watch time and shares — ideal for platform YouTube, only return title plain text in one line and nothing else do not use bold text just plain format normal text and nothing else.
            #     """
            # ]

            # prompt = [
            #     """
            #     Act as a infinite wealth YouTube content creator and expert viral scriptwriter in the Motivation & Self-Improvement niche.

            #     I want you to generate **highly engaging, viral video content ideas and titles** based on the following subcategories (ranked in descending order of profitability and audience interest):

            #     1. Morning Motivation & Daily Routines  
            #     2. Mindset Shifts for Success  
            #     3. Time Management & Productivity  
            #     4. Confidence Building & Overcoming Fear  
            #     5. Goal Setting & Life Planning  
            #     6. Habits of Successful People  
            #     7. Law of Attraction & Manifestation  
            #     8. Stoic Wisdom & Mental Toughness  
            #     9. Self-Discipline & Focus  
            #     10. Inspirational Stories & Life Lessons  

            #     From subcategories listed above, choose randomly one and give me:
            #     - title which is such that user just clicks the video and want to see my video that much hooky catchy and attractive title, he tend to subscribe my video anyhow by this title create that level of best title it should immediately make viewer to watch video mind it i want that level of title. mind it it should be seo optimised
            #     - 1 viral, curiosity-driven YouTube video title 
            #     - do not include me in the title it should be general but specific mind it
            #     - title selected must be most trendy which is most trending on google search or whatever it must be that which people are most searching for so video has highest potential of getting viral
            #     - title must be short not long one, max 50 characters, be precise concise and it should must be engaging title hooky title but short only and it must be valid title not an invalid one please mind it, mind it
            #     Make title attention-grabbing, specific, and designed for high watch time and shares — ideal for platform YouTube, only return title plain text in one line and nothing else do not use bold text just plain format normal text and nothing else.
            #     """
            # ]

            # prompt = [
            #     """
            #     Act as a infinite wealth YouTube content creator and expert viral scriptwriter in the Relationships & Dating Advice niche.

            #     I want you to generate **highly engaging, viral video content ideas and titles** based on the following subcategories (ranked in descending order of profitability and audience interest):

            #     1. Dating Tips & First Impressions  
            #     2. Relationship Communication & Conflict Resolution  
            #     3. Online Dating & Dating Apps Strategies  
            #     4. Building Trust & Emotional Intimacy  
            #     5. Breakup Recovery & Moving On  
            #     6. Signs of Healthy vs Toxic Relationships  
            #     7. Attraction Psychology & Body Language  
            #     8. Long-Distance Relationships Advice  
            #     9. Marriage & Commitment Tips  
            #     10. Self-Love & Personal Growth in Relationships  

            #     From subcategories listed above, choose randomly one and give me:
            #     - title which is such that user just clicks the video and want to see my video that much hooky catchy and attractive title, he tend to subscribe my video anyhow by this title create that level of best title it should immediately make viewer to watch video mind it i want that level of title. mind it it should be seo optimised
            #     - 1 viral, curiosity-driven YouTube video title 
            #     - do not include me in the title it should be general but specific mind it
            #     - title selected must be most trendy which is most trending on google search or whatever it must be that which people are most searching for so video has highest potential of getting viral
            #     - title must be short not long one, max 50 characters, be precise concise and it should must be engaging title hooky title but short only and it must be valid title not an invalid one please mind it, mind it
            #     Make title attention-grabbing, specific, and designed for high watch time and shares — ideal for platform YouTube, only return title plain text in one line and nothing else do not use bold text just plain format normal text and nothing else.
            #     """
            # ]

            # prompt = [
            #     """
            #     Act as a infinite wealth YouTube content creator and expert viral scriptwriter in the Health & Fitness niche.

            #     I want you to generate **highly engaging, viral video content ideas and titles** based on the following subcategories (ranked in descending order of profitability and audience interest):

            #     1. Home Workouts & Exercise Routines  
            #     2. Weight Loss Tips & Fat Burning  
            #     3. Nutrition & Healthy Eating  
            #     4. Mental Health & Stress Relief  
            #     5. Muscle Building & Strength Training  
            #     6. Yoga & Flexibility Exercises  
            #     7. Fitness Challenges & Transformation Stories  
            #     8. Supplements & Vitamins Guide  
            #     9. Sleep Improvement & Recovery  
            #     10. Health Myths & Facts Debunked  

            #     From subcategories listed above, choose randomly one and give me:
            #     - title which is such that user just clicks the video and want to see my video that much hooky catchy and attractive title, he tend to subscribe my video anyhow by this title create that level of best title it should immediately make viewer to watch video mind it i want that level of title. mind it it should be seo optimised
            #     - 1 viral, curiosity-driven YouTube video title 
            #     - do not include me in the title it should be general but specific mind it
            #     - title selected must be most trendy which is most trending on google search or whatever it must be that which people are most searching for so video has highest potential of getting viral
            #     - title must be short not long one, max 50 characters, be precise concise and it should must be engaging title hooky title but short only and it must be valid title not an invalid one please mind it, mind it
            #     Make title attention-grabbing, specific, and designed for high watch time and shares — ideal for platform YouTube, only return title plain text in one line and nothing else do not use bold text just plain format normal text and nothing else.
            #     """
            # ]

            # prompt = [
            #     """
            #     Act as a infinite wealth YouTube content creator and expert viral scriptwriter in the Beauty & Fashion niche.

            #     I want you to generate **highly engaging, viral video content ideas and titles** based on the following subcategories (ranked in descending order of profitability and audience interest):

            #     1. Skincare Routines & Tips  
            #     2. Makeup Tutorials & Hacks  
            #     3. Fashion Trends & Styling Advice  
            #     4. Haircare & Hairstyles  
            #     5. Product Reviews & Recommendations  
            #     6. Sustainable & Ethical Fashion  
            #     7. Beauty Tools & Gadgets  
            #     8. Seasonal Fashion Lookbooks  
            #     9. DIY Beauty Remedies  
            #     10. Celebrity Style Inspirations  

            #     From subcategories listed above, choose randomly one and give me:
            #     - title which is such that user just clicks the video and want to see my video that much hooky catchy and attractive title, he tend to subscribe my video anyhow by this title create that level of best title it should immediately make viewer to watch video mind it i want that level of title. mind it it should be seo optimised
            #     - 1 viral, curiosity-driven YouTube video title 
            #     - do not include me in the title it should be general but specific mind it
            #     - title selected must be most trendy which is most trending on google search or whatever it must be that which people are most searching for so video has highest potential of getting viral
            #     - title must be short not long one, max 50 characters, be precise concise and it should must be engaging title hooky title but short only and it must be valid title not an invalid one please mind it, mind it
            #     Make title attention-grabbing, specific, and designed for high watch time and shares — ideal for platform YouTube, only return title plain text in one line and nothing else do not use bold text just plain format normal text and nothing else.
            #     """
            # ]

            # prompt = [
            #     """
            #     Act as a infinite wealth YouTube content creator and expert viral scriptwriter in the Education & Learning niche.

            #     I want you to generate **highly engaging, viral video content ideas and titles** based on the following subcategories (ranked in descending order of profitability and audience interest):

            #     1. Study Tips & Productivity Hacks  
            #     2. Learning New Skills Fast  
            #     3. Exam Preparation Strategies  
            #     4. Language Learning Tips  
            #     5. Educational Technology & Tools  
            #     6. Science Experiments & Demonstrations  
            #     7. Math Tricks & Problem Solving  
            #     8. Career Advice & Job Skills  
            #     9. History & Fun Facts  
            #     10. Book Summaries & Reviews  

            #     From subcategories listed above, choose randomly one and give me:
            #     - title which is such that user just clicks the video and want to see my video that much hooky catchy and attractive title, he tend to subscribe my video anyhow by this title create that level of best title it should immediately make viewer to watch video mind it i want that level of title. mind it it should be seo optimised
            #     - 1 viral, curiosity-driven YouTube video title 
            #     - do not include me in the title it should be general but specific mind it
            #     - title selected must be most trendy which is most trending on google search or whatever it must be that which people are most searching for so video has highest potential of getting viral
            #     - title must be short not long one, max 50 characters, be precise concise and it should must be engaging title hooky title but short only and it must be valid title not an invalid one please mind it, mind it
            #     Make title attention-grabbing, specific, and designed for high watch time and shares — ideal for platform YouTube, only return title plain text in one line and nothing else do not use bold text just plain format normal text and nothing else.
            #     """
            # ]

            # prompt = [
            #     """
            #     Act as a infinite wealth YouTube content creator and expert viral scriptwriter in the Travel & Adventure niche.

            #     I want you to generate **highly engaging, viral video content ideas and titles** based on the following subcategories (ranked in descending order of profitability and audience interest):

            #     1. Budget Travel Hacks  
            #     2. Hidden Travel Gems  
            #     3. Adventure Sports & Activities  
            #     4. Travel Gear Reviews  
            #     5. Cultural Experiences  
            #     6. Travel Vlogs & Stories  
            #     7. Travel Safety Tips  
            #     8. Eco-Friendly Travel  
            #     9. Food & Cuisine from Around the World  
            #     10. Solo Travel Guides  

            #     From subcategories listed above, choose randomly one and give me:
            #     - title which is such that user just clicks the video and want to see my video that much hooky catchy and attractive title, he tend to subscribe my video anyhow by this title create that level of best title it should immediately make viewer to watch video mind it i want that level of title. mind it it should be seo optimised
            #     - 1 viral, curiosity-driven YouTube video title 
            #     - do not include me in the title it should be general but specific mind it
            #     - title selected must be most trendy which is most trending on google search or whatever it must be that which people are most searching for so video has highest potential of getting viral
            #     - title must be short not long one, max 50 characters, be precise concise and it should must be engaging title hooky title but short only and it must be valid title not an invalid one please mind it, mind it
            #     Make title attention-grabbing, specific, and designed for high watch time and shares — ideal for platform YouTube, only return title plain text in one line and nothing else do not use bold text just plain format normal text and nothing else.
            #     """
            # ]
            # prompt = [
            #     """
            #     Act as a infinite wealth YouTube content creator and expert viral scriptwriter in the Health & Fitness niche.

            #     I want you to generate **highly engaging, viral video content ideas and titles** based on the following subcategories (ranked in descending order of profitability and audience interest):

            #     1. Home Workout Routines  
            #     2. Nutrition & Diet Tips  
            #     3. Weight Loss Strategies  
            #     4. Mental Health & Wellness  
            #     5. Yoga & Meditation  
            #     6. Fitness Challenges  
            #     7. Supplements & Vitamins  
            #     8. Muscle Building Tips  
            #     9. Healthy Recipes  
            #     10. Injury Prevention & Recovery  

            #     From subcategories listed above, choose randomly one and give me:
            #     - title which is such that user just clicks the video and want to see my video that much hooky catchy and attractive title, he tend to subscribe my video anyhow by this title create that level of best title it should immediately make viewer to watch video mind it i want that level of title. mind it it should be seo optimised
            #     - 1 viral, curiosity-driven YouTube video title 
            #     - do not include me in the title it should be general but specific mind it
            #     - title selected must be most trendy which is most trending on google search or whatever it must be that which people are most searching for so video has highest potential of getting viral
            #     - title must be short not long one, max 50 characters, be precise concise and it should must be engaging title hooky title but short only and it must be valid title not an invalid one please mind it, mind it
            #     Make title attention-grabbing, specific, and designed for high watch time and shares — ideal for platform YouTube, only return title plain text in one line and nothing else do not use bold text just plain format normal text and nothing else.
            #     """
            # ]

            # prompt = [
            #     """
            #     Act as a infinite wealth YouTube content creator and expert viral scriptwriter in the Beauty & Fashion niche.

            #     I want you to generate **highly engaging, viral video content ideas and titles** based on the following subcategories (ranked in descending order of profitability and audience interest):

            #     1. Skincare Routines & Tips  
            #     2. Makeup Tutorials  
            #     3. Fashion Trends & Hauls  
            #     4. Hair Care & Styling  
            #     5. Product Reviews & Recommendations  
            #     6. DIY Beauty Hacks  
            #     7. Sustainable Fashion  
            #     8. Men's Grooming Tips  
            #     9. Celebrity Style Inspiration  
            #     10. Seasonal Lookbooks  

            #     From subcategories listed above, choose randomly one and give me:
            #     - title which is such that user just clicks the video and want to see my video that much hooky catchy and attractive title, he tend to subscribe my video anyhow by this title create that level of best title it should immediately make viewer to watch video mind it i want that level of title. mind it it should be seo optimised
            #     - 1 viral, curiosity-driven YouTube video title 
            #     - do not include me in the title it should be general but specific mind it
            #     - title selected must be most trendy which is most trending on google search or whatever it must be that which people are most searching for so video has highest potential of getting viral
            #     - title must be short not long one, max 50 characters, be precise concise and it should must be engaging title hooky title but short only and it must be valid title not an invalid one please mind it, mind it
            #     Make title attention-grabbing, specific, and designed for high watch time and shares — ideal for platform YouTube, only return title plain text in one line and nothing else do not use bold text just plain format normal text and nothing else.
            #     """
            # ]
            #  prompt = [
            #     """
            #     Act as a viral YouTube content creator and expert scriptwriter in the niche Beyond God.

            #     I want you to generate 10 highly engaging, viral YouTube video titles that cover all the following subcategories:

            #     1. Transcendence beyond religion  
            #     2. Mystical experiences and awakening  
            #     3. Philosophy of ultimate reality  
            #     4. Divine consciousness and unity  
            #     5. Ancient spiritual wisdom  
            #     6. Meditation beyond the self  
            #     7. The nature of enlightenment  
            #     8. Cosmic energy and universal life  
            #     9. Higher dimensions and spirituality  
            #     10. The path to absolute truth  

            #     Requirements for each title:
            #     - Max 50 characters  
            #     - Specific, SEO optimized, and highly clickable  
            #     - Engaging and curiosity-driven to maximize watch time and shares  
            #     - No mention of "me" or the creator; keep titles general but specific  
            #     - Reflect current trending searches in these topics  
            #     - Output titles as plain text only, one title per line, no formatting or extras
            #     """,

            #     """
            #     Act as a viral YouTube content creator and expert scriptwriter in the niche Infinity in Everything (Only Good Things).

            #     I want you to generate 10 highly engaging, viral YouTube video titles that cover all the following subcategories:

            #     1. Abundance mindset and positivity  
            #     2. Manifesting infinite good energy  
            #     3. Infinite possibilities in daily life  
            #     4. Gratitude practices for infinite joy  
            #     5. Science of infinite vibrations  
            #     6. Infinite love and compassion  
            #     7. Eternal happiness and peace  
            #     8. Infinite creativity and flow  
            #     9. Limitless growth and self-improvement  
            #     10. Infinite good karma and blessings  

            #     Requirements for each title:
            #     - Max 50 characters  
            #     - Specific, SEO optimized, and highly clickable  
            #     - Engaging and curiosity-driven to maximize watch time and shares  
            #     - No mention of "me" or the creator; keep titles general but specific  
            #     - Reflect current trending searches in these topics  
            #     - Output titles as plain text only, one title per line, no formatting or extras
            #     """,

            #     """
            #     Act as a viral YouTube content creator and expert scriptwriter in the niche Immortality and Eternal Life.

            #     I want you to generate 10 highly engaging, viral YouTube video titles that cover all the following subcategories:

            #     1. Scientific breakthroughs for eternal life  
            #     2. Ancient myths and legends about immortality  
            #     3. Mind and body longevity techniques  
            #     4. Spiritual paths to eternal life  
            #     5. Future technologies for life extension  
            #     6. Anti-aging and youth preservation  
            #     7. Genetic engineering and immortality  
            #     8. Consciousness beyond death  
            #     9. Cryonics and suspended animation  
            #     10. Ethical debates on immortality  

            #     Requirements for each title:
            #     - Max 50 characters  
            #     - Specific, SEO optimized, and highly clickable  
            #     - Engaging and curiosity-driven to maximize watch time and shares  
            #     - No mention of "me" or the creator; keep titles general but specific  
            #     - Reflect current trending searches in these topics  
            #     - Output titles as plain text only, one title per line, no formatting or extras
            #     """
            # ]



            # prompt = (

            #     "generate a YouTube video title on which i cerate video and upload on yt\n"
            #     "- catchy SEO-optimized with high click-through potential\n"
            #     "dont include i as title should be ggeneral but specific but not include me\n"
            #     "i want ot generate infinite wealth by using this title to create my video and post on yt to earn money"
            #     "- Strictly a single-line output (only the video title)\n\n"
            #     "dont include year, The output must be only the video title. Keep it human, and audience-focused."
            # )  
            # )
            # prompt = [
            # """
            #     Act as a viral storytelling YouTube content creator and expert hook-based scriptwriter in the Stories niche.

            #     I want you to generate **highly engaging, curiosity-driven, viral video title ideas** based on the following subcategories (ranked in descending order of popularity and emotional engagement):

            #     1. Real-life Inspirational Stories  
            #     2. Mystery and Unsolved Cases  
            #     3. Horror and Supernatural Stories  
            #     4. Crime and Investigation Narratives  
            #     5. Emotional Love and Relationship Stories  
            #     6. Survival and Near-death Experiences  
            #     7. Historical Events Told as Stories  
            #     8. Revenge and Justice-Based Stories  
            #     9. Motivational Stories of Success After Failure  
            #     10. Moral Stories with a Powerful Twist  

            #     From the subcategories listed above, randomly pick one and give me:
            #     - One ultra-hooky, viral YouTube video title 
            #     - The title should instantly make viewers click and watch the video
            #     - Make sure it's designed to emotionally engage humans — not sound robotic
            #     - Do not mention "me" or "my" — make it relatable and universal
            #     - The title must be the most *currently trending* topic searched online within the stories category
            #     - It should be specific, curiosity-driven, and emotionally charged
            #     - Keep the title under 50 characters — short, concise, powerful, and SEO optimized
            #     - Do not use bold text, just return plain text — one title only

            #     Output only the title in a single line, with no formatting or explanations.
            #     """
            # ]
            # prompt = [
            #     """
            #     Act as a viral storytelling YouTube content creator and expert hook-based scriptwriter in the Ghost Stories niche.

            #     I want you to generate **one highly engaging, viral YouTube video title** based on the following subcategories (ranked by fear factor, audience retention, and viral trends):

            #     1. True Real-Life Ghost Encounters  
            #     2. Haunted Houses and Cursed Places  
            #     3. Possession and Paranormal Activity  
            #     4. Ghost Sightings Caught on Camera  
            #     5. Ancient Ghost Legends from Different Cultures  
            #     6. Ghost Stories from Abandoned Hospitals or Asylums  
            #     7. Supernatural Happenings During Travel or Camping  
            #     8. Real Stories of Spirits Communicating with the Living  
            #     9. Mysterious Deaths Linked to Ghosts  
            #     10. Midnight Horror Tales with Shocking Endings  

            #     From the subcategories above, randomly pick one and give me:
            #     - One short, curiosity-driven, emotionally charged YouTube video title
            #     - The title should immediately compel viewers to click and watch
            #     - Avoid generic phrasing — use specific, SEO-optimized, and trending keywords
            #     - Do **not** use “me”, “my”, or personal terms — make it universally relatable
            #     - Make it sound like a real, chilling story or warning, not clickbait
            #     - Keep the title under 50 characters — ultra-short, powerful, and human-readable
            #     - Output only one plain text title — no formatting, no bullet points, no explanation
            #     """
            # ]

            prompt = ["""
            Act as the richest and most viral YouTuber on Earth — a fake MrBeast.

            You create extreme, attention-grabbing YouTube video ideas that instantly go viral. Your style is wild, over-the-top, and built for mass attention, but your titles are short, real, and emotionally powerful.

            🎯 Your task: Generate **one ultra-viral, curiosity-driven video title** that sounds like something MrBeast would post.

            🚫 Do not include any text like "here is" or quotes — just output the raw title only.

            🎬 Format Guidelines:
            - Max 50 characters
            - No personal pronouns like “my” or “me”
            - Must imply something crazy, expensive, extreme, emotional, or suspenseful
            - Built for maximum watch-time and shares
            - No hashtags, emojis, or clickbait lies

            🧠 Examples for inspiration (do not repeat):
            - I Lived 7 Days in a Haunted Mall  
            - 1,000 People vs 1 Ice Cream Truck  
            - Trapped in a Coffin for 50 Hours  
            - I Paid 100 Strangers’ Rent  
            - We Built a House Underwater  

            Now generate 1 original, never-seen-before title like that. only title nothing else in plain normal text dont use bold text mind it normal plain text only title as output"""]

            # Load existing topics if file exists
            existing_topics = set()
            if os.path.exists(TOPIC_LOG_FILE):
                with open(TOPIC_LOG_FILE, 'r', encoding='utf-8') as file:
                    existing_topics = set(line.strip().lower() for line in file if line.strip())

            # Loop until a unique topic is generated
            while True:
                response = generate_gemini_response(prompt, model_name="gemini-2.5-flash-preview-05-20")
                new_topic = response.text.strip()

                if new_topic.lower() not in existing_topics:
                    with open(TOPIC_LOG_FILE, 'a', encoding='utf-8') as file:
                        file.write(new_topic + '\n')
                    return new_topic
                    
                else:
                    print("Duplicate topic found, regenerating...")

        # Example usage
        my_topic = generate_viral_wealth_topic()
        print("💡 Unique Viral Topic:", my_topic)
        script = generate_youtube_shorts_script(my_topic)
        print("\n📜 Generated Script:\n", script)
        output_dic=create_video_from_script_with_whisperx(script, my_topic)
        
        from pydub import AudioSegment
        import os
        import ffmpeg

        from pydub import AudioSegment
        import os
        import re

        folder_path = "audio"
        combined = AudioSegment.empty()

        # Sort files by the numeric index in the filename
        def extract_scene_number(filename):
            match = re.search(r"scene_(\d+)\.mp3", filename)
            return int(match.group(1)) if match else float('inf')

        mp3_files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith(".mp3")],
            key=extract_scene_number
        )

        for file_name in mp3_files:
            file_path = os.path.join(folder_path, file_name)

            if os.path.getsize(file_path) == 0:
                print(f"⚠️ Skipping empty file: {file_name}")
                continue

            try:
                audio = AudioSegment.from_mp3(file_path)
                combined += audio
            except Exception as e:
                print(f"❌ Could not decode {file_name}: {e}")
                continue

        # Export final merged MP3
        output_path1 = os.path.join(folder_path, "merged_audio.mp3")
        combined.export(output_path1, format="mp3")
        print(f"✅ Merged audio saved to: {output_path1}")



        # # === Step 1: Convert merged_audio.mp3 to WAV ===
        output_path1 = 'audio/merged_audio.mp3'
        output_wav = 'Wav2Lip/merged_audio.wav'
        mp3_to_wav(output_path1, output_wav)

        # from pydub import AudioSegment

        # def speed_up_wav(input_path, output_path, speed=1.5):
        #     sound = AudioSegment.from_wav(input_path)
        #     faster_sound = sound._spawn(sound.raw_data, overrides={
        #         "frame_rate": int(sound.frame_rate * speed)
        #     }).set_frame_rate(sound.frame_rate)
        #     faster_sound.export(output_path, format="wav")

        # # Example usage:
        # speed_up_wav(output_wav1, "faster.wav", speed=1.18)


        # # === Step 2: Run OpenVoice CLI with subprocess (not using !python) ===
        # input_wav = "faster1.wav"
        # output_wav = "/Users/uday/Downloads/VIDEOYT/Wav2Lip/merged_audio_final.wav"
        # openvoice_command = [
        #     "python3", "-m", "openvoice_cli", "single",
        #     "-i", output_wav1,
        #     "-r", input_wav,
        #     "-o", output_wav,
        #     "-d", "cpu"
        # ]

        # try:
        #     subprocess.run(openvoice_command, check=True)
        #     print(f"✅ OpenVoice processed: {output_wav}")
        # except subprocess.CalledProcessError as e:
        #     print("❌ OpenVoice CLI failed:")
        #     print(e)



        import os
        import subprocess

        # Define all paths

        temp_result = "temp/result.avi"
        output_final = "video_created/output_with_audio.mp4"
        output_wav="/Users/uday/Downloads/VIDEOYT/Wav2Lip/merged_audio.wav"
        checkpoint_path = "checkpoints/wav2lip.pth"
        face_video = "Wav2Lip/output_video.mp4"


        run_wav2lip_inference(
            checkpoint_path="checkpoints/wav2lip.pth",
            face_video="Wav2Lip/output_video (4).mp4",
            audio_path="/Users/uday/Downloads/VIDEOYT/Wav2Lip/merged_audio.wav",
            output_video="video_created/output_with_audio1.mp4",
            static=True,
            fps=24,
            wav2lip_batch_size=256,
            resize_factor=4,
            out_height=360
        )



        # Step 2: Merge audio and video using FFmpeg
        print("🎵 Merging final video with audio...")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", temp_result,
            "-i", output_wav,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "320k",         # Increased bitrate for better audio quality
            "-ar", "48000",         # Set sample rate to 48 kHz
            "-ac", "2",  
            "-shortest",
            output_final
        ])

        print(f"✅ Done! Final video with audio is saved to:\n{output_final}")

        import subprocess


        

        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"/Users/uday/Downloads/VIDEOYT/final_video/final_video_{timestamp}.mp4"




        import subprocess

        import subprocess

        command = [
            "ffmpeg", "-y",
            "-i", output_dic,        # Background video (with or without audio)
            "-i", output_final,      # Avatar video (with audio)
            "-filter_complex",
            """
            [1:v]scale=444:667[avatar];
            [0:v]scale=1080:1920[bg];
            [bg][avatar]overlay=main_w-overlay_w-10:main_h-overlay_h-10[outv];
            [0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2[aout]
            """,
            "-map", "[outv]",
            "-map", "[aout]",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            "-shortest",
            output_file
        ]






        print("▶️ Generating video with avatar in bottom-right corner...")
        subprocess.run(command)
        print(f"✅ Final video saved to: {output_file}")



        def save_file_topic_mappings(mappings, filename):
            """
            Save mappings of output file paths and topic names to a text file.
            
            Each line will be:
            output_file_path | topic_name
            
            :param mappings: List of tuples [(file_path1, topic1), (file_path2, topic2), ...]
            :param filename: Text file to save mappings
            """
            with open(filename, 'a') as f:
                for file_path, topic in mappings:
                    f.write(f"{file_path} | {topic}\n")

        # Example usage:
        mappings = [
            (output_file, my_topic)
            
        ]

        save_file_topic_mappings(mappings, "/Users/uday/Downloads/VIDEOYT/file_topic_map.txt")


        # # Call your function
        # video_path, used_urls = create_video_from_script(script, user_topic)

        # If the video was created successfully, append to the permanent URL log file
        # if output_file and used_urls:
        #     permanent_log_path = "all_video_used_urls.txt"  # Define your permanent log file

        #     with open(permanent_log_path, "a") as f:
        #         f.write(f"{os.path.basename(output_file)}\n")
        #         for url in sorted(used_urls):
        #             f.write(url + "\n")
        #         f.write("\n")  # Blank line between entries

        #     print(f"📝 Appended used URLs to: {permanent_log_path}")
        # else:
        #     print("❌ Could not create video or no URLs were used.")
