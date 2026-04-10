import os
import json
import time
import random
from datetime import datetime
from collections import defaultdict
import google.generativeai as genai


# -------------------------
# Gemini response container
# -------------------------
class GeminiResponse:
    def __init__(self, text):
        self.text = text


# -------------------------
# File names & limits
# -------------------------
FAILED_KEYS_FILE = "disabled_keys.json"
USAGE_FILE = "usage_counts.json"

DAILY_LIMIT = 500
PER_MINUTE_LIMIT = 10

minute_usage_tracker = defaultdict(list)


# -------------------------
# JSON Helpers
# -------------------------
def load_json_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}


def save_json_file(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


# -------------------------
# Disabled keys
# -------------------------
def load_disabled_keys():
    today = datetime.now().strftime("%Y-%m-%d")
    data = load_json_file(FAILED_KEYS_FILE)
    return set(data.get(today, []))


def save_disabled_key(api_key):
    today = datetime.now().strftime("%Y-%m-%d")
    data = load_json_file(FAILED_KEYS_FILE)

    if today not in data:
        data[today] = []

    if api_key not in data[today]:
        data[today].append(api_key)

    save_json_file(FAILED_KEYS_FILE, data)


# -------------------------
# Usage tracking
# -------------------------
def increment_usage(api_key):
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")

    # Daily usage
    usage = load_json_file(USAGE_FILE)
    if today not in usage:
        usage[today] = {}

    if api_key not in usage[today]:
        usage[today][api_key] = 0

    usage[today][api_key] += 1
    save_json_file(USAGE_FILE, usage)

    # Per-minute tracking
    minute = now.strftime("%Y-%m-%d %H:%M")
    minute_usage_tracker[api_key] = [
        ts for ts in minute_usage_tracker[api_key] if ts.startswith(minute)
    ]
    minute_usage_tracker[api_key].append(now.strftime("%Y-%m-%d %H:%M:%S"))


def has_exceeded_daily_limit(api_key, limit=DAILY_LIMIT):
    today = datetime.now().strftime("%Y-%m-%d")
    usage = load_json_file(USAGE_FILE)
    return usage.get(today, {}).get(api_key, 0) >= limit


def has_exceeded_minute_limit(api_key, limit=PER_MINUTE_LIMIT):
    current_minute = datetime.now().strftime("%Y-%m-%d %H:%M")
    recent_times = [
        ts for ts in minute_usage_tracker[api_key] if ts.startswith(current_minute)
    ]
    return len(recent_times) >= limit


# -------------------------
# Main function Gemini API
# -------------------------
def generate_gemini_response(prompt, model_name=None, max_retries=5, wait_seconds=5):
    api_keys = [
       "AIzaSyBji4lqM1r7gfzy8_uJKaV2EsNegNdNMSA"
    ]

    model_names = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite"
    ]

    disabled_keys_today = load_disabled_keys()

    for attempt in range(max_retries):

        available_keys = [
            k for k in api_keys
            if k not in disabled_keys_today
            and not has_exceeded_daily_limit(k)
            and not has_exceeded_minute_limit(k)
        ]

        if not available_keys:
            raise RuntimeError("❌ All API keys exhausted / disabled.")

        key = random.choice(available_keys)
        model = model_name or random.choice(model_names)

        try:
            genai.configure(api_key=key)

            print(f"✅ Using model: {model}, key ending with {key[-6:]}")

            gemini = genai.GenerativeModel(model)
            response = gemini.generate_content(prompt)

            increment_usage(key)

            return GeminiResponse(response.text.strip())

        except Exception as e:
            print(f"❌ Failed with key {key[-6:]} | Attempt {attempt + 1} | Error: {e}")
            save_disabled_key(key)
            time.sleep(wait_seconds)

    raise RuntimeError("❌ All attempts failed.")
