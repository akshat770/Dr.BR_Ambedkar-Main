import requests
import os
import uuid
from dotenv import load_dotenv

load_dotenv()  # ✅ REQUIRED

ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # default voice

if not ELEVEN_API_KEY:
    raise RuntimeError("ELEVENLABS_API_KEY not found in environment")

AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

def elevenlabs_tts(text: str) -> str:
    filename = f"{uuid.uuid4()}.wav"
    audio_path = os.path.join(AUDIO_DIR, filename)

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    with open(audio_path, "wb") as f:
        f.write(response.content)

    return filename
