from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from rag import answer_question
import uuid
import os
import requests

print("API loaded")
load_dotenv()

print("ELEVEN KEY VALUE:", os.getenv("ELEVENLABS_API_KEY"))
print("ELEVEN KEY LENGTH:", len(os.getenv("ELEVENLABS_API_KEY") or ""))


# ===================== APP =====================
app = FastAPI(
    title="Dr. Ambedkar RAG API",
    description="RAG-based QA system powered by Qdrant + Gemini + ElevenLabs",
    version="1.0"
)

# Health check (REQUIRED for Render)
@app.get("/")
def health():
    return {"status": "API running"}

# ===================== CORS =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-friendly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== AUDIO =====================
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # default ElevenLabs voice

def elevenlabs_tts(text: str) -> str:
    filename = f"{uuid.uuid4()}.wav"
    audio_path = os.path.join(AUDIO_DIR, filename)

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
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

# ===================== API =====================
class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    answer = answer_question(query.question)

    audio_filename = elevenlabs_tts(answer)

    return {
        "question": query.question,
        "answer": answer,
        "audio_url": f"http://127.0.0.1:8000/audio/{audio_filename}"
    }
