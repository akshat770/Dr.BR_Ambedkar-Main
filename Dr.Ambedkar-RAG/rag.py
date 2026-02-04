from qdrant_client import QdrantClient
from google import genai
from dotenv import load_dotenv
import os
import requests

load_dotenv()
print("RAG loaded")

# ---------- Qdrant ----------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=10
)

COLLECTION_NAME = "ambedkar_rag"

# ---------- Gemini ----------
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------- Retrieve (NO ML, NO TORCH) ----------
def retrieve(query: str, top_k: int = 3):
    url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search"

    payload = {
        "vector": {
            "name": "vector",
            "text": query
        },
        "limit": top_k,
        "with_payload": True
    }

    headers = {
        "api-key": QDRANT_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers, timeout=10)
    response.raise_for_status()

    data = response.json()
    return [point["payload"] for point in data.get("result", [])]

# ---------- Answer ----------
def answer_question(question: str) -> str:
    contexts = retrieve(question, top_k=3)

    if not contexts:
        return "No relevant context found in the Ambedkar corpus."

    context_text = "\n\n".join(
        c.get("text", "") for c in contexts
    )

    prompt = f"""
Answer the question using only the context below.

Context:
{context_text}

Question:
{question}
"""

    response = gemini_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        request_options={"timeout": 10}
    )

    return response.text.strip()
