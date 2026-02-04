from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()
print("RAG loaded")

# ---------- Qdrant ----------
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60
)

# ---------- Embedding ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Gemini ----------
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

COLLECTION_NAME = "ambedkar_rag"

# ---------- Retrieve ----------
def retrieve(query: str, top_k: int = 3):
    vector = embedder.encode(query).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[],               # REQUIRED in v1.16
        query=vector,              # VECTOR GOES HERE
        limit=top_k,
        with_payload=True
    )

    # v1.16 returns object with .points
    return [point.payload for point in results.points]

# ---------- Answer ----------
def answer_question(question: str) -> str:
    contexts = retrieve(question, top_k=3)

    if not contexts:
        return "No relevant context found in the Ambedkar corpus."

    context_text = "\n\n".join(
        f"Source: {c.get('source', 'Unknown')}\nText: {c.get('text', '')}"
        for c in contexts
    )

    prompt = f"""
You are a scholarly assistant answering questions using Dr. B. R. Ambedkar's writings.

Context:
{context_text}

Question:
{question}

Answer in a clear, concise, and academic tone.
If the answer is not found in the context, say so clearly.
"""

    response = gemini_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    return response.text.strip()
