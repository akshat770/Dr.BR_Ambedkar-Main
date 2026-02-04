from qdrant_client import QdrantClient
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

COLLECTION_NAME = "ambedkar_rag"

# ---------- Lazy Embedder ----------
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(1)
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

# ---------- Gemini ----------
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ---------- Retrieve ----------
def retrieve(query: str, top_k: int = 3):
    embedder = get_embedder()
    vector = embedder.encode(query).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[],          # REQUIRED in qdrant-client >=1.16
        query=vector,
        limit=top_k,
        with_payload=True
    )

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
