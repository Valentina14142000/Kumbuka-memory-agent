import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

CHAT_MODELS = [
    m.strip()
    for m in os.getenv("GEMINI_MODELS", "gemini-3.8-flash,gemini-3.5-flash").split(",")
    if m.strip()
]
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "768"))

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")  # optional: use a Qdrant server instead of local files
QDRANT_PATH = os.getenv("QDRANT_PATH", str(BASE_DIR / "qdrant_data"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "developer_memory")

DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", "0.92"))
RECALL_THRESHOLD = float(os.getenv("RECALL_THRESHOLD", "0.55"))
HISTORY_WINDOW = int(os.getenv("HISTORY_WINDOW", "12"))

_client = None


def api_key() -> str:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def get_client():
    global _client
    if _client is None:
        key = api_key()
        if not key:
            raise RuntimeError("No Gemini API key found. Put GEMINI_API_KEY=... in the .env file.")
        _client = genai.Client(api_key=key)
    return _client
