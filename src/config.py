import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None

try:
    from qdrant_client import QdrantClient
except ImportError:  # pragma: no cover
    QdrantClient = None

# Locate the root directory and the .env file
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


def _load_environment() -> str:
    """Loads environment variables and retrieves the Gemini API Key."""
    if load_dotenv is not None:
        load_dotenv(dotenv_path=ENV_PATH, override=True)

    # First check standard developer key, fallback to standard google key
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def _build_ai_client(api_key: str):
    """Initializes the standard Gemini Google GenAI Client."""
    if not api_key or genai is None:
        return None
    return genai.Client(api_key=api_key)


def _build_qdrant_client():
    """Initializes an in-memory local Qdrant Client."""
    if QdrantClient is None:
        return None
    return QdrantClient(":memory:")


# Initialize configurations
api_key = _load_environment()

if not api_key:
    print("\n⚠️  [Config Warning]: No Gemini API key found in the environment or .env file.")
else:
    display_value = api_key[:8] + "..." if len(api_key) > 8 else api_key
    print(f"\n✅ [Config Success]: API Key loaded successfully (starts with: {display_value})")

ai = _build_ai_client(api_key)
qdrant_client = _build_qdrant_client()

# Constant configuration properties
COLLECTION_NAME = "personal_memories"
VECTOR_SIZE = 768  # Native dimension for text-embedding-004