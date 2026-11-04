"""Settings, read from environment variables (and a .env file next to the repo root)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Where uploaded PDFs, extracted pages, chunks and vectors live.
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", str(ROOT / "data")))

# "azure" = Azure OpenAI / AI Foundry embeddings.
# "fake"  = offline hashing embedder, only for tests and demos without keys.
EMBEDDING_BACKEND = os.getenv("RAG_EMBEDDING_BACKEND", "azure").lower()

DEFAULT_CHUNK_SIZE = 400  # tokens; chosen by the evaluation in docs/STATUS.md
MIN_CHUNK_SIZE, MAX_CHUNK_SIZE = 50, 2000
MAX_UPLOAD_MB = int(os.getenv("RAG_MAX_UPLOAD_MB", "100"))

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
