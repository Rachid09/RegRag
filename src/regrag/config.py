"""Paths and settings shared by every module. Everything can be overridden from the env/.env."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.environ.get("REGRAG_DATA_DIR", "data")).resolve()
RAW_DIR = DATA_DIR / "raw"
DOCS_DIR = RAW_DIR / "documents"
CACHE_PATH = DATA_DIR / "cache" / "api_cache.sqlite"

# Embeddings (hosted, via LiteLLM). The rest of the code only sees these names.
EMBED_MODEL = os.environ.get("REGRAG_EMBED_MODEL", "gemini/gemini-embedding-2")
EMBED_DIMS = int(os.environ.get("REGRAG_EMBED_DIMS", "3072"))
EMBED_BATCH = int(os.environ.get("REGRAG_EMBED_BATCH", "50"))  # texts per request
EMBED_RPM = int(os.environ.get("REGRAG_EMBED_RPM", "60"))  # client-side request pacing

# Vector store (Qdrant in Docker, see docker-compose.yml)
QDRANT_URL = os.environ.get("REGRAG_QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("REGRAG_COLLECTION", "obliqa_dense")
