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
EMBED_BATCH = int(os.environ.get("REGRAG_EMBED_BATCH", "25"))  # texts per request
EMBED_RPM = int(os.environ.get("REGRAG_EMBED_RPM", "3"))  # 75 texts/min < free tier 100

# Vector store (Qdrant in Docker, see docker-compose.yml)
QDRANT_URL = os.environ.get("REGRAG_QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("REGRAG_COLLECTION", "obliqa_dense")

# Local embedding models (Ollama in Docker, see docker-compose.yml). Free, no quota, no cost,
# run entirely on this machine; used to compare against the hosted Gemini model above. Each
# gets its own Qdrant collection (name = key) since dims differ and vectors are not comparable
# across models.
OLLAMA_URL = os.environ.get("REGRAG_OLLAMA_URL", "http://localhost:11434")
# nomic-embed-text must be given a task prefix (model card: "search_document: " for passages,
# "search_query: " for questions). Only models with a verified prefix are listed.
LOCAL_MODELS = {
    "nomic-embed-text": {
        "model": "ollama/nomic-embed-text",
        "dims": 768,
        "doc_prefix": "search_document: ",
        "query_prefix": "search_query: ",
    },
}

# Chat API: which retriever answers questions. "bm25" needs nothing but the corpus; "hybrid"
# also needs the local model's Qdrant collection to be built and Ollama to be running.
API_RETRIEVER = os.environ.get("REGRAG_API_RETRIEVER", "bm25")
API_LOCAL_MODEL = os.environ.get("REGRAG_API_LOCAL_MODEL", "nomic-embed-text")

# Answer generation (hosted Gemini via LiteLLM). "cheap" answers; "strong" is the judge in
# phase 5 and can also answer. No temperature: Gemini 3.6+ ignores it (decision 19).
GEN_MODELS = {
    "cheap": os.environ.get("REGRAG_GEN_MODEL_CHEAP", "gemini/gemini-3.5-flash-lite"),
    "strong": os.environ.get("REGRAG_GEN_MODEL_STRONG", "gemini/gemini-3.8-flash"),
}
GEN_MODEL = os.environ.get("REGRAG_GEN_MODEL", "cheap")
CONTEXT_PASSAGES = 5  # retrieved passages shown to the model and cited as [1]..[5]
# Refuse without calling the model when the best BM25 score is below this. Clearly off-topic
# questions score under ~10, real ones over ~15 (dev sample); tuned in phase 5.
MIN_RETRIEVAL_SCORE = float(os.environ.get("REGRAG_MIN_RETRIEVAL_SCORE", "10"))
API_RATE_LIMIT = int(os.environ.get("REGRAG_API_RATE_LIMIT", "10"))  # questions/minute per client
