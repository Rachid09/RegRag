"""Paths shared by every module. Override the root with REGRAG_DATA_DIR."""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("REGRAG_DATA_DIR", "data")).resolve()
RAW_DIR = DATA_DIR / "raw"
DOCS_DIR = RAW_DIR / "documents"
CACHE_PATH = DATA_DIR / "cache" / "api_cache.sqlite"
