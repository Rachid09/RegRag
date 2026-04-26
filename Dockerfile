# RegRAG chat app: one container, BM25 retrieval in memory, answers from Gemini via LiteLLM.
#   docker build -t regrag .
#   docker run -p 7860:7860 -e GEMINI_API_KEY=... regrag      # http://localhost:7860
# Port 7860 is the Hugging Face Spaces default. Qdrant is not needed: the app uses BM25.

FROM ghcr.io/astral-sh/uv:0.13.8 AS uv

# ---- build: install locked dependencies and fetch the corpus (pinned + SHA-256 checked)
FROM python:3.12-slim-bookworm AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
ENV REGRAG_DATA_DIR=/app/data
RUN .venv/bin/python -m regrag.download && rm -f data/raw/ObliQA_train.json

# ---- runtime: no uv, no build tools, non-root user
FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build --chown=app:app /app/.venv /app/.venv
COPY --from=build --chown=app:app /app/src /app/src
COPY --from=build --chown=app:app /app/data /app/data
USER app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    REGRAG_DATA_DIR=/app/data \
    LITELLM_LOCAL_MODEL_COST_MAP=True \
    PORT=7860
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)"
CMD ["sh", "-c", "uvicorn regrag.api:create_app --factory --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
