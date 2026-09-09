FROM python:3.10-slim

# Tesseract is a system binary, not a pip package -- pytesseract just calls
# out to it. This is the one piece a plain `pip install` deployment target
# (Render/Railway native buildpacks, etc.) can't give you, which is the
# whole reason this is a Docker image and not just a requirements.txt.
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so `docker build` caches the pip install layer
# and doesn't re-download everything on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runs as non-root -- no reason this process needs root inside the container.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# .env is deliberately NOT baked into the image (.dockerignore excludes it,
# same reason it's gitignored) -- pass LLM_PROVIDER/API keys/BASIC_AUTH_* at
# `docker run -e ...` / --env-file, or via the platform's env var config.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
