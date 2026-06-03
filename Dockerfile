# RagArt — Turkish Retrieval-Augmented Generation platform.
#   docker build -t ragart .
#   docker run -p 5000:5000 ragart      →  http://localhost:5000
FROM python:3.11-slim

WORKDIR /app

# build-essential: a few transitive wheels still need a C compiler.
# curl: used by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first — this layer is cached unless requirements change.
# requirements.txt already includes gunicorn + gevent for production.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code, then register the `ragart` console command.
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 5000
ENV HOST=0.0.0.0 \
    PORT=5000

# The embedding model (~470 MB) downloads on the first query. Mount a
# volume at /root/.cache/huggingface (see docker-compose.yml) so it is
# downloaded once, not on every container start.

# Liveness/readiness probe — the platform restarts the container if /health
# stops answering. start-period is generous: the first boot may download the
# embedding model.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# Production WSGI server (gunicorn + gevent) — SSE-friendly, no dev-server
# warning. Config lives in gunicorn.conf.py (reads $PORT, $WEB_CONCURRENCY).
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
