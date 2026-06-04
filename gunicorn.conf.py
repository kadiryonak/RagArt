"""Gunicorn config for production deploys (Linux / Docker / PaaS).

    gunicorn -c gunicorn.conf.py app:app

Why these settings:

- worker_class = "gevent": RagArt streams answers over Server-Sent Events
  (/ask/stream). Sync workers would block a whole worker per open stream;
  gevent multiplexes many SSE connections on one worker via greenlets, and
  it monkeypatches the socket so the blocking `requests` calls to the LLM
  provider cooperate instead of stalling the loop.

- workers = 1 (override with WEB_CONCURRENCY): the embedding model (~470 MB)
  and each workspace's ChromaDB client are per-process. One gevent worker
  keeps memory sane on a single small instance while still serving many
  concurrent requests. Scale out with more instances behind a load balancer
  rather than many workers on one box.

- timeout = 0: never kill a worker mid-stream. A long SSE response or a slow
  first-query model download must not trip gunicorn's watchdog.

- bind reads $PORT: Render / Railway / Fly / Heroku inject the port.
"""

from __future__ import annotations

import multiprocessing  # noqa: F401  (kept for easy CPU-based scaling tweaks)
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
worker_class = "gevent"
worker_connections = int(os.environ.get("WORKER_CONNECTIONS", "100"))

# SSE-friendly: no request timeout, generous keep-alive.
timeout = 0
graceful_timeout = 30
keepalive = 75

# Logging to stdout/stderr so the platform's log drain picks it up.
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")

# Preloading would share the model across workers BUT breaks the per-process
# warm-start thread and ChromaDB clients; keep it off so each worker boots
# its own runtime cleanly.
preload_app = False
