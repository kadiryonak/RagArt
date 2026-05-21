# RagArt — Turkish Retrieval-Augmented Generation platform.
#   docker build -t ragart .
#   docker run -p 5000:5000 ragart      →  http://localhost:5000
FROM python:3.11-slim

WORKDIR /app

# build-essential: a few transitive wheels still need a C compiler.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first — this layer is cached unless requirements change.
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
CMD ["ragart", "--no-browser", "--host", "0.0.0.0"]
