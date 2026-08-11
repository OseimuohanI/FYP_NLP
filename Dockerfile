# Dockerfile — replaces FYP_NLP/Dockerfile
#
# Change from the original:
# Added HF_HOME pointing to a fixed path inside the container. On its own
# this doesn't persist anything (containers are still ephemeral by
# default) — the point is that this path is now stable and mountable.
# When you run this on your home server, mount a volume at that path so
# the ~1GB model download only happens once instead of on every
# rebuild/recreate:
#
#   docker run -v nlp_hf_cache:/app/.cache/huggingface ...
#
# or the equivalent volumes: entry in Portainer / docker-compose.

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HF_HOME=/app/.cache/huggingface

COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
