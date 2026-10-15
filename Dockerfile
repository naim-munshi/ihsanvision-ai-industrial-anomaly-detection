# Local CPU-only research prototype. Build/run validation remains pending.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/tmp/.cache \
    HOME=/tmp \
    TMPDIR=/tmp \
    PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir -e /app \
    && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin ihsan

# Runtime cannot write to the project/image; Anomalib's temporary results use /tmp.
USER 10001:10001
WORKDIR /tmp
EXPOSE 8080

# Configuration probe, NOT a full checkpoint deserialization/inference readiness test.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import json,urllib.request; h=json.load(urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=3)); assert h.get('model_configured')" || exit 1

CMD ["python","-m","uvicorn","ihsanvision.api:app","--host","0.0.0.0","--port","8080","--workers","1"]
