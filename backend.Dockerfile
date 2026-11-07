# Build from the ResearchIQ project root:
# docker build -f backend.Dockerfile -t researchiq-backend ./
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    RESEARCHIQ_DATA_DIR=/app/data

WORKDIR /app

# Runtime libraries needed by common scientific / PDF dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN python -c "from pathlib import Path; p=Path('requirements.txt'); p.write_text(p.read_text(encoding='utf-16'), encoding='utf-8')" \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

COPY backend/app ./app
COPY backend/eval ./eval
COPY backend/alembic.ini ./alembic.ini
COPY backend/migrations ./migrations

RUN mkdir -p /app/data/uploads /app/eval/results \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
