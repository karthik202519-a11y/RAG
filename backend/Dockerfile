# API and Celery worker image. Build context is the repository root:  docker build -f backend/Dockerfile .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/backend STUDIO_DATA_DIR=/data
WORKDIR /app/backend

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend /app/backend

# run unprivileged
RUN useradd --create-home --uid 10001 studio && mkdir -p /data && chown -R studio /data /app
USER studio
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "studio.main:app", "--host", "0.0.0.0", "--port", "8000"]
