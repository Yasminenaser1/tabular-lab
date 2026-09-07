FROM python:3.13-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api.py .
COPY models/ models/
COPY static/ static/

CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}
