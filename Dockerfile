
FROM python:3.11-slim AS builder

WORKDIR /build


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt



FROM python:3.11-slim AS runtime


LABEL org.opencontainers.image.title="Adaptive AI Project Manager"
LABEL org.opencontainers.image.description="OpenEnv-compliant RL environment for Agile sprint simulation"
LABEL org.opencontainers.image.version="2.1.0"

RUN useradd -m -u 1000 envuser

WORKDIR /app


COPY --from=builder /install /usr/local


COPY . . 


USER envuser


EXPOSE 7860


HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1


CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 7860"]
