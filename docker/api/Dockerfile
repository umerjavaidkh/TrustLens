FROM python:3.11-slim

# System libs Pillow/scipy occasionally need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[api]"

EXPOSE 8000
ENV API_HOST=0.0.0.0 API_PORT=8000

CMD ["uvicorn", "trustlens.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
