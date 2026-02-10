FROM runpod/pytorch:1.0.3-cu1290-torch280-ubuntu2204

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/runpod-volume/models

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock .python-version ./
COPY server.py download_models.py test_client.py ./

# Install dependencies (uv creates a venv automatically)
RUN uv sync --frozen --no-dev

# Copy startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8000

CMD ["/start.sh"]
