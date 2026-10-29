FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies using pre-built CUDA wheels for llama-cpp-python
COPY requirements.txt /app/
RUN pip install --no-cache-dir --break-system-packages -r requirements-linux.txt

COPY . /app
RUN chmod +x /app/run_linux.sh

EXPOSE 8000

ENTRYPOINT ["/app/run_linux.sh"]
