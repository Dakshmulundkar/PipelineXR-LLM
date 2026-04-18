FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV HF_HOME=/app/.cache

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    "huggingface_hub>=0.23.0" \
    pydantic

COPY app.py .

EXPOSE 7860

CMD ["sh", "-c", "CMAKE_ARGS='-DLLAMA_BLAS=OFF' MAKEFLAGS='-j1' pip install --no-cache-dir llama-cpp-python==0.3.20 && uvicorn app:app --host 0.0.0.0 --port 7860"]
