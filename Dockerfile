FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install llama-cpp-python at BUILD time — eliminates the 4-min compile on every cold start.
# MAKEFLAGS=-j$(nproc) uses all available build cores.
RUN CMAKE_ARGS="-DLLAMA_BLAS=OFF" MAKEFLAGS="-j$(nproc)" \
    pip install --no-cache-dir llama-cpp-python==0.3.20

# Install remaining dependencies
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    "huggingface_hub>=0.23.0" \
    pydantic

COPY app.py .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
