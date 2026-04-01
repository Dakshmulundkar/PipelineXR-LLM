FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install base deps first
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" huggingface_hub pydantic

# Compile llama-cpp-python with limited parallelism to avoid OOM during build
RUN CMAKE_ARGS="-DLLAMA_BLAS=OFF" MAKEFLAGS="-j1" pip install --no-cache-dir llama-cpp-python

COPY app.py .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
