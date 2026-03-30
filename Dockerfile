FROM python:3.11-slim

# Required to compile llama-cpp-python from source
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Build llama-cpp-python for CPU
RUN CMAKE_ARGS="-DLLAMA_BLAS=OFF" pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
