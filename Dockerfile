FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install base deps
RUN pip install --no-cache-dir -r requirements.txt

# Install pre-built llama-cpp-python CPU wheel directly (no compilation)
RUN pip install --no-cache-dir \
    "llama-cpp-python==0.3.2" \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

COPY app.py .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
