FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install pre-built llama-cpp-python wheel (no compilation, no OOM)
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
