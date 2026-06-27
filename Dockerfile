# Pure Python runtime — no C++ build needed
FROM python:3.11-slim

WORKDIR /app

# Copy Python application files
COPY requirements.txt .
COPY main.py .
COPY alien_inference.py .
COPY vocab.txt .
COPY model_checkpoint.bin .
COPY frontend/ ./frontend/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
