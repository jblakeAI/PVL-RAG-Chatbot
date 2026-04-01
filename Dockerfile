# Use official Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (Docker caches this layer separately,
# so rebuilds are faster if only your code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Pre-download the embedding model and cross-encoder model into the image
# This makes cold starts faster - the model won't need to download at runtime
# This way Cloud Run never needs to contact HuggingFace
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2'); \
print('Both models downloaded successfully')"


# Copy all project files into the container
COPY . .

# Cloud Run expects port 8080
EXPOSE 8080

# Start the app
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]






