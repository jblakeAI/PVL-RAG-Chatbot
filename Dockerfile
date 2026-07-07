# Stage 1: buider with complier tools that get thrown out later 
# Use official Python image
FROM python:3.11-slim AS builder

# Set working directory inside the container
WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (Docker caches this layer separately,
# so rebuilds are faster if only your code changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: final image without compiler tools
FROM python:3.11-slim

WORKDIR /app

# Copy only the installed Python packages — no compiler, no build junk
COPY --from=builder /install /usr/local


# # Model cache dir — MUST be /tmp, it's the only guaranteed-writable
# location in Cloud Run's filesystem at runtime
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/hf_cache



# Copy all project files into the container
COPY . .

# Cloud Run expects port 8080
EXPOSE 8080

# Start the app
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]






