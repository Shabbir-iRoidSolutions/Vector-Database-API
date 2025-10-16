# Use official Python image as a base
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install git and system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 appuser

# Copy local code into the container
COPY . .

# Set proper permissions
RUN chown -R appuser:appuser /app && \
    mkdir -p /app/VECTOR_DB && \
    chown -R appuser:appuser /app/VECTOR_DB && \
    chmod -R 777 /app/VECTOR_DB

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Qdrant binary
USER root
RUN mkdir -p /qdrant/storage && \
    chown -R appuser:appuser /qdrant && \
    curl -L -o /tmp/qdrant.zip https://github.com/qdrant/qdrant/releases/latest/download/qdrant-linux-x86_64.zip && \
    unzip /tmp/qdrant.zip -d /usr/local/bin && \
    rm /tmp/qdrant.zip && \
    chmod +x /usr/local/bin/qdrant

# Copy entrypoint that starts Qdrant and the API
COPY docker_container/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown appuser:appuser /entrypoint.sh

# Expose API and Qdrant ports
EXPOSE 2100 6333

# Switch to non-root user
USER appuser
ENTRYPOINT ["/entrypoint.sh"]
