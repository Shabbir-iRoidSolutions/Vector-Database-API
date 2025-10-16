# Vector Database API

A powerful API service for managing vector embeddings and document retrieval using various LLM providers.

## Prerequisites
- Docker (Windows: Docker Desktop; Linux/macOS: Docker Engine + docker compose)

## Quick Start (single container)
Now you can run Qdrant and the API in a single container.

### Run pulled image
```bash
# Create persistent volume for Qdrant storage (once)
docker volume create ai-assistant-qdrant

# Run the single container
docker run -d --name vector-db-api \
  -e QDRANT_API_KEY=your-strong-key \
  -e QDRANT_URL=http://localhost:6333 \
  -p 2100:2100 -p 6333:6333 \
  -v ai-assistant-qdrant:/qdrant/storage \
  ghcr.io/shabbir-iroidsolutions/vector-db-api:latest

# Verify
curl -H "api-key: your-strong-key" http://localhost:6333/healthz
curl http://localhost:2100/
```

### Build locally (if you need to build the image)
```bash
docker build -t vector-db-api:local -f docker_container/Dockerfile .
docker run -d --name vector-db-api \
  -e QDRANT_API_KEY=your-strong-key \
  -e QDRANT_URL=http://localhost:6333 \
  -p 2100:2100 -p 6333:6333 \
  -v ai-assistant-qdrant:/qdrant/storage \
  vector-db-api:local
```

### What gets started
- Qdrant (vector database) with API key authentication, data persisted in the volume `ai-assistant-qdrant`.
- The Vector Database API (port 2100) in the same container.

### Authentication
- Qdrant uses its native API key authentication. The UI and API require the header `api-key: <YOUR_QDRANT_API_KEY>`.

### Updating to a new version
```bash
docker pull ghcr.io/shabbir-iroidsolutions/vector-db-api:latest
docker rm -f vector-db-api || true
docker run -d --name vector-db-api \
  -e QDRANT_API_KEY=your-strong-key \
  -e QDRANT_URL=http://localhost:6333 \
  -p 2100:2100 -p 6333:6333 \
  -v ai-assistant-qdrant:/qdrant/storage \
  ghcr.io/shabbir-iroidsolutions/vector-db-api:latest
```
Your data remains intact because it lives in the external volume.

---

## Docker Setup (compose alternative)

### Prerequisites
- Docker installed on your server
- Sudo access to create directories

### Important Note
> ⚠️ Please execute the commands exactly as shown below. Do not modify any parameters or flags as they are specifically configured for optimal performance.

### Container Management

- View container logs:
```bash
# Monitor the container logs in real-time
docker logs -f vector-db-api
```

## Notes

- The API runs on port 2100 by default

---

## Environment Template

An example env file is provided at `docker_container/env.template`. Copy it to `.env` if you want to pre-provision the key without prompts:

```bash
cp docker_container/env.template docker_container/.env
```
