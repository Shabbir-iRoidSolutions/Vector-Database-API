# Vector Database API

A powerful API service for managing vector embeddings and document retrieval using various LLM providers.

## Prerequisites
- Docker (Windows: Docker Desktop; Linux/macOS: Docker Engine + docker compose)

## Quick Start (recommended)
The easiest way to run everything (API + Qdrant protected by API key) is with Docker Compose.

### One-time setup on Windows (PowerShell)
```powershell
# Run from the repository root (no need to cd into docker_container)
.\docker_container\start.ps1
```
- Enter a strong API key when prompted.
- The script will write a .env with the API key, ensure the external volume exists, and start all services.
- Access Qdrant via: http://localhost:6333 (the UI will ask for the API key).
- The API runs on port 2100.

### One-time setup on Linux/macOS (manual steps)
```bash
# Run all commands from the repository root

# 1) Create bind mount target for API data once (Linux servers)
sudo mkdir -p /var/www/VECTOR_DB && sudo chmod 777 /var/www/VECTOR_DB

# 2) Ensure the external named volume for Qdrant exists
docker volume create ai-assistant-qdrant

# 3) Provide Qdrant API key in the compose env file
echo "QDRANT_API_KEY=your-strong-key" > docker_container/.env

# 4) Start the stack using the compose file without cd-ing
docker compose -f docker_container/docker-compose.yml --env-file docker_container/.env up -d
```

### What gets started
- Qdrant (vector database) with data persisted in the external Docker volume `ai-assistant-qdrant`.
- The Vector Database API (port 2100) pulled from `ghcr.io/blue-elephants-solutions-pte-ltd/vector-db-api:latest`.

### Authentication
- Qdrant uses its native API key authentication. The UI and API require the header `api-key: <YOUR_QDRANT_API_KEY>`.

### Updating to a new version
- Pull the latest images and restart (run from repo root):
```bash
docker compose -f docker_container/docker-compose.yml --env-file docker_container/.env pull
docker compose -f docker_container/docker-compose.yml --env-file docker_container/.env up -d
```
Your data remains intact because it lives in the external volumes.

---

## Docker Setup (alternative: single container)

### Prerequisites
- Docker installed on your server
- Sudo access to create directories

### Important Note
> ⚠️ Please execute the commands exactly as shown below. Do not modify any parameters or flags as they are specifically configured for optimal performance.

### Installation Steps

1. Update system packages and create the required directory for vector storage:
```bash
# Update system packages
sudo apt update

# Create a directory to store vector data with proper permissions
# The -p flag will create all necessary parent directories including /var/www if they don't exist
sudo mkdir -p /var/www/VECTOR_DB
sudo chmod 777 /var/www/VECTOR_DB
```

2. Pull and run the Docker container (alternative to Compose):
```bash

# Download the latest version of the vector-db-api image
docker pull ghcr.io/blue-elephants-solutions-pte-ltd/vector-db-api:latest

# Run the container in detached mode, mapping port 2100 and mounting the vector storage directory
docker run -d -p 2100:2100 --name vector-db-api \
  -v /var/www/VECTOR_DB:/app/VECTOR_DB \
  ghcr.io/blue-elephants-solutions-pte-ltd/vector-db-api:latest

```

### Container Management

- View container logs:
```bash
# Monitor the container logs in real-time
docker logs -f vector-db-api
```

## Notes

- The API runs on port 2100 by default
- Vector data is persisted in the mounted volume at `/var/www/VECTOR_DB`
- Each user's vectors are stored in separate directories
