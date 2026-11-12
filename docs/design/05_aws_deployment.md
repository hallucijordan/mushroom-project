# AWS Deployment Design

## Architecture on AWS

```
                          Internet
                             │
                    ┌────────▼────────┐
                    │   CloudFront    │ ← CDN for React app + cat assets
                    │   Distribution  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Application    │
                    │  Load Balancer  │ ← Routes /api/* to backend
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
        │  ECS Task │ │  ECS Task │ │  ECS Task │  ← Fargate containers
        │ (FastAPI) │ │ (FastAPI) │ │ (FastAPI) │     (auto-scaled)
        └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
        │    S3     │ │   RDS     │ │  Secrets  │
        │ (models,  │ │ (Postgres │ │  Manager  │
        │  assets)  │ │  sessions)│ │ (API keys)│
        └───────────┘ └───────────┘ └───────────┘
```

**For a course project**, we can simplify to:
- A single EC2 instance running Docker Compose (frontend + backend + SQLite)
- S3 for model artifacts
- No load balancer, no RDS

The full architecture above is the production-ready version.

---

## Docker Setup

### docker-compose.yml (Local Development)

```yaml
version: "3.9"

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    volumes:
      - ./frontend/src:/app/src  # Hot reload
    environment:
      - VITE_API_URL=http://localhost:8000

  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app/backend
      - ./ml:/app/ml
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - DATABASE_URL=sqlite:///./data/sessions.db
      - MODEL_DIR=/app/models
      - DATA_DIR=/app/data
      - ENV=development
    depends_on: []
```

### backend/Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install ML package first (changes less often → better layer caching)
COPY ml/ /app/ml/
RUN pip install /app/ml/

# Install backend dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install -r /app/backend/requirements.txt

# Copy backend code
COPY backend/ /app/backend/

# Copy data and models
COPY data/ /app/data/
COPY models/ /app/models/

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### frontend/Dockerfile

```dockerfile
# Build stage
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Serve stage
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 3000
```

### Production Multi-Stage Dockerfile (Single Container)

For the simplified single-EC2 deployment:

```dockerfile
# Stage 1: Build frontend
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
ARG VITE_API_URL=/api
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# Stage 2: Python backend + serve static files
FROM python:3.11-slim
WORKDIR /app

# Install ML package
COPY ml/ /app/ml/
RUN pip install /app/ml/

# Install backend deps
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install -r /app/backend/requirements.txt

# Copy backend
COPY backend/ /app/backend/
COPY data/ /app/data/
COPY models/ /app/models/

# Copy built frontend
COPY --from=frontend-build /app/frontend/dist /app/static

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

The FastAPI app serves the React build from `/app/static` as a fallback:

```python
# backend/app/main.py (relevant snippet)
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="/app/static", html=True), name="static")
```

---

## AWS Services Breakdown

### Simplified (Course Project — Recommended)

| Service | Purpose | Estimated Cost |
|---|---|---|
| **EC2 t3.medium** | Run Docker Compose (backend + frontend) | ~$30/mo |
| **S3 bucket** | Store model artifacts + cat assets | ~$1/mo |
| **Secrets Manager** | Store GEMINI_API_KEY | ~$0.40/mo |
| **Elastic IP** | Stable public IP | Free (if attached) |

Total: ~$31/month

### Full Production (Reference)

| Service | Purpose |
|---|---|
| ECS Fargate | Run backend containers (auto-scaled) |
| CloudFront | CDN for frontend + assets |
| S3 | Static assets + model artifacts |
| RDS PostgreSQL | Session persistence |
| ALB | Load balancer with /api routing |
| Secrets Manager | API keys |
| ECR | Docker image registry |
| CloudWatch | Logging + monitoring |

---

## Environment Configuration

### backend/app/config.py

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Core
    env: str = "development"        # development | staging | production
    debug: bool = True

    # API Keys
    gemini_api_key: str = ""

    # Database
    database_url: str = "sqlite:///./data/sessions.db"

    # ML
    model_dir: str = "./models"
    data_dir: str = "./data"

    # AWS (only used in production)
    aws_region: str = "us-east-1"
    s3_bucket: str = ""
    s3_model_prefix: str = "models/"

    # CORS
    allowed_origins: list[str] = ["http://localhost:3000"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

### .env.example

```bash
# Required
GEMINI_API_KEY=your-gemini-api-key-here

# Database (SQLite for dev, Postgres URL for prod)
DATABASE_URL=sqlite:///./data/sessions.db

# AWS (production only)
AWS_REGION=us-east-1
S3_BUCKET=mushroom-academy-models
S3_MODEL_PREFIX=models/

# App
ENV=development
DEBUG=true
ALLOWED_ORIGINS=http://localhost:3000
```

---

## Deployment Steps (Simplified — EC2)

### 1. Launch EC2 Instance

```bash
# Amazon Linux 2023 or Ubuntu 22.04, t3.medium
# Security group: open ports 80, 443, 22
```

### 2. Install Docker

```bash
sudo yum install docker -y      # Amazon Linux
sudo systemctl start docker
sudo usermod -aG docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
  -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 3. Clone & Deploy

```bash
git clone https://github.com/hallucijordan/mushroom-project.git
cd mushroom-project

# Set env vars
cp deploy/.env.example .env
nano .env  # Fill in GEMINI_API_KEY

# Build and run
docker-compose -f docker-compose.prod.yml up -d --build
```

### 4. Model Artifacts on S3 (Optional)

If models are large or updated separately:

```python
# backend/app/services/model_loader.py
import boto3
import os

def download_model_from_s3(bucket: str, key: str, local_path: str):
    """Download model artifact from S3 if not cached locally."""
    if os.path.exists(local_path):
        return
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
```

---

## CI/CD (GitHub Actions)

```yaml
# .github/workflows/deploy.yml
name: Build & Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ml/[dev]
      - run: pytest ml/tests/

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t mushroom-academy .
      - name: Push to ECR (or Docker Hub)
        # For course project: just SSH and pull on EC2
        run: echo "Push step — configure for your registry"

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - name: SSH and restart
        # For course project, a simple SSH + docker-compose pull + up
        run: echo "Deploy step — SSH into EC2 and restart"
```

---

## Model Update Workflow

When ML models are improved independently:

```
Developer updates ml/ package
    │
    ├── Run ml/tests/ locally
    │
    ├── retrain model: python -m mushroom_ml.train
    │   └── produces models/base_learner.joblib
    │
    ├── Upload to S3: aws s3 cp models/ s3://bucket/models/ --recursive
    │
    └── Backend auto-loads new model on next request
        (or trigger reload via /api/models/reload endpoint)
```

This decouples model development from web development completely.

---

## Security Checklist

- [ ] GEMINI_API_KEY stored in Secrets Manager (prod) or .env (dev), never in code
- [ ] .env in .gitignore
- [ ] CORS restricted to frontend origin
- [ ] No authentication needed (course project), but session IDs are UUIDs (unguessable)
- [ ] Docker runs as non-root user
- [ ] S3 bucket is private (no public access)
- [ ] EC2 security group: only 80/443/22 open
