# Installation Guide

This document describes how to install and deploy the AI Model Platform.

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 16 GB | 32+ GB |
| Storage | 100 GB SSD | 500+ GB SSD |
| GPU | NVIDIA with 8GB VRAM | NVIDIA with 16+ GB VRAM |

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Docker Engine | 24.0+ | Container runtime |
| Docker Compose | V2 (2.20+) | Service orchestration |
| NVIDIA Container Toolkit | Latest | GPU passthrough |
| NVIDIA Driver | 525+ | GPU support |
| Git | 2.30+ | Source code |

### Supported Platforms

- Ubuntu 20.04 / 22.04 LTS
- CentOS 7/8
- RHEL 8/9
- Windows (via Docker Desktop with WSL2)

## Installation Steps

### 1. Install Docker Engine

**Ubuntu/Debian:**

```bash
# Update package index
sudo apt-get update

# Install prerequisites
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker GPG key
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

**CentOS/RHEL:**

```bash
# Install prerequisites
sudo yum install -y yum-utils

# Add Docker repository
sudo yum-config-manager --add-repo \
    https://download.docker.com/linux/centos/docker-ce.repo

# Install Docker
sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### 2. Install NVIDIA Container Toolkit

```bash
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3. Verify GPU Access

```bash
# Check host GPU
nvidia-smi

# Check container GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### 4. Clone and Configure

```bash
# Clone repository
git clone https://github.com/your-org/ai-model-platform.git
cd ai-model-platform

# Create configuration from template
cp infra/.env.example infra/.env

# Edit configuration
nano infra/.env
```

### 5. Create Data Directories

```bash
cd infra
mkdir -p models datasets logs checkpoints postgres redis
```

### 6. Start Services

```bash
# Start infrastructure services first
docker compose up -d postgres redis

# Wait for health checks
docker compose ps

# Run database migrations
docker compose run --rm migrate

# Start all services
docker compose up -d
```

### 7. Verify Installation

```bash
# Check all services are running
docker compose ps

# Check API health
curl http://localhost:8000/health

# Check frontend
curl -I http://localhost:8080
```

## Dependencies

### Python Dependencies

The backend uses the following key dependencies:

- FastAPI - Web framework
- SQLAlchemy - ORM
- Alembic - Database migrations
- Celery - Task queue
- Redis - Message broker
- Ultralytics - YOLO training/inference
- PyTorch - Deep learning framework

### System Dependencies

- PostgreSQL 16 - Metadata storage
- Redis 7 - Task queue and caching
- Nginx - Frontend reverse proxy

## First Startup

### 1. Import a Root Model

```bash
# Copy a YOLO model to the models directory
cp /path/to/yolov8n.pt ./models/

# Import via API
curl -X POST http://localhost:8000/api/models \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "task_type": "object_detection",
    "model_family": "yolo",
    "artifact_path": "/data/models/yolov8n.pt",
    "artifact_hash": "sha256:..."
  }'
```

### 2. Import a Dataset

```bash
# Place YOLO-format dataset
mkdir -p datasets/my-dataset/{images/{train,val,test},labels/{train,val,test}}

# Create data.yaml
cat > datasets/my-dataset/data.yaml << EOF
train: ./images/train
val: ./images/val
test: ./images/test
nc: 1
names: ['fire']
EOF
```

### 3. Test Inference

```bash
curl -X POST http://localhost:8000/api/infer \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "modelBindingId": "mb-fire-001",
    "input": {
      "type": "image",
      "url": "https://example.com/test.jpg"
    }
  }'
```

## Post-Installation

1. Review [Configuration Guide](configuration.md) for production settings
2. Review [Operations Guide](operations.md) for maintenance procedures
3. Set up monitoring and alerting
4. Configure backup strategy

## Next Steps

- [Configuration Guide](configuration.md)
- [Troubleshooting Guide](troubleshooting.md)
- [Operations Guide](operations.md)
