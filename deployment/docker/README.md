# BMI Chat Docker Setup

This directory contains Docker configuration files for running the BMI Chat application in containers. This setup is ideal for testing the application before deploying to a VPS.

## Quick Start

### Prerequisites

- **Docker Desktop** installed and running
  - Download from: https://www.docker.com/products/docker-desktop
  - Ensure Docker Desktop is started and fully initialized
- **OpenAI API key** (GPT-4o recommended)

### Windows Quick Start

1. **Test Docker Setup** (Recommended first step):
   ```bash
   cd deployment/docker
   docker-test.bat
   ```

2. **Setup Environment**:
   ```bash
   copy .env.docker .env
   # Edit .env file and add your OpenAI API key
   ```

3. **Start Application**:
   ```bash
   docker-start.bat
   ```

4. **Access Application**:
   - Frontend: http://localhost:3003
   - Backend API: http://localhost:3006
   - API Documentation: http://localhost:3006/docs

5. **Stop Application**:
   ```bash
   docker-stop.bat
   ```

### Manual Commands

```bash
# Build and start services
docker-compose up --build -d

# Start with nginx reverse proxy (production-like)
docker-compose --profile production up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Stop and remove volumes (CAUTION: deletes data)
docker-compose down -v
```

## Architecture

### Services

1. **Backend** (`bmi-chat-backend`)
   - FastAPI application
   - Port: 3006
   - Health check: `/health`
   - Volumes: `backend_data`, `backend_logs`

2. **Frontend** (`bmi-chat-frontend`)
   - React SPA with Nginx
   - Port: 3003
   - Health check: `/health`

3. **Nginx Proxy** (`bmi-chat-nginx`) - Optional
   - Reverse proxy for production-like setup
   - Port: 80
   - Routes API calls to backend, everything else to frontend

### Volumes

- `backend_data`: Persistent storage for database, uploads, and vectors
- `backend_logs`: Application logs

### Network

- `bmi-chat-network`: Bridge network for inter-service communication

## Configuration

### Environment Variables

Copy `.env.docker` to `.env` and configure:

```env
# Required
OPENAI_API_KEY=your-api-key-here
SECRET_KEY=your-secret-key

# Optional
OPENAI_MODEL=gpt-4o
LOG_LEVEL=info
```

### Development Mode

To run in development mode with debug logging:

```env
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=debug
```

## File Structure

```
deployment/docker/
├── Dockerfile.backend          # Backend container definition
├── Dockerfile.frontend         # Frontend container definition
├── docker-compose.yml          # Service orchestration
├── nginx.conf                  # Frontend nginx config
├── nginx-proxy.conf            # Reverse proxy config
├── .env.docker                 # Environment template
├── docker-start.bat            # Windows start script
├── docker-stop.bat             # Windows stop script
├── docker-logs.bat             # Windows logs script
└── README.md                   # This file
```

## Troubleshooting

### Common Issues

1. **Docker not installed**:
   - Download and install Docker Desktop from https://www.docker.com/products/docker-desktop
   - Restart your computer after installation

2. **Docker not running**:
   - Start Docker Desktop and wait for it to fully initialize
   - Look for the Docker whale icon in your system tray
   - Run `docker info` to verify Docker is running

3. **Port conflicts**:
   - Ensure ports 3003, 3006, and 80 are available
   - Stop other applications using these ports
   - Use `netstat -an | findstr :3003` to check port usage

4. **Build failures**:
   - Check Docker logs with `docker-compose logs`
   - Ensure all source files are present
   - Try `docker system prune` to clean up Docker cache
   - Rebuild with `docker-compose build --no-cache`

5. **API key missing**:
   - Verify OPENAI_API_KEY in .env file
   - Ensure the key starts with "sk-proj-" or "sk-"
   - Check for extra spaces or quotes around the key

6. **Memory issues**:
   - Increase Docker Desktop memory allocation (Settings > Resources > Memory)
   - Recommended: At least 4GB RAM for Docker

7. **Windows-specific issues**:
   - Enable WSL 2 if prompted by Docker Desktop
   - Ensure virtualization is enabled in BIOS
   - Run PowerShell/Command Prompt as Administrator if needed

### Useful Commands

```bash
# Check service status
docker-compose ps

# View specific service logs
docker-compose logs backend
docker-compose logs frontend

# Rebuild specific service
docker-compose up --build backend

# Access container shell
docker-compose exec backend bash
docker-compose exec frontend sh

# Check container resources
docker stats
```

### Health Checks

All services include health checks:
- Backend: `curl http://localhost:3006/health`
- Frontend: `curl http://localhost:3003/health`

## Production Deployment

This Docker setup closely mirrors production deployment:

1. **Multi-stage builds** for optimized images
2. **Non-root users** for security
3. **Health checks** for monitoring
4. **Proper logging** configuration
5. **Volume persistence** for data
6. **Network isolation** between services

## Next Steps

After testing with Docker:

1. **Test all functionality** thoroughly
2. **Verify data persistence** across restarts
3. **Check performance** under load
4. **Review logs** for any issues
5. **Prepare for VPS deployment** using similar configuration
