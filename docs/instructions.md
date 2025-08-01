# **BMI Chatbot – Complete Build & Deployment Guide**

This single‑tenant guide merges the original **road‑map** and the **production stack templates** so you can move from local Augment Code generation to a live deployment at [**https://bmi.engage-360.net**](https://bmi.engage-360.net) with minimal friction.

---

## ✨ 0 · Prompt for Augment Code

```text
Build a React (Vite) SPA that talks to a FastAPI back‑end over REST.  
Back‑end uses SQLite + ChromaDB; **no multitenancy** (one global dataset).  
Provide Docker & docker‑compose for local dev, persisting the SQLite file via a bind‑mount.  
I’ll deploy the same container on a Hostinger VPS behind the domain bmi.engage-360.net.
```

Copy & paste when scaffolding new code or screens with Augment Code.

---

## 🗺️ Phase‑by‑Phase Roadmap

| Phase                       | Goal                    | Key Actions                                                                                      |
| --------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------ |
| **1 – Remove multitenancy** | Single global dataset   | Drop `tenant_id` columns/tables, delete tenant‑specific folders, simplify API paths and filters. |
| **2 – SQLite persistence**  | Same DB file dev → prod | Bind‑mount `./data/sqlite` → `/app/data`; script nightly dumps to S3.                            |
| **3 – Local Docker dev**    | Hot‑reload stacks       | `docker compose up -d --build`, hit `http://localhost:6688/health`.                              |
| **4 – Git workflow**        | Version control         | Push to GitHub, ignore `.env`; optional CI to Docker Hub.                                        |
| **5 – VPS provision**       | First deploy            | `git clone`, create mount dirs, `docker compose up -d --build`.                                  |
| **6 – Domain & HTTPS**      | Public URL              | DNS ➜ VPS IP, Caddy/Certbot for TLS.                                                             |
| **7 – React build**         | Serve static files      | `npm run build` → copy to `/static`; FastAPI already mounts.                                     |
| **8 – Future steps**        | Hardening               | Seed‑loader, admin CLI, backups, CI/CD.                                                          |

---

## 🏗️ Production Stack – `docker-compose.production.yml`

```yaml
version: "3.8"

services:
  bmi-backend:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: bmi-backend
    restart: unless-stopped
    ports: ["6688:8000"]
    environment:
      ENVIRONMENT: production
      HOST: 0.0.0.0
      PORT: 8000
      PYTHONUNBUFFERED: "1"
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      OPENAI_MODEL: ${OPENAI_MODEL:-gpt-4o}
      DB_SQLITE_PATH: /app/data/bmi.db
    volumes:
      - ./data/sqlite:/app/data
      - ./docs:/app/docs
      - ./vectors:/app/vectors
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks: [bmi-net]

  bmi-frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.production
    container_name: bmi-frontend
    environment:
      - NODE_ENV=production
      - VITE_API_URL=https://bmi.engage-360.net/api
      - VITE_WS_URL=wss://bmi.engage-360.net/ws
    networks: [bmi-net]

  nginx:
    image: nginx:alpine
    container_name: bmi-nginx
    restart: unless-stopped
    ports: ["8090:80"]
    volumes:
      - ./frontend/dist:/usr/share/nginx/html
      - ./nginx/conf.d/default.conf:/etc/nginx/conf.d/default.conf
      - ./logs/nginx:/var/log/nginx
    depends_on: [bmi-backend, bmi-frontend]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:80"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks: [bmi-net]

networks:
  bmi-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.30.0.0/16
```

---

## 🐍 Backend `Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
ENV POETRY_VERSION=1.8.2
RUN apt-get update && apt-get install -y gcc build-essential curl && \
    pip install --no-cache-dir poetry==$POETRY_VERSION && \
    rm -rf /var/lib/apt/lists/*
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-dev --no-interaction --no-ansi
COPY ./app ./app
COPY ./start.sh .
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["./start.sh"]
```

``

```bash
#!/usr/bin/env bash
exec poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

---

## 🌐 Inner‑Docker Nginx `default.conf`

```nginx
server {
    listen 80;
    server_name _;
    root /usr/share/nginx/html;
    index index.html;

    location /api/ {
        proxy_pass http://bmi-backend:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /ws/ {
        proxy_pass http://bmi-backend:8000/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

---

## 🛡️ System‑level Nginx (Hostinger)

```nginx
server {
    listen 80;
    server_name bmi.engage-360.net;

    location / {
        proxy_pass http://localhost:8090/;  # Docker Nginx
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Run `sudo certbot --nginx -d bmi.engage-360.net` for HTTPS.

---

## 📦 Frontend `Dockerfile.production`

```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --silent
COPY . .
ARG VITE_API_URL=https://bmi.engage-360.net/api
ARG VITE_WS_URL=wss://bmi.engage-360.net/ws
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_WS_URL=$VITE_WS_URL
ENV NODE_ENV=production
RUN npm run build

FROM nginx:alpine AS production
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
HEALTHCHECK CMD curl -f http://localhost:80 || exit 1
CMD ["nginx", "-g", "daemon off;"]
```

---

## 🗃️ `requirements.txt` (key packages)

```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
aiosqlite>=0.19.0
chromadb>=0.4.0
openai>=1.0.0,<2.0.0
langchain>=0.1.0,<0.2.0
langchain-chroma>=0.1.0,<0.2.0
pydantic==2.5.0
python-dotenv==1.0.0
python-multipart==0.0.6
```

---

## 🚀 Deployment Workflow

### **Local dev**

```bash
docker compose up -d --build
open http://localhost:6688/health
```

### **Commit & push**

```bash
git add .
git commit -m "BMI Chatbot – switch to SQLite stack"
git push origin main
```

### **Server update**

```bash
cd /opt/bmi-chatbot
git pull origin main
docker compose -f docker-compose.production.yml up -d --build
```

*(For zero‑downtime: build + push images in CI, then **`docker compose pull && docker compose up -d`**.)*

---

## ✅ Recap

1. **No multitenancy** – simplified DB & code.
2. **SQLite persisted** – `./data/sqlite/bmi.db` mounts identically in dev and prod.
3. **Stack mirrors your working AI Agent Platform** – same health‑checks, inner Docker Nginx, outer TLS proxy.

You’re now ready to ask Augment Code for the first screen or to wire your existing FastAPI endpoints. Ping me anytime for deep‑dive refactors or troubleshooting!

