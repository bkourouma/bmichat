# 🤖 BMI Chat - Document-Based Chatbot System

An embeddable document-based chatbot system that allows clients to integrate AI-powered document chat functionality into their websites. Users can ask questions about uploaded documents and receive accurate, context-aware responses.

## 🏗️ Architecture

- **Backend**: FastAPI + SQLite + ChromaDB
- **Frontend**: React + Vite + TypeScript
- **Widget**: Vanilla JavaScript (embeddable)
- **AI**: OpenAI GPT-4o with RAG (Retrieval Augmented Generation)
- **Deployment**: Docker + Nginx + HTTPS

## 🚀 Quick Start

### Development

```bash
# Clone repository
git clone <repository-url>
cd bmichat

# Start development environment
docker compose up -d --build

# Backend will be available at: http://localhost:3006
# Frontend will be available at: http://localhost:3003
```

### Production

```bash
# Deploy to production
docker compose -f docker-compose.production.yml up -d --build

# Access at: https://bmi.engage-360.net
```

## 📁 Project Structure

```
bmichat/
├── app/                    # FastAPI backend
│   ├── main.py            # Application entry point
│   ├── models/            # Database models
│   ├── services/          # Business logic
│   ├── routers/           # API endpoints
│   └── config.py          # Configuration
├── frontend/              # React + Vite frontend
│   ├── src/               # Source code
│   ├── public/            # Static assets
│   └── dist/              # Build output
├── widget/                # Embeddable JavaScript widget
│   ├── src/               # Widget source
│   └── dist/              # Widget build
├── tests/                 # Test suites
│   ├── backend/           # Backend tests
│   ├── frontend/          # Frontend tests
│   └── e2e/               # End-to-end tests
├── deployment/            # Deployment configurations
│   ├── docker/            # Docker configurations
│   ├── nginx/             # Nginx configurations
│   └── scripts/           # Deployment scripts
├── data/                  # Persistent data (gitignored)
│   ├── sqlite/            # SQLite database
│   ├── vectors/           # ChromaDB vectors
│   └── uploads/           # Document uploads
└── docs/                  # Documentation
```

## 🔧 Configuration

### Environment Variables

Create `.env` file in the root directory:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.0

# Database Configuration
DB_SQLITE_PATH=data/sqlite/bmi.db
VECTOR_DB_PATH=data/vectors

# API Configuration
API_HOST=0.0.0.0
API_PORT=3006
CORS_ORIGINS=["http://localhost:3003"]

# Frontend Configuration
VITE_API_URL=http://localhost:3006
VITE_WS_URL=ws://localhost:3006/ws
```

## 🧪 Testing

```bash
# Run backend tests
cd tests && python -m pytest backend/

# Run frontend tests
cd frontend && npm test

# Run E2E tests
cd tests && npm run test:e2e
```

## 📦 Deployment

### Local Development
- Backend: `http://localhost:3006`
- Frontend: `http://localhost:3003`

### Production
- Domain: `https://bmi.engage-360.net`
- SSL: Automated with Certbot
- Monitoring: Health checks and logging

## 🔌 Widget Integration

```html
<!-- Add to any website -->
<script>
(function() {
    var script = document.createElement('script');
    script.src = 'https://bmi.engage-360.net/widget/chat-widget.js';
    script.setAttribute('data-api-url', 'https://bmi.engage-360.net');
    script.setAttribute('data-position', 'right');
    script.setAttribute('data-accent-color', '#0056b3');
    script.async = true;
    document.head.appendChild(script);
})();
</script>
```

## 📄 License

MIT License - see LICENSE file for details.
