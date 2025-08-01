# 🤖 Widget Chat Module - Complete Architecture Documentation

## 🏗️ Overview

The Widget Chat Module is an embeddable document-based chatbot system that allows clients to integrate AI-powered document chat functionality into their websites. Users can ask questions about uploaded documents and receive accurate, context-aware responses.

## 📁 File Structure

### Backend Components
```
api/
├── widget_chat.py          # Main widget API endpoints
├── widget_admin.py         # Widget administration & configuration  
└── visitor_documents.py    # Document upload for widgets

visitor_chat/
├── services/
│   ├── document_service.py # Document processing & ingestion
│   ├── chat_service.py     # Chat logic & retrieval
│   └── __init__.py
├── utils/
│   ├── qa_chunker.py       # Intelligent document chunking
│   ├── qa_validator.py     # Content validation
│   └── embedding_utils.py  # OpenAI embeddings wrapper
├── routers/
│   └── chat_router.py      # Chat routing logic
└── config.py               # Configuration settings
```

### Frontend Components
```
dashboardapp/public/
├── document-chat-widget.js     # Production widget
├── document-chat-widget-dev.js # Development widget
├── widget-demo.html            # Demo page
├── widget-admin.html           # Admin interface
└── chat360-widget-wordpress.php # WordPress plugin
```

### Database Schema
```
sqlite_data/chatbot.sqlite
├── tenants                     # Client configuration
├── widget_interactions         # Analytics & logging
├── documents                   # Document metadata
└── visitor_chroma/            # Vector embeddings (ChromaDB)
```

## 🔧 Tech Stack

### Backend Technologies
- **Framework**: FastAPI (Python)
- **Database**: SQLite (primary) + ChromaDB (vectors)
- **Document Processing**: LangChain + PyPDFLoader
- **Embeddings**: OpenAI Embeddings (text-embedding-ada-002)
- **LLM**: OpenAI GPT-3.5-turbo/GPT-4
- **Vector Store**: ChromaDB with persistence

### Frontend Technologies
- **Core**: Vanilla JavaScript (ES6+)
- **Styling**: CSS3 with responsive design
- **Markdown**: Marked.js for response formatting
- **Integration**: Script tag embedding

### Infrastructure
- **Containerization**: Docker + Docker Compose
- **Storage**: File system + vector database
- **API**: RESTful endpoints with CORS support

## 📊 Data Flow Architecture

### 1. Document Upload & Processing

**Process Flow:**
1. **Upload**: PDF/DOCX files uploaded via `/visitor-documents/upload` API
2. **Loading**: PyPDFLoader extracts text content from documents
3. **Chunking**: HybridChunker intelligently splits content into meaningful segments
4. **Embedding**: OpenAI generates vector embeddings for each chunk
5. **Storage**: ChromaDB stores vectors with metadata in tenant-specific collections

```python
# Document ingestion process
@staticmethod
def ingest_document(tenant_id: int, file_path: str) -> None:
    """
    1. Load the document (PDF or DOCX) via appropriate loader
    2. Split it into smaller "chunks" 
    3. Embed those chunks and add to ChromaDB collection
    """
```

### 2. Intelligent Chunking Strategy

**Chunking Methods:**
- **QA Format Detection**: Recognizes `---QA---` delimited content
- **Smart Chunking**: Complete Q&A pairs preserved as single chunks
- **Fallback**: RecursiveCharacterTextSplitter for regular documents
- **Metadata Enrichment**: Adds chunk type, length, content analysis

```python
class HybridChunker:
    """
    Intelligent chunker that automatically chooses between QA and classic chunking
    """
    
    def chunk_document(self, text: str, metadata: Optional[Dict] = None):
        """
        Smart chunking:
        - If ---QA--- format detected: uses QADelimiterChunker
        - Otherwise: uses RecursiveCharacterTextSplitter
        """
```

**Configuration Parameters:**
- **CHUNK_SIZE**: 2000 characters (optimized for detailed content)
- **CHUNK_OVERLAP**: 400 characters (increased overlap for context)
- **DEFAULT_K**: 3 chunks retrieved per query

### 3. Semantic Search & Retrieval

**Search Process:**
1. **Vector Similarity**: ChromaDB performs similarity search on user question
2. **Tenant Isolation**: Results filtered by tenant_id for data security
3. **Relevance Ranking**: Top-K most relevant chunks selected
4. **Context Building**: Chunk contents concatenated for LLM context

```python
@staticmethod
def retrieve_relevant_chunks(tenant_id: int, question: str, k: Optional[int] = None):
    """
    Enhanced search combining similarity and keyword matching
    Returns up to k Document objects with page_content + metadata
    """
```

### 4. Response Generation

**LLM Processing:**
1. **Context Assembly**: Retrieved chunks combined into coherent context
2. **Prompt Engineering**: System prompt + user question + document context
3. **LLM Generation**: OpenAI GPT generates response based on document content
4. **Response Filtering**: Ensures answers stay within document scope

## 🌐 Widget Integration

### Embedding Code
```javascript
// Basic integration
(function() {
    window.docChatWidget = window.docChatWidget || function() {
        (window.docChatWidget.q = window.docChatWidget.q || []).push(arguments);
    };
    
    var script = document.createElement('script');
    script.src = 'https://chat.engage-360.net/static/document-chat-widget.js';
    script.setAttribute('data-tenant-id', 'YOUR_TENANT_ID');
    script.setAttribute('data-api-url', 'https://chat.engage-360.net');
    script.setAttribute('data-position', 'right');
    script.setAttribute('data-accent-color', '#0056b3');
    script.async = true;
    document.head.appendChild(script);
})();

// Initialize the widget
docChatWidget('init', {
    tenantId: YOUR_TENANT_ID,
    apiUrl: 'https://chat.engage-360.net',
    position: 'right',
    accentColor: '#0056b3'
});
```

### Widget Configuration Options
- **Position**: 'right' or 'left' screen positioning
- **Accent Color**: Customizable brand colors
- **Auto Open**: Automatic widget opening
- **Company Name**: Branded assistant name
- **Welcome Message**: Custom greeting message

## 🔌 API Endpoints

### Core Widget APIs
- `POST /widget/chat` - Process chat messages from embedded widgets
- `GET /widget/config/{tenant_id}` - Retrieve widget styling and behavior settings
- `GET /widget/health/{tenant_id}` - Monitor widget availability and document status
- `POST /visitor-documents/upload` - Upload documents for widget knowledge base

### Admin APIs
- `GET /admin/widget/settings` - Retrieve widget configuration
- `POST /admin/widget/settings` - Update widget settings
- `GET /admin/widget/analytics` - Usage statistics and metrics

### Request/Response Models
```python
class WidgetChatRequest(BaseModel):
    tenant_id: int = Field(..., description="Tenant ID for the widget")
    session_id: str = Field(..., description="Unique session identifier")
    question: str = Field(..., description="User's question")
    widget_key: Optional[str] = Field(None, description="Optional authentication key")

class WidgetChatResponse(BaseModel):
    answer: str = Field(..., description="AI-generated response")
    sources: List[str] = Field(default=[], description="Source documents")
    session_id: str = Field(..., description="Session identifier")
    timestamp: str = Field(..., description="Response timestamp")
```

## 💾 Database Schema

### Key Tables
```sql
-- Tenant configuration
CREATE TABLE tenants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    company_name TEXT DEFAULT 'BMI',
    assistant_name TEXT DEFAULT 'Akissi',
    welcome_message TEXT DEFAULT 'Bonjour! Comment puis-je vous aider?',
    accent_color TEXT DEFAULT '#0056b3',
    widget_position TEXT DEFAULT 'right',
    widget_enabled INTEGER DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Widget analytics and interactions
CREATE TABLE widget_interactions (
    id TEXT PRIMARY KEY,
    tenant_id INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY(tenant_id) REFERENCES tenants(id)
);

-- Document metadata
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id INTEGER NOT NULL,
    filename TEXT NOT NULL,
    title TEXT,
    description TEXT,
    uploaded_at TEXT NOT NULL,
    uploaded_by TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY(tenant_id) REFERENCES tenants(id)
);
```

### Vector Storage Structure
```
visitor_chroma/
├── tenant_1/
│   ├── chroma.sqlite3      # ChromaDB metadata
│   └── visitor_collection/ # Vector embeddings
├── tenant_2/
│   ├── chroma.sqlite3
│   └── visitor_collection/
└── ...
```

## 🎨 Frontend Architecture

### Widget Structure
- **Self-contained**: No external dependencies except Marked.js
- **Responsive**: Mobile-optimized design with breakpoints
- **Customizable**: Tenant-specific styling and branding
- **Lightweight**: ~50KB JavaScript bundle

### Key Features
- Real-time chat interface with typing indicators
- Markdown response rendering for rich content
- Session management with persistent conversations
- Error handling & graceful fallbacks
- Analytics tracking for usage insights
- CORS-enabled for cross-domain embedding

### CSS Architecture
```css
.doc-chat-widget {
    position: fixed;
    bottom: 80px;
    width: 380px;
    height: 550px;
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
    z-index: 999999;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto;
}
```

## 🔧 Configuration Parameters

### Environment Variables
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0.0

# Storage Paths
VISITOR_DOCS_DIR=visitor_docs
VISITOR_CHROMA_DIR=visitor_chroma
SQLITE_PATH=sqlite_data/chatbot.sqlite

# Chunking Parameters
VISITOR_CHUNK_SIZE=2000
VISITOR_CHUNK_OVERLAP=400
VISITOR_DEFAULT_K=3
VISITOR_MAX_K=5

# API Configuration
CORS_ORIGINS=["*"]  # Configure for production
```

### Deployment Configuration
- **Development**: 
  - Backend: localhost:3006
  - Frontend: localhost:3003
- **Production**: https://chat.engage-360.net
- **Docker**: Port 6688 mapped to internal 8000

## 🚀 Deployment Architecture

### Docker Configuration
```yaml
# docker-compose.yml
services:
  chatbot:
    build: .
    container_name: bmi-chatbot
    ports:
      - "6688:8000"
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      SQLITE_PATH: /app/data/chatbot.sqlite
    volumes:
      - ./docs:/app/docs                # PDF uploads for RAG
      - ./chroma_db_data:/app/chroma_db # Vector DB persistence
      - ./sqlite_data:/app/data         # SQLite DB persistence
```

### Performance Optimizations
- **Asynchronous Loading**: Non-blocking script loading
- **CDN Ready**: Optimized for content delivery networks
- **Minimal Footprint**: Lightweight JavaScript bundle
- **Caching**: Efficient API response caching
- **Lazy Initialization**: Widget loads only when needed

## 📊 Analytics & Monitoring

### Available Metrics
- Total interactions per tenant
- Daily and weekly usage statistics
- Most common questions and topics
- Average response times
- Widget availability status
- Document coverage analysis

### Monitoring Endpoints
- `/widget/health/{tenant_id}` - Widget operational status
- `/admin/widget/analytics` - Comprehensive usage statistics
- `/admin/widget/test` - Functionality testing interface

## 🎨 Customization Options

### Styling Customization
- **Accent Colors**: Brand-specific color schemes
- **Positioning**: Left or right screen placement
- **Typography**: Custom font families
- **Dimensions**: Adjustable widget size
- **Animations**: Smooth transitions and effects

### Functional Customization
- **Welcome Messages**: Personalized greetings
- **Assistant Names**: Branded AI assistant identity
- **Auto-open Behavior**: Configurable widget activation
- **Language Support**: Multi-language capabilities
- **Integration Hooks**: Custom event handlers

## 🔒 Security Features

### Data Isolation
- **Tenant Separation**: Complete data isolation between clients
- **Session Management**: Secure session handling
- **API Authentication**: Optional widget key authentication
- **CORS Configuration**: Controlled cross-origin access

### Privacy Protection
- **No Personal Data Storage**: Minimal user data collection
- **Anonymized Analytics**: Privacy-preserving metrics
- **Secure Transmission**: HTTPS-only communication
- **Data Retention**: Configurable retention policies

## 🚀 Getting Started

### Quick Setup
1. **Environment Setup**: Configure OpenAI API key and database paths
2. **Database Initialization**: Run `python init_database.py`
3. **Document Upload**: Use `/visitor-documents/upload` to add knowledge base
4. **Widget Integration**: Add embedding code to target website
5. **Configuration**: Customize via admin interface

### Development Mode
```bash
# Start development server
uvicorn app:app --host 0.0.0.0 --port 3006 --reload

# Frontend development
# Serve widget files from localhost:3003
```

### Production Deployment
```bash
# Build and deploy with Docker
docker-compose up -d

# Access at https://chat.engage-360.net
```

This architecture provides a complete, scalable document chat widget system that maintains tenant isolation while delivering accurate, document-based responses through an easily embeddable interface.
