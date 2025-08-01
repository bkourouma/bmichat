"""
BMI Chat Application Configuration

This module handles all configuration settings for the application,
including environment variables, database settings, and API configuration.
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # =============================================================================
    # Application Info
    # =============================================================================
    app_name: str = "BMI Chat"
    app_version: str = "1.0.0"
    app_description: str = "Document-based chatbot system with embeddable widget"
    
    # =============================================================================
    # Environment
    # =============================================================================
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # =============================================================================
    # API Configuration
    # =============================================================================
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=3006, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")
    cors_origins: List[str] = Field(
        default=["http://localhost:3003", "http://127.0.0.1:3003"],
        env="CORS_ORIGINS"
    )
    
    # =============================================================================
    # OpenAI Configuration
    # =============================================================================
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.0, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=2000, env="OPENAI_MAX_TOKENS")
    
    # =============================================================================
    # Database Configuration
    # =============================================================================
    db_sqlite_path: str = Field(default="data/sqlite/bmi.db", env="DB_SQLITE_PATH")
    vector_db_path: str = Field(default="data/vectors", env="VECTOR_DB_PATH")
    upload_dir: str = Field(default="data/uploads", env="UPLOAD_DIR")
    
    # =============================================================================
    # Document Processing
    # =============================================================================
    chunk_size: int = Field(default=2000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=400, env="CHUNK_OVERLAP")
    max_chunks_per_document: int = Field(default=100, env="MAX_CHUNKS_PER_DOCUMENT")
    supported_file_types: List[str] = Field(
        default=["pdf", "txt", "docx", "md"],
        env="SUPPORTED_FILE_TYPES"
    )
    
    # =============================================================================
    # Chat Configuration
    # =============================================================================
    max_chat_history: int = Field(default=10, env="MAX_CHAT_HISTORY")
    default_retrieval_k: int = Field(default=3, env="DEFAULT_RETRIEVAL_K")
    max_retrieval_k: int = Field(default=5, env="MAX_RETRIEVAL_K")
    chat_timeout_seconds: int = Field(default=30, env="CHAT_TIMEOUT_SECONDS")
    
    # =============================================================================
    # Security
    # =============================================================================
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # =============================================================================
    # Logging
    # =============================================================================
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")
    log_max_size: str = Field(default="10MB", env="LOG_MAX_SIZE")
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # =============================================================================
    # Widget Configuration
    # =============================================================================
    widget_default_position: str = Field(default="right", env="WIDGET_DEFAULT_POSITION")
    widget_default_accent_color: str = Field(default="#0056b3", env="WIDGET_DEFAULT_ACCENT_COLOR")
    widget_default_company_name: str = Field(default="BMI", env="WIDGET_DEFAULT_COMPANY_NAME")
    widget_default_assistant_name: str = Field(default="Akissi", env="WIDGET_DEFAULT_ASSISTANT_NAME")
    widget_default_welcome_message: str = Field(
        default="Bonjour! Comment puis-je vous aider?",
        env="WIDGET_DEFAULT_WELCOME_MESSAGE"
    )
    
    # =============================================================================
    # Performance
    # =============================================================================
    max_upload_size: str = Field(default="50MB", env="MAX_UPLOAD_SIZE")
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    cache_ttl_seconds: int = Field(default=300, env="CACHE_TTL_SECONDS")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist."""
        directories = [
            Path(self.db_sqlite_path).parent,
            Path(self.vector_db_path),
            Path(self.upload_dir),
            Path(self.log_file).parent,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def database_url(self) -> str:
        """Get SQLite database URL."""
        return f"sqlite:///{self.db_sqlite_path}"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"


# Global settings instance
settings = Settings()
