"""
BMI Chat Application - Document Model

This module defines the Document model for storing document metadata
and processing information in the BMI Chat application.
"""

from datetime import datetime
from typing import Optional, List
from enum import Enum

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, Float, Enum as SQLEnum, ForeignKey
from sqlalchemy.orm import relationship

from app.models.base import BaseModel


class DocumentStatus(str, Enum):
    """Document processing status enumeration."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    DELETED = "deleted"


class DocumentType(str, Enum):
    """Document type enumeration."""
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    MD = "md"


class Document(BaseModel):
    """
    Document model for storing document metadata and processing information.
    
    Tracks uploaded documents, their processing status, and metadata
    for the knowledge base.
    """
    
    __tablename__ = "documents"
    
    # Basic document information
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(SQLEnum(DocumentType), nullable=False)
    file_size = Column(Integer, nullable=False)  # Size in bytes
    mime_type = Column(String(100), nullable=True)
    
    # Processing information
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.UPLOADED, nullable=False)
    processing_started_at = Column(DateTime, nullable=True)
    processing_completed_at = Column(DateTime, nullable=True)
    processing_error = Column(Text, nullable=True)
    
    # Content analysis
    total_pages = Column(Integer, nullable=True)
    total_words = Column(Integer, nullable=True)
    total_characters = Column(Integer, nullable=True)
    language = Column(String(10), default="fr", nullable=False)
    
    # Chunking information
    chunk_count = Column(Integer, default=0, nullable=False)
    chunk_size = Column(Integer, nullable=True)
    chunk_overlap = Column(Integer, nullable=True)
    
    # Vector database information
    embedding_model = Column(String(100), nullable=True)
    vector_count = Column(Integer, default=0, nullable=False)
    
    # Content metadata
    title = Column(String(500), nullable=True)
    author = Column(String(255), nullable=True)
    subject = Column(String(500), nullable=True)
    keywords = Column(Text, nullable=True)  # JSON array of keywords
    
    # Usage statistics
    query_count = Column(Integer, default=0, nullable=False)
    last_queried = Column(DateTime, nullable=True)
    
    # Flags
    is_active = Column(Boolean, default=True, nullable=False)
    is_public = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    def mark_processing_started(self) -> None:
        """Mark document processing as started."""
        self.status = DocumentStatus.PROCESSING
        self.processing_started_at = datetime.utcnow()
        self.processing_error = None
    
    def mark_processing_completed(self, chunk_count: int = 0, vector_count: int = 0) -> None:
        """Mark document processing as completed."""
        self.status = DocumentStatus.PROCESSED
        self.processing_completed_at = datetime.utcnow()
        self.chunk_count = chunk_count
        self.vector_count = vector_count
        self.processing_error = None
    
    def mark_processing_failed(self, error: str) -> None:
        """Mark document processing as failed."""
        self.status = DocumentStatus.FAILED
        self.processing_error = error
    
    def increment_query_count(self) -> None:
        """Increment query count and update last queried timestamp."""
        self.query_count += 1
        self.last_queried = datetime.utcnow()
    
    @property
    def processing_duration(self) -> Optional[float]:
        """Get processing duration in seconds."""
        if self.processing_started_at and self.processing_completed_at:
            delta = self.processing_completed_at - self.processing_started_at
            return delta.total_seconds()
        return None
    
    @property
    def file_size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.file_size / (1024 * 1024)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_type": self.file_type.value if self.file_type else None,
            "file_size": self.file_size,
            "file_size_mb": self.file_size_mb,
            "mime_type": self.mime_type,
            "status": self.status.value if self.status else None,
            "processing_started_at": self.processing_started_at.isoformat() if self.processing_started_at else None,
            "processing_completed_at": self.processing_completed_at.isoformat() if self.processing_completed_at else None,
            "processing_duration": self.processing_duration,
            "processing_error": self.processing_error,
            "total_pages": self.total_pages,
            "total_words": self.total_words,
            "total_characters": self.total_characters,
            "language": self.language,
            "chunk_count": self.chunk_count,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "vector_count": self.vector_count,
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "keywords": self.keywords,
            "query_count": self.query_count,
            "last_queried": self.last_queried.isoformat() if self.last_queried else None,
            "is_active": self.is_active,
            "is_public": self.is_public,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DocumentChunk(BaseModel):
    """
    Document chunk model for storing individual text chunks.
    
    Represents individual chunks of text extracted from documents
    for embedding and retrieval.
    """
    
    __tablename__ = "document_chunks"
    
    # Relationship to document
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    
    # Chunk information
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_length = Column(Integer, nullable=False)
    
    # Position information
    start_page = Column(Integer, nullable=True)
    end_page = Column(Integer, nullable=True)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    
    # Vector information
    vector_id = Column(String, nullable=True)  # ID in vector database
    embedding_model = Column(String(100), nullable=True)
    
    # Usage statistics
    retrieval_count = Column(Integer, default=0, nullable=False)
    last_retrieved = Column(DateTime, nullable=True)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    def increment_retrieval_count(self) -> None:
        """Increment retrieval count and update timestamp."""
        self.retrieval_count += 1
        self.last_retrieved = datetime.utcnow()
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "content_length": self.content_length,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "vector_id": self.vector_id,
            "embedding_model": self.embedding_model,
            "retrieval_count": self.retrieval_count,
            "last_retrieved": self.last_retrieved.isoformat() if self.last_retrieved else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
