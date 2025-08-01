"""
BMI Chat Application - Chat Models

This module defines chat-related models for storing chat sessions,
messages, and conversation history in the BMI Chat application.
"""

from datetime import datetime
from typing import Optional, List
from enum import Enum

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, Float, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship

from app.models.base import BaseModel


class MessageRole(str, Enum):
    """Message role enumeration."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatSessionStatus(str, Enum):
    """Chat session status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


class ChatSession(BaseModel):
    """
    Chat session model for storing conversation sessions.
    
    Represents individual chat sessions with users,
    tracking session metadata and statistics.
    """
    
    __tablename__ = "chat_sessions"
    
    # User relationship
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    # Session information
    session_name = Column(String(255), nullable=True)
    status = Column(SQLEnum(ChatSessionStatus), default=ChatSessionStatus.ACTIVE, nullable=False)
    
    # Session metadata
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_message_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    
    # Statistics
    message_count = Column(Integer, default=0, nullable=False)
    total_tokens_used = Column(Integer, default=0, nullable=False)
    total_cost = Column(Float, default=0.0, nullable=False)
    
    # Session settings
    model_name = Column(String(100), nullable=True)
    temperature = Column(Float, nullable=True)
    max_tokens = Column(Integer, nullable=True)
    
    # Widget information (if from widget)
    is_widget_session = Column(Boolean, default=False, nullable=False)
    widget_position = Column(String(10), nullable=True)
    widget_theme = Column(String(20), nullable=True)
    
    # Context information
    context_documents = Column(Text, nullable=True)  # JSON array of document IDs
    
    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    
    def add_message_stats(self, tokens_used: int = 0, cost: float = 0.0) -> None:
        """Add message statistics to the session."""
        self.message_count += 1
        self.total_tokens_used += tokens_used
        self.total_cost += cost
        self.last_message_at = datetime.utcnow()
    
    def end_session(self) -> None:
        """End the chat session."""
        self.status = ChatSessionStatus.INACTIVE
        self.ended_at = datetime.utcnow()
    
    def archive_session(self) -> None:
        """Archive the chat session."""
        self.status = ChatSessionStatus.ARCHIVED
        if not self.ended_at:
            self.ended_at = datetime.utcnow()
    
    @property
    def duration_minutes(self) -> Optional[float]:
        """Get session duration in minutes."""
        end_time = self.ended_at or datetime.utcnow()
        if self.started_at:
            delta = end_time - self.started_at
            return delta.total_seconds() / 60
        return None
    
    @property
    def average_tokens_per_message(self) -> float:
        """Get average tokens per message."""
        if self.message_count > 0:
            return self.total_tokens_used / self.message_count
        return 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_name": self.session_name,
            "status": self.status.value if self.status else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_minutes": self.duration_minutes,
            "message_count": self.message_count,
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
            "average_tokens_per_message": self.average_tokens_per_message,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "is_widget_session": self.is_widget_session,
            "widget_position": self.widget_position,
            "widget_theme": self.widget_theme,
            "context_documents": self.context_documents,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ChatMessage(BaseModel):
    """
    Chat message model for storing individual messages.
    
    Represents individual messages within chat sessions,
    including user messages and AI responses.
    """
    
    __tablename__ = "chat_messages"
    
    # Session relationship
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    
    # Message information
    role = Column(SQLEnum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    message_index = Column(Integer, nullable=False)  # Order within session
    
    # AI response metadata
    model_name = Column(String(100), nullable=True)
    tokens_used = Column(Integer, nullable=True)
    cost = Column(Float, nullable=True)
    response_time_ms = Column(Integer, nullable=True)
    
    # Context information
    retrieved_chunks = Column(Text, nullable=True)  # JSON array of chunk IDs
    source_documents = Column(Text, nullable=True)  # JSON array of document IDs
    
    # Message metadata
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    edited_at = Column(DateTime, nullable=True)
    is_edited = Column(Boolean, default=False, nullable=False)
    
    # Quality metrics
    user_rating = Column(Integer, nullable=True)  # 1-5 rating
    user_feedback = Column(Text, nullable=True)
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    
    def edit_content(self, new_content: str) -> None:
        """Edit message content."""
        self.content = new_content
        self.is_edited = True
        self.edited_at = datetime.utcnow()
    
    def add_user_rating(self, rating: int, feedback: str = None) -> None:
        """Add user rating and feedback."""
        if 1 <= rating <= 5:
            self.user_rating = rating
            self.user_feedback = feedback
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role.value if self.role else None,
            "content": self.content,
            "message_index": self.message_index,
            "model_name": self.model_name,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "response_time_ms": self.response_time_ms,
            "retrieved_chunks": self.retrieved_chunks,
            "source_documents": self.source_documents,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "edited_at": self.edited_at.isoformat() if self.edited_at else None,
            "is_edited": self.is_edited,
            "user_rating": self.user_rating,
            "user_feedback": self.user_feedback,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
