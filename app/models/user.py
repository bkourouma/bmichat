"""
BMI Chat Application - User Model

This module defines the User model for storing user information
and session management in the BMI Chat application.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Boolean, DateTime, Text, Integer
from sqlalchemy.orm import relationship

from app.models.base import BaseModel


class User(BaseModel):
    """
    User model for storing user information and preferences.
    
    In the simplified single-tenant architecture, this represents
    individual users/sessions interacting with the chat system.
    """
    
    __tablename__ = "users"
    
    # User identification
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    user_agent = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    
    # User preferences
    preferred_language = Column(String(10), default="fr", nullable=False)
    timezone = Column(String(50), nullable=True)
    
    # Session information
    first_visit = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Usage statistics
    total_messages = Column(Integer, default=0, nullable=False)
    total_sessions = Column(Integer, default=1, nullable=False)
    
    # Widget-specific information
    widget_position = Column(String(10), default="right", nullable=False)
    widget_theme = Column(String(20), default="light", nullable=False)
    
    # Relationships
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def increment_message_count(self) -> None:
        """Increment total message count."""
        self.total_messages += 1
        self.update_activity()
    
    def to_dict(self) -> dict:
        """Convert to dictionary with safe fields only."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "preferred_language": self.preferred_language,
            "timezone": self.timezone,
            "first_visit": self.first_visit.isoformat() if self.first_visit else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "is_active": self.is_active,
            "total_messages": self.total_messages,
            "total_sessions": self.total_sessions,
            "widget_position": self.widget_position,
            "widget_theme": self.widget_theme,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
