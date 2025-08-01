"""
BMI Chat Application - Analytics Model

This module defines analytics models for tracking usage statistics,
performance metrics, and system analytics.
"""

from datetime import datetime, date
from typing import Optional

from sqlalchemy import Column, String, Integer, Float, DateTime, Date, Text, Boolean
from sqlalchemy.dialects.sqlite import JSON

from app.models.base import BaseModel


class DailyAnalytics(BaseModel):
    """
    Daily analytics model for tracking daily usage statistics.
    
    Aggregates daily metrics for monitoring and reporting.
    """
    
    __tablename__ = "daily_analytics"
    
    # Date information
    date = Column(Date, nullable=False, unique=True, index=True)
    
    # User metrics
    total_users = Column(Integer, default=0, nullable=False)
    new_users = Column(Integer, default=0, nullable=False)
    active_users = Column(Integer, default=0, nullable=False)
    returning_users = Column(Integer, default=0, nullable=False)
    
    # Session metrics
    total_sessions = Column(Integer, default=0, nullable=False)
    widget_sessions = Column(Integer, default=0, nullable=False)
    average_session_duration = Column(Float, default=0.0, nullable=False)
    
    # Message metrics
    total_messages = Column(Integer, default=0, nullable=False)
    user_messages = Column(Integer, default=0, nullable=False)
    assistant_messages = Column(Integer, default=0, nullable=False)
    average_messages_per_session = Column(Float, default=0.0, nullable=False)
    
    # Document metrics
    documents_uploaded = Column(Integer, default=0, nullable=False)
    documents_processed = Column(Integer, default=0, nullable=False)
    documents_failed = Column(Integer, default=0, nullable=False)
    total_document_size_mb = Column(Float, default=0.0, nullable=False)
    
    # AI metrics
    total_tokens_used = Column(Integer, default=0, nullable=False)
    total_ai_cost = Column(Float, default=0.0, nullable=False)
    average_response_time_ms = Column(Float, default=0.0, nullable=False)
    
    # Query metrics
    total_queries = Column(Integer, default=0, nullable=False)
    successful_queries = Column(Integer, default=0, nullable=False)
    failed_queries = Column(Integer, default=0, nullable=False)
    average_retrieval_count = Column(Float, default=0.0, nullable=False)
    
    # System metrics
    system_errors = Column(Integer, default=0, nullable=False)
    api_errors = Column(Integer, default=0, nullable=False)
    
    @classmethod
    def get_or_create_today(cls, db_session) -> "DailyAnalytics":
        """Get or create analytics record for today."""
        today = date.today()
        analytics = db_session.query(cls).filter(cls.date == today).first()
        
        if not analytics:
            analytics = cls(
                id=f"analytics_{today.isoformat()}",
                date=today
            )
            db_session.add(analytics)
            db_session.commit()
        
        return analytics
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "date": self.date.isoformat() if self.date else None,
            "total_users": self.total_users,
            "new_users": self.new_users,
            "active_users": self.active_users,
            "returning_users": self.returning_users,
            "total_sessions": self.total_sessions,
            "widget_sessions": self.widget_sessions,
            "average_session_duration": self.average_session_duration,
            "total_messages": self.total_messages,
            "user_messages": self.user_messages,
            "assistant_messages": self.assistant_messages,
            "average_messages_per_session": self.average_messages_per_session,
            "documents_uploaded": self.documents_uploaded,
            "documents_processed": self.documents_processed,
            "documents_failed": self.documents_failed,
            "total_document_size_mb": self.total_document_size_mb,
            "total_tokens_used": self.total_tokens_used,
            "total_ai_cost": self.total_ai_cost,
            "average_response_time_ms": self.average_response_time_ms,
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "average_retrieval_count": self.average_retrieval_count,
            "system_errors": self.system_errors,
            "api_errors": self.api_errors,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class SystemMetrics(BaseModel):
    """
    System metrics model for tracking real-time system performance.
    
    Stores periodic system performance and health metrics.
    """
    
    __tablename__ = "system_metrics"
    
    # Timestamp
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Database metrics
    db_connection_count = Column(Integer, nullable=True)
    db_query_time_avg_ms = Column(Float, nullable=True)
    db_size_mb = Column(Float, nullable=True)
    
    # Vector database metrics
    vector_db_size_mb = Column(Float, nullable=True)
    vector_collection_count = Column(Integer, nullable=True)
    vector_document_count = Column(Integer, nullable=True)
    
    # API metrics
    api_requests_per_minute = Column(Float, nullable=True)
    api_response_time_avg_ms = Column(Float, nullable=True)
    api_error_rate = Column(Float, nullable=True)
    
    # Memory and CPU (if available)
    memory_usage_mb = Column(Float, nullable=True)
    cpu_usage_percent = Column(Float, nullable=True)
    
    # OpenAI API metrics
    openai_requests_count = Column(Integer, default=0, nullable=False)
    openai_tokens_used = Column(Integer, default=0, nullable=False)
    openai_cost = Column(Float, default=0.0, nullable=False)
    openai_avg_response_time_ms = Column(Float, nullable=True)
    
    # File system metrics
    upload_dir_size_mb = Column(Float, nullable=True)
    log_dir_size_mb = Column(Float, nullable=True)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
            "db_connection_count": self.db_connection_count,
            "db_query_time_avg_ms": self.db_query_time_avg_ms,
            "db_size_mb": self.db_size_mb,
            "vector_db_size_mb": self.vector_db_size_mb,
            "vector_collection_count": self.vector_collection_count,
            "vector_document_count": self.vector_document_count,
            "api_requests_per_minute": self.api_requests_per_minute,
            "api_response_time_avg_ms": self.api_response_time_avg_ms,
            "api_error_rate": self.api_error_rate,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "openai_requests_count": self.openai_requests_count,
            "openai_tokens_used": self.openai_tokens_used,
            "openai_cost": self.openai_cost,
            "openai_avg_response_time_ms": self.openai_avg_response_time_ms,
            "upload_dir_size_mb": self.upload_dir_size_mb,
            "log_dir_size_mb": self.log_dir_size_mb,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
