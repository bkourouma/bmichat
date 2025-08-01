#!/usr/bin/env python3
"""
BMI Chat Application - Database Seeding Script

This script seeds the database with sample data for development and testing.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, date
from uuid import uuid4

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import settings
from app.core.database import AsyncSessionLocal
from app.models import (
    User, Document, DocumentChunk, ChatSession, ChatMessage,
    DailyAnalytics, SystemMetrics, DocumentStatus, DocumentType,
    MessageRole, ChatSessionStatus
)
from loguru import logger


async def create_sample_users():
    """Create sample users for testing."""
    async with AsyncSessionLocal() as session:
        try:
            # Sample user 1
            user1 = User(
                id=str(uuid4()),
                session_id="demo_session_001",
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                ip_address="192.168.1.100",
                preferred_language="fr",
                timezone="Europe/Paris",
                widget_position="right",
                widget_theme="light",
                total_messages=15,
                total_sessions=3
            )
            
            # Sample user 2
            user2 = User(
                id=str(uuid4()),
                session_id="demo_session_002",
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                ip_address="192.168.1.101",
                preferred_language="en",
                timezone="America/New_York",
                widget_position="left",
                widget_theme="dark",
                total_messages=8,
                total_sessions=1
            )
            
            session.add(user1)
            session.add(user2)
            await session.commit()
            
            logger.info("✅ Created sample users")
            return [user1, user2]
            
        except Exception as e:
            await session.rollback()
            logger.error(f"❌ Failed to create sample users: {str(e)}")
            raise


async def create_sample_documents():
    """Create sample documents for testing."""
    async with AsyncSessionLocal() as session:
        try:
            # Sample document 1
            doc1 = Document(
                id=str(uuid4()),
                filename="sample_guide.pdf",
                original_filename="BMI Company Guide.pdf",
                file_path="data/uploads/sample_guide.pdf",
                file_type=DocumentType.PDF,
                file_size=2048576,  # 2MB
                mime_type="application/pdf",
                status=DocumentStatus.PROCESSED,
                processing_completed_at=datetime.utcnow(),
                total_pages=25,
                total_words=5000,
                total_characters=30000,
                language="fr",
                chunk_count=15,
                chunk_size=2000,
                chunk_overlap=400,
                embedding_model="text-embedding-ada-002",
                vector_count=15,
                title="BMI Company Guide",
                author="BMI Team",
                subject="Company policies and procedures",
                keywords='["BMI", "guide", "policies", "procedures"]',
                query_count=25,
                last_queried=datetime.utcnow()
            )
            
            # Sample document 2
            doc2 = Document(
                id=str(uuid4()),
                filename="faq.md",
                original_filename="Frequently Asked Questions.md",
                file_path="data/uploads/faq.md",
                file_type=DocumentType.MD,
                file_size=51200,  # 50KB
                mime_type="text/markdown",
                status=DocumentStatus.PROCESSED,
                processing_completed_at=datetime.utcnow(),
                total_pages=1,
                total_words=1200,
                total_characters=8000,
                language="fr",
                chunk_count=5,
                chunk_size=2000,
                chunk_overlap=400,
                embedding_model="text-embedding-ada-002",
                vector_count=5,
                title="FAQ - Questions Fréquentes",
                author="Support Team",
                subject="Frequently asked questions and answers",
                keywords='["FAQ", "questions", "answers", "support"]',
                query_count=12,
                last_queried=datetime.utcnow()
            )
            
            session.add(doc1)
            session.add(doc2)
            await session.commit()
            
            logger.info("✅ Created sample documents")
            return [doc1, doc2]
            
        except Exception as e:
            await session.rollback()
            logger.error(f"❌ Failed to create sample documents: {str(e)}")
            raise


async def create_sample_chat_sessions(users):
    """Create sample chat sessions for testing."""
    async with AsyncSessionLocal() as session:
        try:
            # Chat session 1
            chat1 = ChatSession(
                id=str(uuid4()),
                user_id=users[0].id,
                session_name="Welcome Chat",
                status=ChatSessionStatus.ACTIVE,
                message_count=4,
                total_tokens_used=350,
                total_cost=0.0007,
                model_name="gpt-4o",
                temperature=0.0,
                is_widget_session=True,
                widget_position="right",
                widget_theme="light"
            )
            
            # Chat session 2
            chat2 = ChatSession(
                id=str(uuid4()),
                user_id=users[1].id,
                session_name="Product Inquiry",
                status=ChatSessionStatus.INACTIVE,
                message_count=6,
                total_tokens_used=520,
                total_cost=0.00104,
                model_name="gpt-4o",
                temperature=0.0,
                is_widget_session=True,
                widget_position="left",
                widget_theme="dark"
            )
            
            session.add(chat1)
            session.add(chat2)
            await session.commit()
            
            logger.info("✅ Created sample chat sessions")
            return [chat1, chat2]
            
        except Exception as e:
            await session.rollback()
            logger.error(f"❌ Failed to create sample chat sessions: {str(e)}")
            raise


async def create_sample_analytics():
    """Create sample analytics data."""
    async with AsyncSessionLocal() as session:
        try:
            # Daily analytics for today
            today_analytics = DailyAnalytics(
                id=f"analytics_{date.today().isoformat()}",
                date=date.today(),
                total_users=25,
                new_users=5,
                active_users=18,
                returning_users=20,
                total_sessions=45,
                widget_sessions=40,
                average_session_duration=8.5,
                total_messages=180,
                user_messages=90,
                assistant_messages=90,
                average_messages_per_session=4.0,
                documents_uploaded=3,
                documents_processed=3,
                documents_failed=0,
                total_document_size_mb=15.5,
                total_tokens_used=8500,
                total_ai_cost=0.017,
                average_response_time_ms=1250,
                total_queries=95,
                successful_queries=92,
                failed_queries=3,
                average_retrieval_count=3.2,
                system_errors=2,
                api_errors=1
            )
            
            # System metrics
            system_metrics = SystemMetrics(
                id=str(uuid4()),
                recorded_at=datetime.utcnow(),
                db_connection_count=5,
                db_query_time_avg_ms=15.5,
                db_size_mb=2.1,
                vector_db_size_mb=5.8,
                vector_collection_count=1,
                vector_document_count=20,
                api_requests_per_minute=12.5,
                api_response_time_avg_ms=850,
                api_error_rate=0.02,
                memory_usage_mb=256,
                cpu_usage_percent=15.2,
                openai_requests_count=45,
                openai_tokens_used=8500,
                openai_cost=0.017,
                openai_avg_response_time_ms=1100,
                upload_dir_size_mb=25.3,
                log_dir_size_mb=1.2
            )
            
            session.add(today_analytics)
            session.add(system_metrics)
            await session.commit()
            
            logger.info("✅ Created sample analytics data")
            
        except Exception as e:
            await session.rollback()
            logger.error(f"❌ Failed to create sample analytics: {str(e)}")
            raise


async def main():
    """Main seeding function."""
    try:
        logger.info("🌱 Starting database seeding...")
        
        # Create sample data
        users = await create_sample_users()
        documents = await create_sample_documents()
        chat_sessions = await create_sample_chat_sessions(users)
        await create_sample_analytics()
        
        logger.info("✅ Database seeding completed successfully!")
        logger.info(f"Created {len(users)} users, {len(documents)} documents, {len(chat_sessions)} chat sessions")
        
    except Exception as e:
        logger.error(f"❌ Database seeding failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
