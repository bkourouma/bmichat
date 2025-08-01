"""
BMI Chat Application - Database Configuration

This module handles SQLite database configuration and initialization.
It provides async database operations using SQLAlchemy with aiosqlite.
"""

from typing import AsyncGenerator

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from loguru import logger

from app.config import settings


# Create async engine
async_engine = create_async_engine(
    f"sqlite+aiosqlite:///{settings.db_sqlite_path}",
    echo=settings.debug,
    future=True,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Create sync engine for migrations and initial setup
sync_engine = create_engine(
    f"sqlite:///{settings.db_sqlite_path}",
    echo=settings.debug,
    future=True,
)

# Import the declarative base from models
from app.models.base import Base

# Metadata for table creation
metadata = MetaData()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.
    
    Yields:
        AsyncSession: Database session
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_database() -> None:
    """
    Initialize the database by creating all tables.
    
    This function should be called during application startup.
    """
    try:
        logger.info("🗄️ Initializing database...")
        
        # Import all models to ensure they are registered
        from app.models import (  # noqa: F401
            User, Document, DocumentChunk, ChatSession, ChatMessage,
            DailyAnalytics, SystemMetrics
        )
        
        # Create all tables
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all, checkfirst=True)

        logger.info("✅ Database initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {str(e)}")
        # Don't raise the error if tables already exist
        if "already exists" not in str(e):
            raise


async def close_database() -> None:
    """
    Close database connections.
    
    This function should be called during application shutdown.
    """
    try:
        logger.info("🔒 Closing database connections...")
        await async_engine.dispose()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"❌ Error closing database: {str(e)}")


def create_tables_sync() -> None:
    """
    Create tables synchronously (for testing or initial setup).
    """
    try:
        logger.info("🗄️ Creating tables synchronously...")
        
        # Import all models
        from app.models import (  # noqa: F401
            User, Document, DocumentChunk, ChatSession, ChatMessage,
            DailyAnalytics, SystemMetrics
        )
        
        # Create all tables
        Base.metadata.create_all(bind=sync_engine)
        
        logger.info("✅ Tables created successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to create tables: {str(e)}")
        raise


def drop_tables_sync() -> None:
    """
    Drop all tables synchronously (for testing).
    """
    try:
        logger.warning("🗑️ Dropping all tables...")
        Base.metadata.drop_all(bind=sync_engine)
        logger.info("✅ Tables dropped successfully")
    except Exception as e:
        logger.error(f"❌ Failed to drop tables: {str(e)}")
        raise
