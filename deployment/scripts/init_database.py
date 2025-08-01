#!/usr/bin/env python3
"""
BMI Chat Application - Database Initialization Script

This script initializes the database with tables and optional seed data.
Can be run standalone or as part of the deployment process.
"""

import asyncio
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import settings
from app.core.database import init_database, create_tables_sync
from app.models import *  # Import all models
from loguru import logger


async def main():
    """Main initialization function."""
    try:
        logger.info("🚀 Starting database initialization...")
        logger.info(f"Database path: {settings.db_sqlite_path}")
        logger.info(f"Environment: {settings.environment}")
        
        # Ensure database directory exists
        db_path = Path(settings.db_sqlite_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database asynchronously
        await init_database()
        
        logger.info("✅ Database initialization completed successfully!")
        
        # Print database info
        logger.info(f"📍 Database location: {db_path.absolute()}")
        logger.info(f"📊 Database size: {db_path.stat().st_size if db_path.exists() else 0} bytes")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {str(e)}")
        sys.exit(1)


def sync_main():
    """Synchronous initialization function."""
    try:
        logger.info("🚀 Starting synchronous database initialization...")
        logger.info(f"Database path: {settings.db_sqlite_path}")
        
        # Ensure database directory exists
        db_path = Path(settings.db_sqlite_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create tables synchronously
        create_tables_sync()
        
        logger.info("✅ Synchronous database initialization completed!")
        
    except Exception as e:
        logger.error(f"❌ Synchronous database initialization failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Initialize BMI Chat database")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous initialization (for testing)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of existing database"
    )
    
    args = parser.parse_args()
    
    if args.force:
        db_path = Path(settings.db_sqlite_path)
        if db_path.exists():
            logger.warning(f"🗑️ Removing existing database: {db_path}")
            db_path.unlink()
    
    if args.sync:
        sync_main()
    else:
        asyncio.run(main())
