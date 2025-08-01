"""
BMI Chat Application - Health Check Router

This module provides health check endpoints for monitoring application status,
database connectivity, and external service availability.
"""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from loguru import logger

from app.config import settings
from app.core.database import get_db


router = APIRouter()


@router.get("/health", summary="Basic health check")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns:
        Dict containing basic application status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
        "environment": settings.environment,
        "service": settings.app_name
    }


@router.get("/health/detailed", summary="Detailed health check")
async def detailed_health_check(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """
    Detailed health check with database and service connectivity.
    
    Args:
        db: Database session
        
    Returns:
        Dict containing detailed application status
        
    Raises:
        HTTPException: If any critical service is unavailable
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
        "environment": settings.environment,
        "service": settings.app_name,
        "checks": {}
    }
    
    # Check database connectivity
    try:
        await db.execute(text("SELECT 1"))
        health_status["checks"]["database"] = {
            "status": "healthy",
            "message": "Database connection successful"
        }
        logger.debug("✅ Database health check passed")
    except Exception as e:
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}"
        }
        health_status["status"] = "unhealthy"
        logger.error(f"❌ Database health check failed: {str(e)}")
    
    # Check OpenAI API connectivity
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        # Simple API call to check connectivity
        models = await client.models.list()
        if models:
            health_status["checks"]["openai"] = {
                "status": "healthy",
                "message": "OpenAI API connection successful"
            }
            logger.debug("✅ OpenAI health check passed")
        else:
            raise Exception("No models returned from OpenAI API")
            
    except Exception as e:
        health_status["checks"]["openai"] = {
            "status": "unhealthy",
            "message": f"OpenAI API connection failed: {str(e)}"
        }
        health_status["status"] = "degraded"
        logger.error(f"❌ OpenAI health check failed: {str(e)}")
    
    # Check vector database
    try:
        from app.services.vector_service import VectorService
        vector_service = VectorService()
        
        # Simple check to see if vector service is available
        collection_count = await vector_service.get_collection_count()
        health_status["checks"]["vector_database"] = {
            "status": "healthy",
            "message": f"Vector database available with {collection_count} collections"
        }
        logger.debug("✅ Vector database health check passed")
        
    except Exception as e:
        health_status["checks"]["vector_database"] = {
            "status": "unhealthy",
            "message": f"Vector database check failed: {str(e)}"
        }
        health_status["status"] = "degraded"
        logger.error(f"❌ Vector database health check failed: {str(e)}")
    
    # Check file system permissions
    try:
        import os
        from pathlib import Path
        
        # Check if we can write to upload directory
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        test_file = upload_dir / "health_check.tmp"
        test_file.write_text("health check")
        test_file.unlink()
        
        health_status["checks"]["filesystem"] = {
            "status": "healthy",
            "message": "File system access successful"
        }
        logger.debug("✅ File system health check passed")
        
    except Exception as e:
        health_status["checks"]["filesystem"] = {
            "status": "unhealthy",
            "message": f"File system check failed: {str(e)}"
        }
        health_status["status"] = "degraded"
        logger.error(f"❌ File system health check failed: {str(e)}")
    
    # Return appropriate HTTP status
    if health_status["status"] == "unhealthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=health_status
        )
    
    return health_status


@router.get("/health/ready", summary="Readiness check")
async def readiness_check(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """
    Kubernetes-style readiness check.
    
    Args:
        db: Database session
        
    Returns:
        Dict indicating if the service is ready to accept traffic
    """
    try:
        # Check database
        await db.execute(text("SELECT 1"))
        
        # Check if required directories exist
        from pathlib import Path
        required_dirs = [
            Path(settings.upload_dir),
            Path(settings.vector_db_path),
            Path(settings.log_file).parent
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                raise Exception(f"Required directory does not exist: {directory}")
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Readiness check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not_ready",
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get("/health/live", summary="Liveness check")
async def liveness_check() -> Dict[str, Any]:
    """
    Kubernetes-style liveness check.
    
    Returns:
        Dict indicating if the service is alive
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": "unknown"  # Could be enhanced to track actual uptime
    }
