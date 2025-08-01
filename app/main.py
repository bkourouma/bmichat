"""
BMI Chat Application - FastAPI Main Application

This is the main entry point for the BMI Chat FastAPI application.
It sets up the FastAPI app with all necessary middleware, routers, and configurations.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import settings
from app.core.exceptions import BMIChatException
from app.core.logging import setup_logging
from app.routers import chat, documents, health, widget, search


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Starting BMI Chat application...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"OpenAI Model: {settings.openai_model}")
    
    # Initialize database
    from app.core.database import init_database
    await init_database()
    
    # Initialize vector database
    from app.services.vector_service import VectorService
    vector_service = VectorService()
    await vector_service.initialize()
    
    logger.info("✅ Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down BMI Chat application...")
    logger.info("✅ Application shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    # Setup logging
    setup_logging()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # Add middleware
    setup_middleware(app)
    
    # Add routers
    setup_routers(app)
    
    # Add static files
    setup_static_files(app)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    return app


def setup_middleware(app: FastAPI) -> None:
    """
    Configure application middleware.
    
    Args:
        app: FastAPI application instance
    """
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware (production only)
    if settings.is_production:
        allowed_hosts = ["bmi.engage-360.net", "www.bmi.engage-360.net", "localhost", "127.0.0.1"]
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=allowed_hosts
        )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all HTTP requests."""
        start_time = logger.bind(request_id=id(request)).info(
            f"🌐 {request.method} {request.url.path}"
        )
        
        response = await call_next(request)
        
        logger.bind(request_id=id(request)).info(
            f"✅ {request.method} {request.url.path} - {response.status_code}"
        )
        
        return response


def setup_routers(app: FastAPI) -> None:
    """
    Configure application routers.
    
    Args:
        app: FastAPI application instance
    """
    # Health check router (no prefix)
    app.include_router(health.router, tags=["Health"])
    
    # API routers with /api prefix
    app.include_router(chat.router, prefix="/api", tags=["Chat"])
    app.include_router(documents.router, prefix="/api", tags=["Documents"])
    app.include_router(search.router, prefix="/api", tags=["Search"])

    # Widget router (no prefix for embedding)
    app.include_router(widget.router, prefix="/widget", tags=["Widget"])


def setup_static_files(app: FastAPI) -> None:
    """
    Configure static file serving.
    
    Args:
        app: FastAPI application instance
    """
    # Serve frontend build files
    try:
        app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")
        logger.info("📁 Static files mounted at /static")
    except RuntimeError:
        logger.warning("⚠️ Frontend dist directory not found, skipping static files")
    
    # Serve widget files
    try:
        app.mount("/widget", StaticFiles(directory="widget/dist"), name="widget")
        logger.info("🔧 Widget files mounted at /widget")
    except RuntimeError:
        logger.warning("⚠️ Widget dist directory not found, skipping widget files")


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Configure global exception handlers.
    
    Args:
        app: FastAPI application instance
    """
    @app.exception_handler(BMIChatException)
    async def bmi_chat_exception_handler(request: Request, exc: BMIChatException):
        """Handle custom BMI Chat exceptions."""
        logger.error(f"BMI Chat Exception: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "details": exc.details
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {str(exc)}")
        
        if settings.debug:
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An internal server error occurred",
                "details": str(exc) if settings.debug else None
            }
        )


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.api_workers,
        log_level=settings.log_level.lower(),
    )
