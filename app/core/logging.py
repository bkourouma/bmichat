"""
BMI Chat Application - Logging Configuration

This module configures logging for the BMI Chat application using Loguru.
It provides structured logging with different levels for development and production.
"""

import sys
from pathlib import Path
from typing import Dict, Any

from loguru import logger

from app.config import settings


def setup_logging() -> None:
    """
    Configure application logging using Loguru.
    
    Sets up different logging configurations for development and production:
    - Development: Console output with DEBUG level
    - Production: File output with WARNING level + structured JSON format
    """
    # Remove default logger
    logger.remove()
    
    # Console logging configuration
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # File logging configuration
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
    
    # JSON format for production
    json_format = "{time} | {level} | {name}:{function}:{line} | {message}"
    
    if settings.is_development:
        # Development logging: Console with DEBUG level
        logger.add(
            sys.stdout,
            format=console_format,
            level="DEBUG",
            colorize=True,
            backtrace=True,
            diagnose=True,
        )
        
        # Also log to file in development for debugging
        logger.add(
            settings.log_file,
            format=file_format,
            level="DEBUG",
            rotation=settings.log_max_size,
            retention=f"{settings.log_backup_count} days",
            compression="zip",
            backtrace=True,
            diagnose=True,
        )
        
    else:
        # Production logging: File only with WARNING level
        logger.add(
            sys.stdout,
            format=console_format,
            level="INFO",
            colorize=False,
            backtrace=False,
            diagnose=False,
        )
        
        logger.add(
            settings.log_file,
            format=json_format,
            level=settings.log_level,
            rotation=settings.log_max_size,
            retention=f"{settings.log_backup_count} days",
            compression="zip",
            serialize=True,  # JSON format
            backtrace=False,
            diagnose=False,
        )
    
    # Configure specific loggers
    configure_external_loggers()
    
    logger.info(f"🔧 Logging configured for {settings.environment} environment")


def configure_external_loggers() -> None:
    """Configure logging for external libraries."""
    import logging
    
    # Suppress noisy loggers in production
    if settings.is_production:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("chromadb").setLevel(logging.WARNING)
    
    # Intercept standard logging and redirect to loguru
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            
            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1
            
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )
    
    # Replace standard logging handlers
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def get_logger(name: str) -> Any:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(module=name)


def log_function_call(func_name: str, **kwargs) -> None:
    """
    Log function calls with parameters.
    
    Args:
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    if settings.debug:
        logger.debug(f"🔧 Calling {func_name} with params: {kwargs}")


def log_performance(operation: str, duration: float, **context) -> None:
    """
    Log performance metrics.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        **context: Additional context information
    """
    logger.info(
        f"⏱️ {operation} completed in {duration:.3f}s",
        extra={"operation": operation, "duration": duration, **context}
    )


def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """
    Log errors with context.
    
    Args:
        error: Exception that occurred
        context: Additional context information
    """
    context = context or {}
    logger.error(
        f"❌ Error: {str(error)}",
        extra={"error_type": type(error).__name__, **context}
    )
