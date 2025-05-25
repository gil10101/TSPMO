"""
Structured logging system for TSPMO Smart Stock Forecasting System.

This module provides a comprehensive logging setup with structured formatting,
multiple handlers, and configuration based on environment settings.
"""

import logging
import logging.config
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from pythonjsonlogger import jsonlogger

from .config import get_settings


class StructuredFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        if not log_record.get('timestamp'):
            log_record['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Add service information
        log_record['service'] = 'tspmo'
        log_record['version'] = get_settings().app.app_version
        log_record['environment'] = get_settings().environment
        
        # Add level name
        if log_record.get('level'):
            log_record['level'] = log_record['level'].upper()
        else:
            log_record['level'] = record.levelname
        
        # Add module and function information
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add process and thread information
        log_record['process_id'] = record.process
        log_record['thread_id'] = record.thread


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output in development."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Add color to level name
        level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"
        
        # Format the message
        formatted = super().format(record)
        
        return formatted


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_json: bool = False,
    enable_console: bool = True
) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_json: Enable JSON structured logging
        enable_console: Enable console logging
    """
    settings = get_settings()
    
    # Use provided log level or default from settings
    if log_level is None:
        log_level = settings.app.log_level
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        
        if settings.is_development and not enable_json:
            # Use colored formatter for development
            console_formatter = ColoredFormatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            # Use JSON formatter for production or when explicitly requested
            console_formatter = StructuredFormatter()
        
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file or settings.is_production:
        if log_file is None:
            log_file = logs_dir / f"tspmo_{datetime.now(timezone.utc).strftime('%Y%m%d')}.log"
        
        file_handler = logging.FileHandler(log_file)
        
        # Always use JSON formatter for file logging
        file_formatter = StructuredFormatter()
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(file_handler)
    
    # Error file handler (separate file for errors)
    if settings.is_production:
        error_file = logs_dir / f"tspmo_errors_{datetime.now(timezone.utc).strftime('%Y%m%d')}.log"
        error_handler = logging.FileHandler(error_file)
        error_formatter = StructuredFormatter()
        error_handler.setFormatter(error_formatter)
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
    
    # Set specific logger levels
    _configure_logger_levels()
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging system initialized",
        extra={
            "log_level": log_level,
            "environment": settings.environment,
            "json_logging": enable_json or settings.is_production,
            "console_logging": enable_console,
            "file_logging": log_file is not None or settings.is_production
        }
    )


def _configure_logger_levels() -> None:
    """Configure specific logger levels to reduce noise."""
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)
    
    # Set specific levels for our modules
    logging.getLogger("tspmo.api").setLevel(logging.INFO)
    logging.getLogger("tspmo.ml").setLevel(logging.INFO)
    logging.getLogger("tspmo.data").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")


def log_execution_time(func):
    """Decorator to log function execution time."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.info(
                f"Function {func.__name__} executed successfully",
                extra={
                    "function": func.__name__,
                    "execution_time": execution_time,
                    "module": func.__module__
                }
            )
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(
                f"Function {func.__name__} failed",
                extra={
                    "function": func.__name__,
                    "execution_time": execution_time,
                    "module": func.__module__,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise
    
    return wrapper


def log_async_execution_time(func):
    """Decorator to log async function execution time."""
    import functools
    import time
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.info(
                f"Async function {func.__name__} executed successfully",
                extra={
                    "function": func.__name__,
                    "execution_time": execution_time,
                    "module": func.__module__,
                    "async": True
                }
            )
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            
            logger.error(
                f"Async function {func.__name__} failed",
                extra={
                    "function": func.__name__,
                    "execution_time": execution_time,
                    "module": func.__module__,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "async": True
                }
            )
            raise
    
    return wrapper


class ContextualLogger:
    """Logger with contextual information."""
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        """
        Initialize contextual logger.
        
        Args:
            logger: Base logger instance
            context: Contextual information to include in all log messages
        """
        self.logger = logger
        self.context = context
    
    def _log_with_context(self, level: int, message: str, *args, **kwargs) -> None:
        """Log message with context."""
        extra = kwargs.get('extra', {})
        extra.update(self.context)
        kwargs['extra'] = extra
        self.logger.log(level, message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message with context."""
        self._log_with_context(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message with context."""
        self._log_with_context(logging.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message with context."""
        self._log_with_context(logging.CRITICAL, message, *args, **kwargs)


def get_contextual_logger(name: str, context: Dict[str, Any]) -> ContextualLogger:
    """
    Get a contextual logger with additional context information.
    
    Args:
        name: Logger name
        context: Contextual information to include in all log messages
        
    Returns:
        ContextualLogger: Logger with context
    """
    base_logger = get_logger(name)
    return ContextualLogger(base_logger, context)


# Initialize logging on module import
if not logging.getLogger().handlers:
    setup_logging() 