"""
Domain-specific exceptions for TSPMO Smart Stock Forecasting System.

This module defines a comprehensive hierarchy of exceptions that can occur
throughout the application, providing clear error handling and debugging information.
"""

from typing import Any, Dict, Optional


class TSPMOException(Exception):
    """Base exception class for all TSPMO-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize TSPMO exception.
        
        Args:
            message: Human-readable error message
            error_code: Unique error code for programmatic handling
            context: Additional context information
            original_exception: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.original_exception = original_exception
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        base_msg = f"[{self.error_code}] {self.message}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg += f" (Context: {context_str})"
        return base_msg
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "original_exception": str(self.original_exception) if self.original_exception else None
        }


# Configuration and Settings Exceptions
class ConfigurationError(TSPMOException):
    """Raised when there's an error in system configuration."""
    pass


class EnvironmentError(ConfigurationError):
    """Raised when environment variables are missing or invalid."""
    pass


class SettingsValidationError(ConfigurationError):
    """Raised when settings validation fails."""
    pass


# Data-related Exceptions
class DataError(TSPMOException):
    """Base class for data-related errors."""
    pass


class DataValidationError(DataError):
    """Raised when data validation fails."""
    pass


class DataCollectionError(DataError):
    """Raised when data collection from external sources fails."""
    pass


class DataProcessingError(DataError):
    """Raised when data processing operations fail."""
    pass


class DataStorageError(DataError):
    """Raised when data storage operations fail."""
    pass


class DataNotFoundError(DataError):
    """Raised when requested data is not found."""
    pass


class DataQualityError(DataError):
    """Raised when data quality checks fail."""
    pass


# API and External Service Exceptions
class APIError(TSPMOException):
    """Base class for API-related errors."""
    pass


class ExternalAPIError(APIError):
    """Raised when external API calls fail."""
    
    def __init__(
        self,
        message: str,
        api_name: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize external API error.
        
        Args:
            message: Error message
            api_name: Name of the external API
            status_code: HTTP status code
            response_data: Response data from the API
            **kwargs: Additional arguments for base exception
        """
        context = kwargs.get('context', {})
        context.update({
            "api_name": api_name,
            "status_code": status_code,
            "response_data": response_data
        })
        kwargs['context'] = context
        super().__init__(message, **kwargs)
        self.api_name = api_name
        self.status_code = status_code
        self.response_data = response_data


class RateLimitError(APIError):
    """Raised when API rate limits are exceeded."""
    pass


class AuthenticationError(APIError):
    """Raised when API authentication fails."""
    pass


class AuthorizationError(APIError):
    """Raised when API authorization fails."""
    pass


# Database Exceptions
class DatabaseError(TSPMOException):
    """Base class for database-related errors."""
    pass


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class QueryError(DatabaseError):
    """Raised when database query execution fails."""
    pass


class TransactionError(DatabaseError):
    """Raised when database transaction fails."""
    pass


class MigrationError(DatabaseError):
    """Raised when database migration fails."""
    pass


# Machine Learning Exceptions
class ModelError(TSPMOException):
    """Base class for machine learning model errors."""
    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""
    pass


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""
    pass


class ModelValidationError(ModelError):
    """Raised when model validation fails."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found."""
    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails."""
    pass


class ModelSaveError(ModelError):
    """Raised when model saving fails."""
    pass


class FeatureEngineeringError(ModelError):
    """Raised when feature engineering fails."""
    pass


class HyperparameterError(ModelError):
    """Raised when hyperparameter optimization fails."""
    pass


# Portfolio and Risk Management Exceptions
class PortfolioError(TSPMOException):
    """Base class for portfolio-related errors."""
    pass


class InsufficientFundsError(PortfolioError):
    """Raised when there are insufficient funds for an operation."""
    pass


class InvalidPositionError(PortfolioError):
    """Raised when an invalid position is requested."""
    pass


class RiskLimitExceededError(PortfolioError):
    """Raised when risk limits are exceeded."""
    pass


class PositionSizingError(PortfolioError):
    """Raised when position sizing calculations fail."""
    pass


# Business Logic Exceptions
class BusinessLogicError(TSPMOException):
    """Base class for business logic errors."""
    pass


class InvalidSymbolError(BusinessLogicError):
    """Raised when an invalid stock symbol is provided."""
    pass


class InvalidTimeframeError(BusinessLogicError):
    """Raised when an invalid timeframe is specified."""
    pass


class PredictionError(BusinessLogicError):
    """Raised when prediction generation fails."""
    pass


class BacktestError(BusinessLogicError):
    """Raised when backtesting fails."""
    pass


# Cache and Performance Exceptions
class CacheError(TSPMOException):
    """Base class for cache-related errors."""
    pass


class CacheConnectionError(CacheError):
    """Raised when cache connection fails."""
    pass


class CacheOperationError(CacheError):
    """Raised when cache operations fail."""
    pass


# Validation and Input Exceptions
class ValidationError(TSPMOException):
    """Base class for validation errors."""
    pass


class InputValidationError(ValidationError):
    """Raised when input validation fails."""
    pass


class SchemaValidationError(ValidationError):
    """Raised when schema validation fails."""
    pass


# System and Infrastructure Exceptions
class SystemError(TSPMOException):
    """Base class for system-level errors."""
    pass


class ResourceError(SystemError):
    """Raised when system resources are unavailable."""
    pass


class TimeoutError(SystemError):
    """Raised when operations timeout."""
    pass


class ServiceUnavailableError(SystemError):
    """Raised when a required service is unavailable."""
    pass


# Utility functions for exception handling
def handle_external_exception(
    exc: Exception,
    context: Optional[Dict[str, Any]] = None,
    error_mapping: Optional[Dict[type, type]] = None
) -> TSPMOException:
    """
    Convert external exceptions to TSPMO exceptions.
    
    Args:
        exc: Original exception
        context: Additional context information
        error_mapping: Mapping of external exception types to TSPMO exception types
        
    Returns:
        TSPMOException: Converted TSPMO exception
    """
    if isinstance(exc, TSPMOException):
        return exc
    
    # Default error mapping
    default_mapping = {
        ValueError: DataValidationError,
        TypeError: DataValidationError,
        KeyError: DataNotFoundError,
        FileNotFoundError: DataNotFoundError,
        ConnectionRefusedError: ConnectionError,
        TimeoutError: TimeoutError,
    }
    
    # Use provided mapping or default
    mapping = error_mapping or default_mapping
    
    # Find appropriate TSPMO exception type
    tspmo_exc_type = mapping.get(type(exc), TSPMOException)
    
    return tspmo_exc_type(
        message=str(exc),
        context=context,
        original_exception=exc
    )


def create_error_response(
    exc: TSPMOException,
    include_traceback: bool = False
) -> Dict[str, Any]:
    """
    Create standardized error response from exception.
    
    Args:
        exc: TSPMO exception
        include_traceback: Whether to include traceback information
        
    Returns:
        Dict: Standardized error response
    """
    import traceback
    
    response = {
        "error": True,
        "error_type": exc.__class__.__name__,
        "error_code": exc.error_code,
        "message": exc.message,
        "context": exc.context
    }
    
    if include_traceback:
        response["traceback"] = traceback.format_exc()
    
    return response


def log_exception(
    exc: Exception,
    logger,
    context: Optional[Dict[str, Any]] = None,
    level: str = "error"
) -> None:
    """
    Log exception with proper formatting and context.
    
    Args:
        exc: Exception to log
        logger: Logger instance
        context: Additional context information
        level: Log level (debug, info, warning, error, critical)
    """
    import traceback
    
    log_context = {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }
    
    if context:
        log_context.update(context)
    
    if isinstance(exc, TSPMOException):
        log_context.update({
            "error_code": exc.error_code,
            "tspmo_context": exc.context
        })
    
    log_method = getattr(logger, level.lower(), logger.error)
    log_method(
        f"Exception occurred: {exc}",
        extra=log_context,
        exc_info=True
    ) 