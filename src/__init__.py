"""
TSPMO Smart Stock Forecasting System.

Technical Stock Prediction and Management Operations (TSPMO) is a
stock forecasting system built with Domain-Driven Design principles.
"""

__version__ = "1.0.0"
__author__ = "TSPMO Development Team"
__description__ = "Technical Stock Prediction and Management Operations"

# Import core components for easy access
from .core import (
    # Configuration
    Settings,
    get_settings,
    # Exceptions
    TSPMOException,
    ConfigurationError,
    DataValidationError,
    ModelError,
    APIError,
    DatabaseError,
    # Logging
    get_logger,
    setup_logging,
    # Constants
    DEFAULT_PREDICTION_HORIZON,
    SUPPORTED_SYMBOLS,
    MODEL_TYPES,
    RISK_LEVELS,
    DATA_SOURCES,
)

__all__ = [
    # Package metadata
    "__version__",
    "__author__", 
    "__description__",
    # Configuration
    "Settings",
    "get_settings",
    # Exceptions
    "TSPMOException",
    "ConfigurationError",
    "DataValidationError", 
    "ModelError",
    "APIError",
    "DatabaseError",
    # Logging
    "get_logger",
    "setup_logging",
    # Constants
    "DEFAULT_PREDICTION_HORIZON",
    "SUPPORTED_SYMBOLS",
    "MODEL_TYPES",
    "RISK_LEVELS",
    "DATA_SOURCES",
] 