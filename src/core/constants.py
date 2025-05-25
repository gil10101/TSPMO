"""
System-wide constants for TSPMO Smart Stock Forecasting System.

This module contains all the constants used throughout the application
including supported symbols, model configurations, risk parameters, and data sources.
"""

from enum import Enum
from typing import Any, Dict, List, Set


# Application Constants
APP_NAME = "TSPMO Smart Stock Forecasting"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Technical Stock Prediction and Management Operations"

# Default Configuration
DEFAULT_PREDICTION_HORIZON = 30  # days
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEQUENCE_LENGTH = 60

# Supported Stock Symbols (Major US Stocks)
SUPPORTED_SYMBOLS: Set[str] = {
    # Technology
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "META", "NVDA", "NFLX", "ADBE",
    "CRM", "ORCL", "INTC", "AMD", "PYPL", "UBER", "LYFT", "ZOOM", "SHOP", "SQ",
    
    # Financial
    "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "COF",
    "AXP", "BLK", "SCHW", "CB", "AIG", "MET", "PRU", "ALL", "TRV", "PGR",
    
    # Healthcare
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN",
    "GILD", "CVS", "ANTM", "CI", "HUM", "CNC", "WBA", "CVX", "XOM", "COP",
    
    # Consumer
    "KO", "PEP", "WMT", "HD", "MCD", "SBUX", "NKE", "DIS", "CMCSA", "VZ",
    "T", "COST", "TGT", "LOW", "TJX", "BKNG", "MAR", "HLT", "YUM", "CMG",
    
    # Industrial
    "BA", "CAT", "GE", "MMM", "HON", "UPS", "FDX", "LMT", "RTX", "NOC",
    "DE", "EMR", "ETN", "ITW", "PH", "ROK", "DOV", "XYL", "FTV", "AME",
    
    # ETFs
    "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "AGG", "BND", "TLT",
    "GLD", "SLV", "USO", "XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLU"
}

# Model Types
class ModelType(Enum):
    """Enumeration of supported model types."""
    CHRONOS = "chronos"
    LSTM = "lstm"
    LIGHTGBM = "lightgbm"
    ENSEMBLE = "ensemble"


MODEL_TYPES: Dict[str, str] = {
    "chronos": "Chronos-T5 Time Series Model",
    "lstm": "Long Short-Term Memory Neural Network",
    "lightgbm": "Light Gradient Boosting Machine",
    "ensemble": "Weighted Ensemble Model"
}

# Risk Levels
class RiskLevel(Enum):
    """Enumeration of risk levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


RISK_LEVELS: Dict[str, Dict[str, float]] = {
    "conservative": {
        "max_position_size": 0.05,
        "max_portfolio_risk": 0.01,
        "stop_loss": 0.03,
        "take_profit": 0.06,
        "max_correlation": 0.5
    },
    "moderate": {
        "max_position_size": 0.10,
        "max_portfolio_risk": 0.02,
        "stop_loss": 0.05,
        "take_profit": 0.10,
        "max_correlation": 0.7
    },
    "aggressive": {
        "max_position_size": 0.20,
        "max_portfolio_risk": 0.05,
        "stop_loss": 0.08,
        "take_profit": 0.15,
        "max_correlation": 0.8
    }
}

# Data Sources
class DataSource(Enum):
    """Enumeration of data sources."""
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO_FINANCE = "yahoo_finance"
    FRED = "fred"
    WEBSOCKET = "websocket"


DATA_SOURCES: Dict[str, Dict[str, str]] = {
    "alpha_vantage": {
        "name": "Alpha Vantage",
        "type": "REST API",
        "description": "Primary market data source",
        "base_url": "https://www.alphavantage.co/query"
    },
    "yahoo_finance": {
        "name": "Yahoo Finance",
        "type": "Python Library",
        "description": "Backup market data source",
        "base_url": "https://finance.yahoo.com"
    },
    "fred": {
        "name": "Federal Reserve Economic Data",
        "type": "REST API",
        "description": "Economic indicators and macro data",
        "base_url": "https://api.stlouisfed.org/fred"
    },
    "websocket": {
        "name": "Real-time WebSocket",
        "type": "WebSocket",
        "description": "Real-time market data streaming",
        "base_url": "wss://stream.data.alpaca.markets"
    }
}

# Time Intervals
class TimeInterval(Enum):
    """Enumeration of time intervals."""
    MINUTE_1 = "1min"
    MINUTE_5 = "5min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR_1 = "1hour"
    HOUR_4 = "4hour"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


TIME_INTERVALS: Dict[str, int] = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1hour": 60,
    "4hour": 240,
    "daily": 1440,
    "weekly": 10080,
    "monthly": 43200
}

# Technical Indicators
TECHNICAL_INDICATORS: List[str] = [
    # Trend Indicators
    "SMA", "EMA", "WMA", "DEMA", "TEMA", "TRIMA",
    "KAMA", "MAMA", "T3", "HT_TRENDLINE",
    
    # Momentum Indicators
    "RSI", "STOCH", "STOCHF", "STOCHRSI", "MACD", "MACDEXT", "MACDFIX",
    "PPO", "ROC", "ROCP", "ROCR", "ROCR100", "MOM", "TSF", "CMO",
    "WILLR", "CCI", "DX", "MINUS_DI", "PLUS_DI", "MINUS_DM", "PLUS_DM",
    "BBANDS", "MIDPOINT", "MIDPRICE", "SAR", "SAREXT", "APO", "TRIX",
    "ULTOSC", "AROON", "AROONOSC", "MFI", "ADX", "ADXR",
    
    # Volume Indicators
    "AD", "ADOSC", "OBV", "CHAIKIN", "AVGPRICE", "MEDPRICE", "TYPPRICE",
    "WCLPRICE", "HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR", "HT_SINE",
    "HT_TRENDMODE",
    
    # Volatility Indicators
    "ATR", "NATR", "TRANGE", "STDDEV", "VAR"
]

# Feature Categories
FEATURE_CATEGORIES: Dict[str, List[str]] = {
    "price": ["open", "high", "low", "close", "adjusted_close"],
    "volume": ["volume", "volume_sma", "volume_ema", "volume_ratio"],
    "returns": ["returns", "log_returns", "cumulative_returns"],
    "technical": TECHNICAL_INDICATORS,
    "fundamental": ["pe_ratio", "pb_ratio", "dividend_yield", "market_cap"],
    "sentiment": ["news_sentiment", "social_sentiment", "analyst_rating"],
    "macro": ["interest_rate", "inflation", "gdp_growth", "unemployment"]
}

# Model Parameters
CHRONOS_MODELS: Dict[str, Dict[str, Any]] = {
    "small": {
        "model_name": "amazon/chronos-t5-small",
        "context_length": 512,
        "prediction_length": 24,
        "parameters": "20M"
    },
    "base": {
        "model_name": "amazon/chronos-t5-base",
        "context_length": 512,
        "prediction_length": 24,
        "parameters": "200M"
    },
    "large": {
        "model_name": "amazon/chronos-t5-large",
        "context_length": 512,
        "prediction_length": 24,
        "parameters": "710M"
    }
}

LSTM_PARAMETERS: Dict[str, Any] = {
    "sequence_length": 60,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10
}

LIGHTGBM_PARAMETERS: Dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": 42
}

# API Configuration
API_RATE_LIMITS: Dict[str, Dict[str, int]] = {
    "alpha_vantage": {
        "requests_per_minute": 5,
        "requests_per_day": 500
    },
    "yahoo_finance": {
        "requests_per_minute": 60,
        "requests_per_day": 2000
    },
    "fred": {
        "requests_per_minute": 120,
        "requests_per_day": 1000
    }
}

# Database Configuration
DATABASE_TABLES: List[str] = [
    "market_data",
    "predictions",
    "models",
    "portfolios",
    "positions",
    "risk_metrics",
    "backtests",
    "alerts",
    "users",
    "api_keys"
]

# Cache Configuration
CACHE_KEYS: Dict[str, str] = {
    "market_data": "market_data:{symbol}:{interval}:{date}",
    "predictions": "predictions:{symbol}:{model}:{date}",
    "model_metadata": "model:{model_type}:{version}",
    "portfolio": "portfolio:{user_id}",
    "risk_metrics": "risk:{portfolio_id}:{date}",
    "technical_indicators": "indicators:{symbol}:{interval}:{date}"
}

CACHE_TTL: Dict[str, int] = {
    "market_data": 300,      # 5 minutes
    "predictions": 3600,     # 1 hour
    "model_metadata": 86400, # 24 hours
    "portfolio": 60,         # 1 minute
    "risk_metrics": 1800,    # 30 minutes
    "technical_indicators": 600  # 10 minutes
}

# File Paths
DATA_PATHS: Dict[str, str] = {
    "raw_data": "data/raw",
    "processed_data": "data/processed",
    "models": "data/models",
    "backtest_results": "data/backtest",
    "logs": "logs",
    "config": "config",
    "temp": "temp"
}

# Validation Rules
VALIDATION_RULES: Dict[str, Dict[str, Any]] = {
    "symbol": {
        "pattern": r"^[A-Z]{1,5}$",
        "max_length": 5,
        "allowed_chars": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    },
    "price": {
        "min_value": 0.01,
        "max_value": 100000.0,
        "decimal_places": 2
    },
    "volume": {
        "min_value": 0,
        "max_value": 1000000000
    },
    "percentage": {
        "min_value": -100.0,
        "max_value": 100.0,
        "decimal_places": 4
    },
    "confidence": {
        "min_value": 0.0,
        "max_value": 1.0,
        "decimal_places": 4
    }
}

# Error Codes
ERROR_CODES: Dict[str, str] = {
    "INVALID_SYMBOL": "E001",
    "DATA_NOT_FOUND": "E002",
    "API_RATE_LIMIT": "E003",
    "MODEL_NOT_FOUND": "E004",
    "PREDICTION_FAILED": "E005",
    "INSUFFICIENT_DATA": "E006",
    "INVALID_TIMEFRAME": "E007",
    "RISK_LIMIT_EXCEEDED": "E008",
    "AUTHENTICATION_FAILED": "E009",
    "AUTHORIZATION_FAILED": "E010",
    "DATABASE_ERROR": "E011",
    "CACHE_ERROR": "E012",
    "CONFIGURATION_ERROR": "E013",
    "VALIDATION_ERROR": "E014",
    "SYSTEM_ERROR": "E015"
}

# Status Codes
STATUS_CODES: Dict[str, str] = {
    "PENDING": "pending",
    "RUNNING": "running",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "cancelled",
    "PAUSED": "paused"
}

# Alert Types
ALERT_TYPES: Dict[str, str] = {
    "PRICE_ALERT": "price_alert",
    "VOLUME_ALERT": "volume_alert",
    "PREDICTION_ALERT": "prediction_alert",
    "RISK_ALERT": "risk_alert",
    "MODEL_ALERT": "model_alert",
    "SYSTEM_ALERT": "system_alert"
}

# Performance Metrics
PERFORMANCE_METRICS: List[str] = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "mse",
    "rmse",
    "mae",
    "mape",
    "directional_accuracy",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "information_ratio",
    "treynor_ratio",
    "jensen_alpha",
    "beta",
    "var_95",
    "cvar_95"
]

# Default Values
DEFAULT_VALUES: Dict[str, Any] = {
    "prediction_horizon": DEFAULT_PREDICTION_HORIZON,
    "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
    "batch_size": DEFAULT_BATCH_SIZE,
    "sequence_length": DEFAULT_SEQUENCE_LENGTH,
    "risk_level": RiskLevel.MODERATE.value,
    "model_type": ModelType.ENSEMBLE.value,
    "time_interval": TimeInterval.DAILY.value,
    "data_source": DataSource.ALPHA_VANTAGE.value
} 