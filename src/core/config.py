"""
Configuration management for TSPMO Smart Stock Forecasting System.

This module provides Pydantic-based settings with validation for all system
components including API keys, database connections, ML models, and data sources.
"""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field, field_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # SQLite settings (development)
    sqlite_url: str = Field(
        default="sqlite:///./data/tspmo.db",
        description="SQLite database URL for development"
    )
    
    # PostgreSQL settings (production)
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_user: str = Field(default="tspmo", description="PostgreSQL username")
    postgres_password: SecretStr = Field(default="", description="PostgreSQL password")
    postgres_db: str = Field(default="tspmo", description="PostgreSQL database name")
    
    # Connection settings
    pool_size: int = Field(default=10, description="Database connection pool size")
    max_overflow: int = Field(default=20, description="Maximum connection overflow")
    pool_timeout: int = Field(default=30, description="Connection pool timeout")
    
    @property
    def postgres_url(self) -> str:
        """Generate PostgreSQL connection URL."""
        password = self.postgres_password.get_secret_value()
        return f"postgresql://{self.postgres_user}:{password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def async_postgres_url(self) -> str:
        """Generate async PostgreSQL connection URL."""
        password = self.postgres_password.get_secret_value()
        return f"postgresql+asyncpg://{self.postgres_user}:{password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"


class RedisSettings(BaseSettings):
    """Redis cache configuration settings."""
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: Optional[SecretStr] = Field(default=None, description="Redis password")
    db: int = Field(default=0, description="Redis database number")
    max_connections: int = Field(default=20, description="Maximum Redis connections")
    
    @property
    def url(self) -> str:
        """Generate Redis connection URL."""
        auth = ""
        if self.password:
            auth = f":{self.password.get_secret_value()}@"
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class APISettings(BaseSettings):
    """External API configuration settings."""
    
    # Alpha Vantage (primary data source)
    alpha_vantage_api_key: SecretStr = Field(
        default="",
        description="Alpha Vantage API key"
    )
    alpha_vantage_base_url: str = Field(
        default="https://www.alphavantage.co/query",
        description="Alpha Vantage API base URL"
    )
    
    # Yahoo Finance (backup data source)
    yahoo_finance_enabled: bool = Field(
        default=True,
        description="Enable Yahoo Finance as backup data source"
    )
    
    # Federal Reserve Economic Data
    fred_api_key: Optional[SecretStr] = Field(
        default=None,
        description="FRED API key for economic data"
    )
    
    # Rate limiting
    requests_per_minute: int = Field(
        default=5,
        description="API requests per minute limit"
    )
    retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for failed API calls"
    )
    retry_delay: float = Field(
        default=1.0,
        description="Delay between retry attempts in seconds"
    )


class MLModelSettings(BaseSettings):
    """Machine Learning model configuration settings."""
    
    # Model types
    enabled_models: List[str] = Field(
        default=["chronos", "lstm", "lightgbm"],
        description="List of enabled ML models"
    )
    
    # Chronos-T5 settings
    chronos_model_name: str = Field(
        default="amazon/chronos-t5-small",
        description="Chronos model name from HuggingFace"
    )
    chronos_context_length: int = Field(
        default=512,
        description="Context length for Chronos model"
    )
    chronos_prediction_length: int = Field(
        default=24,
        description="Prediction length for Chronos model"
    )
    
    # LSTM settings
    lstm_sequence_length: int = Field(
        default=60,
        description="LSTM input sequence length"
    )
    lstm_hidden_size: int = Field(
        default=128,
        description="LSTM hidden layer size"
    )
    lstm_num_layers: int = Field(
        default=2,
        description="Number of LSTM layers"
    )
    lstm_dropout: float = Field(
        default=0.2,
        description="LSTM dropout rate"
    )
    
    # LightGBM settings
    lgb_num_leaves: int = Field(
        default=31,
        description="LightGBM number of leaves"
    )
    lgb_learning_rate: float = Field(
        default=0.1,
        description="LightGBM learning rate"
    )
    lgb_feature_fraction: float = Field(
        default=0.9,
        description="LightGBM feature fraction"
    )
    
    # Training settings
    train_test_split: float = Field(
        default=0.8,
        description="Train/test split ratio"
    )
    validation_split: float = Field(
        default=0.2,
        description="Validation split ratio"
    )
    batch_size: int = Field(
        default=32,
        description="Training batch size"
    )
    max_epochs: int = Field(
        default=100,
        description="Maximum training epochs"
    )
    early_stopping_patience: int = Field(
        default=10,
        description="Early stopping patience"
    )
    
    @field_validator("enabled_models")
    @classmethod
    def validate_enabled_models(cls, v):
        """Validate enabled models list."""
        valid_models = {"chronos", "lstm", "lightgbm"}
        for model in v:
            if model not in valid_models:
                raise ValueError(f"Invalid model: {model}. Must be one of {valid_models}")
        return v


class RiskSettings(BaseSettings):
    """Risk management configuration settings."""
    
    # Position sizing
    max_position_size: float = Field(
        default=0.1,
        description="Maximum position size as fraction of portfolio"
    )
    max_portfolio_risk: float = Field(
        default=0.02,
        description="Maximum portfolio risk per trade"
    )
    
    # Stop loss and take profit
    default_stop_loss: float = Field(
        default=0.05,
        description="Default stop loss percentage"
    )
    default_take_profit: float = Field(
        default=0.10,
        description="Default take profit percentage"
    )
    
    # Risk metrics
    var_confidence_level: float = Field(
        default=0.95,
        description="Value at Risk confidence level"
    )
    max_drawdown_threshold: float = Field(
        default=0.15,
        description="Maximum drawdown threshold"
    )
    
    # Correlation limits
    max_correlation: float = Field(
        default=0.7,
        description="Maximum correlation between positions"
    )


class AppSettings(BaseSettings):
    """Application-level configuration settings."""
    
    # Application info
    app_name: str = Field(
        default="TSPMO Smart Stock Forecasting",
        description="Application name"
    )
    app_version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    debug: bool = Field(
        default=False,
        description="Debug mode flag"
    )
    
    # API settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")
    
    # Dashboard settings
    dashboard_host: str = Field(default="0.0.0.0", description="Dashboard host")
    dashboard_port: int = Field(default=8501, description="Dashboard port")
    
    # Security
    secret_key: SecretStr = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT tokens"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration time in minutes"
    )
    
    # Data settings
    data_retention_days: int = Field(
        default=365,
        description="Data retention period in days"
    )
    prediction_horizon_days: int = Field(
        default=30,
        description="Default prediction horizon in days"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""
    
    # Environment
    environment: str = Field(
        default="development",
        description="Application environment"
    )
    
    # Sub-settings
    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api: APISettings = Field(default_factory=APISettings)
    ml: MLModelSettings = Field(default_factory=MLModelSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment setting."""
        valid_envs = {"development", "testing", "staging", "production"}
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return v.lower()
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    @property
    def database_url(self) -> str:
        """Get appropriate database URL based on environment."""
        if self.is_production:
            return self.database.postgres_url
        return self.database.sqlite_url
    
    @property
    def async_database_url(self) -> str:
        """Get appropriate async database URL based on environment."""
        if self.is_production:
            return self.database.async_postgres_url
        return self.database.sqlite_url.replace("sqlite://", "sqlite+aiosqlite://")


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Returns:
        Settings: Application configuration settings
    """
    return Settings()


def get_config_path() -> Path:
    """
    Get the configuration directory path.
    
    Returns:
        Path: Path to the config directory
    """
    return Path(__file__).parent.parent.parent / "config"


def load_yaml_config(filename: str) -> Dict:
    """
    Load YAML configuration file.
    
    Args:
        filename: Name of the YAML file to load
        
    Returns:
        Dict: Configuration data from YAML file
    """
    import yaml
    
    config_path = get_config_path() / filename
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) 