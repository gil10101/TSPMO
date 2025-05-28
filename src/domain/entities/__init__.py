"""
Domain entities package for TSPMO Smart Stock Forecasting System.

This package contains the core domain entities that represent the fundamental
business concepts in the stock forecasting domain.

Domain Entity Modules:
- market_data.py: Stock, OHLCV, Volume models
- prediction.py: Forecast, Confidence models  
- portfolio.py: Position, Holdings models
- risk.py: Risk metrics, Limits models
"""

# Market Data Entities
from .market_data import (
    # Core entities
    Stock,
    OHLCV,
    Volume,
    MarketDataSummary,
    
    # Enumerations
    MarketType,
    Exchange,
    Currency,
    TradingStatus,
    
    # Factory functions
    create_stock,
    create_ohlcv,
    validate_ohlcv_sequence,
)

# Prediction Entities (to be implemented)
# from .prediction import (
#     # Core entities
#     Prediction,
#     Forecast,
#     Confidence,
#     PredictionResult,
#     ModelPrediction,
#     
#     # Enumerations
#     PredictionType,
#     ModelType,
#     ConfidenceLevel,
#     ForecastHorizon,
#     
#     # Factory functions
#     create_prediction,
#     create_forecast,
#     validate_prediction_sequence,
# )

# Portfolio Entities (to be implemented)
# from .portfolio import (
#     # Core entities
#     Portfolio,
#     Position,
#     Holdings,
#     Trade,
#     Order,
#     
#     # Enumerations
#     PositionType,
#     OrderType,
#     OrderStatus,
#     TradeDirection,
#     
#     # Factory functions
#     create_portfolio,
#     create_position,
#     create_trade,
# )

# Risk Entities (to be implemented)
# from .risk import (
#     # Core entities
#     RiskMetrics,
#     RiskLimits,
#     RiskAssessment,
#     VaRCalculation,
#     DrawdownMetrics,
#     
#     # Enumerations
#     RiskLevel,
#     RiskType,
#     RiskMeasure,
#     
#     # Factory functions
#     create_risk_metrics,
#     create_risk_limits,
#     assess_portfolio_risk,
# )

# Currently available exports (market_data only)
__all__ = [
    # Market Data entities
    "Stock",
    "OHLCV", 
    "Volume",
    "MarketDataSummary",
    
    # Market Data enumerations
    "MarketType",
    "Exchange", 
    "Currency",
    "TradingStatus",
    
    # Market Data factory functions
    "create_stock",
    "create_ohlcv",
    "validate_ohlcv_sequence",
    
    # Prediction entities (to be added when implemented)
    # "Prediction",
    # "Forecast", 
    # "Confidence",
    # "PredictionResult",
    # "ModelPrediction",
    # "PredictionType",
    # "ModelType",
    # "ConfidenceLevel",
    # "ForecastHorizon",
    # "create_prediction",
    # "create_forecast",
    # "validate_prediction_sequence",
    
    # Portfolio entities (to be added when implemented)
    # "Portfolio",
    # "Position",
    # "Holdings",
    # "Trade",
    # "Order",
    # "PositionType",
    # "OrderType",
    # "OrderStatus",
    # "TradeDirection",
    # "create_portfolio",
    # "create_position",
    # "create_trade",
    
    # Risk entities (to be added when implemented)
    # "RiskMetrics",
    # "RiskLimits",
    # "RiskAssessment",
    # "VaRCalculation",
    # "DrawdownMetrics",
    # "RiskLevel",
    # "RiskType",
    # "RiskMeasure",
    # "create_risk_metrics",
    # "create_risk_limits",
    # "assess_portfolio_risk",
]


def get_available_entities():
    """
    Get list of currently available entity types.
    
    Returns:
        dict: Dictionary of available entity categories and their types
    """
    return {
        "market_data": [
            "Stock", "OHLCV", "Volume", "MarketDataSummary"
        ],
        "prediction": [],  # To be populated when implemented
        "portfolio": [],   # To be populated when implemented
        "risk": []         # To be populated when implemented
    }


def get_entity_info():
    """
    Get information about the domain entities package.
    
    Returns:
        dict: Package information and status
    """
    return {
        "package": "src.domain.entities",
        "description": "Domain entities for Smart Stock Forecasting System",
        "implemented_modules": ["market_data"],
        "planned_modules": ["prediction", "portfolio", "risk"],
        "total_entities": len(__all__),
        "available_entities": get_available_entities()
    }


# Future imports helper function
def _prepare_future_imports():
    """
    Helper function to prepare for future entity imports.
    This will be called when additional entity modules are implemented.
    
    Usage:
        When implementing new entity modules, uncomment the relevant
        import statements above and add the new entities to __all__.
    """
    # This function will be updated as new entity modules are added
    # Placeholder for dynamic import logic if needed
    pass

"""

1. prediction.py:
   - Prediction: Core prediction entity with confidence intervals
   - Forecast: Time series forecast with multiple horizons  
   - Confidence: Confidence metrics and uncertainty quantification
   - PredictionResult: Aggregated prediction results
   - ModelPrediction: Individual model prediction results

2. portfolio.py:
   - Portfolio: Portfolio entity with holdings and performance
   - Position: Individual stock position with entry/exit data
   - Holdings: Current holdings with market values
   - Trade: Trade execution entity with timestamps
   - Order: Order management entity with status tracking

3. risk.py:
   - RiskMetrics: Comprehensive risk calculations (VaR, CVaR, etc.)
   - RiskLimits: Risk limit definitions and enforcement
   - RiskAssessment: Portfolio risk assessment results
   - VaRCalculation: Value at Risk specific calculations
   - DrawdownMetrics: Drawdown analysis and metrics

IMPLEMENTATION GUIDELINES:
Each module should follow the same patterns as market_data.py:
- Pydantic BaseModel with ConfigDict(frozen=True) for immutability
- Comprehensive field validation with custom validators
- Computed fields for derived properties
- Factory functions for entity creation
- Business logic methods within entities
- Proper error handling with domain exceptions
- Rich string representations and hash functions
- Comprehensive docstrings following Google style

""" 