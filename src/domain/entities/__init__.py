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
    DataQuality,  # Added missing enum
    
    # Factory functions
    create_stock,
    create_ohlcv,
    create_volume,  # Added missing factory function
    validate_ohlcv_sequence,
    calculate_summary_statistics,  # Added missing calculation function
)

# Prediction Entities
from .prediction import (
    # Core entities
    Forecast,
    Confidence,
    PredictionResult,
    ModelPrediction,
    
    # Enumerations
    PredictionType,
    ModelType,
    ConfidenceLevel,
    ForecastHorizon,
    PredictionStatus,
    UncertaintyMethod,
    
    # Factory functions
    create_confidence,
    create_forecast,
    create_forecast_async,  # Added async factory function
    create_model_prediction,
    create_ensemble_prediction,
    
    # Validation functions
    validate_prediction_sequence,
    validate_finite_number,  # Added utility validation function
    
    # Calculation functions
    calculate_prediction_metrics,
    
    # Utility functions
    safe_decimal_conversion,  # Added utility function
    sanitize_string_input,    # Added security utility function
    
    # Statistical validation (for advanced use cases)
    BasicStatisticalValidator,  # Added statistical validator class
)

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
#     PortfolioType,
#     RebalanceStrategy,
#     
#     # Factory functions
#     create_portfolio,
#     create_position,
#     create_trade,
#     create_order,
#     create_holdings,
#     
#     # Calculation functions
#     calculate_portfolio_metrics,
#     calculate_position_sizing,
#     calculate_rebalance_weights,
#     
#     # Validation functions
#     validate_portfolio_constraints,
#     validate_position_limits,
# )

# Risk Entities (to be implemented)
# from .risk import (
#     # Core entities
#     RiskMetrics,
#     RiskLimits,
#     RiskAssessment,
#     VaRCalculation,
#     DrawdownMetrics,
#     CorrelationMatrix,
#     StressTestResult,
#     
#     # Enumerations
#     RiskLevel,
#     RiskType,
#     RiskMeasure,
#     VaRMethod,
#     StressTestType,
#     
#     # Factory functions
#     create_risk_metrics,
#     create_risk_limits,
#     create_var_calculation,
#     assess_portfolio_risk,
#     
#     # Calculation functions
#     calculate_var,
#     calculate_cvar,
#     calculate_maximum_drawdown,
#     calculate_correlation_matrix,
#     calculate_beta,
#     calculate_sharpe_ratio,
#     
#     # Validation functions
#     validate_risk_parameters,
#     validate_correlation_matrix,
# )

# Currently available exports (market_data and prediction)
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
    "DataQuality",  # Added missing enum
    
    # Market Data factory functions
    "create_stock",
    "create_ohlcv",
    "create_volume",  # Added missing factory function
    "validate_ohlcv_sequence",
    "calculate_summary_statistics",  # Added missing calculation function
    
    # Prediction entities
    "Forecast",
    "Confidence", 
    "PredictionResult",
    "ModelPrediction",
    
    # Prediction enumerations
    "PredictionType",
    "ModelType",
    "ConfidenceLevel",
    "ForecastHorizon",
    "PredictionStatus",
    "UncertaintyMethod",
    
    # Prediction factory functions
    "create_confidence",
    "create_forecast",
    "create_forecast_async",  # Added async factory function
    "create_model_prediction",
    "create_ensemble_prediction",
    
    # Prediction validation functions
    "validate_prediction_sequence",
    "validate_finite_number",  # Added utility validation function
    
    # Prediction calculation functions
    "calculate_prediction_metrics",
    
    # Prediction utility functions
    "safe_decimal_conversion",  # Added utility function
    "sanitize_string_input",    # Added security utility function
    "BasicStatisticalValidator",  # Added statistical validator class
    
    # Future Portfolio entities (to be added when implemented)
    # "Portfolio",
    # "Position",
    # "Holdings",
    # "Trade",
    # "Order",
    # "PositionType",
    # "OrderType",
    # "OrderStatus",
    # "TradeDirection",
    # "PortfolioType",
    # "RebalanceStrategy",
    # "create_portfolio",
    # "create_position",
    # "create_trade",
    # "create_order",
    # "create_holdings",
    # "calculate_portfolio_metrics",
    # "calculate_position_sizing",
    # "calculate_rebalance_weights",
    # "validate_portfolio_constraints",
    # "validate_position_limits",
    
    # Future Risk entities (to be added when implemented)
    # "RiskMetrics",
    # "RiskLimits",
    # "RiskAssessment",
    # "VaRCalculation",
    # "DrawdownMetrics",
    # "CorrelationMatrix",
    # "StressTestResult",
    # "RiskLevel",
    # "RiskType",
    # "RiskMeasure",
    # "VaRMethod",
    # "StressTestType",
    # "create_risk_metrics",
    # "create_risk_limits",
    # "create_var_calculation",
    # "assess_portfolio_risk",
    # "calculate_var",
    # "calculate_cvar",
    # "calculate_maximum_drawdown",
    # "calculate_correlation_matrix",
    # "calculate_beta",
    # "calculate_sharpe_ratio",
    # "validate_risk_parameters",
    # "validate_correlation_matrix",
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
        "prediction": [
            "Forecast", "Confidence", "PredictionResult", "ModelPrediction"
        ],
        "portfolio": [],   # To be populated when implemented
        "risk": []         # To be populated when implemented
    }


def get_available_enums():
    """
    Get list of currently available enumerations.
    
    Returns:
        dict: Dictionary of available enumeration categories and their types
    """
    return {
        "market_data": [
            "MarketType", "Exchange", "Currency", "TradingStatus", "DataQuality"
        ],
        "prediction": [
            "PredictionType", "ModelType", "ConfidenceLevel", "ForecastHorizon",
            "PredictionStatus", "UncertaintyMethod"
        ],
        "portfolio": [],   # To be populated when implemented
        "risk": []         # To be populated when implemented
    }


def get_available_factory_functions():
    """
    Get list of currently available factory functions.
    
    Returns:
        dict: Dictionary of available factory function categories and their functions
    """
    return {
        "market_data": [
            "create_stock", "create_ohlcv", "create_volume"
        ],
        "prediction": [
            "create_confidence", "create_forecast", "create_forecast_async",
            "create_model_prediction", "create_ensemble_prediction"
        ],
        "portfolio": [],   # To be populated when implemented
        "risk": []         # To be populated when implemented
    }


def get_available_validation_functions():
    """
    Get list of currently available validation functions.
    
    Returns:
        dict: Dictionary of available validation function categories and their functions
    """
    return {
        "market_data": [
            "validate_ohlcv_sequence"
        ],
        "prediction": [
            "validate_prediction_sequence", "validate_finite_number"
        ],
        "portfolio": [],   # To be populated when implemented
        "risk": []         # To be populated when implemented
    }


def get_available_calculation_functions():
    """
    Get list of currently available calculation functions.
    
    Returns:
        dict: Dictionary of available calculation function categories and their functions
    """
    return {
        "market_data": [
            "calculate_summary_statistics"
        ],
        "prediction": [
            "calculate_prediction_metrics"
        ],
        "portfolio": [],   # To be populated when implemented
        "risk": []         # To be populated when implemented
    }


def get_entity_info():
    """
    Get comprehensive information about the domain entities package.
    
    Returns:
        dict: Package information and status
    """
    return {
        "package": "src.domain.entities",
        "description": "Domain entities for Smart Stock Forecasting System",
        "implemented_modules": ["market_data", "prediction"],
        "planned_modules": ["portfolio", "risk"],
        "total_entities": len([item for item in __all__ if not item.startswith('create_') and not item.startswith('validate_') and not item.startswith('calculate_')]),
        "available_entities": get_available_entities(),
        "available_enums": get_available_enums(),
        "available_factory_functions": get_available_factory_functions(),
        "available_validation_functions": get_available_validation_functions(),
        "available_calculation_functions": get_available_calculation_functions(),
        "implementation_status": {
            "market_data": "✓ COMPLETED",
            "prediction": "✓ COMPLETED", 
            "portfolio": "⏳ PLANNED",
            "risk": "⏳ PLANNED"
        }
    }


def get_module_dependencies():
    """
    Get dependency information between modules.
    
    Returns:
        dict: Module dependency information
    """
    return {
        "market_data": {
            "depends_on": ["src.core.exceptions"],
            "imported_by": ["prediction", "portfolio", "risk"]
        },
        "prediction": {
            "depends_on": ["src.core.exceptions", "market_data"],
            "imported_by": ["portfolio", "risk"]
        },
        "portfolio": {
            "depends_on": ["src.core.exceptions", "market_data", "prediction"],
            "imported_by": ["risk"]
        },
        "risk": {
            "depends_on": ["src.core.exceptions", "market_data", "prediction", "portfolio"],
            "imported_by": []
        }
    }


# Future imports helper function
def _prepare_future_imports():
    """
    Helper function to prepare for future entity imports.
    This will be called when additional entity modules are implemented.
    
    Usage:
        When implementing new entity modules:
        1. Uncomment the relevant import statements above
        2. Add the new entities to __all__
        3. Update the get_available_* functions
        4. Test the imports
        
    Implementation Order (recommended):
        1. portfolio.py - Core portfolio management entities
        2. risk.py - Risk assessment and management entities
    """
    pass


# Validation helper for cross-entity consistency
def validate_entity_consistency():
    """
    Validate consistency across all entity types.
    
    This function performs cross-entity validation to ensure
    that related entities are consistent with each other.
    
    Returns:
        bool: True if all entities are consistent
    
    Raises:
        ValidationError: If inconsistencies are found
    """
    # TODO: Implement cross-entity validation when more modules are available
    # Examples:
    # - Validate that Portfolio entities reference valid Stock entities
    # - Validate that Prediction entities reference valid Stock entities
    # - Validate that Risk entities reference valid Portfolio entities
    return True


def check_import_health():
    """
    Check the health of all imports and return status.
    
    Returns:
        dict: Import health status for each module
    """
    status = {}
    
    # Check market_data imports
    try:
        from .market_data import Stock, OHLCV, Volume
        status["market_data"] = "✓ HEALTHY"
    except ImportError as e:
        status["market_data"] = f"✗ ERROR: {e}"
    
    # Check prediction imports
    try:
        from .prediction import Forecast, Confidence, PredictionResult
        status["prediction"] = "✓ HEALTHY"
    except ImportError as e:
        status["prediction"] = f"✗ ERROR: {e}"
    
    # Future modules
    status["portfolio"] = "⏳ NOT IMPLEMENTED"
    status["risk"] = "⏳ NOT IMPLEMENTED"
    
    return status

"""

ENTITY IMPLEMENTATION ROADMAP:

1. portfolio.py (Next Priority):
   - Portfolio: Core portfolio entity with holdings and performance tracking
   - Position: Individual stock position with entry/exit data and P&L
   - Holdings: Current holdings with real-time market values
   - Trade: Trade execution entity with timestamps and fees
   - Order: Order management entity with status tracking and execution
   - PositionType: Enum for long/short/options positions
   - OrderType: Enum for market/limit/stop orders
   - OrderStatus: Enum for pending/filled/cancelled orders
   - TradeDirection: Enum for buy/sell/cover directions

2. risk.py (Final Priority):
   - RiskMetrics: Comprehensive risk calculations (VaR, CVaR, Sharpe, etc.)
   - RiskLimits: Risk limit definitions and enforcement rules
   - RiskAssessment: Portfolio risk assessment results and recommendations
   - VaRCalculation: Value at Risk specific calculations with multiple methods
   - DrawdownMetrics: Drawdown analysis and maximum drawdown tracking
   - CorrelationMatrix: Asset correlation calculations
   - StressTestResult: Stress testing scenarios and results

IMPLEMENTATION GUIDELINES:
Each module should follow the same patterns as market_data.py and prediction.py:
- Pydantic BaseModel with ConfigDict(frozen=True) for immutability
- Comprehensive field validation with custom validators
- Computed fields for derived properties  
- Factory functions for entity creation with type conversion
- Business logic methods within entities
- Proper error handling with domain exceptions
- Rich string representations and hash functions
- Comprehensive docstrings following Google style
- Performance optimizations with caching where appropriate
- Async support for expensive operations
- Statistical validation and NaN handling
- Enhanced security and input sanitization
""" 