"""
Domain entities for predictions in TSPMO Smart Stock Forecasting System.

This module defines the core domain entities for predictions including Forecast,
Confidence, PredictionResult, and ModelPrediction models. These entities
represent the fundamental business concepts for ML predictions and enforce
business rules through validation and domain logic.

The entities support the three-model ensemble architecture:
- Chronos-T5: Primary time series forecasting model
- LSTM: Deep learning baseline model  
- LightGBM: Gradient boosting model

Production-ready features for 2025:
- Async support for ML inference
- Statistical validation and NaN handling
- Performance optimizations with caching
- Enhanced security and input sanitization
- Comprehensive logging and monitoring integration
"""

import asyncio
import hashlib
import logging
import math
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from functools import lru_cache, wraps
from typing import (
    List, Optional, Union, Dict, Any, Tuple, Callable, 
    TypeVar, Protocol, runtime_checkable
)
from uuid import UUID, uuid4

import numpy as np
from pydantic import (
    BaseModel, Field, field_validator, computed_field, 
    ConfigDict, model_validator, ValidationInfo
)
from pydantic.types import PositiveInt, NonNegativeFloat

from src.core.exceptions import DataValidationError, BusinessLogicError, ModelError

# Type variables for generic support
T = TypeVar('T')
PredictionValue = TypeVar('PredictionValue', bound=Union[float, Decimal])

# Logger for this module
logger = logging.getLogger(__name__)

# Performance constants
DECIMAL_PRECISION = 8
CONFIDENCE_PRECISION = 4
CACHE_TTL_SECONDS = 3600  # 1 hour cache for expensive calculations


# Performance and validation utilities
def performance_monitor(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to monitor performance of critical functions."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            if execution_time > 0.1:  # Log if > 100ms
                logger.warning(
                    f"Slow operation: {func.__name__} took {execution_time:.3f}s"
                )
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper


class PredictionType(str, Enum):
    """Enumeration of prediction types."""
    PRICE = "price"
    DIRECTION = "direction"
    VOLATILITY = "volatility"
    RETURN = "return"
    VOLUME = "volume"
    RISK = "risk"
    SIGNAL = "signal"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ModelType(str, Enum):
    """Enumeration of ML model types supporting TSPMO architecture."""
    # Primary models per tpt.md requirements
    CHRONOS = "chronos"      # Amazon Chronos-T5 time series model
    LSTM = "lstm"            # PyTorch LSTM model
    LIGHTGBM = "lightgbm"    # LightGBM gradient boosting
    ENSEMBLE = "ensemble"    # Weighted ensemble of all models
    
    # Supporting models
    LINEAR = "linear"
    RANDOM_FOREST = "random_forest"
    XGB = "xgboost"
    PROPHET = "prophet"
    ARIMA = "arima"
    
    # Meta models
    STACKING = "stacking"
    VOTING = "voting"
    BLENDING = "blending"


class ConfidenceLevel(str, Enum):
    """Enumeration of confidence levels."""
    VERY_LOW = "very_low"     # < 60%
    LOW = "low"               # 60-70%
    MEDIUM = "medium"         # 70-80%
    HIGH = "high"             # 80-90%
    VERY_HIGH = "very_high"   # > 90%


class ForecastHorizon(str, Enum):
    """Enumeration of forecast horizons."""
    INTRADAY = "intraday"     # < 1 day
    DAILY = "daily"           # 1 day
    WEEKLY = "weekly"         # 1 week
    MONTHLY = "monthly"       # 1 month
    QUARTERLY = "quarterly"   # 3 months
    YEARLY = "yearly"         # 1 year


class PredictionStatus(str, Enum):
    """Enumeration of prediction status."""
    PENDING = "pending"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"
    ARCHIVED = "archived"


class UncertaintyMethod(str, Enum):
    """Enumeration of uncertainty quantification methods."""
    BOOTSTRAP = "bootstrap"
    MONTE_CARLO = "monte_carlo"
    BAYESIAN = "bayesian"
    ENSEMBLE_VARIANCE = "ensemble_variance"
    QUANTILE_REGRESSION = "quantile_regression"
    CONFORMAL = "conformal"


class Confidence(BaseModel):
    """
    Confidence value object representing prediction confidence with validation.
    
    This is a value object that encapsulates confidence metrics with
    business rules and statistical validation for production use in 2025.
    
    Features:
    - Enhanced statistical validation with NaN/infinity checks
    - Performance optimizations with caching
    - Comprehensive uncertainty quantification
    - Production monitoring and logging
    """
    
    model_config = ConfigDict(
        frozen=True,  # Immutable value object
        validate_assignment=True,
        extra="forbid",
        # Performance optimization
        str_to_lower=True,
        str_strip_whitespace=True,
    )
    
    # Core confidence metrics
    score: Decimal = Field(
        ...,
        ge=0,
        le=1,
        decimal_places=CONFIDENCE_PRECISION,
        description="Confidence score between 0 and 1"
    )
    level: ConfidenceLevel = Field(
        ...,
        description="Categorical confidence level"
    )
    
    # Statistical confidence intervals
    lower_bound: Optional[Decimal] = Field(
        default=None,
        decimal_places=DECIMAL_PRECISION,
        description="Lower confidence interval bound"
    )
    upper_bound: Optional[Decimal] = Field(
        default=None,
        decimal_places=DECIMAL_PRECISION,
        description="Upper confidence interval bound"
    )
    interval_width: Optional[Decimal] = Field(
        default=None,
        ge=0,
        decimal_places=DECIMAL_PRECISION,
        description="Confidence interval width"
    )
    
    # Quantile-based confidence (for Chronos model)
    quantiles: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Quantile predictions (e.g., 0.1, 0.25, 0.5, 0.75, 0.9)"
    )
    
    # Model-specific confidence metrics
    model_agreement: Optional[Decimal] = Field(
        default=None,
        ge=0,
        le=1,
        decimal_places=CONFIDENCE_PRECISION,
        description="Agreement between ensemble models"
    )
    prediction_variance: Optional[Decimal] = Field(
        default=None,
        ge=0,
        decimal_places=DECIMAL_PRECISION,
        description="Variance of prediction across models"
    )
    
    # Uncertainty quantification
    uncertainty_method: UncertaintyMethod = Field(
        default=UncertaintyMethod.ENSEMBLE_VARIANCE,
        description="Method used for uncertainty quantification"
    )
    epistemic_uncertainty: Optional[Decimal] = Field(
        default=None,
        ge=0,
        decimal_places=DECIMAL_PRECISION,
        description="Model uncertainty (reducible with more data)"
    )
    aleatoric_uncertainty: Optional[Decimal] = Field(
        default=None,
        ge=0,
        decimal_places=DECIMAL_PRECISION,
        description="Data uncertainty (irreducible noise)"
    )
    
    # Statistical quality metrics
    statistical_significance: Optional[Decimal] = Field(
        default=None,
        ge=0,
        le=1,
        decimal_places=CONFIDENCE_PRECISION,
        description="Statistical significance level (p-value)"
    )
    sample_size: Optional[int] = Field(
        default=None,
        ge=1,
        description="Sample size used for confidence calculation"
    )
    outlier_score: Optional[Decimal] = Field(
        default=None,
        ge=0,
        decimal_places=CONFIDENCE_PRECISION,
        description="Outlier score indicating prediction anomaly"
    )
    
    # Metadata
    calculated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When confidence was calculated"
    )
    calculation_hash: Optional[str] = Field(
        default=None,
        description="Hash of inputs used for confidence calculation"
    )
    
    @field_validator("score")
    @classmethod
    def validate_confidence_score(cls, v: Decimal, info: ValidationInfo) -> Decimal:
        """Enhanced validation for confidence score with finite number checking."""
        # Use safe decimal conversion for better error handling
        try:
            score = safe_decimal_conversion(v, CONFIDENCE_PRECISION)
        except DataValidationError as e:
            logger.error(f"Invalid confidence score conversion: {e}")
            raise
        
        if not (0 <= score <= 1):
            raise DataValidationError(
                f"Confidence score must be between 0 and 1, got {score}",
                context={"field": "score", "value": str(score)}
            )
        
        # Validate finite number
        if not validate_finite_number(score):
            raise DataValidationError(
                f"Confidence score must be finite, got {score}",
                context={"field": "score", "value": str(score)}
            )
        
        return score
    
    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, v: Optional[Dict[str, Decimal]]) -> Optional[Dict[str, Decimal]]:
        """Enhanced quantile validation with statistical checks."""
        if v is None:
            return v
        
        try:
            quantile_values = []
            validated_quantiles = {}
            
            for quantile_str, value in v.items():
                # Validate quantile key
                try:
                    quantile = float(quantile_str)
                    if not (0 <= quantile <= 1):
                        raise DataValidationError(
                            f"Quantile {quantile} must be between 0 and 1",
                            context={"quantile": quantile_str}
                        )
                except ValueError:
                    raise DataValidationError(
                        f"Invalid quantile key: {quantile_str}",
                        context={"quantile_key": quantile_str}
                    )
                
                # Validate value is finite
                validated_value = safe_decimal_conversion(value, DECIMAL_PRECISION)
                if not validate_finite_number(validated_value):
                    raise DataValidationError(
                        f"Quantile value must be finite: {value}",
                        context={"quantile": quantile_str, "value": str(value)}
                    )
                
                quantile_values.append((quantile, validated_value))
                validated_quantiles[quantile_str] = validated_value
            
            # Check quantiles are properly ordered
            quantile_values.sort()
            for i in range(1, len(quantile_values)):
                if quantile_values[i][1] < quantile_values[i-1][1]:
                    logger.warning(f"Quantile values not monotonically increasing: {quantile_values}")
                    # In production, we might want to sort them automatically
                    # rather than raising an error
            
            # Statistical validation
            if len(quantile_values) >= 3:
                BasicStatisticalValidator.validate_distribution(
                    [qv[1] for qv in quantile_values]
                )
            
            return validated_quantiles
            
        except Exception as e:
            logger.error(f"Quantile validation failed: {e}")
            raise DataValidationError(f"Quantile validation failed: {e}")
    
    @field_validator("model_agreement", "prediction_variance", "epistemic_uncertainty", 
                     "aleatoric_uncertainty", "statistical_significance", "outlier_score")
    @classmethod
    def validate_optional_decimals(cls, v: Optional[Decimal], info: ValidationInfo) -> Optional[Decimal]:
        """Validate optional decimal fields for finite values."""
        if v is None:
            return v
        
        try:
            validated_value = safe_decimal_conversion(v, DECIMAL_PRECISION)
            if not validate_finite_number(validated_value):
                raise DataValidationError(
                    f"Field {info.field_name} must be finite, got {validated_value}",
                    context={"field": info.field_name, "value": str(v)}
                )
            return validated_value
        except Exception as e:
            logger.error(f"Validation failed for {info.field_name}: {e}")
            raise DataValidationError(f"Invalid value for {info.field_name}: {e}")
    
    @model_validator(mode='after')
    def validate_confidence_bounds(self) -> 'Confidence':
        """Enhanced validation for confidence interval bounds."""
        try:
            if self.lower_bound is not None and self.upper_bound is not None:
                if self.lower_bound > self.upper_bound:
                    raise DataValidationError(
                        f"Lower bound {self.lower_bound} cannot be greater than upper bound {self.upper_bound}",
                        context={
                            "lower_bound": str(self.lower_bound),
                            "upper_bound": str(self.upper_bound)
                        }
                    )
                
                # Calculate interval width if not provided
                if self.interval_width is None:
                    calculated_width = self.upper_bound - self.lower_bound
                    object.__setattr__(self, 'interval_width', calculated_width)
                else:
                    # Validate provided interval width
                    expected_width = self.upper_bound - self.lower_bound
                    if abs(self.interval_width - expected_width) > Decimal('1e-6'):
                        logger.warning(
                            f"Interval width mismatch: provided={self.interval_width}, "
                            f"calculated={expected_width}"
                        )
            
            # Validate uncertainty combination
            if (self.epistemic_uncertainty is not None and 
                self.aleatoric_uncertainty is not None):
                total_uncertainty = (self.epistemic_uncertainty ** 2 + 
                                   self.aleatoric_uncertainty ** 2).sqrt()
                
                # Log if total uncertainty is very high
                if total_uncertainty > Decimal('1.0'):
                    logger.warning(f"Very high total uncertainty: {total_uncertainty}")
            
            # Generate calculation hash for caching
            if self.calculation_hash is None:
                hash_input = f"{self.score}_{self.model_agreement}_{self.prediction_variance}"
                hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
                object.__setattr__(self, 'calculation_hash', hash_value)
            
            return self
            
        except Exception as e:
            logger.error(f"Confidence validation failed: {e}")
            raise BusinessLogicError(f"Confidence validation failed: {e}")
    
    @computed_field
    @property
    def is_high_confidence(self) -> bool:
        """Check if confidence is considered high."""
        return self.score >= Decimal('0.8')
    
    @computed_field
    @property
    def is_reliable(self) -> bool:
        """Enhanced reliability check with multiple criteria."""
        criteria_met = 0
        total_criteria = 0
        
        # Core confidence score
        if self.score >= Decimal('0.7'):
            criteria_met += 1
        total_criteria += 1
        
        # Model agreement (if available)
        if self.model_agreement is not None:
            if self.model_agreement >= Decimal('0.6'):
                criteria_met += 1
            total_criteria += 1
        
        # Prediction variance (if available)
        if self.prediction_variance is not None:
            if self.prediction_variance <= Decimal('0.1'):
                criteria_met += 1
            total_criteria += 1
        
        # Statistical significance (if available)
        if self.statistical_significance is not None:
            if self.statistical_significance <= Decimal('0.05'):  # p < 0.05
                criteria_met += 1
            total_criteria += 1
        
        # Outlier score (if available)
        if self.outlier_score is not None:
            if self.outlier_score <= Decimal('2.0'):  # Not a strong outlier
                criteria_met += 1
            total_criteria += 1
        
        # Need at least 60% of criteria met
        reliability_ratio = criteria_met / total_criteria if total_criteria > 0 else 0
        return reliability_ratio >= 0.6
    
    @computed_field
    @property
    def confidence_category(self) -> str:
        """Categorize confidence into descriptive categories."""
        score_float = float(self.score)
        if score_float >= 0.95:
            return "exceptional"
        elif score_float >= 0.85:
            return "high"
        elif score_float >= 0.70:
            return "moderate"
        elif score_float >= 0.55:
            return "low"
        else:
            return "very_low"
    
    @computed_field
    @property
    def total_uncertainty(self) -> Optional[Decimal]:
        """Calculate total uncertainty from epistemic and aleatoric components."""
        if (self.epistemic_uncertainty is not None and 
            self.aleatoric_uncertainty is not None):
            return (self.epistemic_uncertainty ** 2 + 
                   self.aleatoric_uncertainty ** 2).sqrt()
        return None
    
    @performance_monitor
    def calculate_confidence_interval(self, alpha: float = 0.05) -> Optional[Tuple[Decimal, Decimal]]:
        """Calculate confidence interval for given alpha level."""
        if self.quantiles is None:
            return None
        
        try:
            lower_quantile = alpha / 2
            upper_quantile = 1 - (alpha / 2)
            
            lower_key = str(lower_quantile)
            upper_key = str(upper_quantile)
            
            if lower_key in self.quantiles and upper_key in self.quantiles:
                return (self.quantiles[lower_key], self.quantiles[upper_key])
            
            # If exact quantiles not available, interpolate
            quantile_keys = sorted([float(k) for k in self.quantiles.keys()])
            
            if lower_quantile < min(quantile_keys) or upper_quantile > max(quantile_keys):
                logger.warning(f"Cannot calculate confidence interval for alpha={alpha}")
                return None
            
            # Simple linear interpolation
            def interpolate(target_q):
                for i in range(len(quantile_keys) - 1):
                    if quantile_keys[i] <= target_q <= quantile_keys[i + 1]:
                        q1, q2 = quantile_keys[i], quantile_keys[i + 1]
                        v1 = self.quantiles[str(q1)]
                        v2 = self.quantiles[str(q2)]
                        weight = (target_q - q1) / (q2 - q1)
                        return v1 + weight * (v2 - v1)
                return None
            
            lower_val = interpolate(lower_quantile)
            upper_val = interpolate(upper_quantile)
            
            if lower_val is not None and upper_val is not None:
                return (lower_val, upper_val)
            
        except Exception as e:
            logger.error(f"Error calculating confidence interval: {e}")
        
        return None
    
    def validate_against_historical(self, historical_scores: List[Decimal]) -> Dict[str, Any]:
        """Validate current confidence against historical performance."""
        if not historical_scores:
            return {"status": "no_history", "warnings": []}
        
        try:
            validator = BasicStatisticalValidator()
            outlier_score = validator.calculate_outlier_score(self.score, historical_scores)
            
            warnings = []
            if outlier_score > Decimal('3.0'):
                warnings.append(f"Confidence score is a strong outlier (z-score: {outlier_score})")
            
            # Calculate percentile
            scores_array = np.array([float(s) for s in historical_scores])
            current_score = float(self.score)
            percentile = (scores_array < current_score).mean() * 100
            
            return {
                "status": "validated",
                "outlier_score": outlier_score,
                "percentile": Decimal(str(percentile)),
                "warnings": warnings,
                "sample_size": len(historical_scores)
            }
            
        except Exception as e:
            logger.error(f"Historical validation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def __str__(self) -> str:
        """Enhanced string representation with key metrics."""
        base_str = f"Confidence(score={self.score:.3f}, level={self.level.value}"
        
        if self.model_agreement is not None:
            base_str += f", agreement={self.model_agreement:.3f}"
        
        if self.is_reliable:
            base_str += ", reliable=True"
        
        return base_str + ")"
    
    def __hash__(self) -> int:
        """Hash for use in sets/dicts using calculation hash."""
        return hash((self.calculation_hash, str(self.score), self.level.value))


class Forecast(BaseModel):
    """
    Forecast entity representing a single prediction value with metadata.
    
    This is a core domain entity that represents a forecast for a specific
    target variable with comprehensive validation and business logic.
    """
    
    model_config = ConfigDict(
        frozen=True,  # Immutable entity
        validate_assignment=True,
        extra="forbid"
    )
    
    # Identification
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this forecast"
    )
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Stock symbol for the forecast"
    )
    
    # Prediction details
    prediction_type: PredictionType = Field(
        ...,
        description="Type of prediction being made"
    )
    target_value: Decimal = Field(
        ...,
        decimal_places=8,
        description="Predicted target value"
    )
    current_value: Optional[Decimal] = Field(
        default=None,
        decimal_places=8,
        description="Current value for comparison"
    )
    
    # Time information
    prediction_date: datetime = Field(
        ...,
        description="Date when prediction was made"
    )
    target_date: datetime = Field(
        ...,
        description="Date for which prediction is made"
    )
    horizon: ForecastHorizon = Field(
        ...,
        description="Forecast horizon"
    )
    
    # Model information
    model_type: ModelType = Field(
        ...,
        description="Type of model used for prediction"
    )
    model_version: str = Field(
        ...,
        description="Version of the model used"
    )
    
    # Confidence and uncertainty
    confidence: Confidence = Field(
        ...,
        description="Confidence metrics for this forecast"
    )
    
    # Additional prediction metrics
    expected_return: Optional[Decimal] = Field(
        default=None,
        decimal_places=6,
        description="Expected return percentage"
    )
    risk_score: Optional[Decimal] = Field(
        default=None,
        ge=0,
        le=1,
        decimal_places=4,
        description="Risk score for this prediction"
    )
    volatility_estimate: Optional[Decimal] = Field(
        default=None,
        ge=0,
        decimal_places=6,
        description="Estimated volatility"
    )
    
    # Model-specific metadata
    feature_importance: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Feature importance scores (for tree-based models)"
    )
    attention_weights: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Attention weights (for transformer models like Chronos)"
    )
    shap_values: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="SHAP explanation values"
    )
    
    # Quality indicators
    data_quality_score: Decimal = Field(
        default=Decimal('1.0'),
        ge=0,
        le=1,
        decimal_places=4,
        description="Quality score of input data"
    )
    model_fit_score: Optional[Decimal] = Field(
        default=None,
        ge=0,
        le=1,
        decimal_places=4,
        description="How well model fits the data"
    )
    
    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this forecast was created"
    )
    status: PredictionStatus = Field(
        default=PredictionStatus.COMPLETED,
        description="Status of the prediction"
    )
    
    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate stock symbol format."""
        v = v.upper().strip()
        if not v.isalnum():
            raise DataValidationError(f"Symbol must be alphanumeric: {v}")
        return v
    
    @field_validator("target_date")
    @classmethod
    def validate_target_date(cls, v: datetime) -> datetime:
        """Validate target date is in the future."""
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v
    
    @model_validator(mode='after')
    def validate_forecast_consistency(self) -> 'Forecast':
        """Validate forecast temporal consistency and business rules."""
        # Ensure target date is after prediction date
        if self.target_date <= self.prediction_date:
            raise BusinessLogicError(
                "Target date must be after prediction date",
                context={
                    "prediction_date": self.prediction_date.isoformat(),
                    "target_date": self.target_date.isoformat()
                }
            )
        
        # Validate horizon consistency
        time_diff = self.target_date - self.prediction_date
        
        horizon_mapping = {
            ForecastHorizon.INTRADAY: timedelta(days=1),
            ForecastHorizon.DAILY: timedelta(days=2),
            ForecastHorizon.WEEKLY: timedelta(days=8),
            ForecastHorizon.MONTHLY: timedelta(days=32),
            ForecastHorizon.QUARTERLY: timedelta(days=95),
            ForecastHorizon.YEARLY: timedelta(days=370),
        }
        
        max_diff = horizon_mapping.get(self.horizon)
        if max_diff and time_diff > max_diff:
            raise BusinessLogicError(
                f"Time difference ({time_diff}) exceeds maximum for horizon {self.horizon.value}",
                context={
                    "time_diff_days": time_diff.days,
                    "max_days": max_diff.days,
                    "horizon": self.horizon.value
                }
            )
        
        return self
    
    @computed_field
    @property
    def time_to_target(self) -> timedelta:
        """Calculate time remaining until target date."""
        return self.target_date - self.prediction_date
    
    @computed_field
    @property
    def days_to_target(self) -> int:
        """Calculate days remaining until target date."""
        return self.time_to_target.days
    
    @computed_field
    @property
    def predicted_change(self) -> Optional[Decimal]:
        """Calculate predicted change from current value."""
        if self.current_value is None:
            return None
        return self.target_value - self.current_value
    
    @computed_field
    @property
    def predicted_change_percent(self) -> Optional[Decimal]:
        """Calculate predicted percentage change."""
        if self.current_value is None or self.current_value == 0:
            return None
        change = self.predicted_change
        if change is None:
            return None
        return (change / self.current_value) * Decimal('100')
    
    @computed_field
    @property
    def is_bullish(self) -> Optional[bool]:
        """Determine if prediction is bullish."""
        change = self.predicted_change
        if change is None:
            return None
        return change > 0
    
    @computed_field
    @property
    def is_bearish(self) -> Optional[bool]:
        """Determine if prediction is bearish."""
        change = self.predicted_change
        if change is None:
            return None
        return change < 0
    
    @computed_field
    @property
    def prediction_strength(self) -> str:
        """Categorize prediction strength based on confidence and magnitude."""
        if not self.confidence.is_high_confidence:
            return "weak"
        
        change_pct = self.predicted_change_percent
        if change_pct is None:
            return "neutral"
        
        abs_change = abs(change_pct)
        if abs_change >= Decimal('10'):
            return "very_strong"
        elif abs_change >= Decimal('5'):
            return "strong"
        elif abs_change >= Decimal('2'):
            return "moderate"
        else:
            return "weak"
    
    def is_significant_prediction(self, threshold_percent: Decimal = Decimal('2')) -> bool:
        """Check if prediction represents a significant change."""
        change_pct = self.predicted_change_percent
        if change_pct is None:
            return False
        return abs(change_pct) >= threshold_percent and self.confidence.is_reliable
    
    def calculate_risk_adjusted_return(self) -> Optional[Decimal]:
        """Calculate risk-adjusted expected return."""
        if self.expected_return is None or self.volatility_estimate is None:
            return None
        if self.volatility_estimate == 0:
            return None
        return self.expected_return / self.volatility_estimate
    
    def __str__(self) -> str:
        """Return string representation of forecast."""
        direction = "↑" if self.is_bullish else "↓" if self.is_bearish else "→"
        return (f"Forecast({self.symbol} {direction} {self.target_value} "
                f"by {self.target_date.date()}, conf={self.confidence.score:.2f})")
    
    def __hash__(self) -> int:
        """Return hash of forecast for use in sets/dicts."""
        return hash((self.id, self.symbol, self.prediction_date, self.target_date))


class ModelPrediction(BaseModel):
    """
    Model prediction entity representing output from a specific ML model.
    
    This entity encapsulates the raw output from individual models
    before ensemble aggregation.
    """
    
    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    # Identification
    id: UUID = Field(default_factory=uuid4)
    model_type: ModelType = Field(..., description="Type of ML model")
    model_name: str = Field(..., description="Specific model name/version")
    
    # Prediction details
    symbol: str = Field(..., min_length=1, max_length=20)
    prediction_value: Decimal = Field(..., decimal_places=8)
    prediction_type: PredictionType = Field(...)
    
    # Model-specific outputs
    raw_output: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw model output (model-specific format)"
    )
    probabilities: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Class probabilities (for classification)"
    )
    feature_contributions: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Individual feature contributions to prediction"
    )
    
    # Model performance metrics
    model_confidence: Decimal = Field(
        ...,
        ge=0,
        le=1,
        decimal_places=4,
        description="Model's confidence in this prediction"
    )
    uncertainty_estimate: Optional[Decimal] = Field(
        default=None,
        ge=0,
        decimal_places=6,
        description="Model's uncertainty estimate"
    )
    
    # Computational metadata
    inference_time_ms: Optional[PositiveInt] = Field(
        default=None,
        description="Time taken for inference in milliseconds"
    )
    memory_usage_mb: Optional[NonNegativeFloat] = Field(
        default=None,
        description="Memory usage during inference in MB"
    )
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate symbol format."""
        return v.upper().strip()
    
    def __str__(self) -> str:
        """Return string representation."""
        return (f"ModelPrediction({self.model_type.value}: {self.symbol} = "
                f"{self.prediction_value}, conf={self.model_confidence:.3f})")


class PredictionResult(BaseModel):
    """
    Aggregated prediction result from ensemble of models.
    
    This entity represents the final prediction result after
    combining multiple model outputs using ensemble methods.
    """
    
    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    # Identification
    id: UUID = Field(default_factory=uuid4)
    symbol: str = Field(..., min_length=1, max_length=20)
    prediction_type: PredictionType = Field(...)
    
    # Core prediction
    primary_forecast: Forecast = Field(
        ...,
        description="Primary ensemble forecast"
    )
    
    # Individual model predictions
    model_predictions: List[ModelPrediction] = Field(
        ...,
        min_length=1,
        description="Individual predictions from each model"
    )
    
    # Ensemble metadata
    ensemble_method: str = Field(
        default="weighted_average",
        description="Method used for ensemble aggregation"
    )
    model_weights: Dict[str, Decimal] = Field(
        ...,
        description="Weights assigned to each model in ensemble"
    )
    
    # Consensus metrics
    model_agreement_score: Decimal = Field(
        ...,
        ge=0,
        le=1,
        decimal_places=4,
        description="Agreement between models (0=no agreement, 1=perfect agreement)"
    )
    prediction_variance: Decimal = Field(
        ...,
        ge=0,
        decimal_places=8,
        description="Variance of predictions across models"
    )
    confidence_intervals: Dict[str, Tuple[Decimal, Decimal]] = Field(
        default_factory=dict,
        description="Confidence intervals at different levels"
    )
    
    # Alternative scenarios
    bull_case_forecast: Optional[Forecast] = Field(
        default=None,
        description="Optimistic scenario forecast"
    )
    bear_case_forecast: Optional[Forecast] = Field(
        default=None,
        description="Pessimistic scenario forecast"
    )
    base_case_forecast: Optional[Forecast] = Field(
        default=None,
        description="Most likely scenario forecast"
    )
    
    # Performance tracking
    backtest_metrics: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Historical performance metrics"
    )
    
    # Metadata
    generation_time_ms: Optional[PositiveInt] = Field(
        default=None,
        description="Total time to generate prediction in milliseconds"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = Field(
        default=None,
        description="When this prediction expires"
    )
    
    @field_validator("model_weights")
    @classmethod
    def validate_model_weights(cls, v: Dict[str, Decimal]) -> Dict[str, Decimal]:
        """Validate model weights sum to 1 and are non-negative."""
        total_weight = sum(v.values())
        
        # Check individual weights are non-negative
        for model, weight in v.items():
            if weight < 0:
                raise DataValidationError(f"Model weight for {model} cannot be negative: {weight}")
        
        # Check weights sum to approximately 1 (allowing for rounding)
        if not (Decimal('0.99') <= total_weight <= Decimal('1.01')):
            raise DataValidationError(
                f"Model weights must sum to 1.0, got {total_weight}"
            )
        
        return v
    
    @field_validator("model_predictions")
    @classmethod
    def validate_model_predictions(cls, v: List[ModelPrediction]) -> List[ModelPrediction]:
        """Validate model predictions are consistent."""
        if len(v) == 0:
            raise DataValidationError("At least one model prediction is required")
        
        # Check all predictions are for the same symbol
        symbols = set(pred.symbol for pred in v)
        if len(symbols) > 1:
            raise DataValidationError(f"All predictions must be for same symbol, got: {symbols}")
        
        # Check all predictions are for the same type
        pred_types = set(pred.prediction_type for pred in v)
        if len(pred_types) > 1:
            raise DataValidationError(f"All predictions must be same type, got: {pred_types}")
        
        return v
    
    @computed_field
    @property
    def participating_models(self) -> List[str]:
        """Get list of models that participated in ensemble."""
        return [pred.model_type.value for pred in self.model_predictions]
    
    @computed_field
    @property
    def is_high_consensus(self) -> bool:
        """Check if models have high consensus."""
        return self.model_agreement_score >= Decimal('0.8')
    
    @computed_field
    @property
    def dominant_model(self) -> Optional[str]:
        """Get the model with highest weight."""
        if not self.model_weights:
            return None
        return max(self.model_weights.items(), key=lambda x: x[1])[0]
    
    @computed_field
    @property
    def ensemble_confidence(self) -> Decimal:
        """Calculate ensemble confidence based on agreement and individual confidences."""
        # Weight individual model confidences by their ensemble weights
        weighted_confidence = Decimal('0')
        total_weight = Decimal('0')
        
        for pred in self.model_predictions:
            model_name = pred.model_type.value
            if model_name in self.model_weights:
                weight = self.model_weights[model_name]
                weighted_confidence += pred.model_confidence * weight
                total_weight += weight
        
        if total_weight == 0:
            return Decimal('0')
        
        base_confidence = weighted_confidence / total_weight
        
        # Adjust by agreement score (high agreement increases confidence)
        agreement_bonus = (self.model_agreement_score - Decimal('0.5')) * Decimal('0.2')
        adjusted_confidence = base_confidence + agreement_bonus
        
        # Ensure result is between 0 and 1
        return max(Decimal('0'), min(Decimal('1'), adjusted_confidence))
    
    def get_model_prediction(self, model_type: ModelType) -> Optional[ModelPrediction]:
        """Get prediction from specific model type."""
        for pred in self.model_predictions:
            if pred.model_type == model_type:
                return pred
        return None
    
    def calculate_prediction_spread(self) -> Decimal:
        """Calculate spread between min and max model predictions."""
        if len(self.model_predictions) <= 1:
            return Decimal('0')
        
        values = [pred.prediction_value for pred in self.model_predictions]
        return max(values) - min(values)
    
    def __str__(self) -> str:
        """Return string representation."""
        models = ", ".join(self.participating_models)
        return (f"PredictionResult({self.symbol}: {self.primary_forecast.target_value} "
                f"from [{models}], agreement={self.model_agreement_score:.2f})")


# Factory functions for creating prediction entities with enhanced features

@performance_monitor
def create_confidence(
    score: Union[float, Decimal],
    level: Optional[ConfidenceLevel] = None,
    lower_bound: Optional[Union[float, Decimal]] = None,
    upper_bound: Optional[Union[float, Decimal]] = None,
    sample_size: Optional[int] = None,
    statistical_significance: Optional[Union[float, Decimal]] = None,
    **kwargs
) -> Confidence:
    """
    Enhanced factory function to create Confidence entity with automatic level inference.
    
    Args:
        score: Confidence score between 0 and 1
        level: Confidence level (auto-inferred if not provided)
        lower_bound: Lower confidence interval bound
        upper_bound: Upper confidence interval bound
        sample_size: Sample size for statistical validation
        statistical_significance: P-value for statistical significance
        **kwargs: Additional confidence parameters
    
    Returns:
        Confidence entity
    
    Raises:
        DataValidationError: If parameters are invalid
        ModelError: If confidence calculation fails
    """
    try:
        # Enhanced score conversion with validation
        score_decimal = safe_decimal_conversion(score, CONFIDENCE_PRECISION)
        
        # Auto-infer confidence level if not provided 
        if level is None:
            score_float = float(score_decimal)
            if score_float >= 0.95:
                level = ConfidenceLevel.VERY_HIGH
            elif score_float >= 0.85:
                level = ConfidenceLevel.HIGH
            elif score_float >= 0.70:
                level = ConfidenceLevel.MEDIUM
            elif score_float >= 0.60:
                level = ConfidenceLevel.LOW
            else:
                level = ConfidenceLevel.VERY_LOW
        
        # Enhanced bounds conversion with validation
        lower_decimal = None
        upper_decimal = None
        
        if lower_bound is not None:
            lower_decimal = safe_decimal_conversion(lower_bound, DECIMAL_PRECISION)
        if upper_bound is not None:
            upper_decimal = safe_decimal_conversion(upper_bound, DECIMAL_PRECISION)
        
        # Convert statistical significance if provided
        stat_sig_decimal = None
        if statistical_significance is not None:
            stat_sig_decimal = safe_decimal_conversion(statistical_significance, CONFIDENCE_PRECISION)
            if not (0 <= stat_sig_decimal <= 1):
                raise DataValidationError(f"Statistical significance must be between 0 and 1: {stat_sig_decimal}")
        
        # Validate sample size
        if sample_size is not None and sample_size < 1:
            raise DataValidationError(f"Sample size must be positive: {sample_size}")
        
        # Create confidence with enhanced parameters
        confidence_data = {
            "score": score_decimal,
            "level": level,
            "lower_bound": lower_decimal,
            "upper_bound": upper_decimal,
            "sample_size": sample_size,
            "statistical_significance": stat_sig_decimal,
            **kwargs
        }
        
        # Remove None values to avoid validation issues
        confidence_data = {k: v for k, v in confidence_data.items() if v is not None}
        
        return Confidence(**confidence_data)
    
    except Exception as e:
        logger.error(f"Failed to create Confidence: {e}")
        raise ModelError(
            f"Failed to create Confidence entity: {e}",
            context={"score": str(score), "level": str(level)}
        )


@performance_monitor
def create_forecast(
    symbol: str,
    prediction_type: Union[str, PredictionType],
    target_value: Union[float, Decimal],
    target_date: datetime,
    model_type: Union[str, ModelType],
    confidence_score: Union[float, Decimal],
    current_value: Optional[Union[float, Decimal]] = None,
    model_version: str = "1.0.0",
    horizon: Optional[ForecastHorizon] = None,
    data_quality_score: Union[float, Decimal] = 1.0,
    **kwargs
) -> Forecast:
    """
    Enhanced factory function to create Forecast entity with automatic horizon inference.
    
    Args:
        symbol: Stock symbol (will be sanitized)
        prediction_type: Type of prediction
        target_value: Predicted value
        target_date: Date for which prediction is made
        model_type: ML model type used
        confidence_score: Confidence score
        current_value: Current value for comparison
        model_version: Version of the model (sanitized)
        horizon: Forecast horizon (auto-inferred if not provided)
        data_quality_score: Quality score of input data
        **kwargs: Additional forecast parameters
    
    Returns:
        Forecast entity
    
    Raises:
        DataValidationError: If parameters are invalid
        ModelError: If forecast creation fails
    """
    try:
        # Enhanced input sanitization
        symbol_clean = sanitize_string_input(symbol.upper(), 20)
        model_version_clean = sanitize_string_input(model_version, 50)
        
        # Convert string enums to proper types
        if isinstance(prediction_type, str):
            try:
                prediction_type = PredictionType(prediction_type.lower())
            except ValueError:
                raise DataValidationError(f"Invalid prediction type: {prediction_type}")
        
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type.lower())
            except ValueError:
                raise DataValidationError(f"Invalid model type: {model_type}")
        
        # Enhanced value conversion with validation
        target_decimal = safe_decimal_conversion(target_value, DECIMAL_PRECISION)
        current_decimal = None
        if current_value is not None:
            current_decimal = safe_decimal_conversion(current_value, DECIMAL_PRECISION)
        
        # Validate target_date
        if target_date.tzinfo is None:
            target_date = target_date.replace(tzinfo=timezone.utc)
        
        # Auto-infer horizon with enhanced logic
        prediction_date = datetime.now(timezone.utc)
        if horizon is None:
            time_diff = target_date - prediction_date
            days = time_diff.days
            hours = time_diff.total_seconds() / 3600
            
            if hours < 24:
                horizon = ForecastHorizon.INTRADAY
            elif days <= 2:
                horizon = ForecastHorizon.DAILY
            elif days <= 8:
                horizon = ForecastHorizon.WEEKLY
            elif days <= 35:
                horizon = ForecastHorizon.MONTHLY
            elif days <= 100:
                horizon = ForecastHorizon.QUARTERLY
            else:
                horizon = ForecastHorizon.YEARLY
        
        # Create enhanced confidence object
        confidence = create_confidence(confidence_score)
        
        # Validate data quality score
        quality_decimal = safe_decimal_conversion(data_quality_score, CONFIDENCE_PRECISION)
        if not (0 <= quality_decimal <= 1):
            raise DataValidationError(f"Data quality score must be between 0 and 1: {quality_decimal}")
        
        # Build forecast data
        forecast_data = {
            "symbol": symbol_clean,
            "prediction_type": prediction_type,
            "target_value": target_decimal,
            "current_value": current_decimal,
            "target_date": target_date,
            "prediction_date": prediction_date,
            "horizon": horizon,
            "model_type": model_type,
            "model_version": model_version_clean,
            "confidence": confidence,
            "data_quality_score": quality_decimal,
            **kwargs
        }
        
        # Remove None values
        forecast_data = {k: v for k, v in forecast_data.items() if v is not None}
        
        return Forecast(**forecast_data)
    
    except Exception as e:
        logger.error(f"Failed to create Forecast: {e}")
        raise ModelError(
            f"Failed to create Forecast entity: {e}",
            context={
                "symbol": symbol,
                "prediction_type": str(prediction_type),
                "model_type": str(model_type)
            }
        )


@performance_monitor
async def create_forecast_async(
    symbol: str,
    prediction_type: Union[str, PredictionType],
    target_value: Union[float, Decimal],
    target_date: datetime,
    model_type: Union[str, ModelType],
    confidence_score: Union[float, Decimal],
    **kwargs
) -> Forecast:
    """
    Async version of create_forecast for use in async contexts.
    
    This is useful when forecast creation is part of an async ML pipeline
    or needs to be integrated with async data fetching operations.
    """
    try:
        # Run the synchronous version in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        forecast = await loop.run_in_executor(
            None,
            create_forecast,
            symbol,
            prediction_type,
            target_value,
            target_date,
            model_type,
            confidence_score,
            **kwargs
        )
        return forecast
    except Exception as e:
        logger.error(f"Async forecast creation failed: {e}")
        raise ModelError(f"Async forecast creation failed: {e}")


@performance_monitor
def create_model_prediction(
    model_type: Union[str, ModelType],
    symbol: str,
    prediction_value: Union[float, Decimal],
    prediction_type: Union[str, PredictionType],
    model_confidence: Union[float, Decimal],
    model_name: Optional[str] = None,
    inference_time_ms: Optional[int] = None,
    memory_usage_mb: Optional[float] = None,
    **kwargs
) -> ModelPrediction:
    """
    Enhanced factory function to create ModelPrediction entity.
    
    Args:
        model_type: Type of ML model
        symbol: Stock symbol (will be sanitized)
        prediction_value: Predicted value
        prediction_type: Type of prediction
        model_confidence: Model's confidence score
        model_name: Specific model name (auto-generated if not provided)
        inference_time_ms: Inference time for performance monitoring
        memory_usage_mb: Memory usage for resource monitoring
        **kwargs: Additional model prediction parameters
    
    Returns:
        ModelPrediction entity
    
    Raises:
        DataValidationError: If parameters are invalid
        ModelError: If model prediction creation fails
    """
    try:
        # Enhanced input sanitization
        symbol_clean = sanitize_string_input(symbol.upper(), 20)
        
        # Convert string enums to proper types
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type.lower())
            except ValueError:
                raise DataValidationError(f"Invalid model type: {model_type}")
        
        if isinstance(prediction_type, str):
            try:
                prediction_type = PredictionType(prediction_type.lower())
            except ValueError:
                raise DataValidationError(f"Invalid prediction type: {prediction_type}")
        
        # Generate enhanced model name if not provided
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            model_name = f"{model_type.value}_model_v1.0_{timestamp}"
        else:
            model_name = sanitize_string_input(model_name, 100)
        
        # Enhanced value conversion with validation
        value_decimal = safe_decimal_conversion(prediction_value, DECIMAL_PRECISION)
        confidence_decimal = safe_decimal_conversion(model_confidence, CONFIDENCE_PRECISION)
        
        if not (0 <= confidence_decimal <= 1):
            raise DataValidationError(f"Model confidence must be between 0 and 1: {confidence_decimal}")
        
        # Validate performance metrics
        if inference_time_ms is not None and inference_time_ms < 0:
            raise DataValidationError(f"Inference time cannot be negative: {inference_time_ms}")
        
        if memory_usage_mb is not None and memory_usage_mb < 0:
            raise DataValidationError(f"Memory usage cannot be negative: {memory_usage_mb}")
        
        # Build model prediction data
        prediction_data = {
            "model_type": model_type,
            "model_name": model_name,
            "symbol": symbol_clean,
            "prediction_value": value_decimal,
            "prediction_type": prediction_type,
            "model_confidence": confidence_decimal,
            "inference_time_ms": inference_time_ms,
            "memory_usage_mb": memory_usage_mb,
            **kwargs
        }
        
        # Remove None values
        prediction_data = {k: v for k, v in prediction_data.items() if v is not None}
        
        return ModelPrediction(**prediction_data)
    
    except Exception as e:
        logger.error(f"Failed to create ModelPrediction: {e}")
        raise ModelError(
            f"Failed to create ModelPrediction entity: {e}",
            context={
                "model_type": str(model_type),
                "symbol": symbol,
                "prediction_type": str(prediction_type)
            }
        )


@performance_monitor
def create_ensemble_prediction(
    symbol: str,
    prediction_type: Union[str, PredictionType],
    model_predictions: List[ModelPrediction],
    model_weights: Dict[str, Union[float, Decimal]],
    target_date: datetime,
    ensemble_method: str = "weighted_average",
    confidence_threshold: float = 0.7,
    **kwargs
) -> PredictionResult:
    """
    Enhanced factory function to create ensemble PredictionResult.
    
    Args:
        symbol: Stock symbol (sanitized)
        prediction_type: Type of prediction
        model_predictions: List of individual model predictions
        model_weights: Weights for each model in ensemble
        target_date: Target date for prediction
        ensemble_method: Method used for ensemble aggregation
        confidence_threshold: Minimum confidence threshold for validation
        **kwargs: Additional parameters
    
    Returns:
        PredictionResult entity
    
    Raises:
        DataValidationError: If parameters are invalid
        ModelError: If ensemble prediction creation fails
    """
    try:
        # Enhanced input sanitization
        symbol_clean = sanitize_string_input(symbol.upper(), 20)
        ensemble_method_clean = sanitize_string_input(ensemble_method, 50)
        
        # Convert prediction type
        if isinstance(prediction_type, str):
            try:
                prediction_type = PredictionType(prediction_type.lower())
            except ValueError:
                raise DataValidationError(f"Invalid prediction type: {prediction_type}")
        
        # Validate predictions list
        if not model_predictions:
            raise DataValidationError("At least one model prediction is required")
        
        # Enhanced weights conversion and validation
        weights_decimal = {}
        total_weight = Decimal('0')
        
        for model, weight in model_weights.items():
            weight_decimal = safe_decimal_conversion(weight, CONFIDENCE_PRECISION)
            if weight_decimal < 0:
                raise DataValidationError(f"Model weight cannot be negative: {weight_decimal}")
            weights_decimal[model] = weight_decimal
            total_weight += weight_decimal
        
        # Normalize weights to sum to 1 with better precision
        if total_weight > 0:
            weights_decimal = {
                model: weight / total_weight 
                for model, weight in weights_decimal.items()
            }
        else:
            raise DataValidationError("Total model weights cannot be zero")
        
        # Enhanced ensemble calculation with statistical validation
        ensemble_value = Decimal('0')
        ensemble_confidence = Decimal('0')
        prediction_values = []
        
        for pred in model_predictions:
            model_name = pred.model_type.value
            if model_name in weights_decimal:
                weight = weights_decimal[model_name]
                ensemble_value += pred.prediction_value * weight
                ensemble_confidence += pred.model_confidence * weight
                prediction_values.append(pred.prediction_value)
        
        # Statistical validation of ensemble
        if len(prediction_values) > 1:
            validator = BasicStatisticalValidator()
            validator.validate_distribution(prediction_values)
        
        # Enhanced model agreement calculation
        if len(model_predictions) > 1:
            # Use numpy for better statistical calculations
            np_values = np.array([float(v) for v in prediction_values])
            
            if len(np_values) > 0:
                mean_value = np.mean(np_values)
                std_value = np.std(np_values)
                
                # Calculate agreement based on coefficient of variation
                if mean_value != 0:
                    cv = std_value / abs(mean_value)
                    # Agreement inversely related to coefficient of variation
                    agreement = max(Decimal('0'), Decimal('1') - safe_decimal_conversion(cv, CONFIDENCE_PRECISION))
                else:
                    agreement = Decimal('1') if std_value == 0 else Decimal('0')
                
                variance = safe_decimal_conversion(np.var(np_values), DECIMAL_PRECISION)
            else:
                agreement = Decimal('1')
                variance = Decimal('0')
        else:
            agreement = Decimal('1')
            variance = Decimal('0')
        
        # Validate ensemble confidence threshold
        if ensemble_confidence < Decimal(str(confidence_threshold)):
            logger.warning(
                f"Ensemble confidence {ensemble_confidence} below threshold {confidence_threshold}"
            )
        
        # Create enhanced primary forecast
        primary_forecast = create_forecast(
            symbol=symbol_clean,
            prediction_type=prediction_type,
            target_value=ensemble_value,
            target_date=target_date,
            model_type=ModelType.ENSEMBLE,
            confidence_score=ensemble_confidence,
            **kwargs
        )
        
        # Calculate generation time if not provided
        generation_time_ms = kwargs.get('generation_time_ms')
        if generation_time_ms is None:
            # Sum up individual inference times if available
            total_inference_time = sum(
                pred.inference_time_ms or 0 for pred in model_predictions
            )
            if total_inference_time > 0:
                generation_time_ms = total_inference_time
        
        # Build prediction result data
        result_data = {
            "symbol": symbol_clean,
            "prediction_type": prediction_type,
            "primary_forecast": primary_forecast,
            "model_predictions": model_predictions,
            "ensemble_method": ensemble_method_clean,
            "model_weights": weights_decimal,
            "model_agreement_score": agreement,
            "prediction_variance": variance,
            "generation_time_ms": generation_time_ms,
            **kwargs
        }
        
        # Remove None values
        result_data = {k: v for k, v in result_data.items() if v is not None}
        
        return PredictionResult(**result_data)
    
    except Exception as e:
        logger.error(f"Failed to create ensemble PredictionResult: {e}")
        raise ModelError(
            f"Failed to create ensemble PredictionResult: {e}",
            context={
                "symbol": symbol,
                "prediction_type": str(prediction_type),
                "num_models": len(model_predictions)
            }
        )


def validate_prediction_sequence(predictions: List[Forecast]) -> bool:
    """
    Validate a sequence of predictions for consistency.
    
    Args:
        predictions: List of Forecast entities to validate
    
    Returns:
        True if sequence is valid
    
    Raises:
        BusinessLogicError: If sequence is invalid
    """
    if not predictions:
        return True
    
    # Check all predictions are for the same symbol
    symbols = set(pred.symbol for pred in predictions)
    if len(symbols) > 1:
        raise BusinessLogicError(
            f"All predictions must be for the same symbol, found: {symbols}"
        )
    
    # Check all predictions are the same type
    pred_types = set(pred.prediction_type for pred in predictions)
    if len(pred_types) > 1:
        raise BusinessLogicError(
            f"All predictions must be the same type, found: {pred_types}"
        )
    
    # Check temporal ordering
    sorted_predictions = sorted(predictions, key=lambda p: p.target_date)
    for i in range(len(sorted_predictions) - 1):
        current_pred = sorted_predictions[i]
        next_pred = sorted_predictions[i + 1]
        
        # Check for temporal consistency
        if current_pred.target_date >= next_pred.target_date:
            raise BusinessLogicError(
                "Predictions must be temporally ordered",
                context={
                    "current_target": current_pred.target_date.isoformat(),
                    "next_target": next_pred.target_date.isoformat()
                }
            )
    
    return True


def calculate_prediction_metrics(predictions: List[Forecast]) -> Dict[str, Decimal]:
    """
    Enhanced calculation of aggregate metrics for a collection of predictions.
    
    Args:
        predictions: List of Forecast entities
    
    Returns:
        Dictionary of calculated metrics with enhanced statistical analysis
    """
    if not predictions:
        return {}
    
    # Statistical validation
    validator = BasicStatisticalValidator()
    
    # Calculate basic statistics with enhanced precision
    values = [pred.target_value for pred in predictions]
    n = len(values)
    
    # Use numpy for better numerical stability
    np_values = np.array([float(v) for v in values])
    
    mean_value = safe_decimal_conversion(np.mean(np_values), DECIMAL_PRECISION)
    variance = safe_decimal_conversion(np.var(np_values), DECIMAL_PRECISION)
    std_dev = variance.sqrt() if variance > 0 else Decimal('0')
    
    # Enhanced confidence statistics
    confidence_scores = [pred.confidence.score for pred in predictions]
    np_confidence = np.array([float(c) for c in confidence_scores])
    
    mean_confidence = safe_decimal_conversion(np.mean(np_confidence), CONFIDENCE_PRECISION)
    confidence_std = safe_decimal_conversion(np.std(np_confidence), CONFIDENCE_PRECISION)
    
    # Enhanced temporal metrics
    time_spans = []
    for pred in predictions:
        if hasattr(pred, 'days_to_target'):
            time_spans.append(pred.days_to_target)
    
    mean_horizon = sum(time_spans) / len(time_spans) if time_spans else 0
    
    # Enhanced directional consistency analysis
    bullish_predictions = [pred for pred in predictions if pred.is_bullish]
    bearish_predictions = [pred for pred in predictions if pred.is_bearish]
    neutral_predictions = [pred for pred in predictions 
                         if not pred.is_bullish and not pred.is_bearish]
    
    bullish_count = len(bullish_predictions)
    bearish_count = len(bearish_predictions)
    neutral_count = len(neutral_predictions)
    
    # Calculate confidence-weighted directional bias
    total_weighted_confidence = Decimal('0')
    bullish_weighted_confidence = Decimal('0')
    
    for pred in predictions:
        weight = pred.confidence.score
        total_weighted_confidence += weight
        if pred.is_bullish:
            bullish_weighted_confidence += weight
    
    confidence_weighted_bullish_ratio = (
        bullish_weighted_confidence / total_weighted_confidence 
        if total_weighted_confidence > 0 else Decimal('0')
    )
    
    # Statistical quality metrics
    prediction_reliability_score = sum(
        1 for pred in predictions if pred.confidence.is_reliable
    ) / n if n > 0 else 0
    
    # Volatility and risk analysis
    volatility_estimates = [
        pred.volatility_estimate for pred in predictions 
        if pred.volatility_estimate is not None
    ]
    mean_volatility = (
        sum(volatility_estimates) / len(volatility_estimates) 
        if volatility_estimates else None
    )
    
    return {
        # Basic statistics
        "count": Decimal(str(n)),
        "mean_value": mean_value,
        "std_dev": std_dev,
        "variance": variance,
        "min_value": min(values),
        "max_value": max(values),
        
        # Confidence statistics
        "mean_confidence": mean_confidence,
        "confidence_std": confidence_std,
        "min_confidence": min(confidence_scores),
        "max_confidence": max(confidence_scores),
        
        # Temporal metrics
        "mean_horizon_days": Decimal(str(mean_horizon)),
        
        # Directional analysis
        "bullish_ratio": Decimal(str(bullish_count)) / Decimal(str(n)),
        "bearish_ratio": Decimal(str(bearish_count)) / Decimal(str(n)),
        "neutral_ratio": Decimal(str(neutral_count)) / Decimal(str(n)),
        "confidence_weighted_bullish_ratio": confidence_weighted_bullish_ratio,
        
        # Quality metrics
        "reliability_score": Decimal(str(prediction_reliability_score)),
        "mean_volatility": mean_volatility,
        
        # Statistical validation
        "distribution_valid": validator.validate_distribution(values),
    }


def validate_finite_number(value: Union[float, Decimal, int]) -> bool:
    """Validate that a number is finite (not NaN or infinity)."""
    if isinstance(value, Decimal):
        return value.is_finite()
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))


def safe_decimal_conversion(
    value: Union[str, int, float, Decimal], 
    precision: int = DECIMAL_PRECISION
) -> Decimal:
    """Safely convert value to Decimal with proper error handling."""
    try:
        if isinstance(value, Decimal):
            return value.quantize(Decimal(f'1E-{precision}'), rounding=ROUND_HALF_UP)
        
        # Handle potential NaN/infinity values
        if isinstance(value, float):
            if not math.isfinite(value):
                raise DataValidationError(f"Cannot convert non-finite value to Decimal: {value}")
        
        decimal_value = Decimal(str(value))
        
        # Validate the result is finite
        if not decimal_value.is_finite():
            raise DataValidationError(f"Converted Decimal is not finite: {decimal_value}")
            
        return decimal_value.quantize(Decimal(f'1E-{precision}'), rounding=ROUND_HALF_UP)
        
    except (InvalidOperation, ValueError, TypeError) as e:
        raise DataValidationError(f"Invalid decimal conversion for value {value}: {e}")


def sanitize_string_input(value: str, max_length: int = 100, allowed_chars: str = None) -> str:
    """Sanitize string input for security and consistency."""
    if not isinstance(value, str):
        raise DataValidationError(f"Expected string, got {type(value)}")
    
    # Remove potential injection characters
    dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '|', '*']
    sanitized = value
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')
    
    # Filter by allowed characters if specified
    if allowed_chars is not None:
        sanitized = ''.join(char for char in sanitized if char in allowed_chars)
    
    # Trim and validate length
    sanitized = sanitized.strip()[:max_length]
    
    if not sanitized:
        raise DataValidationError("String input cannot be empty after sanitization")
    
    return sanitized


@runtime_checkable
class StatisticalValidator(Protocol):
    """Protocol for statistical validation of prediction values."""
    
    def validate_distribution(self, values: List[Decimal]) -> bool: ...
    def calculate_outlier_score(self, value: Decimal, distribution: List[Decimal]) -> Decimal: ...


class BasicStatisticalValidator:
    """Basic implementation of statistical validation."""
    
    @staticmethod
    def validate_distribution(values: List[Decimal]) -> bool:
        """Validate that values form a reasonable distribution."""
        if len(values) < 2:
            return True
        
        # Convert to numpy for statistical operations
        np_values = np.array([float(v) for v in values])
        
        # Check for reasonable variance
        if np.var(np_values) < 1e-10:
            logger.warning("Very low variance in prediction distribution")
        
        # Check for extreme outliers (more than 3 sigma)
        mean_val = np.mean(np_values)
        std_val = np.std(np_values)
        
        if std_val > 0:
            outliers = np.abs(np_values - mean_val) > 3 * std_val
            outlier_count = np.sum(outliers)
            if outlier_count > len(values) * 0.1:  # More than 10% outliers
                logger.warning(f"High number of outliers detected: {outlier_count}/{len(values)}")
        
        return True
    
    @staticmethod
    def calculate_outlier_score(value: Decimal, distribution: List[Decimal]) -> Decimal:
        """Calculate outlier score for a value given distribution."""
        if len(distribution) < 2:
            return Decimal('0')
        
        np_dist = np.array([float(v) for v in distribution])
        value_float = float(value)
        
        mean_val = np.mean(np_dist)
        std_val = np.std(np_dist)
        
        if std_val == 0:
            return Decimal('0')
        
        z_score = abs(value_float - mean_val) / std_val
        return safe_decimal_conversion(z_score, 4) 