"""
Domain entities for market data in TSPMO Smart Stock Forecasting System.

This module defines the core domain entities for market data including Stock,
OHLCV (Open, High, Low, Close, Volume), and Volume models. These entities
represent the fundamental business concepts and enforce business rules through
validation and domain logic.
"""

from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import List, Optional, Union, Dict, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, computed_field, ConfigDict, model_validator
from pydantic.types import PositiveInt

from src.core.exceptions import DataValidationError, BusinessLogicError


class MarketType(str, Enum):
    """Enumeration of market types."""
    STOCK = "stock"
    ETF = "etf"
    INDEX = "index"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    BOND = "bond"
    OPTION = "option"
    FUTURE = "future"
    REIT = "reit"
    MUTUAL_FUND = "mutual_fund"
    WARRANT = "warrant"
    STRUCTURED_PRODUCT = "structured_product"


class Exchange(str, Enum):
    """Enumeration of major stock exchanges."""
    # Traditional Exchanges
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    AMEX = "AMEX"
    LSE = "LSE"
    TSE = "TSE"
    HKEX = "HKEX"
    SSE = "SSE"
    SZSE = "SZSE"
    BSE = "BSE"
    NSE = "NSE"
    TSX = "TSX"
    ASX = "ASX"
    EURONEXT = "EURONEXT"
    XETRA = "XETRA"
    SIX = "SIX"
    
    # Crypto Exchanges
    BINANCE = "BINANCE"
    COINBASE = "COINBASE"
    KRAKEN = "KRAKEN"
    BITFINEX = "BITFINEX"
    HUOBI = "HUOBI"
    OKEX = "OKEX"
    
    # Alternative Trading Systems
    IEX = "IEX"
    BATS = "BATS"
    ARCA = "ARCA"
    
    # Other
    OTHER = "OTHER"


class Currency(str, Enum):
    """Enumeration of major currencies."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CAD = "CAD"
    AUD = "AUD"
    CHF = "CHF"
    CNY = "CNY"
    HKD = "HKD"
    INR = "INR"
    KRW = "KRW"
    SGD = "SGD"
    BRL = "BRL"
    MXN = "MXN"
    RUB = "RUB"
    # Crypto currencies
    BTC = "BTC"
    ETH = "ETH"
    USDT = "USDT"
    USDC = "USDC"


class TradingStatus(str, Enum):
    """Enumeration of trading status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELISTED = "delisted"
    HALTED = "halted"
    PENDING = "pending"
    PRE_MARKET = "pre_market"
    POST_MARKET = "post_market"
    CIRCUIT_BREAKER = "circuit_breaker"


class DataQuality(str, Enum):
    """Enumeration of data quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    QUESTIONABLE = "questionable"
    INVALID = "invalid"


class Stock(BaseModel):
    """
    Stock entity representing a tradeable security.
    
    This is a rich domain entity that encapsulates all the essential
    information about a stock and enforces business rules.
    """
    
    model_config = ConfigDict(
        frozen=True,  # Immutable entity
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    # Core identification
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Stock symbol (ticker)"
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Company or security name"
    )
    
    # Market classification
    market_type: MarketType = Field(
        default=MarketType.STOCK,
        description="Type of market instrument"
    )
    exchange: Exchange = Field(
        ...,
        description="Primary exchange where the stock is traded"
    )
    currency: Currency = Field(
        default=Currency.USD,
        description="Trading currency"
    )
    
    # Business information
    sector: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Business sector"
    )
    industry: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Industry classification"
    )
    country: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Country of incorporation"
    )
    
    # Trading information
    trading_status: TradingStatus = Field(
        default=TradingStatus.ACTIVE,
        description="Current trading status"
    )
    shares_outstanding: Optional[PositiveInt] = Field(
        default=None,
        description="Number of shares outstanding"
    )
    market_cap: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Market capitalization"
    )
    
    # Additional identifiers
    isin: Optional[str] = Field(
        default=None,
        max_length=12,
        description="International Securities Identification Number"
    )
    cusip: Optional[str] = Field(
        default=None,
        max_length=9,
        description="Committee on Uniform Securities Identification Procedures number"
    )
    figi: Optional[str] = Field(
        default=None,
        max_length=12,
        description="Financial Instrument Global Identifier"
    )
    
    # ESG and sustainability metrics
    esg_score: Optional[Decimal] = Field(
        default=None,
        ge=0,
        le=100,
        description="Environmental, Social, and Governance score"
    )
    
    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Entity creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp"
    )
    
    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate stock symbol format."""
        if not v:
            raise DataValidationError("Stock symbol cannot be empty")
        
        # Remove whitespace and convert to uppercase
        symbol = v.strip().upper()
        
        # Enhanced symbol validation for modern markets
        # Allow alphanumeric, dots, hyphens, carets, and forward slashes (for crypto pairs)
        allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-^/")
        if not all(c in allowed_chars for c in symbol):
            raise DataValidationError(
                f"Invalid stock symbol format: {symbol}. Only alphanumeric characters, dots, hyphens, carets, and forward slashes are allowed.",
                context={"symbol": symbol}
            )
        
        return symbol
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate company name."""
        if not v or not v.strip():
            raise DataValidationError("Company name cannot be empty")
        return v.strip()
    
    @field_validator("market_cap")
    @classmethod
    def validate_market_cap(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Validate market capitalization."""
        if v is not None and v < 0:
            raise DataValidationError("Market cap cannot be negative")
        return v
    
    @field_validator("isin")
    @classmethod
    def validate_isin(cls, v: Optional[str]) -> Optional[str]:
        """Validate ISIN format."""
        if v is None:
            return v
        
        v = v.strip().upper()
        if len(v) != 12:
            raise DataValidationError("ISIN must be exactly 12 characters")
        
        if not v[:2].isalpha() or not v[2:].isalnum():
            raise DataValidationError("ISIN format invalid: first 2 chars must be letters, rest alphanumeric")
        
        return v
    
    @field_validator("cusip")
    @classmethod
    def validate_cusip(cls, v: Optional[str]) -> Optional[str]:
        """Validate CUSIP format."""
        if v is None:
            return v
        
        v = v.strip().upper()
        if len(v) != 9:
            raise DataValidationError("CUSIP must be exactly 9 characters")
        
        if not v.isalnum():
            raise DataValidationError("CUSIP must be alphanumeric")
        
        return v
    
    @computed_field
    @property
    def is_tradeable(self) -> bool:
        """Check if the stock is currently tradeable."""
        return self.trading_status in {TradingStatus.ACTIVE, TradingStatus.PRE_MARKET, TradingStatus.POST_MARKET}
    
    @computed_field
    @property
    def display_name(self) -> str:
        """Get display name for the stock."""
        return f"{self.symbol} - {self.name}"
    
    @computed_field
    @property
    def is_crypto(self) -> bool:
        """Check if this is a cryptocurrency."""
        return self.market_type == MarketType.CRYPTO or self.exchange in {
            Exchange.BINANCE, Exchange.COINBASE, Exchange.KRAKEN, 
            Exchange.BITFINEX, Exchange.HUOBI, Exchange.OKEX
        }
    
    def __str__(self) -> str:
        """String representation of the stock."""
        return f"Stock({self.symbol}, {self.name}, {self.exchange.value})"
    
    def __hash__(self) -> int:
        """Hash based on symbol and exchange for uniqueness."""
        return hash((self.symbol, self.exchange))


class Volume(BaseModel):
    """
    Volume value object representing trading volume with validation.
    
    This is a value object that encapsulates volume data with
    business rules and validation.
    """
    
    model_config = ConfigDict(
        frozen=True,  # Immutable value object
        validate_assignment=True,
        extra="forbid"
    )
    
    value: PositiveInt = Field(
        ...,
        description="Trading volume (number of shares/units traded)"
    )
    currency_volume: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Volume in currency terms (value traded)"
    )
    
    # Modern volume metrics
    buy_volume: Optional[PositiveInt] = Field(
        default=None,
        description="Volume from buy orders"
    )
    sell_volume: Optional[PositiveInt] = Field(
        default=None,
        description="Volume from sell orders"
    )
    
    @field_validator("value")
    @classmethod
    def validate_volume_value(cls, v: int) -> int:
        """Validate volume value."""
        if v < 0:
            raise DataValidationError("Volume cannot be negative")
        return v
    
    @field_validator("currency_volume")
    @classmethod
    def validate_currency_volume(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Validate currency volume."""
        if v is not None and v < 0:
            raise DataValidationError("Currency volume cannot be negative")
        return v
    
    @model_validator(mode='after')
    def validate_buy_sell_volumes(self) -> 'Volume':
        """Validate that buy and sell volumes sum to total volume."""
        if self.buy_volume is not None and self.sell_volume is not None:
            if self.buy_volume + self.sell_volume != self.value:
                raise DataValidationError(
                    "Buy volume + sell volume must equal total volume",
                    context={
                        "total": self.value,
                        "buy": self.buy_volume,
                        "sell": self.sell_volume
                    }
                )
        return self
    
    @computed_field
    @property
    def is_significant(self) -> bool:
        """Check if volume is significant (> 0)."""
        return self.value > 0
    
    @computed_field
    @property
    def buy_sell_ratio(self) -> Optional[Decimal]:
        """Calculate buy to sell volume ratio."""
        if self.buy_volume is not None and self.sell_volume is not None and self.sell_volume > 0:
            return Decimal(self.buy_volume) / Decimal(self.sell_volume)
        return None
    
    def __str__(self) -> str:
        """String representation of volume."""
        if self.currency_volume:
            return f"Volume({self.value:,} shares, ${self.currency_volume:,.2f})"
        return f"Volume({self.value:,} shares)"


class OHLCV(BaseModel):
    """
    OHLCV (Open, High, Low, Close, Volume) entity representing price data.
    
    This is the core market data entity that represents a single period
    of trading data with comprehensive validation and business logic.
    """
    
    model_config = ConfigDict(
        frozen=True,  # Immutable entity
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True
    )
    
    # Identification
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this OHLCV record"
    )
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Stock symbol"
    )
    
    # Time information
    timestamp: datetime = Field(
        ...,
        description="Timestamp for this data point"
    )
    timeframe: str = Field(
        ...,
        description="Timeframe (1s, 1min, 5min, 1hour, daily, etc.)"
    )
    
    # Price data (using Decimal for precision)
    open_price: Decimal = Field(
        ...,
        gt=0,
        decimal_places=8,  # Increased for crypto precision
        description="Opening price"
    )
    high_price: Decimal = Field(
        ...,
        gt=0,
        decimal_places=8,
        description="Highest price during the period"
    )
    low_price: Decimal = Field(
        ...,
        gt=0,
        decimal_places=8,
        description="Lowest price during the period"
    )
    close_price: Decimal = Field(
        ...,
        gt=0,
        decimal_places=8,
        description="Closing price"
    )
    
    # Volume information
    volume: Volume = Field(
        ...,
        description="Trading volume for the period"
    )
    
    # Additional price data
    adjusted_close: Optional[Decimal] = Field(
        default=None,
        gt=0,
        decimal_places=8,
        description="Adjusted closing price (for dividends, splits)"
    )
    vwap: Optional[Decimal] = Field(
        default=None,
        gt=0,
        decimal_places=8,
        description="Volume Weighted Average Price"
    )
    
    # Trade count (if available)
    trade_count: Optional[PositiveInt] = Field(
        default=None,
        description="Number of trades during the period"
    )
    
    # Modern data science fields
    bid_price: Optional[Decimal] = Field(
        default=None,
        gt=0,
        decimal_places=8,
        description="Best bid price at close"
    )
    ask_price: Optional[Decimal] = Field(
        default=None,
        gt=0,
        decimal_places=8,
        description="Best ask price at close"
    )
    spread: Optional[Decimal] = Field(
        default=None,
        ge=0,
        decimal_places=8,
        description="Bid-ask spread"
    )
    
    # Data quality indicators
    is_complete: bool = Field(
        default=True,
        description="Whether this is complete data or partial"
    )
    data_source: str = Field(
        default="unknown",
        description="Source of the data"
    )
    data_quality: DataQuality = Field(
        default=DataQuality.HIGH,
        description="Quality assessment of the data"
    )
    
    # Data lineage for modern data science
    source_timestamp: Optional[datetime] = Field(
        default=None,
        description="Original timestamp from data source"
    )
    processing_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this data was processed"
    )
    data_lineage: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Data lineage information"
    )
    
    # Metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this record was created"
    )
    
    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate stock symbol."""
        if not v:
            raise DataValidationError("Symbol cannot be empty")
        return v.strip().upper()
    
    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: datetime) -> datetime:
        """Validate timestamp."""
        # Allow future timestamps for forward-looking data (e.g., futures)
        # but not more than 10 years in the future
        max_future = datetime.now(timezone.utc).replace(year=datetime.now().year + 10)
        if v > max_future:
            raise DataValidationError("Timestamp cannot be more than 10 years in the future")
        
        # Ensure timezone awareness
        if v.tzinfo is None:
            raise DataValidationError("Timestamp must be timezone-aware")
        
        return v
    
    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """Validate timeframe format."""
        valid_timeframes = {
            # Sub-minute
            "1s", "5s", "10s", "15s", "30s",
            # Minutes
            "1min", "2min", "3min", "5min", "10min", "15min", "30min",
            # Hours
            "1hour", "2hour", "3hour", "4hour", "6hour", "8hour", "12hour",
            # Days and longer
            "daily", "weekly", "monthly", "quarterly", "yearly",
            # Alternative formats
            "1d", "1w", "1M", "1Q", "1Y"
        }
        if v not in valid_timeframes:
            raise DataValidationError(
                f"Invalid timeframe: {v}. Must be one of {sorted(valid_timeframes)}"
            )
        return v
    
    @model_validator(mode='after')
    def validate_price_relationships(self) -> 'OHLCV':
        """Validate OHLCV price relationships using model validator."""
        # High must be >= all other prices
        prices = [self.open_price, self.low_price, self.close_price]
        if self.high_price < max(prices):
            raise DataValidationError(
                "High price must be >= all other prices",
                context={
                    "high": float(self.high_price),
                    "open": float(self.open_price),
                    "low": float(self.low_price),
                    "close": float(self.close_price)
                }
            )
        
        # Low must be <= all other prices
        if self.low_price > min(prices):
            raise DataValidationError(
                "Low price must be <= all other prices",
                context={
                    "low": float(self.low_price),
                    "open": float(self.open_price),
                    "high": float(self.high_price),
                    "close": float(self.close_price)
                }
            )
        
        # Validate bid/ask relationship
        if self.bid_price is not None and self.ask_price is not None:
            if self.bid_price > self.ask_price:
                raise DataValidationError(
                    "Bid price cannot be greater than ask price",
                    context={
                        "bid": float(self.bid_price),
                        "ask": float(self.ask_price)
                    }
                )
            
            # Validate spread calculation
            if self.spread is not None:
                calculated_spread = self.ask_price - self.bid_price
                if abs(self.spread - calculated_spread) > Decimal('0.0001'):
                    raise DataValidationError(
                        "Spread does not match bid-ask difference",
                        context={
                            "spread": float(self.spread),
                            "calculated": float(calculated_spread)
                        }
                    )
        
        return self
    
    @computed_field
    @property
    def price_range(self) -> Decimal:
        """Calculate the price range (high - low)."""
        return self.high_price - self.low_price
    
    @computed_field
    @property
    def price_change(self) -> Decimal:
        """Calculate the price change (close - open)."""
        return self.close_price - self.open_price
    
    @computed_field
    @property
    def price_change_percent(self) -> Decimal:
        """Calculate the percentage price change."""
        if self.open_price == 0:
            return Decimal('0')
        return ((self.close_price - self.open_price) / self.open_price * 100).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP
        )
    
    @computed_field
    @property
    def is_bullish(self) -> bool:
        """Check if the period was bullish (close > open)."""
        return self.close_price > self.open_price
    
    @computed_field
    @property
    def is_bearish(self) -> bool:
        """Check if the period was bearish (close < open)."""
        return self.close_price < self.open_price
    
    @computed_field
    @property
    def is_doji(self) -> bool:
        """Check if this is a doji (open ≈ close)."""
        # Consider it a doji if the difference is less than 0.1% of the open price
        threshold = self.open_price * Decimal('0.001')
        return abs(self.close_price - self.open_price) <= threshold
    
    @computed_field
    @property
    def body_size(self) -> Decimal:
        """Calculate the size of the candlestick body."""
        return abs(self.close_price - self.open_price)
    
    @computed_field
    @property
    def upper_shadow(self) -> Decimal:
        """Calculate the upper shadow length."""
        return self.high_price - max(self.open_price, self.close_price)
    
    @computed_field
    @property
    def lower_shadow(self) -> Decimal:
        """Calculate the lower shadow length."""
        return min(self.open_price, self.close_price) - self.low_price
    
    @computed_field
    @property
    def typical_price(self) -> Decimal:
        """Calculate the typical price (HLC/3)."""
        return ((self.high_price + self.low_price + self.close_price) / 3).quantize(
            Decimal('0.0001'), rounding=ROUND_HALF_UP
        )
    
    @computed_field
    @property
    def weighted_close(self) -> Decimal:
        """Calculate the weighted close (OHLC/4)."""
        return ((self.open_price + self.high_price + self.low_price + self.close_price) / 4).quantize(
            Decimal('0.0001'), rounding=ROUND_HALF_UP
        )
    
    @computed_field
    @property
    def volatility_estimate(self) -> Decimal:
        """Calculate Parkinson volatility estimator."""
        if self.high_price <= 0 or self.low_price <= 0:
            return Decimal('0')
        
        # Parkinson estimator: ln(H/L)^2 / (4 * ln(2))
        import math
        ln_hl_ratio = math.log(float(self.high_price / self.low_price))
        parkinson = (ln_hl_ratio ** 2) / (4 * math.log(2))
        return Decimal(str(parkinson)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
    
    @computed_field
    @property
    def liquidity_score(self) -> Decimal:
        """Calculate a simple liquidity score based on volume and spread."""
        base_score = Decimal('50')  # Base score
        
        # Volume component (higher volume = higher liquidity)
        volume_score = min(Decimal('30'), Decimal(str(self.volume.value)) / Decimal('1000000') * 10)
        
        # Spread component (lower spread = higher liquidity)
        spread_score = Decimal('20')
        if self.spread is not None and self.close_price > 0:
            spread_pct = (self.spread / self.close_price) * 100
            spread_score = max(Decimal('0'), Decimal('20') - spread_pct * 10)
        
        return (base_score + volume_score + spread_score).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP
        )
    
    def calculate_true_range(self, previous_close: Optional[Decimal] = None) -> Decimal:
        """
        Calculate True Range for this period.
        
        Args:
            previous_close: Previous period's closing price
            
        Returns:
            True Range value
        """
        if previous_close is None:
            return self.price_range
        
        tr1 = self.high_price - self.low_price
        tr2 = abs(self.high_price - previous_close)
        tr3 = abs(self.low_price - previous_close)
        
        return max(tr1, tr2, tr3)
    
    def is_gap_up(self, previous_close: Decimal, threshold_percent: Decimal = Decimal('0.5')) -> bool:
        """
        Check if this period represents a gap up.
        
        Args:
            previous_close: Previous period's closing price
            threshold_percent: Minimum gap percentage to consider significant
            
        Returns:
            True if gap up detected
        """
        if previous_close <= 0:
            return False
        gap_percent = ((self.open_price - previous_close) / previous_close * 100)
        return gap_percent > threshold_percent
    
    def is_gap_down(self, previous_close: Decimal, threshold_percent: Decimal = Decimal('0.5')) -> bool:
        """
        Check if this period represents a gap down.
        
        Args:
            previous_close: Previous period's closing price
            threshold_percent: Minimum gap percentage to consider significant
            
        Returns:
            True if gap down detected
        """
        if previous_close <= 0:
            return False
        gap_percent = ((previous_close - self.open_price) / previous_close * 100)
        return gap_percent > threshold_percent
    
    def calculate_returns(self, previous_close: Optional[Decimal] = None) -> Dict[str, Decimal]:
        """
        Calculate various return metrics.
        
        Args:
            previous_close: Previous period's closing price
            
        Returns:
            Dictionary of return metrics
        """
        returns = {
            "intraday_return": self.price_change_percent,
            "body_return": ((max(self.open_price, self.close_price) - min(self.open_price, self.close_price)) / self.open_price * 100).quantize(Decimal('0.01')),
        }
        
        if previous_close is not None and previous_close > 0:
            returns["period_return"] = ((self.close_price - previous_close) / previous_close * 100).quantize(Decimal('0.01'))
            returns["gap_return"] = ((self.open_price - previous_close) / previous_close * 100).quantize(Decimal('0.01'))
        
        return returns
    
    def __str__(self) -> str:
        """String representation of OHLCV data."""
        return (
            f"OHLCV({self.symbol}, {self.timestamp.strftime('%Y-%m-%d %H:%M')}, "
            f"O:{self.open_price}, H:{self.high_price}, L:{self.low_price}, "
            f"C:{self.close_price}, V:{self.volume.value:,})"
        )
    
    def __hash__(self) -> int:
        """Hash based on symbol, timestamp, and timeframe for uniqueness."""
        return hash((self.symbol, self.timestamp, self.timeframe))


class MarketDataSummary(BaseModel):
    """
    Summary statistics for a collection of OHLCV data.
    
    This value object provides aggregated statistics and insights
    for a series of market data points with modern data science metrics.
    """
    
    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    symbol: str = Field(..., description="Stock symbol")
    period_start: datetime = Field(..., description="Start of the period")
    period_end: datetime = Field(..., description="End of the period")
    timeframe: str = Field(..., description="Data timeframe")
    
    # Price statistics
    highest_price: Decimal = Field(..., description="Highest price in period")
    lowest_price: Decimal = Field(..., description="Lowest price in period")
    opening_price: Decimal = Field(..., description="First opening price")
    closing_price: Decimal = Field(..., description="Last closing price")
    average_price: Decimal = Field(..., description="Average closing price")
    median_price: Optional[Decimal] = Field(None, description="Median closing price")
    
    # Volume statistics
    total_volume: int = Field(..., description="Total volume traded")
    average_volume: int = Field(..., description="Average volume per period")
    max_volume: int = Field(..., description="Maximum volume in single period")
    min_volume: int = Field(..., description="Minimum volume in single period")
    volume_std: Optional[Decimal] = Field(None, description="Volume standard deviation")
    
    # Performance metrics
    total_return: Decimal = Field(..., description="Total return percentage")
    volatility: Decimal = Field(..., description="Price volatility (std dev)")
    sharpe_ratio: Optional[Decimal] = Field(None, description="Sharpe ratio")
    max_drawdown: Optional[Decimal] = Field(None, description="Maximum drawdown percentage")
    
    # Modern data science metrics
    skewness: Optional[Decimal] = Field(None, description="Return distribution skewness")
    kurtosis: Optional[Decimal] = Field(None, description="Return distribution kurtosis")
    var_95: Optional[Decimal] = Field(None, description="Value at Risk (95%)")
    cvar_95: Optional[Decimal] = Field(None, description="Conditional Value at Risk (95%)")
    
    # Technical indicators
    rsi: Optional[Decimal] = Field(None, description="Relative Strength Index")
    bollinger_position: Optional[Decimal] = Field(None, description="Position within Bollinger Bands")
    
    # Data quality metrics
    data_points: int = Field(..., description="Number of data points")
    complete_periods: int = Field(..., description="Number of complete periods")
    quality_score: Decimal = Field(default=Decimal('100'), description="Overall data quality score")
    anomaly_count: int = Field(default=0, description="Number of detected anomalies")
    
    # Market microstructure
    average_spread: Optional[Decimal] = Field(None, description="Average bid-ask spread")
    average_liquidity_score: Optional[Decimal] = Field(None, description="Average liquidity score")
    
    @computed_field
    @property
    def data_completeness(self) -> Decimal:
        """Calculate data completeness percentage."""
        if self.data_points == 0:
            return Decimal('0')
        return (Decimal(self.complete_periods) / Decimal(self.data_points) * 100).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP
        )
    
    @computed_field
    @property
    def price_range_percent(self) -> Decimal:
        """Calculate price range as percentage of opening price."""
        if self.opening_price == 0:
            return Decimal('0')
        range_value = self.highest_price - self.lowest_price
        return (range_value / self.opening_price * 100).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP
        )
    
    @computed_field
    @property
    def volume_consistency(self) -> Decimal:
        """Calculate volume consistency score."""
        if self.volume_std is None or self.average_volume == 0:
            return Decimal('50')  # Neutral score
        
        cv = self.volume_std / Decimal(self.average_volume)  # Coefficient of variation
        # Lower CV = higher consistency, score from 0-100
        consistency = max(Decimal('0'), Decimal('100') - cv * 50)
        return consistency.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    @computed_field
    @property
    def risk_adjusted_return(self) -> Decimal:
        """Calculate risk-adjusted return."""
        if self.volatility == 0:
            return self.total_return
        return (self.total_return / self.volatility).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP
        )
    
    @computed_field
    @property
    def market_efficiency_score(self) -> Decimal:
        """Calculate a market efficiency score based on various metrics."""
        base_score = Decimal('50')
        
        # Liquidity component
        liquidity_component = Decimal('0')
        if self.average_liquidity_score is not None:
            liquidity_component = self.average_liquidity_score * Decimal('0.3')
        
        # Spread component (lower spread = higher efficiency)
        spread_component = Decimal('15')
        if self.average_spread is not None and self.average_price > 0:
            spread_pct = (self.average_spread / self.average_price) * 100
            spread_component = max(Decimal('0'), Decimal('15') - spread_pct * 5)
        
        # Data quality component
        quality_component = self.quality_score * Decimal('0.2')
        
        total_score = base_score + liquidity_component + spread_component + quality_component
        return min(Decimal('100'), total_score).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    
    def __str__(self) -> str:
        """String representation of market data summary."""
        return (
            f"MarketDataSummary({self.symbol}, {self.data_points} points, "
            f"Return: {self.total_return}%, Vol: {self.volatility}%, "
            f"Quality: {self.quality_score}%)"
        )


# Factory functions for creating entities
def create_stock(
    symbol: str,
    name: str,
    exchange: Union[str, Exchange],
    **kwargs
) -> Stock:
    """
    Factory function to create a Stock entity with validation.
    
    Args:
        symbol: Stock symbol
        name: Company name
        exchange: Exchange (string or enum)
        **kwargs: Additional stock attributes
        
    Returns:
        Stock entity
        
    Raises:
        DataValidationError: If validation fails
    """
    try:
        if isinstance(exchange, str):
            exchange = Exchange(exchange.upper())
        
        return Stock(
            symbol=symbol,
            name=name,
            exchange=exchange,
            **kwargs
        )
    except ValueError as e:
        raise DataValidationError(f"Invalid stock data: {e}")


def create_ohlcv(
    symbol: str,
    timestamp: datetime,
    open_price: Union[float, Decimal],
    high_price: Union[float, Decimal],
    low_price: Union[float, Decimal],
    close_price: Union[float, Decimal],
    volume: Union[int, Volume],
    timeframe: str = "daily",
    **kwargs
) -> OHLCV:
    """
    Factory function to create an OHLCV entity with validation.
    
    Args:
        symbol: Stock symbol
        timestamp: Data timestamp
        open_price: Opening price
        high_price: High price
        low_price: Low price
        close_price: Closing price
        volume: Trading volume (int or Volume object)
        timeframe: Data timeframe
        **kwargs: Additional OHLCV attributes
        
    Returns:
        OHLCV entity
        
    Raises:
        DataValidationError: If validation fails
    """
    try:
        # Convert prices to Decimal for precision
        open_price = Decimal(str(open_price))
        high_price = Decimal(str(high_price))
        low_price = Decimal(str(low_price))
        close_price = Decimal(str(close_price))
        
        # Create Volume object if needed
        if isinstance(volume, int):
            volume = Volume(value=volume)
        
        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        return OHLCV(
            symbol=symbol,
            timestamp=timestamp,
            timeframe=timeframe,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume,
            **kwargs
        )
    except (ValueError, TypeError) as e:
        raise DataValidationError(f"Invalid OHLCV data: {e}")


def create_volume(
    value: int,
    currency_volume: Optional[Union[float, Decimal]] = None,
    buy_volume: Optional[int] = None,
    sell_volume: Optional[int] = None
) -> Volume:
    """
    Factory function to create a Volume value object.
    
    Args:
        value: Total volume
        currency_volume: Volume in currency terms
        buy_volume: Buy volume
        sell_volume: Sell volume
        
    Returns:
        Volume value object
        
    Raises:
        DataValidationError: If validation fails
    """
    try:
        currency_vol = None
        if currency_volume is not None:
            currency_vol = Decimal(str(currency_volume))
        
        return Volume(
            value=value,
            currency_volume=currency_vol,
            buy_volume=buy_volume,
            sell_volume=sell_volume
        )
    except (ValueError, TypeError) as e:
        raise DataValidationError(f"Invalid volume data: {e}")


def validate_ohlcv_sequence(ohlcv_list: List[OHLCV]) -> bool:
    """
    Validate a sequence of OHLCV data for consistency.
    
    Args:
        ohlcv_list: List of OHLCV entities
        
    Returns:
        True if sequence is valid
        
    Raises:
        BusinessLogicError: If sequence validation fails
    """
    if not ohlcv_list:
        return True
    
    # Check that all entries are for the same symbol
    symbols = {ohlcv.symbol for ohlcv in ohlcv_list}
    if len(symbols) > 1:
        raise BusinessLogicError(
            f"OHLCV sequence contains multiple symbols: {symbols}"
        )
    
    # Check that all entries have the same timeframe
    timeframes = {ohlcv.timeframe for ohlcv in ohlcv_list}
    if len(timeframes) > 1:
        raise BusinessLogicError(
            f"OHLCV sequence contains multiple timeframes: {timeframes}"
        )
    
    # Check that timestamps are in order
    timestamps = [ohlcv.timestamp for ohlcv in ohlcv_list]
    if timestamps != sorted(timestamps):
        raise BusinessLogicError("OHLCV sequence timestamps are not in chronological order")
    
    # Check for duplicate timestamps
    if len(timestamps) != len(set(timestamps)):
        raise BusinessLogicError("OHLCV sequence contains duplicate timestamps")
    
    # Validate price continuity (no extreme gaps without explanation)
    for i in range(1, len(ohlcv_list)):
        prev_close = ohlcv_list[i-1].close_price
        curr_open = ohlcv_list[i].open_price
        
        # Check for extreme price gaps (>50% change)
        if prev_close > 0:
            gap_percent = abs((curr_open - prev_close) / prev_close * 100)
            if gap_percent > 50:
                # This might be valid for stock splits, but should be flagged
                import warnings
                warnings.warn(
                    f"Large price gap detected: {gap_percent:.2f}% between "
                    f"{ohlcv_list[i-1].timestamp} and {ohlcv_list[i].timestamp}. "
                    f"This might indicate a stock split or data error.",
                    UserWarning
                )
    
    return True


def calculate_summary_statistics(ohlcv_list: List[OHLCV]) -> MarketDataSummary:
    """
    Calculate comprehensive summary statistics for a list of OHLCV data.
    
    Args:
        ohlcv_list: List of OHLCV entities
        
    Returns:
        MarketDataSummary with calculated statistics
        
    Raises:
        BusinessLogicError: If calculation fails
    """
    if not ohlcv_list:
        raise BusinessLogicError("Cannot calculate statistics for empty OHLCV list")
    
    # Validate sequence first
    validate_ohlcv_sequence(ohlcv_list)
    
    # Sort by timestamp
    sorted_data = sorted(ohlcv_list, key=lambda x: x.timestamp)
    
    # Basic statistics
    symbol = sorted_data[0].symbol
    timeframe = sorted_data[0].timeframe
    period_start = sorted_data[0].timestamp
    period_end = sorted_data[-1].timestamp
    
    # Price statistics
    closing_prices = [float(ohlcv.close_price) for ohlcv in sorted_data]
    highest_price = max(float(ohlcv.high_price) for ohlcv in sorted_data)
    lowest_price = min(float(ohlcv.low_price) for ohlcv in sorted_data)
    opening_price = float(sorted_data[0].open_price)
    closing_price = float(sorted_data[-1].close_price)
    average_price = sum(closing_prices) / len(closing_prices)
    
    # Volume statistics
    volumes = [ohlcv.volume.value for ohlcv in sorted_data]
    total_volume = sum(volumes)
    average_volume = total_volume // len(volumes)
    max_volume = max(volumes)
    min_volume = min(volumes)
    
    # Calculate volatility and returns
    if len(closing_prices) > 1:
        returns = []
        for i in range(1, len(closing_prices)):
            if closing_prices[i-1] > 0:
                ret = (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1] * 100
                returns.append(ret)
        
        if returns:
            import statistics
            volatility = statistics.stdev(returns) if len(returns) > 1 else 0
            total_return = (closing_price - opening_price) / opening_price * 100
        else:
            volatility = 0
            total_return = 0
    else:
        volatility = 0
        total_return = 0
    
    # Data quality metrics
    complete_periods = sum(1 for ohlcv in sorted_data if ohlcv.is_complete)
    data_points = len(sorted_data)
    
    # Calculate quality score based on completeness and data consistency
    completeness_score = (complete_periods / data_points) * 100 if data_points > 0 else 0
    
    # Check for data anomalies
    anomaly_count = 0
    for ohlcv in sorted_data:
        # Check for zero volume (potential anomaly)
        if ohlcv.volume.value == 0:
            anomaly_count += 1
        # Check for extreme price movements within a period
        if ohlcv.price_range > ohlcv.open_price * Decimal('0.2'):  # >20% intraday range
            anomaly_count += 1
    
    quality_score = max(0, completeness_score - (anomaly_count / data_points * 100))
    
    return MarketDataSummary(
        symbol=symbol,
        period_start=period_start,
        period_end=period_end,
        timeframe=timeframe,
        highest_price=Decimal(str(highest_price)),
        lowest_price=Decimal(str(lowest_price)),
        opening_price=Decimal(str(opening_price)),
        closing_price=Decimal(str(closing_price)),
        average_price=Decimal(str(average_price)),
        total_volume=total_volume,
        average_volume=average_volume,
        max_volume=max_volume,
        min_volume=min_volume,
        total_return=Decimal(str(total_return)).quantize(Decimal('0.01')),
        volatility=Decimal(str(volatility)).quantize(Decimal('0.01')),
        data_points=data_points,
        complete_periods=complete_periods,
        quality_score=Decimal(str(quality_score)).quantize(Decimal('0.01')),
        anomaly_count=anomaly_count
    ) 