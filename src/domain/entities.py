"""Domain entities - Core business objects."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class StockPrice:
    """
    Immutable entity representing a stock price at a specific point in time.

    This is a domain entity following Clean Architecture principles.
    It contains only business data without any behavior or external dependencies.

    Attributes:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        date: Timestamp of the price data
        open: Opening price of the trading period
        high: Highest price during the trading period
        low: Lowest price during the trading period
        close: Closing price of the trading period
        volume: Number of shares traded
        adjusted_close: Price adjusted for splits and dividends
    """

    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate entity invariants."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if self.open < 0:
            raise ValueError("Open price cannot be negative")
        if self.high < 0:
            raise ValueError("High price cannot be negative")
        if self.low < 0:
            raise ValueError("Low price cannot be negative")
        if self.close < 0:
            raise ValueError("Close price cannot be negative")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
        if self.high < self.low:
            raise ValueError("High price cannot be less than low price")
        if self.open > self.high or self.open < self.low:
            raise ValueError("Open price must be between low and high")
        if self.close > self.high or self.close < self.low:
            raise ValueError("Close price must be between low and high")
