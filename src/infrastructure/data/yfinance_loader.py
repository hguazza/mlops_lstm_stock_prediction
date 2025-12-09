"""YFinance data loader implementation."""

from datetime import datetime
from typing import Optional

import pandas as pd
import structlog
import yfinance as yf
from pandera.errors import SchemaError

from . import StockDataSchema

logger = structlog.get_logger(__name__)


class DataLoaderError(Exception):
    """Base exception for data loader errors."""

    pass


class InvalidSymbolError(DataLoaderError):
    """Raised when stock symbol is invalid or not found."""

    pass


class NoDataError(DataLoaderError):
    """Raised when no data is available for the given parameters."""

    pass


class ValidationError(DataLoaderError):
    """Raised when data fails Pandera schema validation."""

    pass


class YFinanceDataLoader:
    """
    Data loader for fetching stock price data using yfinance.

    Implements Clean Architecture principles by providing a concrete
    implementation of data fetching in the infrastructure layer.
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize the data loader.

        Args:
            timeout: Request timeout in seconds (default: 30)
        """
        self.timeout = timeout
        self.logger = logger.bind(component="YFinanceDataLoader")

    def fetch_stock_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        validate: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch stock price data for a given symbol.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period: Period to fetch (e.g., '1mo', '3mo', '1y', '60d')
                    Used if start_date/end_date not provided
            start_date: Start date for data fetching
            end_date: End date for data fetching
            validate: Whether to validate data with Pandera schema (default: True)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            Index: Date (datetime)

        Raises:
            InvalidSymbolError: If symbol is invalid or not found
            NoDataError: If no data is available
            ValidationError: If data fails schema validation
            DataLoaderError: For other data loading errors

        Example:
            >>> loader = YFinanceDataLoader()
            >>> df = loader.fetch_stock_data('AAPL', period='60d')
            >>> print(df.head())
        """
        symbol = symbol.upper().strip()

        self.logger.info(
            "fetching_stock_data",
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
        )

        try:
            ticker = yf.Ticker(symbol)

            # Fetch historical data
            if start_date and end_date:
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    timeout=self.timeout,
                )
            elif period:
                df = ticker.history(period=period, timeout=self.timeout)
            else:
                # Default: fetch 60 days as per spec
                df = ticker.history(period="60d", timeout=self.timeout)

            # Check if data was retrieved
            if df is None or df.empty:
                self.logger.error(
                    "no_data_retrieved",
                    symbol=symbol,
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                )
                raise NoDataError(
                    f"No data available for symbol '{symbol}'. "
                    "Please check if the symbol is valid."
                )

            # Verify symbol exists by checking ticker info
            try:
                info = ticker.info
                if not info or "symbol" not in info:
                    raise InvalidSymbolError(f"Invalid or unknown symbol: {symbol}")
            except Exception:
                # If we got data, symbol is probably valid
                # Some symbols may not have full info but still have price data
                pass

            # Clean the data
            df = self._clean_data(df)

            # Validate with Pandera schema
            if validate:
                try:
                    df = StockDataSchema.validate(df)
                    self.logger.info(
                        "data_validated",
                        symbol=symbol,
                        rows=len(df),
                    )
                except SchemaError as e:
                    self.logger.error(
                        "validation_failed",
                        symbol=symbol,
                        error=str(e),
                    )
                    raise ValidationError(
                        f"Data validation failed for {symbol}: {e}"
                    ) from e

            self.logger.info(
                "data_fetched_successfully",
                symbol=symbol,
                rows=len(df),
                date_range=f"{df.index[0]} to {df.index[-1]}",
            )

            return df

        except (InvalidSymbolError, NoDataError, ValidationError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            self.logger.error(
                "data_fetch_error",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise DataLoaderError(f"Failed to fetch data for {symbol}: {str(e)}") from e

    def fetch_latest_data(
        self, symbol: str, num_days: int = 60, validate: bool = True
    ) -> pd.DataFrame:
        """
        Fetch the most recent stock data.

        Args:
            symbol: Stock ticker symbol
            num_days: Number of most recent days to fetch (default: 60)
            validate: Whether to validate data (default: True)

        Returns:
            DataFrame with recent stock data

        Raises:
            Same as fetch_stock_data
        """
        self.logger.info(
            "fetching_latest_data",
            symbol=symbol,
            num_days=num_days,
        )

        return self.fetch_stock_data(
            symbol=symbol, period=f"{num_days}d", validate=validate
        )

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare the DataFrame.

        Args:
            df: Raw DataFrame from yfinance

        Returns:
            Cleaned DataFrame with proper types and no missing values
        """
        # Ensure index is named 'Date'
        df.index.name = "Date"

        # Select only the columns we need
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        available_columns = [col for col in required_columns if col in df.columns]

        if not available_columns:
            raise ValidationError(
                f"Required columns not found. Available: {df.columns.tolist()}"
            )

        df = df[available_columns].copy()

        # Remove rows with any NaN values
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)

        if dropped_rows > 0:
            self.logger.warning(
                "dropped_rows_with_nan",
                count=dropped_rows,
            )

        # Ensure proper data types
        df["Volume"] = df["Volume"].astype(int)

        # Ensure numeric columns are float
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

        return df

    def get_latest_price(self, symbol: str) -> float:
        """
        Get the most recent closing price for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Latest closing price

        Raises:
            Same as fetch_stock_data
        """
        df = self.fetch_latest_data(symbol, num_days=5, validate=False)
        return float(df["Close"].iloc[-1])
