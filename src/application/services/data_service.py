"""Data service - Orchestrates data fetching and preprocessing."""

from typing import Optional

import pandas as pd
import structlog

from src.infrastructure.data.yfinance_loader import (
    DataLoaderError,
    InvalidSymbolError,
    NoDataError,
    YFinanceDataLoader,
)

logger = structlog.get_logger(__name__)


class DataFetchError(Exception):
    """Raised when data fetching fails at application level."""

    pass


class DataService:
    """
    Service for orchestrating stock data operations.

    Provides high-level data operations for use cases,
    abstracting infrastructure details.
    """

    def __init__(self, data_loader: Optional[YFinanceDataLoader] = None):
        """
        Initialize data service.

        Args:
            data_loader: YFinanceDataLoader instance (creates default if None)
        """
        self.data_loader = data_loader or YFinanceDataLoader()
        self.logger = logger.bind(component="DataService")

    def fetch_stock_data(
        self, symbol: str, periods: int = 60, validate: bool = True
    ) -> pd.DataFrame:
        """
        Fetch stock data for a given symbol.

        Args:
            symbol: Stock ticker symbol
            periods: Number of days to fetch (default: 60)
            validate: Whether to validate data with Pandera

        Returns:
            DataFrame with stock data

        Raises:
            DataFetchError: If data fetching fails
        """
        self.logger.info(
            "fetching_stock_data",
            symbol=symbol,
            periods=periods,
        )

        try:
            data = self.data_loader.fetch_latest_data(
                symbol=symbol, num_days=periods, validate=validate
            )

            self.logger.info(
                "stock_data_fetched",
                symbol=symbol,
                rows=len(data),
                date_range=f"{data.index[0]} to {data.index[-1]}",
            )

            return data

        except InvalidSymbolError as e:
            self.logger.error("invalid_symbol", symbol=symbol, error=str(e))
            raise DataFetchError(f"Invalid stock symbol: {symbol}") from e

        except NoDataError as e:
            self.logger.error("no_data_available", symbol=symbol, error=str(e))
            raise DataFetchError(
                f"No data available for symbol: {symbol}. "
                "Please check if the symbol is valid and try again."
            ) from e

        except DataLoaderError as e:
            self.logger.error("data_loader_error", symbol=symbol, error=str(e))
            raise DataFetchError(f"Failed to fetch data for {symbol}: {str(e)}") from e

        except Exception as e:
            self.logger.error(
                "unexpected_error",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise DataFetchError(
                f"Unexpected error fetching data for {symbol}: {str(e)}"
            ) from e

    def prepare_data_for_training(
        self, symbol: str, periods: int = 120
    ) -> pd.DataFrame:
        """
        Fetch and prepare data specifically for model training.

        Fetches more data than needed for prediction to ensure
        sufficient training samples.

        Args:
            symbol: Stock ticker symbol
            periods: Number of days to fetch (default: 120 for training)

        Returns:
            DataFrame ready for training

        Raises:
            DataFetchError: If data preparation fails
        """
        self.logger.info(
            "preparing_training_data",
            symbol=symbol,
            periods=periods,
        )

        data = self.fetch_stock_data(symbol=symbol, periods=periods, validate=True)

        # Ensure we have enough data for training
        min_required = 80  # Minimum for reasonable train/val/test split
        if len(data) < min_required:
            raise DataFetchError(
                f"Insufficient data for training: got {len(data)} days, "
                f"need at least {min_required}"
            )

        self.logger.info(
            "training_data_prepared",
            symbol=symbol,
            rows=len(data),
        )

        return data

    def get_latest_prices(self, symbol: str, periods: int = 60) -> pd.DataFrame:
        """
        Get latest stock prices for prediction.

        Args:
            symbol: Stock ticker symbol
            periods: Number of recent days to fetch (default: 60)

        Returns:
            DataFrame with recent prices

        Raises:
            DataFetchError: If fetching fails
        """
        self.logger.info(
            "getting_latest_prices",
            symbol=symbol,
            periods=periods,
        )

        data = self.fetch_stock_data(symbol=symbol, periods=periods, validate=True)

        # Ensure we have exactly the requested number of periods
        if len(data) < periods:
            self.logger.warning(
                "insufficient_data_points",
                symbol=symbol,
                requested=periods,
                got=len(data),
            )

        return data

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists and has data.

        Args:
            symbol: Stock ticker symbol to validate

        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            data = self.data_loader.fetch_latest_data(
                symbol=symbol, num_days=5, validate=False
            )
            return len(data) > 0
        except (InvalidSymbolError, NoDataError):
            return False
        except Exception:
            return False

    def get_latest_price(self, symbol: str) -> float:
        """
        Get the most recent closing price for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Latest closing price

        Raises:
            DataFetchError: If fetching fails
        """
        try:
            price = self.data_loader.get_latest_price(symbol)
            self.logger.info("latest_price_retrieved", symbol=symbol, price=price)
            return price
        except Exception as e:
            self.logger.error("failed_to_get_latest_price", symbol=symbol, error=str(e))
            raise DataFetchError(
                f"Failed to get latest price for {symbol}: {str(e)}"
            ) from e
