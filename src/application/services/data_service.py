"""Data service - Orchestrates data fetching and preprocessing."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

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

    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        periods: int = 252,
        validate: bool = True,
        align_dates: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers in parallel.

        This method fetches data for multiple tickers simultaneously using
        ThreadPoolExecutor for improved performance. All tickers are aligned
        to have the same date range (intersection of all dates).

        Args:
            tickers: List of stock ticker symbols
            periods: Number of days to fetch (default: 252, ~1 trading year)
            validate: Whether to validate data with Pandera
            align_dates: Whether to align all tickers to common date range

        Returns:
            Dictionary mapping ticker symbols to DataFrames

        Raises:
            DataFetchError: If fetching fails for any ticker

        Example:
            >>> service = DataService()
            >>> data = service.fetch_multiple_tickers(['AAPL', 'GOOGL', 'MSFT'])
            >>> print(data['AAPL'].head())
        """
        self.logger.info(
            "fetching_multiple_tickers",
            tickers=tickers,
            periods=periods,
            num_tickers=len(tickers),
        )

        if not tickers:
            raise DataFetchError("tickers list cannot be empty")

        # Remove duplicates while preserving order
        unique_tickers = list(dict.fromkeys(tickers))
        if len(unique_tickers) != len(tickers):
            self.logger.warning(
                "duplicate_tickers_removed",
                original=len(tickers),
                unique=len(unique_tickers),
            )

        # Fetch data in parallel
        ticker_data = {}
        failed_tickers = []

        with ThreadPoolExecutor(max_workers=min(len(unique_tickers), 10)) as executor:
            # Submit all fetch tasks
            future_to_ticker = {
                executor.submit(
                    self._fetch_single_ticker_safe,
                    ticker,
                    periods,
                    validate,
                ): ticker
                for ticker in unique_tickers
            }

            # Collect results as they complete
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data is not None:
                        ticker_data[ticker] = data
                    else:
                        failed_tickers.append(ticker)
                except Exception as e:
                    self.logger.error(
                        "ticker_fetch_failed",
                        ticker=ticker,
                        error=str(e),
                    )
                    failed_tickers.append(ticker)

        # Check if we got data for all tickers
        if failed_tickers:
            self.logger.warning(
                "some_tickers_failed",
                failed_tickers=failed_tickers,
                successful=len(ticker_data),
            )

        if not ticker_data:
            raise DataFetchError(
                f"Failed to fetch data for any ticker. Attempted: {tickers}"
            )

        # Align dates if requested
        if align_dates and len(ticker_data) > 1:
            ticker_data = self._align_ticker_dates(ticker_data)

        self.logger.info(
            "multiple_tickers_fetched",
            successful_tickers=list(ticker_data.keys()),
            failed_tickers=failed_tickers,
        )

        return ticker_data

    def _fetch_single_ticker_safe(
        self, ticker: str, periods: int, validate: bool
    ) -> Optional[pd.DataFrame]:
        """
        Safely fetch data for a single ticker.

        Returns None if fetch fails instead of raising exception.

        Args:
            ticker: Stock ticker symbol
            periods: Number of days to fetch
            validate: Whether to validate data

        Returns:
            DataFrame or None if failed
        """
        try:
            return self.data_loader.fetch_latest_data(
                symbol=ticker,
                num_days=periods,
                validate=validate,
            )
        except Exception as e:
            self.logger.error(
                "single_ticker_fetch_failed",
                ticker=ticker,
                error=str(e),
            )
            return None

    def _align_ticker_dates(
        self, ticker_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Align all tickers to common date range (intersection).

        Ensures all tickers have data for the same dates to prevent
        misalignment in multivariate models.

        Args:
            ticker_data: Dictionary of ticker DataFrames

        Returns:
            Dictionary of aligned DataFrames

        Raises:
            DataFetchError: If date intersection is too small
        """
        if len(ticker_data) < 2:
            return ticker_data

        # Find common dates (intersection)
        common_dates = None
        for ticker, df in ticker_data.items():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))

        if not common_dates:
            raise DataFetchError(
                "No overlapping dates found across tickers. "
                "They may have different trading calendars or data availability."
            )

        # Check minimum overlap
        min_required_days = 60
        if len(common_dates) < min_required_days:
            raise DataFetchError(
                f"Insufficient overlapping dates: {len(common_dates)} days. "
                f"Need at least {min_required_days} days of common data across all tickers."
            )

        # Sort dates
        common_dates = sorted(common_dates)

        # Align all DataFrames
        aligned_data = {}
        for ticker, df in ticker_data.items():
            aligned_df = df.loc[common_dates].copy()
            aligned_data[ticker] = aligned_df

        self.logger.info(
            "tickers_aligned",
            num_tickers=len(aligned_data),
            common_dates=len(common_dates),
            date_range=f"{common_dates[0]} to {common_dates[-1]}",
        )

        return aligned_data
