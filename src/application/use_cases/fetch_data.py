"""Fetch stock data use case."""

from typing import Optional

import pandas as pd
import structlog

from src.application.services.data_service import DataFetchError, DataService

logger = structlog.get_logger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


class FetchStockDataUseCase:
    """
    Use case for fetching stock data.

    Business logic: Retrieve and validate historical stock price data.
    """

    def __init__(self, data_service: Optional[DataService] = None):
        """
        Initialize use case.

        Args:
            data_service: DataService instance (creates default if None)
        """
        self.data_service = data_service or DataService()
        self.logger = logger.bind(component="FetchStockDataUseCase")

    def execute(self, symbol: str, periods: int = 60) -> pd.DataFrame:
        """
        Execute the use case to fetch stock data.

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            periods: Number of days to fetch (default: 60)

        Returns:
            DataFrame with stock data

        Raises:
            ValidationError: If input validation fails
            DataFetchError: If data fetching fails
        """
        self.logger.info(
            "executing_fetch_stock_data",
            symbol=symbol,
            periods=periods,
        )

        # Validate inputs
        self._validate_inputs(symbol, periods)

        try:
            # Fetch data via service
            data = self.data_service.fetch_stock_data(symbol=symbol, periods=periods)

            self.logger.info(
                "fetch_stock_data_completed",
                symbol=symbol,
                rows_fetched=len(data),
            )

            return data

        except DataFetchError:
            # Re-raise service errors
            raise

        except Exception as e:
            self.logger.error(
                "unexpected_error_in_use_case",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise DataFetchError(f"Unexpected error: {str(e)}") from e

    def _validate_inputs(self, symbol: str, periods: int) -> None:
        """
        Validate use case inputs.

        Args:
            symbol: Stock ticker symbol
            periods: Number of periods

        Raises:
            ValidationError: If validation fails
        """
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol must be a non-empty string")

        symbol = symbol.strip()
        if not symbol:
            raise ValidationError("Symbol cannot be empty or whitespace")

        if len(symbol) > 10:
            raise ValidationError("Symbol too long (max 10 characters)")

        # Validate periods
        if not isinstance(periods, int):
            raise ValidationError("Periods must be an integer")

        if periods < 10:
            raise ValidationError("Periods must be at least 10")

        if periods > 1000:
            raise ValidationError("Periods cannot exceed 1000")

        self.logger.debug("input_validation_passed", symbol=symbol, periods=periods)
