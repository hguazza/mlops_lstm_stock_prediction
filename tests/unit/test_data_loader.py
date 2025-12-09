"""Unit tests for YFinanceDataLoader."""

from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.infrastructure.data.yfinance_loader import (
    DataLoaderError,
    NoDataError,
    ValidationError,
    YFinanceDataLoader,
)


@pytest.mark.unit
class TestYFinanceDataLoaderInit:
    """Test YFinanceDataLoader initialization."""

    def test_init_default_timeout(self):
        """Test initialization with default timeout."""
        loader = YFinanceDataLoader()
        assert loader.timeout == 30

    def test_init_custom_timeout(self):
        """Test initialization with custom timeout."""
        loader = YFinanceDataLoader(timeout=60)
        assert loader.timeout == 60


@pytest.mark.unit
class TestFetchStockData:
    """Test fetch_stock_data method."""

    @patch("src.infrastructure.data.yfinance_loader.yf.Ticker")
    def test_fetch_stock_data_with_period(self, mock_ticker, sample_stock_data):
        """Test successful data fetch with period parameter."""
        # Arrange
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_stock_data
        mock_ticker_instance.info = {"symbol": "AAPL"}
        mock_ticker.return_value = mock_ticker_instance

        loader = YFinanceDataLoader()

        # Act
        result = loader.fetch_stock_data("AAPL", period="60d", validate=False)

        # Assert
        assert not result.empty
        assert len(result) == len(sample_stock_data)
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
        mock_ticker.assert_called_once_with("AAPL")
        mock_ticker_instance.history.assert_called_once()

    @patch("src.infrastructure.data.yfinance_loader.yf.Ticker")
    def test_fetch_stock_data_with_dates(self, mock_ticker, sample_stock_data):
        """Test data fetch with start and end dates."""
        # Arrange
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_stock_data
        mock_ticker_instance.info = {"symbol": "AAPL"}
        mock_ticker.return_value = mock_ticker_instance

        loader = YFinanceDataLoader()
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 3, 31)

        # Act
        result = loader.fetch_stock_data(
            "AAPL", start_date=start_date, end_date=end_date, validate=False
        )

        # Assert
        assert not result.empty
        mock_ticker_instance.history.assert_called_once_with(
            start=start_date, end=end_date, timeout=30
        )

    @patch("src.infrastructure.data.yfinance_loader.yf.Ticker")
    def test_fetch_stock_data_default_period(self, mock_ticker, sample_stock_data):
        """Test data fetch uses 60d as default period."""
        # Arrange
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_stock_data
        mock_ticker_instance.info = {"symbol": "AAPL"}
        mock_ticker.return_value = mock_ticker_instance

        loader = YFinanceDataLoader()

        # Act
        result = loader.fetch_stock_data("AAPL", validate=False)

        # Assert
        assert not result.empty
        mock_ticker_instance.history.assert_called_once_with(period="60d", timeout=30)

    @patch("src.infrastructure.data.yfinance_loader.yf.Ticker")
    def test_invalid_symbol_raises_error(self, mock_ticker):
        """Test that invalid symbol raises NoDataError."""
        # Arrange
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()  # Empty DataFrame
        mock_ticker.return_value = mock_ticker_instance

        loader = YFinanceDataLoader()

        # Act & Assert
        with pytest.raises(NoDataError, match="No data available"):
            loader.fetch_stock_data("INVALID", period="30d", validate=False)

    @patch("src.infrastructure.data.yfinance_loader.yf.Ticker")
    def test_symbol_case_insensitive(self, mock_ticker, sample_stock_data):
        """Test that symbol is converted to uppercase."""
        # Arrange
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_stock_data
        mock_ticker_instance.info = {"symbol": "AAPL"}
        mock_ticker.return_value = mock_ticker_instance

        loader = YFinanceDataLoader()

        # Act
        result = loader.fetch_stock_data("aapl", period="30d", validate=False)

        # Assert
        assert not result.empty
        # Should be called with uppercase
        mock_ticker.assert_called_once_with("AAPL")

    @patch("src.infrastructure.data.yfinance_loader.yf.Ticker")
    def test_fetch_handles_general_exception(self, mock_ticker):
        """Test that general exceptions are wrapped in DataLoaderError."""
        # Arrange
        mock_ticker.side_effect = Exception("Network error")
        loader = YFinanceDataLoader()

        # Act & Assert
        with pytest.raises(DataLoaderError, match="Failed to fetch data"):
            loader.fetch_stock_data("AAPL", period="30d", validate=False)


@pytest.mark.unit
class TestFetchLatestData:
    """Test fetch_latest_data method."""

    @patch("src.infrastructure.data.yfinance_loader.yf.Ticker")
    def test_fetch_latest_data_default(self, mock_ticker, sample_stock_data):
        """Test fetch latest data with default 60 days."""
        # Arrange
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_stock_data
        mock_ticker_instance.info = {"symbol": "MSFT"}
        mock_ticker.return_value = mock_ticker_instance

        loader = YFinanceDataLoader()

        # Act
        result = loader.fetch_latest_data("MSFT", validate=False)

        # Assert
        assert not result.empty
        mock_ticker_instance.history.assert_called_once_with(period="60d", timeout=30)

    @patch("src.infrastructure.data.yfinance_loader.yf.Ticker")
    def test_fetch_latest_data_custom_days(self, mock_ticker, sample_stock_data):
        """Test fetch latest data with custom number of days."""
        # Arrange
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_stock_data
        mock_ticker_instance.info = {"symbol": "GOOGL"}
        mock_ticker.return_value = mock_ticker_instance

        loader = YFinanceDataLoader()

        # Act
        result = loader.fetch_latest_data("GOOGL", num_days=90, validate=False)

        # Assert
        assert not result.empty
        mock_ticker_instance.history.assert_called_once_with(period="90d", timeout=30)


@pytest.mark.unit
class TestCleanData:
    """Test _clean_data private method."""

    def test_clean_data_removes_nan(self):
        """Test that NaN values are removed."""
        # Arrange
        data = pd.DataFrame(
            {
                "Open": [100, 101, None, 103],
                "High": [102, 103, 104, 105],
                "Low": [99, 100, 101, 102],
                "Close": [101, 102, 103, 104],
                "Volume": [1000, 1100, 1200, 1300],
            }
        )
        loader = YFinanceDataLoader()

        # Act
        result = loader._clean_data(data)

        # Assert
        assert len(result) == 3  # One row removed
        assert result["Open"].isna().sum() == 0

    def test_clean_data_proper_types(self):
        """Test that data types are correct after cleaning."""
        # Arrange
        data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [102.0, 103.0, 104.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [101.0, 102.0, 103.0],
                "Volume": [1000.5, 1100.5, 1200.5],  # Float volume
            }
        )
        loader = YFinanceDataLoader()

        # Act
        result = loader._clean_data(data)

        # Assert
        assert result["Volume"].dtype == "int64"
        assert result["Open"].dtype == "float64"
        assert result["Close"].dtype == "float64"

    def test_clean_data_index_name(self):
        """Test that index is named 'Date'."""
        # Arrange
        data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [102, 103, 104],
                "Low": [99, 100, 101],
                "Close": [101, 102, 103],
                "Volume": [1000, 1100, 1200],
            }
        )
        loader = YFinanceDataLoader()

        # Act
        result = loader._clean_data(data)

        # Assert
        assert result.index.name == "Date"

    def test_clean_data_missing_columns(self):
        """Test that missing required columns raise error."""
        # Arrange
        data = pd.DataFrame(
            {
                "SomeColumn": [1, 2, 3],
            }
        )
        loader = YFinanceDataLoader()

        # Act & Assert
        with pytest.raises(ValidationError, match="Required columns not found"):
            loader._clean_data(data)


@pytest.mark.unit
class TestGetLatestPrice:
    """Test get_latest_price method."""

    @patch("src.infrastructure.data.yfinance_loader.yf.Ticker")
    def test_get_latest_price(self, mock_ticker, sample_stock_data):
        """Test getting the latest closing price."""
        # Arrange
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_stock_data
        mock_ticker_instance.info = {"symbol": "AAPL"}
        mock_ticker.return_value = mock_ticker_instance

        loader = YFinanceDataLoader()

        # Act
        result = loader.get_latest_price("AAPL")

        # Assert
        assert isinstance(result, float)
        assert result == sample_stock_data["Close"].iloc[-1]


@pytest.mark.unit
class TestDataValidation:
    """Test data validation with Pandera."""

    @patch("src.infrastructure.data.yfinance_loader.yf.Ticker")
    def test_validation_with_valid_data(self, mock_ticker, sample_stock_data):
        """Test that valid data passes Pandera validation."""
        # Arrange
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_stock_data
        mock_ticker_instance.info = {"symbol": "AAPL"}
        mock_ticker.return_value = mock_ticker_instance

        loader = YFinanceDataLoader()

        # Act - should not raise
        result = loader.fetch_stock_data("AAPL", period="30d", validate=True)

        # Assert
        assert not result.empty

    @patch("src.infrastructure.data.yfinance_loader.yf.Ticker")
    def test_validation_with_invalid_data(self, mock_ticker):
        """Test that invalid data raises ValidationError."""
        # Arrange - create data that violates constraints
        invalid_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [90, 91, 92],  # High < Low - INVALID
                "Low": [99, 100, 101],
                "Close": [101, 102, 103],
                "Volume": [1000, 1100, 1200],
            }
        )
        invalid_data.index.name = "Date"

        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = invalid_data
        mock_ticker_instance.info = {"symbol": "AAPL"}
        mock_ticker.return_value = mock_ticker_instance

        loader = YFinanceDataLoader()

        # Act & Assert
        with pytest.raises(ValidationError, match="Data validation failed"):
            loader.fetch_stock_data("AAPL", period="30d", validate=True)
