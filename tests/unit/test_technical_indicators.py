"""Unit tests for technical indicators calculator."""

import numpy as np
import pandas as pd
import pytest

from src.infrastructure.features.technical_indicators import (
    FeatureEngineeringError,
    InsufficientDataError,
    TechnicalIndicatorCalculator,
)


class TestTechnicalIndicatorCalculator:
    """Test suite for TechnicalIndicatorCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create calculator instance with leakage prevention."""
        return TechnicalIndicatorCalculator(prevent_leakage=True)

    @pytest.fixture
    def sample_prices(self):
        """Create sample price series."""
        np.random.seed(42)
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 100)
        prices = base_price * np.exp(np.cumsum(returns))
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        return pd.Series(prices, index=dates, name="Close")

    @pytest.fixture
    def sample_volume(self):
        """Create sample volume series."""
        np.random.seed(42)
        volumes = np.random.randint(1000000, 10000000, 100)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        return pd.Series(volumes, index=dates, name="Volume")

    def test_calculate_log_returns(self, calculator, sample_prices):
        """Test log returns calculation."""
        log_returns = calculator.calculate_log_returns(sample_prices)

        # Check length
        assert len(log_returns) == len(sample_prices)

        # Check first value is 0 (filled)
        assert log_returns.iloc[0] == 0.0

        # Check values are reasonable
        assert log_returns.std() > 0
        assert log_returns.mean() < 0.1  # Should be small

    def test_calculate_rsi(self, calculator, sample_prices):
        """Test RSI calculation."""
        rsi = calculator.calculate_rsi(sample_prices, period=14)

        # Check length
        assert len(rsi) == len(sample_prices)

        # Check range (0-100)
        assert rsi.min() >= 0
        assert rsi.max() <= 100

        # Check no NaN
        assert not rsi.isna().any()

    def test_calculate_macd(self, calculator, sample_prices):
        """Test MACD calculation."""
        macd_line, signal_line, histogram = calculator.calculate_macd(sample_prices)

        # Check lengths
        assert len(macd_line) == len(sample_prices)
        assert len(signal_line) == len(sample_prices)
        assert len(histogram) == len(sample_prices)

        # Check histogram = macd - signal
        np.testing.assert_array_almost_equal(
            histogram.values,
            (macd_line - signal_line).values,
            decimal=5,
        )

    def test_calculate_realized_volatility(self, calculator, sample_prices):
        """Test realized volatility calculation."""
        log_returns = calculator.calculate_log_returns(sample_prices)
        volatility = calculator.calculate_realized_volatility(log_returns, window=20)

        # Check length
        assert len(volatility) == len(log_returns)

        # Check all positive
        assert (volatility >= 0).all()

        # Check no NaN
        assert not volatility.isna().any()

    def test_calculate_normalized_volume(self, calculator, sample_volume):
        """Test normalized volume calculation."""
        norm_volume = calculator.calculate_normalized_volume(sample_volume, window=20)

        # Check length
        assert len(norm_volume) == len(sample_volume)

        # Check all positive
        assert (norm_volume > 0).all()

        # Check mean around 1 (normalized)
        assert 0.5 < norm_volume.mean() < 2.0

    def test_create_feature_matrix(self, calculator, sample_prices, sample_volume):
        """Test feature matrix creation."""
        # Create ticker data
        ticker_data = {}
        for ticker in ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]:
            df = pd.DataFrame(
                {"Close": sample_prices.values, "Volume": sample_volume.values},
                index=sample_prices.index,
            )
            ticker_data[ticker] = df

        # Create feature matrix
        feature_df, feature_names = calculator.create_feature_matrix(
            ticker_data=ticker_data,
            target_ticker="NVDA",
        )

        # Check shape
        assert len(feature_df) > 0
        assert len(feature_names) == 4 * 5  # 4 input tickers × 5 features each

        # Check all features present
        for ticker in ["AAPL", "GOOGL", "MSFT", "AMZN"]:
            assert f"{ticker}_return" in feature_names
            assert f"{ticker}_rsi" in feature_names
            assert f"{ticker}_macd" in feature_names
            assert f"{ticker}_volatility" in feature_names
            assert f"{ticker}_volume" in feature_names

        # Check target column
        assert "target" in feature_df.columns

        # Check no NaN
        assert not feature_df.isna().any().any()

    def test_no_future_data_in_features(self, calculator, sample_prices, sample_volume):
        """
        CRITICAL TEST: Ensure features don't leak future information.

        This test verifies that features at time t only use data up to t-1.
        """
        # Create ticker data
        ticker_data = {}
        for ticker in ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"]:
            df = pd.DataFrame(
                {"Close": sample_prices.values, "Volume": sample_volume.values},
                index=sample_prices.index,
            )
            ticker_data[ticker] = df

        # Create feature matrix WITH leakage prevention
        calculator_with_shift = TechnicalIndicatorCalculator(prevent_leakage=True)
        feature_df_shifted, _ = calculator_with_shift.create_feature_matrix(
            ticker_data=ticker_data,
            target_ticker="NVDA",
        )

        # Create feature matrix WITHOUT leakage prevention
        calculator_no_shift = TechnicalIndicatorCalculator(prevent_leakage=False)
        feature_df_not_shifted, _ = calculator_no_shift.create_feature_matrix(
            ticker_data=ticker_data,
            target_ticker="NVDA",
        )

        # Features at time t WITH shift should equal features at time t+1 WITHOUT shift
        # This proves shift is working correctly
        for i in range(len(feature_df_shifted) - 1):
            date_shifted = feature_df_shifted.index[i]
            date_not_shifted_next = feature_df_not_shifted.index[i + 1]

            if date_shifted == date_not_shifted_next:
                # Compare features (excluding target)
                features_shifted = feature_df_shifted.iloc[i].drop("target")
                features_not_shifted = feature_df_not_shifted.iloc[i].drop("target")

                # They should be approximately equal (features at t with shift = features at t-1 without shift)
                np.testing.assert_array_almost_equal(
                    features_shifted.values,
                    features_not_shifted.values,
                    decimal=5,
                    err_msg=f"Data leakage detected at date {date_shifted}",
                )

    def test_insufficient_data_error(self, calculator):
        """Test error handling for insufficient data."""
        # Create very small dataset
        prices = pd.Series(
            [100, 101, 102], index=pd.date_range("2023-01-01", periods=3)
        )
        volume = pd.Series(
            [1000, 1100, 1200], index=pd.date_range("2023-01-01", periods=3)
        )

        ticker_data = {
            "AAPL": pd.DataFrame({"Close": prices, "Volume": volume}),
        }

        with pytest.raises(InsufficientDataError):
            calculator.create_feature_matrix(
                ticker_data=ticker_data,
                target_ticker="AAPL",
            )

    def test_validate_ticker_data_missing_columns(self, calculator):
        """Test validation of ticker data with missing columns."""
        ticker_data = {
            "AAPL": pd.DataFrame({"Open": [100, 101, 102]}),  # Missing Close and Volume
        }

        with pytest.raises(FeatureEngineeringError):
            calculator.create_feature_matrix(
                ticker_data=ticker_data,
                target_ticker="AAPL",
            )

    def test_align_ticker_dates(self, calculator, sample_prices, sample_volume):
        """Test date alignment across tickers."""
        # Create tickers with different date ranges
        ticker_data = {
            "AAPL": pd.DataFrame(
                {
                    "Close": sample_prices[:80].values,
                    "Volume": sample_volume[:80].values,
                },
                index=sample_prices[:80].index,
            ),
            "GOOGL": pd.DataFrame(
                {
                    "Close": sample_prices[20:].values,
                    "Volume": sample_volume[20:].values,
                },
                index=sample_prices[20:].index,
            ),
            "MSFT": pd.DataFrame(
                {"Close": sample_prices.values, "Volume": sample_volume.values},
                index=sample_prices.index,
            ),
        }

        # Align dates
        aligned_data = calculator._align_ticker_dates(ticker_data)

        # Check all have same length
        lengths = [len(df) for df in aligned_data.values()]
        assert len(set(lengths)) == 1  # All same length

        # Check common dates only (20:80)
        expected_length = 60
        assert lengths[0] == expected_length

    def test_feature_config_customization(
        self, calculator, sample_prices, sample_volume
    ):
        """Test feature configuration customization."""
        ticker_data = {
            "AAPL": pd.DataFrame(
                {"Close": sample_prices.values, "Volume": sample_volume.values},
                index=sample_prices.index,
            ),
            "GOOGL": pd.DataFrame(
                {"Close": sample_prices.values, "Volume": sample_volume.values},
                index=sample_prices.index,
            ),
        }

        # Only use returns and RSI
        feature_config = {
            "log_returns": True,
            "rsi": True,
            "macd": False,
            "volatility": False,
            "volume": False,
        }

        feature_df, feature_names = calculator.create_feature_matrix(
            ticker_data=ticker_data,
            target_ticker="GOOGL",
            feature_config=feature_config,
        )

        # Check only 2 features per ticker (return + RSI)
        assert len(feature_names) == 1 * 2  # 1 input ticker × 2 features

        # Check feature names
        assert "AAPL_return" in feature_names
        assert "AAPL_rsi" in feature_names
        assert "AAPL_macd" not in feature_names
