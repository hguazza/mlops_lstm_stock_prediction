"""Technical indicators calculator for feature engineering.

This module implements financial technical indicators for multivariate time series
prediction. All indicators are calculated with proper temporal alignment to prevent
data leakage.

Key Features:
- Log-returns for stationarity
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Realized volatility
- Normalized volume
- Automatic data leakage prevention with shift operations
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class FeatureEngineeringError(Exception):
    """Base exception for feature engineering errors."""

    pass


class InsufficientDataError(FeatureEngineeringError):
    """Raised when insufficient data is available for feature calculation."""

    pass


class TechnicalIndicatorCalculator:
    """
    Calculator for technical indicators used in stock prediction.

    Implements industry-standard technical indicators with proper temporal
    alignment to prevent data leakage in time series predictions.

    All features are calculated using only historical data available at time t-1
    to predict time t.
    """

    def __init__(self, prevent_leakage: bool = True):
        """
        Initialize calculator.

        Args:
            prevent_leakage: If True, shifts features to prevent future data leakage
                           (default: True for production use)
        """
        self.prevent_leakage = prevent_leakage
        self.logger = logger.bind(component="TechnicalIndicatorCalculator")

    def calculate_log_returns(
        self, prices: pd.Series, fill_method: str = "zero"
    ) -> pd.Series:
        """
        Calculate log returns for price stationarity.

        Formula: log_return(t) = log(price(t) / price(t-1))

        Log returns are preferred over simple returns because:
        - They are time-additive
        - More symmetric (up/down moves)
        - Approximate normal distribution

        Args:
            prices: Price series (Close prices)
            fill_method: Method to fill first NaN value
                        - 'zero': Fill with 0 (default)
                        - 'drop': Drop first row
                        - 'forward': Forward fill from second value

        Returns:
            Series of log returns
        """
        log_returns = np.log(prices / prices.shift(1))

        # Handle first NaN
        if fill_method == "zero":
            log_returns = log_returns.fillna(0)
        elif fill_method == "drop":
            log_returns = log_returns.dropna()
        elif fill_method == "forward":
            log_returns = log_returns.fillna(method="bfill", limit=1)

        self.logger.debug(
            "log_returns_calculated",
            length=len(log_returns),
            mean=float(log_returns.mean()),
            std=float(log_returns.std()),
        )

        return log_returns

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        RSI measures momentum on a 0-100 scale:
        - RSI > 70: Overbought (potential sell signal)
        - RSI < 30: Oversold (potential buy signal)

        Formula:
            RSI = 100 - (100 / (1 + RS))
            RS = Average Gain / Average Loss over period

        Args:
            prices: Price series
            period: Lookback period (default: 14, standard)

        Returns:
            RSI series (0-100 scale)
        """
        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)

        # Calculate exponential moving averages
        avg_gains = gains.ewm(span=period, adjust=False).mean()
        avg_losses = losses.ewm(span=period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        # Fill initial NaN with 50 (neutral)
        rsi = rsi.fillna(50)

        self.logger.debug(
            "rsi_calculated",
            period=period,
            mean=float(rsi.mean()),
            min=float(rsi.min()),
            max=float(rsi.max()),
        )

        return rsi

    def calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        MACD shows trend changes and momentum:
        - MACD line crosses above signal: Bullish
        - MACD line crosses below signal: Bearish

        Components:
        1. MACD Line = EMA(12) - EMA(26)
        2. Signal Line = EMA(9) of MACD Line
        3. Histogram = MACD Line - Signal Line

        Args:
            prices: Price series
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        # Calculate EMAs
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        # Fill NaN with 0
        macd_line = macd_line.fillna(0)
        signal_line = signal_line.fillna(0)
        histogram = histogram.fillna(0)

        self.logger.debug(
            "macd_calculated",
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
        )

        return macd_line, signal_line, histogram

    def calculate_realized_volatility(
        self, returns: pd.Series, window: int = 20
    ) -> pd.Series:
        """
        Calculate realized volatility (rolling standard deviation of returns).

        Volatility measures price uncertainty/risk:
        - High volatility: Large price swings
        - Low volatility: Stable prices

        Formula: volatility(t) = std(returns[t-window:t])

        Args:
            returns: Return series (log returns or simple returns)
            window: Rolling window size (default: 20, ~1 trading month)

        Returns:
            Realized volatility series
        """
        volatility = returns.rolling(window=window).std()

        # Fill NaN with overall std
        volatility = volatility.fillna(returns.std())

        self.logger.debug(
            "volatility_calculated",
            window=window,
            mean=float(volatility.mean()),
            max=float(volatility.max()),
        )

        return volatility

    def calculate_normalized_volume(
        self, volume: pd.Series, window: int = 20
    ) -> pd.Series:
        """
        Calculate normalized volume (volume relative to moving average).

        Normalized volume identifies unusual trading activity:
        - > 1: Higher than average volume
        - < 1: Lower than average volume

        Formula: norm_vol(t) = volume(t) / mean(volume[t-window:t])

        Args:
            volume: Volume series
            window: Rolling window for average (default: 20)

        Returns:
            Normalized volume series
        """
        avg_volume = volume.rolling(window=window).mean()

        # Avoid division by zero
        avg_volume = avg_volume.replace(0, 1)

        normalized = volume / avg_volume

        # Fill NaN with 1 (neutral)
        normalized = normalized.fillna(1.0)

        self.logger.debug(
            "normalized_volume_calculated",
            window=window,
            mean=float(normalized.mean()),
        )

        return normalized

    def create_feature_matrix(
        self,
        ticker_data: Dict[str, pd.DataFrame],
        target_ticker: str,
        feature_config: Optional[Dict[str, bool]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create feature matrix from multiple ticker data.

        Combines data from multiple tickers into a single feature matrix with
        technical indicators. All features are temporally aligned to prevent
        data leakage.

        Args:
            ticker_data: Dictionary mapping ticker symbols to DataFrames
                        Each DataFrame must have 'Close' and 'Volume' columns
            target_ticker: Target ticker for which to predict returns
            feature_config: Optional dict to enable/disable features
                          Default: all features enabled
                          Keys: 'log_returns', 'rsi', 'macd', 'volatility', 'volume'

        Returns:
            Tuple of:
            - feature_df: DataFrame with all features, aligned by date
            - feature_names: List of feature column names

        Raises:
            InsufficientDataError: If not enough data to calculate features
            FeatureEngineeringError: If data processing fails
        """
        if feature_config is None:
            feature_config = {
                "log_returns": True,
                "rsi": True,
                "macd": True,
                "volatility": True,
                "volume": True,
            }

        self.logger.info(
            "creating_feature_matrix",
            num_tickers=len(ticker_data),
            target_ticker=target_ticker,
            feature_config=feature_config,
        )

        try:
            # Validate input data
            self._validate_ticker_data(ticker_data)

            # Align all tickers to common date range (intersection)
            aligned_data = self._align_ticker_dates(ticker_data)

            # Calculate features for each ticker
            all_features = []
            feature_names = []

            for ticker, df in aligned_data.items():
                ticker_features = {}
                prices = df["Close"]
                volume = df["Volume"]

                # 1. Log Returns
                if feature_config.get("log_returns", True):
                    log_returns = self.calculate_log_returns(prices)
                    ticker_features[f"{ticker}_return"] = log_returns
                    feature_names.append(f"{ticker}_return")

                # 2. RSI
                if feature_config.get("rsi", True):
                    rsi = self.calculate_rsi(prices, period=14)
                    # Normalize to [-1, 1] range for neural network
                    rsi_normalized = (rsi - 50) / 50
                    ticker_features[f"{ticker}_rsi"] = rsi_normalized
                    feature_names.append(f"{ticker}_rsi")

                # 3. MACD Histogram (most informative component)
                if feature_config.get("macd", True):
                    _, _, macd_hist = self.calculate_macd(prices)
                    # Normalize MACD histogram
                    macd_normalized = macd_hist / (prices.std() + 1e-8)
                    ticker_features[f"{ticker}_macd"] = macd_normalized
                    feature_names.append(f"{ticker}_macd")

                # 4. Realized Volatility
                if feature_config.get("volatility", True):
                    log_returns = self.calculate_log_returns(prices)
                    volatility = self.calculate_realized_volatility(
                        log_returns, window=20
                    )
                    ticker_features[f"{ticker}_volatility"] = volatility
                    feature_names.append(f"{ticker}_volatility")

                # 5. Normalized Volume
                if feature_config.get("volume", True):
                    norm_volume = self.calculate_normalized_volume(volume, window=20)
                    # Log transform to handle extreme values
                    norm_volume_log = np.log1p(norm_volume)
                    ticker_features[f"{ticker}_volume"] = norm_volume_log
                    feature_names.append(f"{ticker}_volume")

                # Combine ticker features
                ticker_df = pd.DataFrame(ticker_features, index=df.index)
                all_features.append(ticker_df)

            # Combine all features into single DataFrame
            feature_df = pd.concat(all_features, axis=1)

            # CRITICAL: Shift features by 1 to prevent data leakage
            # Features at time t should only use data up to t-1
            if self.prevent_leakage:
                feature_df = feature_df.shift(1)
                self.logger.info("features_shifted_to_prevent_leakage")

            # Create target variable (returns of target ticker)
            if target_ticker in aligned_data:
                target_prices = aligned_data[target_ticker]["Close"]
                target_returns = self.calculate_log_returns(target_prices)
                feature_df["target"] = target_returns

            # Drop rows with NaN (from shift and rolling calculations)
            initial_rows = len(feature_df)
            feature_df = feature_df.dropna()
            dropped_rows = initial_rows - len(feature_df)

            if len(feature_df) < 60:
                raise InsufficientDataError(
                    f"Insufficient data after feature calculation: {len(feature_df)} rows. "
                    f"Need at least 60 rows for model training."
                )

            self.logger.info(
                "feature_matrix_created",
                num_features=len(feature_names),
                num_samples=len(feature_df),
                dropped_rows=dropped_rows,
                date_range=f"{feature_df.index[0]} to {feature_df.index[-1]}",
            )

            return feature_df, feature_names

        except InsufficientDataError:
            raise
        except Exception as e:
            self.logger.error(
                "feature_matrix_creation_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise FeatureEngineeringError(
                f"Failed to create feature matrix: {str(e)}"
            ) from e

    def _validate_ticker_data(self, ticker_data: Dict[str, pd.DataFrame]) -> None:
        """
        Validate ticker data has required columns and sufficient data.

        Args:
            ticker_data: Dictionary of ticker DataFrames

        Raises:
            FeatureEngineeringError: If validation fails
        """
        if not ticker_data:
            raise FeatureEngineeringError("ticker_data cannot be empty")

        required_columns = ["Close", "Volume"]

        for ticker, df in ticker_data.items():
            # Check required columns exist
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise FeatureEngineeringError(
                    f"Ticker {ticker} missing required columns: {missing_cols}"
                )

            # Check minimum data
            if len(df) < 30:
                raise InsufficientDataError(
                    f"Ticker {ticker} has insufficient data: {len(df)} rows. "
                    f"Need at least 30 rows."
                )

            # Check for all NaN
            if df["Close"].isna().all():
                raise FeatureEngineeringError(
                    f"Ticker {ticker} has all NaN values in Close prices"
                )

    def _align_ticker_dates(
        self, ticker_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Align all tickers to common date range (intersection).

        Ensures all tickers have data for the same dates to avoid misalignment.

        Args:
            ticker_data: Dictionary of ticker DataFrames

        Returns:
            Dictionary of aligned DataFrames

        Raises:
            InsufficientDataError: If date intersection is too small
        """
        # Find common date range (intersection)
        common_dates = None

        for ticker, df in ticker_data.items():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))

        if not common_dates or len(common_dates) < 30:
            raise InsufficientDataError(
                f"Insufficient overlapping dates across tickers: {len(common_dates) if common_dates else 0} days. "
                f"Need at least 30 days of common data."
            )

        # Sort dates
        common_dates = sorted(common_dates)

        # Align all DataFrames to common dates
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
