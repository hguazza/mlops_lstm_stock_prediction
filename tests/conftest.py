"""Pytest configuration and shared fixtures."""

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from src.infrastructure.model import ModelConfig
from src.presentation.main import app


@pytest.fixture
def sample_stock_data():
    """
    Create sample stock data DataFrame for testing.

    Returns:
        DataFrame with realistic stock price data
    """
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    np.random.seed(42)

    # Generate realistic-looking stock prices
    base_price = 150.0
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = base_price * (1 + returns).cumprod()

    # Ensure all constraints are met
    open_vals = prices * (1 + np.random.uniform(-0.005, 0.005, len(dates)))
    high_vals = prices * (1 + np.random.uniform(0.005, 0.015, len(dates)))
    low_vals = prices * (1 + np.random.uniform(-0.015, -0.005, len(dates)))
    close_vals = prices

    data = pd.DataFrame(
        {
            "Open": open_vals,
            "High": high_vals,
            "Low": low_vals,
            "Close": close_vals,
            "Volume": np.random.randint(1_000_000, 10_000_000, len(dates)),
        },
        index=dates,
    )

    data.index.name = "Date"

    # Ensure constraints: High >= Open, Close, Low and Low <= Open, Close
    data["High"] = data[["Open", "High", "Low", "Close"]].max(axis=1)
    data["Low"] = data[["Open", "High", "Low", "Close"]].min(axis=1)

    return data


@pytest.fixture
def sample_prices():
    """
    Create sample price array for testing.

    Returns:
        NumPy array of prices
    """
    np.random.seed(42)
    base_price = 150.0
    returns = np.random.normal(0.001, 0.02, 100)
    prices = base_price * (1 + returns).cumprod()
    return prices


@pytest.fixture
def lstm_config():
    """
    Create test ModelConfig for LSTM.

    Returns:
        ModelConfig with small settings for fast testing
    """
    return ModelConfig(
        input_size=1,
        hidden_size=16,  # Small for fast testing
        num_layers=1,
        dropout=0.2,
        sequence_length=20,  # Shorter for faster tests
        learning_rate=0.001,
        batch_size=8,
        epochs=5,
        early_stopping_patience=3,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )


@pytest.fixture
def small_stock_data():
    """
    Create small stock data for quick tests.

    Returns:
        DataFrame with 30 days of data
    """
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    np.random.seed(42)

    prices = np.linspace(100, 120, 30) + np.random.normal(0, 2, 30)

    data = pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": np.random.randint(1_000_000, 5_000_000, 30),
        },
        index=dates,
    )

    data.index.name = "Date"
    return data


@pytest_asyncio.fixture
async def client():
    """
    Create AsyncClient for API integration tests.

    Returns:
        AsyncClient for making HTTP requests to the API
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest_asyncio.fixture
async def db_session():
    """
    Create database session for tests.

    Returns:
        AsyncSession for database operations
    """
    from src.infrastructure.database.connection import async_session_maker

    async with async_session_maker() as session:
        yield session
        await session.rollback()  # Rollback any changes after test
