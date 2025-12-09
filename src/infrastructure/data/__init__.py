"""Data infrastructure - External data sources and validation."""

import pandera as pa
from pandera import Column, DataFrameSchema


class StockDataSchema:
    """
    Pandera schema for validating stock price data from yfinance.

    Ensures data integrity and business rules compliance.
    """

    schema = DataFrameSchema(
        columns={
            "Open": Column(
                float,
                checks=[
                    pa.Check.greater_than(0, error="Open price must be positive"),
                ],
                nullable=False,
                description="Opening price of the trading period",
            ),
            "High": Column(
                float,
                checks=[
                    pa.Check.greater_than(0, error="High price must be positive"),
                ],
                nullable=False,
                description="Highest price during the trading period",
            ),
            "Low": Column(
                float,
                checks=[
                    pa.Check.greater_than(0, error="Low price must be positive"),
                ],
                nullable=False,
                description="Lowest price during the trading period",
            ),
            "Close": Column(
                float,
                checks=[
                    pa.Check.greater_than(0, error="Close price must be positive"),
                ],
                nullable=False,
                description="Closing price of the trading period",
            ),
            "Volume": Column(
                int,
                checks=[
                    pa.Check.greater_than_or_equal_to(
                        0, error="Volume must be non-negative"
                    ),
                ],
                nullable=False,
                description="Number of shares traded",
            ),
        },
        checks=[
            # High must be >= Low
            pa.Check(
                lambda df: (df["High"] >= df["Low"]).all(),
                error="High price must be >= Low price for all rows",
            ),
            # Open and Close must be between Low and High
            pa.Check(
                lambda df: (
                    (df["Open"] >= df["Low"]) & (df["Open"] <= df["High"])
                ).all(),
                error="Open price must be between Low and High",
            ),
            pa.Check(
                lambda df: (
                    (df["Close"] >= df["Low"]) & (df["Close"] <= df["High"])
                ).all(),
                error="Close price must be between Low and High",
            ),
        ],
        strict=False,  # Allow additional columns like Dividends, Stock Splits
        coerce=True,  # Coerce column types
        index=pa.Index(name="Date", nullable=False),
    )

    @classmethod
    def validate(cls, df, lazy: bool = False):
        """
        Validate a DataFrame against the stock data schema.

        Args:
            df: DataFrame to validate
            lazy: If True, collect all validation errors before raising

        Returns:
            Validated DataFrame

        Raises:
            pandera.errors.SchemaError: If validation fails
        """
        return cls.schema.validate(df, lazy=lazy)


__all__ = ["StockDataSchema"]
