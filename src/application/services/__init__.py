"""Application services - Orchestrate infrastructure components."""

from .data_service import DataService
from .prediction_service import PredictionService


def create_data_service() -> DataService:
    """
    Factory function to create DataService instance.

    Returns:
        Configured DataService instance
    """
    return DataService()


def create_prediction_service() -> PredictionService:
    """
    Factory function to create PredictionService instance.

    Returns:
        Configured PredictionService instance
    """
    return PredictionService()


__all__ = [
    "DataService",
    "PredictionService",
    "create_data_service",
    "create_prediction_service",
]
