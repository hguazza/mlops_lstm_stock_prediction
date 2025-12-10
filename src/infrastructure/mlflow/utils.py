"""MLflow utility functions for model management and comparison."""

from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd
import structlog
from mlflow.tracking import MlflowClient

logger = structlog.get_logger(__name__)


def get_best_model(
    experiment_name: str,
    metric: str = "best_val_loss",
    ascending: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Get the best model from an experiment based on a metric.

    Args:
        experiment_name: Name of the experiment
        metric: Metric to optimize (default: 'best_val_loss')
        ascending: True if lower is better (default: True for loss)

    Returns:
        Dictionary with run_id, metrics, and params, or None if not found
    """
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            logger.warning("experiment_not_found", experiment_name=experiment_name)
            return None

        # Search for runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )

        if not runs:
            logger.warning("no_runs_found", experiment_name=experiment_name)
            return None

        best_run = runs[0]
        return {
            "run_id": best_run.info.run_id,
            "metrics": best_run.data.metrics,
            "params": best_run.data.params,
            "tags": best_run.data.tags,
            "artifact_uri": best_run.info.artifact_uri,
        }

    except Exception as e:
        logger.error("get_best_model_failed", error=str(e))
        return None


def load_model_from_registry(
    model_name: str,
    stage: str = "Production",
) -> Optional[Any]:
    """
    Load model from MLflow model registry.

    Args:
        model_name: Name of the registered model
        stage: Model stage ("Production", "Staging", "Archived", or version number)

    Returns:
        Loaded PyTorch model or None if not found
    """
    try:
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pytorch.load_model(model_uri)
        logger.info(
            "model_loaded_from_registry",
            model_name=model_name,
            stage=stage,
        )
        return model
    except Exception as e:
        logger.error(
            "load_model_failed",
            model_name=model_name,
            stage=stage,
            error=str(e),
        )
        return None


def compare_runs(
    run_ids: List[str],
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compare multiple MLflow runs.

    Args:
        run_ids: List of run IDs to compare
        metrics: Optional list of specific metrics to compare

    Returns:
        DataFrame with run comparisons
    """
    try:
        client = MlflowClient()
        data = []

        for run_id in run_ids:
            run = client.get_run(run_id)
            row = {
                "run_id": run_id,
                "run_name": run.data.tags.get("mlflow.runName", "N/A"),
                "start_time": run.info.start_time,
                "status": run.info.status,
            }

            # Add all metrics or filtered metrics
            if metrics:
                for metric in metrics:
                    row[f"metric_{metric}"] = run.data.metrics.get(metric, None)
            else:
                for key, value in run.data.metrics.items():
                    row[f"metric_{key}"] = value

            # Add key parameters
            for key, value in run.data.params.items():
                row[f"param_{key}"] = value

            data.append(row)

        df = pd.DataFrame(data)
        logger.info("runs_compared", n_runs=len(run_ids))
        return df

    except Exception as e:
        logger.error("compare_runs_failed", error=str(e))
        return pd.DataFrame()


def get_experiment_metrics(
    experiment_name: str,
    max_results: int = 100,
) -> pd.DataFrame:
    """
    Get all metrics from an experiment.

    Args:
        experiment_name: Name of the experiment
        max_results: Maximum number of runs to retrieve

    Returns:
        DataFrame with all runs and their metrics
    """
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            logger.warning("experiment_not_found", experiment_name=experiment_name)
            return pd.DataFrame()

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results,
        )

        data = []
        for run in runs:
            row = {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", "N/A"),
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "status": run.info.status,
                "artifact_uri": run.info.artifact_uri,
            }

            # Add metrics
            for key, value in run.data.metrics.items():
                row[key] = value

            # Add tags
            for key, value in run.data.tags.items():
                if not key.startswith("mlflow."):
                    row[f"tag_{key}"] = value

            data.append(row)

        df = pd.DataFrame(data)
        logger.info(
            "experiment_metrics_retrieved",
            experiment_name=experiment_name,
            n_runs=len(df),
        )
        return df

    except Exception as e:
        logger.error("get_experiment_metrics_failed", error=str(e))
        return pd.DataFrame()


def get_model_versions(model_name: str) -> List[Dict[str, Any]]:
    """
    Get all versions of a registered model.

    Args:
        model_name: Name of the registered model

    Returns:
        List of model version dictionaries
    """
    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")

        version_list = []
        for version in versions:
            version_list.append(
                {
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "status": version.status,
                    "creation_timestamp": version.creation_timestamp,
                }
            )

        logger.info(
            "model_versions_retrieved",
            model_name=model_name,
            n_versions=len(version_list),
        )
        return version_list

    except Exception as e:
        logger.error("get_model_versions_failed", error=str(e))
        return []


def delete_experiment_runs(
    experiment_name: str,
    confirm: bool = False,
) -> int:
    """
    Delete all runs from an experiment.

    Args:
        experiment_name: Name of the experiment
        confirm: Must be True to actually delete (safety check)

    Returns:
        Number of runs deleted
    """
    if not confirm:
        logger.warning("delete_requires_confirmation")
        return 0

    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            logger.warning("experiment_not_found", experiment_name=experiment_name)
            return 0

        runs = client.search_runs(experiment_ids=[experiment.experiment_id])

        count = 0
        for run in runs:
            client.delete_run(run.info.run_id)
            count += 1

        logger.info(
            "experiment_runs_deleted",
            experiment_name=experiment_name,
            n_deleted=count,
        )
        return count

    except Exception as e:
        logger.error("delete_experiment_runs_failed", error=str(e))
        return 0
