"""MLflow service for experiment tracking and model registry."""

from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
import numpy as np
import structlog
import torch.nn as nn
from mlflow.models import infer_signature

from . import MLflowConfig

logger = structlog.get_logger(__name__)


class MLflowServiceError(Exception):
    """Base exception for MLflow service errors."""

    pass


class MLflowService:
    """
    Service for MLflow experiment tracking and model registry.

    Provides a clean interface for logging experiments, metrics,
    parameters, and models with proper error handling and logging.
    """

    def __init__(self, config: Optional[MLflowConfig] = None):
        """
        Initialize MLflow service.

        Args:
            config: MLflow configuration (uses defaults if None)
        """
        self.config = config or MLflowConfig()
        self.logger = logger.bind(component="MLflowService")
        self._experiment_id: Optional[str] = None
        self._active_run = None
        self._setup_complete = False
        self._last_run_id: Optional[str] = None

    def setup_tracking(self) -> None:
        """
        Setup MLflow tracking URI and experiment.

        Creates experiment if it doesn't exist.

        Raises:
            MLflowServiceError: If setup fails
        """
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config.tracking_uri)
            self.logger.info(
                "mlflow_tracking_configured", tracking_uri=self.config.tracking_uri
            )

            # Create artifact location directory
            artifact_path = Path(self.config.artifact_location)
            artifact_path.mkdir(parents=True, exist_ok=True)

            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)

            if experiment is None:
                self._experiment_id = mlflow.create_experiment(
                    name=self.config.experiment_name,
                    artifact_location=self.config.artifact_location,
                )
                self.logger.info(
                    "mlflow_experiment_created",
                    experiment_name=self.config.experiment_name,
                    experiment_id=self._experiment_id,
                )
            else:
                self._experiment_id = experiment.experiment_id
                self.logger.info(
                    "mlflow_experiment_loaded",
                    experiment_name=self.config.experiment_name,
                    experiment_id=self._experiment_id,
                )

            # Enable autologging if configured
            if self.config.enable_autolog:
                self.enable_autologging()

            self._setup_complete = True

        except Exception as e:
            self.logger.error("mlflow_setup_failed", error=str(e))
            raise MLflowServiceError(f"Failed to setup MLflow: {str(e)}") from e

    def enable_autologging(self) -> None:
        """
        Enable PyTorch autologging.

        Automatically logs model parameters, metrics, and artifacts.
        """
        try:
            mlflow.pytorch.autolog(
                log_models=True,
                log_every_n_epoch=1,
                log_every_n_step=0,
                disable=False,
            )
            self.logger.info("pytorch_autologging_enabled")
        except Exception as e:
            self.logger.warning("autologging_failed", error=str(e))

    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Start a new MLflow run.

        Args:
            run_name: Optional name for the run
            tags: Optional tags to add to the run

        Raises:
            MLflowServiceError: If run start fails
        """
        if not self._setup_complete:
            self.setup_tracking()

        try:
            self._active_run = mlflow.start_run(
                experiment_id=self._experiment_id, run_name=run_name
            )

            # Store the run_id for later access
            self._last_run_id = self._active_run.info.run_id

            if tags:
                mlflow.set_tags(tags)

            self.logger.info(
                "mlflow_run_started",
                run_id=self._active_run.info.run_id,
                run_name=run_name,
                experiment_id=self._experiment_id,
            )

        except Exception as e:
            self.logger.error("run_start_failed", error=str(e))
            raise MLflowServiceError(f"Failed to start MLflow run: {str(e)}") from e

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to current run.

        Args:
            params: Dictionary of parameters to log

        Raises:
            MLflowServiceError: If logging fails
        """
        if self._active_run is None:
            raise MLflowServiceError("No active run. Call start_run() first.")

        try:
            mlflow.log_params(params)
            self.logger.debug("params_logged", n_params=len(params))
        except Exception as e:
            self.logger.error("param_logging_failed", error=str(e))
            raise MLflowServiceError(f"Failed to log parameters: {str(e)}") from e

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log metrics to current run.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number (e.g., epoch number)

        Raises:
            MLflowServiceError: If logging fails
        """
        if self._active_run is None:
            raise MLflowServiceError("No active run. Call start_run() first.")

        try:
            mlflow.log_metrics(metrics, step=step)
            self.logger.debug("metrics_logged", n_metrics=len(metrics), step=step)
        except Exception as e:
            self.logger.error("metric_logging_failed", error=str(e))
            raise MLflowServiceError(f"Failed to log metrics: {str(e)}") from e

    def log_artifact(
        self, local_path: str, artifact_path: Optional[str] = None
    ) -> None:
        """
        Log artifact file to current run.

        Args:
            local_path: Path to local file
            artifact_path: Optional path within artifacts directory

        Raises:
            MLflowServiceError: If logging fails
        """
        if self._active_run is None:
            raise MLflowServiceError("No active run. Call start_run() first.")

        try:
            mlflow.log_artifact(local_path, artifact_path)
            self.logger.debug("artifact_logged", local_path=local_path)
        except Exception as e:
            self.logger.error("artifact_logging_failed", error=str(e))
            raise MLflowServiceError(f"Failed to log artifact: {str(e)}") from e

    def infer_signature(self, input_data: np.ndarray, output_data: np.ndarray) -> Any:
        """
        Infer model signature from input/output examples.

        Args:
            input_data: Sample input data
            output_data: Sample output data

        Returns:
            MLflow ModelSignature

        Raises:
            MLflowServiceError: If signature inference fails
        """
        try:
            signature = infer_signature(input_data, output_data)
            self.logger.debug("signature_inferred")
            return signature
        except Exception as e:
            self.logger.error("signature_inference_failed", error=str(e))
            raise MLflowServiceError(f"Failed to infer signature: {str(e)}") from e

    def log_model(
        self,
        model: nn.Module,
        artifact_path: str = "model",
        signature: Optional[Any] = None,
        input_example: Optional[np.ndarray] = None,
        registered_model_name: Optional[str] = None,
    ) -> str:
        """
        Log PyTorch model to current run.

        Args:
            model: PyTorch model to log
            artifact_path: Path within artifacts directory
            signature: Optional model signature
            input_example: Optional input example
            registered_model_name: Optional name for model registry

        Returns:
            Model URI

        Raises:
            MLflowServiceError: If logging fails
        """
        if self._active_run is None:
            raise MLflowServiceError("No active run. Call start_run() first.")

        try:
            self.logger.info("starting_model_logging", artifact_path=artifact_path)
            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name,
            )

            self.logger.info(
                "model_logged",
                artifact_path=artifact_path,
                model_uri=model_info.model_uri,
            )

            return model_info.model_uri

        except Exception as e:
            self.logger.error("model_logging_failed", error=str(e))
            raise MLflowServiceError(f"Failed to log model: {str(e)}") from e

    def register_model(self, model_uri: str, model_name: str) -> Any:
        """
        Register model to MLflow model registry.

        Args:
            model_uri: URI of the model to register
            model_name: Name for the registered model

        Returns:
            ModelVersion object

        Raises:
            MLflowServiceError: If registration fails
        """
        try:
            model_version = mlflow.register_model(model_uri, model_name)
            self.logger.info(
                "model_registered",
                model_name=model_name,
                version=model_version.version,
            )
            return model_version
        except Exception as e:
            self.logger.error("model_registration_failed", error=str(e))
            raise MLflowServiceError(f"Failed to register model: {str(e)}") from e

    def transition_model_stage(self, model_name: str, version: int, stage: str) -> None:
        """
        Transition model to a different stage.

        Args:
            model_name: Name of the registered model
            version: Version number
            stage: Target stage ("Staging", "Production", "Archived")

        Raises:
            MLflowServiceError: If transition fails
        """
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=False,
            )
            self.logger.info(
                "model_stage_transitioned",
                model_name=model_name,
                version=version,
                stage=stage,
            )
        except Exception as e:
            self.logger.error("stage_transition_failed", error=str(e))
            raise MLflowServiceError(
                f"Failed to transition model stage: {str(e)}"
            ) from e

    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Add tags to current run.

        Args:
            tags: Dictionary of tags

        Raises:
            MLflowServiceError: If setting tags fails
        """
        if self._active_run is None:
            raise MLflowServiceError("No active run. Call start_run() first.")

        try:
            mlflow.set_tags(tags)
            self.logger.debug("tags_set", n_tags=len(tags))
        except Exception as e:
            self.logger.error("tag_setting_failed", error=str(e))
            raise MLflowServiceError(f"Failed to set tags: {str(e)}") from e

    def get_run_info(self) -> Optional[Any]:
        """
        Get information about current run.

        Returns:
            RunInfo object or None if no active run
        """
        if self._active_run is None:
            return None
        return self._active_run.info

    def get_last_run_id(self) -> Optional[str]:
        """
        Get the run_id of the last run that was started.

        This persists even after the run is ended, unlike get_run_info().

        Returns:
            Last run_id or None if no run was started
        """
        return self._last_run_id

    def end_run(self, status: str = "FINISHED") -> None:
        """
        End current MLflow run.

        Args:
            status: Run status ("FINISHED", "FAILED", "KILLED")

        Raises:
            MLflowServiceError: If ending run fails
        """
        if self._active_run is None:
            self.logger.warning("no_active_run_to_end")
            return

        try:
            mlflow.end_run(status=status)
            run_id = self._active_run.info.run_id
            self._active_run = None
            self.logger.info("mlflow_run_ended", run_id=run_id, status=status)
        except Exception as e:
            self.logger.error("run_end_failed", error=str(e))
            raise MLflowServiceError(f"Failed to end run: {str(e)}") from e

    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.end_run(status="FAILED")
        else:
            self.end_run(status="FINISHED")
        return False
