from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


from .models import ColoringInstance, ColoringResult
from .solvers.base import BaseColoringSolver


@dataclass(frozen=True, slots=True)
class MLflowConfig:
    enabled: bool = True
    tracking_uri: str | None = None
    experiment_name: str = "vertex-coloring"
    run_name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    log_instance_file: bool = True


@dataclass(frozen=True, slots=True)
class ExperimentOutcome:
    result: ColoringResult
    mlflow_run_id: str | None = None


class TrackingUnavailableError(RuntimeError):
    """Raised when MLflow tracking is requested but MLflow is unavailable."""


class ExperimentRunner:
    """Run one solver/instance pair and record a standardized MLflow run."""

    def __init__(self, config: MLflowConfig) -> None:
        self.config = config

    def run(
        self,
        instance: ColoringInstance,
        solver: BaseColoringSolver,
    ) -> ExperimentOutcome:
        if not self.config.enabled:
            return ExperimentOutcome(result=solver.solve(instance))

        try:
            import mlflow
        except ImportError as exc:
            raise TrackingUnavailableError(
                "MLflow tracking is enabled, but mlflow is not installed."
            ) from exc

        if self.config.tracking_uri:
            mlflow.set_tracking_uri(self.config.tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)

        run_name = self.config.run_name or f"{solver.name}-{instance.name}"
        tags = {
            "problem_type": "vertex_coloring",
            "solver": solver.name,
            "instance": instance.name,
            "instance_fingerprint": instance.fingerprint,
            **self.config.tags,
        }

        with mlflow.start_run(run_name=run_name, tags=tags) as active_run:
            mlflow.log_params(
                {
                    "method": solver.name,
                    "instance_name": instance.name,
                    "num_vertices": instance.num_vertices,
                    "num_edges": instance.num_edges,
                    "time_limit_seconds": solver.config.time_limit_seconds,
                    "seed": solver.config.seed,
                    "threads": solver.config.threads,
                    "verbose": solver.config.verbose,
                    "solver_options": json.dumps(solver.config.options, sort_keys=True),
                }
            )
            mlflow.log_dict(instance.summary(), "instance/summary.json")
            mlflow.log_dict(solver.config.as_dict(), "solver/config.json")

            if (
                self.config.log_instance_file
                and instance.source_path
                and Path(instance.source_path).is_file()
            ):
                mlflow.log_artifact(str(instance.source_path), artifact_path="instance/source")

            try:
                result = solver.solve(instance)
            except Exception:
                mlflow.set_tag("solve_status", "error")
                mlflow.log_text(traceback.format_exc(), "errors/traceback.txt")
                raise

            valid, validation_error = result.validate(instance) if result.has_solution else (False, None)
            metrics: dict[str, float] = {
                "runtime_seconds": result.runtime_seconds,
                "solution_valid": float(valid),
            }
            if result.num_colors is not None:
                metrics["num_colors"] = float(result.num_colors)
            if result.lower_bound is not None:
                metrics["lower_bound"] = float(result.lower_bound)
            if result.upper_bound is not None:
                metrics["upper_bound"] = float(result.upper_bound)
            if result.lower_bound is not None and result.upper_bound is not None:
                metrics["absolute_gap"] = float(result.upper_bound - result.lower_bound)

            mlflow.log_metrics(metrics)
            mlflow.set_tags(
                {
                    "solve_status": result.status.value,
                    "solver_version": result.solver_version or "unknown",
                    "validation_error": validation_error or "",
                }
            )
            mlflow.log_dict(result.to_dict(), "result/result.json")

            return ExperimentOutcome(
                result=result,
                mlflow_run_id=active_run.info.run_id,
            )
