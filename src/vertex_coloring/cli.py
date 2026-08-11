from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .io import load_instance
from .models import SolveStatus, SolverConfig
from .registry import available_solvers, create_solver
from .solvers.base import SolverUnavailableError
from .tracking import ExperimentRunner, MLflowConfig, TrackingUnavailableError


def _parse_key_value(values: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, received {value!r}")
        key, raw = value.split("=", maxsplit=1)
        key = key.strip()
        if not key:
            raise argparse.ArgumentTypeError("Option key cannot be empty")
        try:
            parsed[key] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[key] = raw
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vertex-coloring",
        description="Run a pluggable vertex-coloring solver and track the experiment in MLflow.",
    )
    parser.add_argument("--method", "--solver", dest="method", choices=available_solvers())
    parser.add_argument("--instance", type=Path, help="Path to a DIMACS .col or edge-list file")
    parser.add_argument(
        "--format",
        choices=("auto", "dimacs", "edgelist"),
        default="auto",
        help="Problem-instance format",
    )
    parser.add_argument("--time-limit", type=float, default=None, help="Time limit in seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument(
        "--solver-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Backend-specific option; may be repeated",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow URI, for example http://localhost:5000 or sqlite:///mlflow.db",
    )
    parser.add_argument("--experiment-name", default="vertex-coloring")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--tag", action="append", default=[], metavar="KEY=VALUE", help="MLflow tag"
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
    parser.add_argument("--output", type=Path, default=None, help="Also write result JSON here")
    parser.add_argument("--list-methods", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_methods:
        print("\n".join(available_solvers()))
        return 0
    if args.method is None or args.instance is None:
        parser.error("--method and --instance are required unless --list-methods is used")

    try:
        solver_options = _parse_key_value(args.solver_option)
        tags = {str(k): str(v) for k, v in _parse_key_value(args.tag).items()}
        instance = load_instance(args.instance, args.format)
        solver = create_solver(
            args.method,
            SolverConfig(
                time_limit_seconds=args.time_limit,
                seed=args.seed,
                threads=args.threads,
                verbose=args.verbose_solver,
                options=solver_options,
            ),
        )
        outcome = ExperimentRunner(
            MLflowConfig(
                enabled=not args.no_mlflow,
                tracking_uri=args.tracking_uri,
                experiment_name=args.experiment_name,
                run_name=args.run_name,
                tags=tags,
            )
        ).run(instance, solver)
    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        SolverUnavailableError,
        TrackingUnavailableError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = outcome.result.to_dict()
    payload["instance"] = instance.summary()
    payload["mlflow_run_id"] = outcome.mlflow_run_id
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    return 0 if outcome.result.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE} else 3


if __name__ == "__main__":
    raise SystemExit(main())
