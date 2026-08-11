from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import SolverConfig
    from .solvers.base import BaseColoringSolver

_BUILTIN_SOLVERS: dict[str, str] = {
    "greedy": "vertex_coloring.solvers.greedy:GreedyColoringSolver",
    "backtracking": "vertex_coloring.solvers.backtracking:BacktrackingColoringSolver",
    "violation": "vertex_coloring.solvers.violation:ViolationColoringSolver",
    "ortools": "vertex_coloring.solvers.ortools_cp_sat:ORToolsCPSATColoringSolver",
    "pulp": "vertex_coloring.solvers.pulp_mip:PuLPMIPColoringSolver",
    "pyomo-scip": "vertex_coloring.solvers.pyomo_mip:PyomoMIPColoringSolver",
    "pyomo-scip-representative": "vertex_coloring.solvers.pyomo_mip_representative:PyomoMIPRepresentativeColoringSolver",
    "gurobi": "vertex_coloring.solvers.gurobi_mip:GurobiMIPColoringSolver",
    "gurobi-representative": "vertex_coloring.solvers.gurobi_mip_representative:GurobiMIPRepresentativeColoringSolver",
}
_EXTERNAL_SOLVERS: dict[str, type[BaseColoringSolver]] = {}


def register_solver(name: str, solver_class: type[BaseColoringSolver]) -> None:
    """Register an application-specific solver implementation at runtime."""

    key = name.strip().lower()
    if not key:
        raise ValueError("Solver name cannot be empty")
    if key in _BUILTIN_SOLVERS or key in _EXTERNAL_SOLVERS:
        raise ValueError(f"A solver named {key!r} is already registered")
    _EXTERNAL_SOLVERS[key] = solver_class


def available_solvers() -> tuple[str, ...]:
    return tuple(sorted({*_BUILTIN_SOLVERS, *_EXTERNAL_SOLVERS}))


def create_solver(name: str, config: SolverConfig) -> BaseColoringSolver:
    key = name.strip().lower()
    if key in _EXTERNAL_SOLVERS:
        return _EXTERNAL_SOLVERS[key](config)

    target = _BUILTIN_SOLVERS.get(key)
    if target is None:
        choices = ", ".join(available_solvers())
        raise KeyError(f"Unknown solver {name!r}. Available solvers: {choices}")

    module_name, class_name = target.split(":", maxsplit=1)
    module = import_module(module_name)
    solver_class = getattr(module, class_name)
    return solver_class(config)
