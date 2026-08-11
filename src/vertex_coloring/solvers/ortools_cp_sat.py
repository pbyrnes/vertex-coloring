from __future__ import annotations

import math

import networkx as nx

from vertex_coloring.models import ColoringInstance, ColoringResult, SolveStatus

from .base import (
    BaseColoringSolver,
    SolverUnavailableError,
    greedy_clique_lower_bound,
    normalize_coloring,
)


class ORToolsCPSATColoringSolver(BaseColoringSolver):
    """Exact/anytime CP-SAT formulation using integer color variables."""

    name = "ortools"
    package_name = "ortools"

    def _solve(self, instance: ColoringInstance) -> ColoringResult:
        try:
            from ortools.sat.python import cp_model
        except ImportError as exc:
            raise SolverUnavailableError(
                'OR-Tools is not installed. Install with: pip install -e ".[ortools]"'
            ) from exc

        if instance.num_vertices == 0:
            return ColoringResult(self.name, SolveStatus.OPTIMAL, {}, 0, 0, 0)

        nodes = sorted(instance.graph.nodes)
        greedy = nx.coloring.greedy_color(
            instance.graph, strategy="saturation_largest_first"
        )
        heuristic_upper_bound = max(greedy.values(), default=0) + 1
        lower_bound = greedy_clique_lower_bound(instance)

        model = cp_model.CpModel()
        color = {
            node: model.new_int_var(0, heuristic_upper_bound - 1, f"color_{i}")
            for i, node in enumerate(nodes)
        }
        for u, v in instance.graph.edges:
            model.add(color[u] != color[v])

        # Any coloring can be relabeled so the first node has color zero.
        model.add(color[nodes[0]] == 0)
        max_color = model.new_int_var(0, heuristic_upper_bound - 1, "max_color")
        model.add_max_equality(max_color, [color[node] for node in nodes])
        model.minimize(max_color)

        solver = cp_model.CpSolver()
        if self.config.time_limit_seconds is not None:
            solver.parameters.max_time_in_seconds = self.config.time_limit_seconds
        if self.config.threads is not None:
            solver.parameters.num_search_workers = self.config.threads
        solver.parameters.random_seed = self.config.seed
        solver.parameters.log_search_progress = self.config.verbose

        for option_name, option_value in self.config.options.items():
            if option_name in {"strategy", "interchange"}:
                continue
            if not hasattr(solver.parameters, option_name):
                raise ValueError(f"Unknown CP-SAT parameter: {option_name}")
            setattr(solver.parameters, option_name, option_value)

        status_code = solver.solve(model)
        if status_code == cp_model.OPTIMAL:
            status = SolveStatus.OPTIMAL
        elif status_code == cp_model.FEASIBLE:
            status = SolveStatus.FEASIBLE
        elif status_code == cp_model.INFEASIBLE:
            status = SolveStatus.INFEASIBLE
        else:
            status = SolveStatus.UNKNOWN

        if status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE}:
            raw = {node: int(solver.value(color[node])) for node in nodes}
            coloring = normalize_coloring(raw)
            num_colors = len(set(coloring.values()))
            bound = max(lower_bound, math.ceil(float(solver.best_objective_bound)) + 1)
            if status is SolveStatus.OPTIMAL:
                bound = num_colors
            upper_bound = num_colors
        else:
            coloring = {}
            num_colors = None
            bound = lower_bound
            upper_bound = heuristic_upper_bound

        return ColoringResult(
            solver_name=self.name,
            status=status,
            coloring=coloring,
            num_colors=num_colors,
            lower_bound=bound,
            upper_bound=upper_bound,
            metadata={
                "cp_sat_status": solver.status_name(status_code),
                "conflicts": solver.num_conflicts,
                "branches": solver.num_branches,
                "wall_time": solver.wall_time,
                "heuristic_upper_bound": heuristic_upper_bound,
            },
        )
