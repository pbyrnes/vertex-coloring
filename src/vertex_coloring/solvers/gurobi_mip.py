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


class GurobiMIPColoringSolver(BaseColoringSolver):
    """Binary MIP formulation using the native Gurobi Python API."""

    name = "gurobi"
    package_name = "gurobipy"

    def _solve(self, instance: ColoringInstance) -> ColoringResult:
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:
            raise SolverUnavailableError(
                'gurobipy is not installed. Install with: pip install -e ".[gurobi]"'
            ) from exc

        if instance.num_vertices == 0:
            return ColoringResult(self.name, SolveStatus.OPTIMAL, {}, 0, 0, 0)

        nodes = sorted(instance.graph.nodes)
        node_index = {node: i for i, node in enumerate(nodes)}
        greedy = nx.coloring.greedy_color(
            instance.graph, strategy="saturation_largest_first"
        )
        max_colors = max(greedy.values(), default=0) + 1
        lower_bound = greedy_clique_lower_bound(instance)

        try:
            model = gp.Model("vertex_coloring")
        except gp.GurobiError as exc:
            raise SolverUnavailableError(f"Gurobi could not start: {exc}") from exc

        model.Params.OutputFlag = int(self.config.verbose)
        model.Params.Seed = self.config.seed
        if self.config.time_limit_seconds is not None:
            model.Params.TimeLimit = self.config.time_limit_seconds
        if self.config.threads is not None:
            model.Params.Threads = self.config.threads

        for option_name, option_value in self.config.options.items():
            try:
                model.setParam(option_name, option_value)
            except gp.GurobiError as exc:
                raise ValueError(f"Invalid Gurobi parameter {option_name!r}") from exc

        x = model.addVars(len(nodes), max_colors, vtype=GRB.BINARY, name="x")
        y = model.addVars(max_colors, vtype=GRB.BINARY, name="y")
        model.setObjective(gp.quicksum(y[c] for c in range(max_colors)), GRB.MINIMIZE)

        model.addConstrs(
            (gp.quicksum(x[i, c] for c in range(max_colors)) == 1 for i in range(len(nodes))),
            name="assign",
        )
        model.addConstrs(
            (x[i, c] <= y[c] for i in range(len(nodes)) for c in range(max_colors)),
            name="activate",
        )
        model.addConstrs(
            (
                x[node_index[u], c] + x[node_index[v], c] <= 1
                for u, v in instance.graph.edges
                for c in range(max_colors)
            ),
            name="edge",
        )
        model.addConstrs((y[c] >= y[c + 1] for c in range(max_colors - 1)), name="order")
        model.addConstr(x[0, 0] == 1, name="first_color")
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            status = SolveStatus.OPTIMAL
        elif model.Status == GRB.INFEASIBLE:
            status = SolveStatus.INFEASIBLE
        elif model.SolCount > 0:
            status = SolveStatus.FEASIBLE
        else:
            status = SolveStatus.UNKNOWN

        if model.SolCount > 0:
            raw = {
                node: next(c for c in range(max_colors) if x[node_index[node], c].X > 0.5)
                for node in nodes
            }
            coloring = normalize_coloring(raw)
            num_colors = len(set(coloring.values()))
            upper_bound = num_colors
            result_lower_bound = max(lower_bound, math.ceil(float(model.ObjBound)))
            if status is SolveStatus.OPTIMAL:
                result_lower_bound = num_colors
        else:
            coloring = {}
            num_colors = None
            upper_bound = max_colors
            result_lower_bound = lower_bound

        return ColoringResult(
            solver_name=self.name,
            status=status,
            coloring=coloring,
            num_colors=num_colors,
            lower_bound=result_lower_bound,
            upper_bound=upper_bound,
            metadata={
                "gurobi_status": int(model.Status),
                "node_count": float(model.NodeCount),
                "solution_count": int(model.SolCount),
                "mip_gap": float(model.MIPGap) if model.SolCount > 0 else None,
                "heuristic_upper_bound": max_colors,
            },
        )
