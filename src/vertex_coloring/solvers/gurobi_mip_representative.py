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


class GurobiMIPRepresentativeColoringSolver(BaseColoringSolver):
    """Binary MIP formulation using the native Gurobi Python API."""

    name = "gurobi-representative"
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
        x = dict()
        for i in range(len(nodes)):
            for v in nx.non_neighbors(instance.graph, nodes[i]):
                j = node_index[v]
                if i <= j:
                    x[i, j] = model.addVar(name=f'x_{i}_{j}', vtype=gp.GRB.BINARY)
            x[i, i] = model.addVar(name=f'x_{i}_{i}', vtype=gp.GRB.BINARY)
        model.setObjective(gp.quicksum(x[i, i] for i in range(len(nodes))), GRB.MINIMIZE)
        model.addConstrs(
            (gp.quicksum(x[node_index[j], i] for j in nx.non_neighbors(instance.graph, nodes[i]) if node_index[j] <= i) + x[i, i] == 1 for i in range(len(nodes))),
        )
        pairs_to_check = set(x.keys())
        for u, v in instance.graph.edges:
            for w in set(nx.non_neighbors(instance.graph, u)).intersection(set(nx.non_neighbors(instance.graph, v))):
                i, j, k = node_index[u], node_index[v], node_index[w]
                if k <= i and k <= j:
                    model.addConstr(x[k, i] + x[k, j] <= x[k, k])
                    pairs_to_check.discard((k, i))
                    pairs_to_check.discard((k, j))
        for i, j in pairs_to_check:
            if i != j:
                model.addConstr(x[i, j] <= x[i, i])

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
                node: next(node_index[v] for v in nx.non_neighbors(instance.graph, node) if node_index[v] < node_index[node] and x[node_index[v], node_index[node]].X > 0.5) for node in nodes if x[node_index[node], node_index[node]].X < 0.5
            }
            for node in nodes:
                if x[node_index[node], node_index[node]].X > 0.5:
                    raw[node] = node_index[node]
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
