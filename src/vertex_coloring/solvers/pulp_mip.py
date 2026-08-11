from __future__ import annotations

import networkx as nx

from vertex_coloring.models import ColoringInstance, ColoringResult, SolveStatus

from .base import (
    BaseColoringSolver,
    SolverUnavailableError,
    greedy_clique_lower_bound,
    normalize_coloring,
)


class PuLPMIPColoringSolver(BaseColoringSolver):
    """Binary MIP formulation solved through PuLP's CBC interface."""

    name = "pulp"
    package_name = "pulp"

    def _solve(self, instance: ColoringInstance) -> ColoringResult:
        try:
            import pulp
        except ImportError as exc:
            raise SolverUnavailableError(
                'PuLP is not installed. Install with: pip install -e ".[pulp]"'
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
        colors = range(max_colors)

        model = pulp.LpProblem("vertex_coloring", pulp.LpMinimize)
        x = pulp.LpVariable.dicts(
            "x", (range(len(nodes)), colors), lowBound=0, upBound=1, cat=pulp.LpBinary
        )
        y = pulp.LpVariable.dicts(
            "y", colors, lowBound=0, upBound=1, cat=pulp.LpBinary
        )

        model += pulp.lpSum(y[c] for c in colors)
        for i in range(len(nodes)):
            model += pulp.lpSum(x[i][c] for c in colors) == 1
            for c in colors:
                model += x[i][c] <= y[c]

        for u, v in instance.graph.edges:
            i, j = node_index[u], node_index[v]
            for c in colors:
                model += x[i][c] + x[j][c] <= 1

        for c in range(max_colors - 1):
            model += y[c] >= y[c + 1]
        model += x[0][0] == 1

        solver_kwargs: dict[str, object] = {"msg": self.config.verbose}
        if self.config.time_limit_seconds is not None:
            solver_kwargs["timeLimit"] = self.config.time_limit_seconds
        if self.config.threads is not None:
            solver_kwargs["threads"] = self.config.threads
        if "gapRel" in self.config.options:
            solver_kwargs["gapRel"] = float(self.config.options["gapRel"])
        if "gapAbs" in self.config.options:
            solver_kwargs["gapAbs"] = float(self.config.options["gapAbs"])

        solver = pulp.PULP_CBC_CMD(**solver_kwargs)
        model.solve(solver)
        pulp_status = pulp.LpStatus.get(model.status, str(model.status))

        raw_coloring: dict[str, int] = {}
        for node, i in node_index.items():
            selected = [c for c in colors if (pulp.value(x[i][c]) or 0.0) > 0.5]
            if len(selected) == 1:
                raw_coloring[node] = selected[0]

        has_incumbent = len(raw_coloring) == len(nodes)
        if pulp_status == "Optimal":
            status = SolveStatus.OPTIMAL
        elif pulp_status == "Infeasible":
            status = SolveStatus.INFEASIBLE
        elif has_incumbent:
            status = SolveStatus.FEASIBLE
        else:
            status = SolveStatus.UNKNOWN

        if has_incumbent:
            coloring = normalize_coloring(raw_coloring)
            num_colors = len(set(coloring.values()))
            upper_bound = num_colors
            result_lower_bound = num_colors if status is SolveStatus.OPTIMAL else lower_bound
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
                "pulp_status": pulp_status,
                "heuristic_upper_bound": max_colors,
                "objective_value": pulp.value(model.objective),
            },
        )
