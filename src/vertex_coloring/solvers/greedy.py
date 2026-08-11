from __future__ import annotations

import networkx as nx

from vertex_coloring.models import ColoringInstance, ColoringResult, SolveStatus

from .base import BaseColoringSolver, greedy_clique_lower_bound, normalize_coloring


class GreedyColoringSolver(BaseColoringSolver):
    """Fast NetworkX greedy coloring heuristic."""

    name = "greedy"
    package_name = "networkx"

    def _solve(self, instance: ColoringInstance) -> ColoringResult:
        if instance.num_vertices == 0:
            return ColoringResult(self.name, SolveStatus.OPTIMAL, {}, 0, 0, 0)

        strategy = str(self.config.options.get("strategy", "saturation_largest_first"))
        interchange = bool(self.config.options.get("interchange", False))
        coloring = nx.coloring.greedy_color(
            instance.graph,
            strategy=strategy,
            interchange=interchange,
        )
        normalized = normalize_coloring({str(node): int(color) for node, color in coloring.items()})
        num_colors = len(set(normalized.values()))
        lower_bound = greedy_clique_lower_bound(instance)
        status = SolveStatus.OPTIMAL if num_colors == lower_bound else SolveStatus.FEASIBLE

        return ColoringResult(
            solver_name=self.name,
            status=status,
            coloring=normalized,
            num_colors=num_colors,
            lower_bound=lower_bound,
            upper_bound=num_colors,
            metadata={"strategy": strategy, "interchange": interchange},
        )
