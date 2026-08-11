from __future__ import annotations

from time import perf_counter

import networkx as nx

from vertex_coloring.models import ColoringInstance, ColoringResult, SolveStatus

from .base import BaseColoringSolver, greedy_clique_lower_bound, normalize_coloring


class BacktrackingColoringSolver(BaseColoringSolver):
    """Exact DSATUR-style branch-and-bound solver implemented in pure Python."""

    name = "backtracking"
    package_name = "vertex-coloring-framework"

    def _solve(self, instance: ColoringInstance) -> ColoringResult:
        graph = instance.graph
        if instance.num_vertices == 0:
            return ColoringResult(self.name, SolveStatus.OPTIMAL, {}, 0, 0, 0)

        initial = nx.coloring.greedy_color(graph, strategy="saturation_largest_first")
        best_coloring = normalize_coloring({str(k): int(v) for k, v in initial.items()})
        best_num_colors = len(set(best_coloring.values()))
        lower_bound = greedy_clique_lower_bound(instance)

        if best_num_colors == lower_bound:
            return ColoringResult(
                self.name,
                SolveStatus.OPTIMAL,
                best_coloring,
                best_num_colors,
                best_num_colors,
                best_num_colors,
                metadata={"search_nodes": 0, "timed_out": False},
            )

        adjacency = {node: set(graph.neighbors(node)) for node in graph.nodes}
        assignment: dict[str, int] = {}
        started = perf_counter()
        timed_out = False
        search_nodes = 0

        def time_exceeded() -> bool:
            limit = self.config.time_limit_seconds
            return limit is not None and perf_counter() - started >= limit

        def choose_vertex() -> str:
            uncolored = (node for node in graph.nodes if node not in assignment)
            return max(
                uncolored,
                key=lambda node: (
                    len({assignment[nbr] for nbr in adjacency[node] if nbr in assignment}),
                    graph.degree[node],
                    str(node),
                ),
            )

        def search(colors_in_use: int) -> None:
            nonlocal best_coloring, best_num_colors, timed_out, search_nodes
            if timed_out or best_num_colors == lower_bound:
                return
            if time_exceeded():
                timed_out = True
                return

            search_nodes += 1
            if len(assignment) == instance.num_vertices:
                if colors_in_use < best_num_colors:
                    best_num_colors = colors_in_use
                    best_coloring = normalize_coloring(dict(assignment))
                return

            if colors_in_use >= best_num_colors:
                return

            node = choose_vertex()
            forbidden = {assignment[nbr] for nbr in adjacency[node] if nbr in assignment}

            # Existing colors first, then at most one new color. This removes color-label symmetry.
            for color in range(min(colors_in_use + 1, best_num_colors)):
                if color in forbidden:
                    continue
                next_colors_in_use = max(colors_in_use, color + 1)
                if next_colors_in_use >= best_num_colors:
                    continue
                assignment[node] = color
                search(next_colors_in_use)
                del assignment[node]
                if timed_out or best_num_colors == lower_bound:
                    return

        search(0)
        optimal = not timed_out or best_num_colors == lower_bound
        return ColoringResult(
            solver_name=self.name,
            status=SolveStatus.OPTIMAL if optimal else SolveStatus.FEASIBLE,
            coloring=best_coloring,
            num_colors=best_num_colors,
            lower_bound=best_num_colors if optimal else lower_bound,
            upper_bound=best_num_colors,
            metadata={"search_nodes": search_nodes, "timed_out": timed_out},
        )
