from __future__ import annotations

import networkx as nx

from vertex_coloring.models import ColoringInstance, SolveStatus, SolverConfig
from vertex_coloring.registry import create_solver


def solve(method: str, graph: nx.Graph):
    instance = ColoringInstance(name="test", graph=graph)
    return create_solver(method, SolverConfig()).solve(instance)


def test_greedy_colors_cycle_five() -> None:
    result = solve("greedy", nx.cycle_graph(5))
    assert result.num_colors == 3
    assert result.status in {SolveStatus.FEASIBLE, SolveStatus.OPTIMAL}


def test_backtracking_proves_cycle_five_optimal() -> None:
    result = solve("backtracking", nx.cycle_graph(5))
    assert result.status is SolveStatus.OPTIMAL
    assert result.num_colors == 3
    assert result.lower_bound == 3
    assert result.upper_bound == 3


def test_backtracking_complete_graph() -> None:
    result = solve("backtracking", nx.complete_graph(6))
    assert result.status is SolveStatus.OPTIMAL
    assert result.num_colors == 6


def test_backtracking_empty_graph() -> None:
    result = solve("backtracking", nx.Graph())
    assert result.status is SolveStatus.OPTIMAL
    assert result.num_colors == 0
    assert result.coloring == {}
