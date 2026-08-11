from __future__ import annotations

from abc import ABC, abstractmethod
from importlib.metadata import PackageNotFoundError, version
from time import perf_counter

from vertex_coloring.models import ColoringInstance, ColoringResult, SolverConfig


class SolverUnavailableError(RuntimeError):
    """Raised when an optional solver dependency is not installed or licensed."""


class BaseColoringSolver(ABC):
    """Base class for all vertex-coloring solution methods."""

    name = "base"
    package_name: str | None = None

    def __init__(self, config: SolverConfig) -> None:
        self.config = config

    @property
    def solver_version(self) -> str | None:
        if not self.package_name:
            return None
        try:
            return version(self.package_name)
        except PackageNotFoundError:
            return None

    def solve(self, instance: ColoringInstance) -> ColoringResult:
        started = perf_counter()
        result = self._solve(instance)
        result.runtime_seconds = perf_counter() - started
        result.solver_name = self.name
        result.solver_version = self.solver_version

        if result.has_solution:
            valid, error = result.validate(instance)
            if not valid:
                raise RuntimeError(f"{self.name} returned an invalid coloring: {error}")
        return result

    @abstractmethod
    def _solve(self, instance: ColoringInstance) -> ColoringResult:
        """Implement the solver-specific optimization logic."""


def normalize_coloring(coloring: dict[str, int]) -> dict[str, int]:
    """Map arbitrary color IDs to consecutive integers in first-use order."""

    remap: dict[int, int] = {}
    normalized: dict[str, int] = {}
    for node in sorted(coloring):
        old = int(coloring[node])
        if old not in remap:
            remap[old] = len(remap)
        normalized[node] = remap[old]
    return normalized


def greedy_clique_lower_bound(instance: ColoringInstance) -> int:
    """Return the size of a greedily constructed clique, hence a valid lower bound."""

    graph = instance.graph
    if graph.number_of_nodes() == 0:
        return 0

    best = 1
    ordered = sorted(graph.nodes, key=lambda node: (-graph.degree[node], str(node)))
    for first in ordered:
        clique = [first]
        candidates = sorted(
            graph.neighbors(first), key=lambda node: (-graph.degree[node], str(node))
        )
        for candidate in candidates:
            if all(graph.has_edge(candidate, member) for member in clique):
                clique.append(candidate)
        best = max(best, len(clique))
    return best
