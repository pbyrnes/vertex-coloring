from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import networkx as nx


class SolveStatus(str, Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Configuration shared by all solver implementations."""

    time_limit_seconds: float | None = None
    seed: int = 0
    threads: int | None = None
    verbose: bool = False
    options: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_limit_seconds": self.time_limit_seconds,
            "seed": self.seed,
            "threads": self.threads,
            "verbose": self.verbose,
            "options": self.options,
        }


@dataclass(slots=True)
class ColoringInstance:
    """A graph-coloring problem instance.

    Node labels are normalized to strings to make result artifacts deterministic and
    JSON-compatible.
    """

    name: str
    graph: nx.Graph
    source_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.graph.is_directed():
            raise ValueError("Vertex coloring requires an undirected graph.")
        if nx.number_of_selfloops(self.graph):
            raise ValueError("Self-loops make a proper vertex coloring infeasible.")

        mapping = {node: str(node) for node in self.graph.nodes}
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("Node labels are not unique after conversion to strings.")
        self.graph = nx.relabel_nodes(self.graph, mapping, copy=True)

    @property
    def num_vertices(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    @property
    def density(self) -> float:
        return float(nx.density(self.graph)) if self.num_vertices > 1 else 0.0

    @property
    def fingerprint(self) -> str:
        nodes = sorted(self.graph.nodes)
        edges = sorted(tuple(sorted((str(u), str(v)))) for u, v in self.graph.edges)
        payload = json.dumps({"nodes": nodes, "edges": edges}, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_vertices": self.num_vertices,
            "num_edges": self.num_edges,
            "density": self.density,
            "fingerprint": self.fingerprint,
            "source_path": str(self.source_path) if self.source_path else None,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ColoringResult:
    solver_name: str
    status: SolveStatus
    coloring: dict[str, int] = field(default_factory=dict)
    num_colors: int | None = None
    lower_bound: int | None = None
    upper_bound: int | None = None
    runtime_seconds: float = 0.0
    solver_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_solution(self) -> bool:
        return self.status in {SolveStatus.OPTIMAL, SolveStatus.FEASIBLE} and bool(
            self.coloring or self.num_colors == 0
        )

    def validate(self, instance: ColoringInstance) -> tuple[bool, str | None]:
        expected = set(instance.graph.nodes)
        supplied = set(self.coloring)
        if expected != supplied:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            return False, f"Coloring node mismatch; missing={missing}, extra={extra}"

        for node, color in self.coloring.items():
            if not isinstance(color, int) or color < 0:
                return False, f"Node {node!r} has invalid color {color!r}"

        for u, v in instance.graph.edges:
            if self.coloring[u] == self.coloring[v]:
                return False, f"Edge ({u}, {v}) has equal endpoint colors"

        used = len(set(self.coloring.values()))
        if self.num_colors != used:
            return False, f"num_colors={self.num_colors}, but coloring uses {used} colors"
        return True, None

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver_name": self.solver_name,
            "solver_version": self.solver_version,
            "status": self.status.value,
            "num_colors": self.num_colors,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "runtime_seconds": self.runtime_seconds,
            "coloring": dict(sorted(self.coloring.items())),
            "metadata": self.metadata,
        }
