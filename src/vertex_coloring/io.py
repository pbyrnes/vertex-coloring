from __future__ import annotations

from pathlib import Path

import networkx as nx

from .models import ColoringInstance


def load_instance(path: str | Path, file_format: str = "auto") -> ColoringInstance:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Problem instance not found: {source}")

    fmt = file_format.lower()
    if fmt == "auto":
        fmt = "dimacs" if source.suffix.lower() in {".col", ".dimacs"} else "edgelist"

    if fmt == "dimacs":
        graph, metadata = _read_dimacs(source)
    elif fmt == "edgelist":
        graph, metadata = _read_edgelist(source)
    else:
        raise ValueError(f"Unknown instance format: {file_format!r}")

    return ColoringInstance(
        name=source.stem,
        graph=graph,
        source_path=source,
        metadata={"format": fmt, **metadata},
    )


def _read_dimacs(path: Path) -> tuple[nx.Graph, dict[str, int]]:
    graph = nx.Graph()
    declared_vertices: int | None = None
    declared_edges: int | None = None

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("c"):
                continue

            tokens = line.split()
            marker = tokens[0].lower()
            if marker == "p":
                if len(tokens) < 4 or tokens[1].lower() not in {"edge", "col"}:
                    raise ValueError(f"Invalid DIMACS problem line at {path}:{line_number}")
                declared_vertices = int(tokens[2])
                declared_edges = int(tokens[3])
                graph.add_nodes_from(str(i) for i in range(1, declared_vertices + 1))
            elif marker == "e":
                if len(tokens) != 3:
                    raise ValueError(f"Invalid DIMACS edge line at {path}:{line_number}")
                u, v = tokens[1], tokens[2]
                if u == v:
                    raise ValueError(f"Self-loop at {path}:{line_number}")
                graph.add_edge(u, v)
            else:
                raise ValueError(f"Unknown DIMACS record at {path}:{line_number}: {marker!r}")

    if declared_vertices is not None and graph.number_of_nodes() != declared_vertices:
        raise ValueError("DIMACS vertex count does not match the parsed graph")
    if declared_edges is not None and graph.number_of_edges() != declared_edges:
        raise ValueError("DIMACS edge count does not match the parsed graph")

    return graph, {
        "declared_vertices": declared_vertices or graph.number_of_nodes(),
        "declared_edges": declared_edges if declared_edges is not None else graph.number_of_edges(),
    }


def _read_edgelist(path: Path) -> tuple[nx.Graph, dict[str, int]]:
    graph = nx.Graph()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith(("#", "%", "//")):
                continue
            tokens = line.split()
            if len(tokens) == 1:
                graph.add_node(tokens[0])
            elif len(tokens) >= 2:
                u, v = tokens[0], tokens[1]
                if u == v:
                    raise ValueError(f"Self-loop at {path}:{line_number}")
                graph.add_edge(u, v)
            else:
                raise ValueError(f"Invalid edge-list line at {path}:{line_number}")

    return graph, {}
