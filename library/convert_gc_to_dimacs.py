#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

_COMMENT_PREFIXES = ("#", "%", "//")


def _data_lines(path: Path) -> Iterable[tuple[int, str]]:
    """Yield nonempty, noncomment lines as ``(line_number, text)`` pairs."""
    with path.open("r", encoding="utf-8-sig", newline=None) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith(_COMMENT_PREFIXES):
                continue
            yield line_number, line


def read_simple_graph(
    path: str | Path,
    *,
    input_index_base: int = 0,
) -> tuple[int, list[tuple[int, int]]]:
    """Read the project's example source format.

    The expected format is::

        <number of vertices> <number of edges>
        <u_1> <v_1>
        ...
        <u_m> <v_m>

    Vertex labels must be consecutive integers starting at ``input_index_base``.
    The graph is treated as finite, simple, and undirected.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input graph not found: {source}")
    if input_index_base not in {0, 1}:
        raise ValueError("input_index_base must be 0 or 1")

    records = iter(_data_lines(source))
    try:
        header_line_number, header = next(records)
    except StopIteration as exc:
        raise ValueError(f"Input graph is empty: {source}") from exc

    header_tokens = header.split()
    if len(header_tokens) != 2:
        raise ValueError(
            f"Expected '<vertices> <edges>' at {source}:{header_line_number}, got {header!r}"
        )

    try:
        num_vertices, declared_edges = map(int, header_tokens)
    except ValueError as exc:
        raise ValueError(
            f"Header values must be integers at {source}:{header_line_number}"
        ) from exc

    if num_vertices < 0 or declared_edges < 0:
        raise ValueError(
            f"Vertex and edge counts must be nonnegative at {source}:{header_line_number}"
        )

    minimum_vertex = input_index_base
    maximum_vertex = input_index_base + num_vertices - 1
    edges: list[tuple[int, int]] = []
    seen_edges: set[tuple[int, int]] = set()

    for line_number, line in records:
        tokens = line.split()
        if len(tokens) != 2:
            raise ValueError(
                f"Expected one edge '<u> <v>' at {source}:{line_number}, got {line!r}"
            )
        try:
            u, v = map(int, tokens)
        except ValueError as exc:
            raise ValueError(f"Edge endpoints must be integers at {source}:{line_number}") from exc

        if not minimum_vertex <= u <= maximum_vertex:
            raise ValueError(
                f"Vertex {u} at {source}:{line_number} is outside "
                f"[{minimum_vertex}, {maximum_vertex}]"
            )
        if not minimum_vertex <= v <= maximum_vertex:
            raise ValueError(
                f"Vertex {v} at {source}:{line_number} is outside "
                f"[{minimum_vertex}, {maximum_vertex}]"
            )
        if u == v:
            raise ValueError(f"Self-loop ({u}, {v}) at {source}:{line_number}")

        normalized = (u, v) if u < v else (v, u)
        if normalized in seen_edges:
            raise ValueError(f"Duplicate undirected edge {normalized} at {source}:{line_number}")
        seen_edges.add(normalized)
        edges.append((u, v))

    if len(edges) != declared_edges:
        raise ValueError(
            f"Header declares {declared_edges} edges, but {len(edges)} edge records were read "
            f"from {source}"
        )

    return num_vertices, edges


def convert_simple_to_dimacs(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    input_index_base: int = 0,
    overwrite: bool = False,
) -> Path:
    """Convert a header edge list to a DIMACS ``.col`` graph file.

    DIMACS vertex labels are emitted as one-based integers, regardless of the input base.
    The output path defaults to replacing the input suffix with ``.col``.
    """
    source = Path(input_path).expanduser().resolve()
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else source.with_suffix(".col")
    )

    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {destination}. Pass --overwrite to replace it."
        )
    if source == destination:
        raise ValueError("Input and output paths must be different")

    num_vertices, edges = read_simple_graph(source, input_index_base=input_index_base)
    shift = 1 - input_index_base

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"c Converted from {source.name}\n")
        handle.write(
            "c Source format: '<vertices> <edges>' header followed by integer edge pairs\n"
        )
        handle.write(f"p edge {num_vertices} {len(edges)}\n")
        for u, v in edges:
            handle.write(f"e {u + shift} {v + shift}\n")

    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vertex-coloring-convert",
        description=(
            "Convert a graph whose first line is '<vertices> <edges>' and whose remaining "
            "lines are integer edge pairs into DIMACS .col format."
        ),
    )
    parser.add_argument("input", type=Path, help="Source graph file")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Destination .col file; defaults to <input filename>.col",
    )
    parser.add_argument(
        "--input-index-base",
        type=int,
        choices=(0, 1),
        default=0,
        help="Smallest source vertex ID (default: 0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        destination = convert_simple_to_dimacs(
            args.input,
            args.output,
            input_index_base=args.input_index_base,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(destination)
    return 0


if __name__ == "__main__":


    raise SystemExit(main())
