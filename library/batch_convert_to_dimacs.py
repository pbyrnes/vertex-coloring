#!/usr/bin/env python3
"""Convert every graph instance in a data folder to DIMACS format."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run convert_gc_to_dimacs.py for every file in a folder."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Folder containing source graph instances (default: data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples"),
        help="Folder for converted .col files (default: data_dimacs).",
    )
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path("library/convert_gc_to_dimacs.py"),
        help="Path to the converter script.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process files in subdirectories recursively.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.converter.is_file():
        print(f"Error: converter not found: {args.converter}", file=sys.stderr)
        return 2

    if not args.data_dir.is_dir():
        print(f"Error: data folder not found: {args.data_dir}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = args.data_dir.rglob("*") if args.recursive else args.data_dir.iterdir()
    source_files = sorted(path for path in paths if path.is_file())

    if not source_files:
        print(f"No files found in {args.data_dir}")
        return 0

    converted = 0
    skipped = 0
    failed = 0

    for source in source_files:
        relative_path = source.relative_to(args.data_dir)
        output = args.output_dir / relative_path.with_suffix(".col")
        output.parent.mkdir(parents=True, exist_ok=True)

        if output.exists() and not args.overwrite:
            print(f"SKIP  {source} -> {output} (already exists)")
            skipped += 1
            continue

        command = [
            sys.executable,
            str(args.converter),
            str(source),
            str(output),
        ]

        print(f"CONVERT  {source} -> {output}")
        result = subprocess.run(command, check=False)

        if result.returncode == 0:
            converted += 1
        else:
            print(
                f"FAILED  {source} (converter exit code {result.returncode})",
                file=sys.stderr,
            )
            failed += 1

    print(
        f"\nFinished: {converted} converted, "
        f"{skipped} skipped, {failed} failed."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
