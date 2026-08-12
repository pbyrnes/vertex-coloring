# Vertex Coloring Framework

An installable Python framework for running vertex-coloring experiments through a common
solver API and recording comparable results in MLflow.

## Included methods

| Method                      | Type                                                                   | Dependency | Optimality |
|-----------------------------|------------------------------------------------------------------------|---|---|
| `greedy`                    | NetworkX greedy/DSATUR heuristic                                       | Base install | Usually feasible only |
| `backtracking`              | Pure-Python DSATUR branch-and-bound                                    | Base install | Exact unless timed out |
| `violation`                 | Pure-Python violation reduction search                                 | Base install | Exact unless timed out |
| `ortools`                   | OR-Tools CP-SAT                                                        | Optional | Exact/anytime |
| `pulp`                      | Binary MIP through PuLP/CBC                                            | Optional | Exact/anytime |
| `gurobi`                    | Native Gurobi binary MIP                                               | Optional + license | Exact/anytime |
| `gurobi-representative`     | Native Gurobi binary MIP using a representative-based MIP formulation  | Optional + license | Exact/anytime |
| `pyomo-scip`                | SCIP binary MIP via Pyomo                                              | Optional | Exact/anytime |
| `pyomo-scip-representative` | SCIP binary MIP via Pyomo using a representative-based MIP formulation | Optional | Exact/anytime |

Every method subclasses `BaseColoringSolver` and returns the same `ColoringResult` model.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
```

Install optional open-source optimization backends:

```bash
pip install -e ".[all-open-source]"
```

Or install one backend at a time:

```bash
pip install -e ".[ortools]"
pip install -e ".[pulp]"
pip install -e ".[gurobi]"  # A working Gurobi license is also required.
```

## Run experiments

Run without MLflow:

```bash
vertex-coloring \
  --method backtracking \
  --instance examples/cycle5.col \
  --no-mlflow
```

Start a local MLflow server and UI in another terminal:

```bash
mlflow server --port 5000
```

Then run a tracked CP-SAT experiment:

```bash
vertex-coloring \
  --method ortools \
  --instance examples/cycle5.col \
  --tracking-uri http://localhost:5000 \
  --experiment-name coloring-benchmarks \
  --time-limit 60 \
  --threads 8 \
  --tag dataset=examples
```

A direct local SQLite tracking URI also works without starting an HTTP server first:

```bash
vertex-coloring \
  --method greedy \
  --instance examples/path4.edgelist \
  --tracking-uri sqlite:///mlflow.db
```

List registered methods:

```bash
vertex-coloring --list-methods
```

Pass backend-specific settings as JSON-compatible `KEY=VALUE` pairs:

```bash
vertex-coloring \
  --method ortools \
  --instance examples/cycle5.col \
  --solver-option relative_gap_limit=0.01 \
  --solver-option cp_model_presolve=true \
  --tracking-uri http://localhost:5000
```

For the greedy method:

```bash
vertex-coloring \
  --method greedy \
  --instance examples/cycle5.col \
  --solver-option strategy='"largest_first"' \
  --no-mlflow
```

## Input formats

### DIMACS `.col`

```text
c comment
p edge 5 5
e 1 2
e 2 3
e 3 4
e 4 5
e 5 1
```

### Whitespace edge list

```text
# Single-token lines declare isolated vertices.
A B
B C
C D
isolated_vertex
```

The format is inferred from the extension, or selected with `--format dimacs` or
`--format edgelist`.

## What is logged to MLflow

Each run logs:

- Parameters: method, instance size, seed, thread count, time limit, and solver options.
- Metrics: color count, lower/upper bounds, absolute gap, runtime, and validity.
- Tags: solver, status, instance name, graph fingerprint, and custom CLI tags.
- Artifacts: normalized result JSON, solver config, instance summary, original instance file,
  and a traceback artifact on failure.

## Add another solver

Create a subclass and implement `_solve`:

```python
from vertex_coloring.models import ColoringResult, SolveStatus
from vertex_coloring.solvers.base import BaseColoringSolver


class MyColoringSolver(BaseColoringSolver):
    name = "my_solver"

    def _solve(self, instance):
        coloring = {node: i for i, node in enumerate(instance.graph.nodes)}
        return ColoringResult(
            solver_name=self.name,
            status=SolveStatus.FEASIBLE,
            coloring=coloring,
            num_colors=len(coloring),
            upper_bound=len(coloring),
        )
```

Register it before calling `create_solver`:

```python
from vertex_coloring import register_solver

register_solver("my_solver", MyColoringSolver)
```

For a permanent built-in method, add its import target to `_BUILTIN_SOLVERS` in
`src/vertex_coloring/registry.py`. Optional libraries should be imported inside `_solve`, so
users can install the base framework without every backend.

## Test

```bash
pip install -e ".[dev]"
pytest
```
