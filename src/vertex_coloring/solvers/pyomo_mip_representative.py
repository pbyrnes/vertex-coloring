from __future__ import annotations

import math
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

import networkx as nx

from vertex_coloring.models import ColoringInstance, ColoringResult, SolveStatus

from .base import (
    BaseColoringSolver,
    SolverUnavailableError,
    greedy_clique_lower_bound,
    normalize_coloring,
)


class PyomoMIPRepresentativeColoringSolver(BaseColoringSolver):
    name = "pyomo-representative"
    package_name = "pyomo"

    def _solve(self, instance: ColoringInstance) -> ColoringResult:
        if instance.num_vertices == 0:
            return ColoringResult(self.name, SolveStatus.OPTIMAL, {}, 0, 0, 0)

        nodes = sorted(instance.graph.nodes)
        node_index = {node: i for i, node in enumerate(nodes)}
        greedy = nx.coloring.greedy_color(
            instance.graph, strategy="saturation_largest_first"
        )
        max_colors = max(greedy.values(), default=0) + 1
        lower_bound = greedy_clique_lower_bound(instance)

        model = pyo.ConcreteModel()

        # the nodes
        N = list(range(len(nodes)))
        model.N = pyo.Set(initialize=N)

        non_edges = set((i, j) for i in range(len(nodes)) for j in range(i+1, len(nodes)) if not instance.graph.has_edge(nodes[i], nodes[j]))
        extra_non_edges = non_edges.copy()
        non_adjacent_edges = []
        for u, v in instance.graph.edges:
            for w in set(nx.non_neighbors(instance.graph, u)).intersection(nx.non_neighbors(instance.graph, v)):
                i, j, k = node_index[u], node_index[v], node_index[w]
                if k < i and k < j:
                    non_adjacent_edges.append((i, j, k))
                    extra_non_edges.discard((k, i))
                    extra_non_edges.discard((k, j))
        model.non_adjacent_edges = pyo.Set(initialize=non_adjacent_edges, dimen=3)
        model.extra_non_edges = pyo.Set(initialize=list(extra_non_edges), dimen=2)
        model.non_edges = pyo.Set(initialize=list(non_edges), dimen=2)

        non_neighbor_dict = dict()
        for i in range(len(nodes)):
            non_neighbor_dict[i] = [j for j in range(i) if not instance.graph.has_edge(nodes[i], nodes[j])]

        def non_neighbor_for_node(model, i):
            return non_neighbor_dict[i]

        model.non_neighbors = pyo.Set(model.N, initialize=non_neighbor_for_node)

        # y[i] == 1 iff node i is representative for color i
        model.y = pyo.Var(model.N, within=pyo.Binary)
        model.x = pyo.Var(model.non_edges, within=pyo.Binary)

        def objective_function(model):
            return sum(model.y[i] for i in model.N)

        model.objective = pyo.Objective(expr=objective_function, sense=pyo.minimize)

        def edge_constraint(model, i, j, k):
            return model.x[k, i] + model.x[k, j] <= model.y[k]

        model.edge_constraint = pyo.Constraint(model.non_adjacent_edges, rule=edge_constraint)

        def valid_rep_constraint(model, k, i):
            return model.x[k, i] <= model.y[k]

        model.valid_rep_constraint = pyo.Constraint(model.extra_non_edges, rule=valid_rep_constraint)

        def assign_color_constraint(model, j):
            return model.y[j] + sum(model.x[i, j] for i in model.non_neighbors[j]) == 1

        model.assign_color_constraints = pyo.Constraint(model.N, rule=assign_color_constraint)

        solver = SolverFactory('scip', solver_io='nl')

        options = dict()
        if self.config.time_limit_seconds is not None:
            options['limits/time'] = self.config.time_limit_seconds
        results = solver.solve(model, options=options)

        if results.solver.status == SolverStatus.ok and results.solver.termination_condition == TerminationCondition.optimal:
            status = SolveStatus.OPTIMAL
        elif results.solver.termination_condition == TerminationCondition.feasible:
            status = SolveStatus.FEASIBLE
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            status = SolveStatus.INFEASIBLE
        else:
            status = SolveStatus.UNKNOWN

        if status == SolveStatus.OPTIMAL or status == SolveStatus.FEASIBLE:
            raw = {
                nodes[i]: i for i in range(len(nodes)) if pyo.value(model.y[i]) > 0.5
            }
            for i in range(len(nodes)):
                if nodes[i] not in raw:
                    raw[nodes[i]] = next(k for k in model.non_neighbors[i] if pyo.value(model.x[k, i]) > 0.5)
            coloring = normalize_coloring(raw)
            num_colors = len(set(coloring.values()))
            upper_bound = num_colors
            result_lower_bound = max(lower_bound, math.ceil(float(pyo.value(model.objective))))
            if status is SolveStatus.OPTIMAL:
                result_lower_bound = num_colors
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
                "scip_status": results.solver.status,
                # "node_count": float(model.getNTotalNodes()),
                # "solution_count": int(model.SolCount),
                # "mip_gap": float(model.MIPGap) if model.SolCount > 0 else None,
                "heuristic_upper_bound": max_colors,
            },
        )
