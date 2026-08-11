from __future__ import annotations

import math
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition, WriterFactory

import networkx as nx

from vertex_coloring.models import ColoringInstance, ColoringResult, SolveStatus

from .base import (
    BaseColoringSolver,
    SolverUnavailableError,
    greedy_clique_lower_bound,
    normalize_coloring,
)


class PyomoMIPColoringSolver(BaseColoringSolver):
    name = "pyomo"
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
        I = list(range(len(nodes)))
        model.I = pyo.Set(initialize=I)
        colors = list(range(max_colors))
        model.colors = pyo.Set(initialize=colors)
        colors2 = list(range(max_colors-1))
        model.colors2 = pyo.Set(initialize=colors2)
        E = list((node_index[i], node_index[j]) for i, j in instance.graph.edges)
        model.E = pyo.Set(initialize=E, dimen=2)

        model.x = pyo.Var(model.I, model.colors, within=pyo.Binary)
        model.y = pyo.Var(model.colors, within=pyo.Binary)

        def objective_function(model):
            return sum(model.y[c] for c in model.colors)

        model.objective = pyo.Objective(expr=objective_function, sense=pyo.minimize)

        def assignment_constraint(model, i):
            return sum(model.x[i, c] for c in model.colors) == 1

        model.assignment_constraint = pyo.Constraint(model.I, rule=assignment_constraint)

        def activate_constraint(model, i, c):
            return model.x[i, c] <= model.y[c]

        model.assign_constraint = pyo.Constraint(model.I, model.colors, rule=activate_constraint)

        def edge_constraint(model, c, e1, e2):
            return model.x[e1, c] + model.x[e2, c] <= 1

        model.edge_constraint = pyo.Constraint(model.colors, model.E, rule=edge_constraint)

        def order_constraint(model, c):
            return model.y[c] >= model.y[c+1]

        model.order_constraint = pyo.Constraint(model.colors2, rule=order_constraint)

        def first_color_constraint(model):
            return model.x[0, 0] == 1

        model.first_color_constraint = pyo.Constraint(rule=first_color_constraint)

        # writer = WriterFactory('nl')
        model.write('scip_mip.mps')

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
        #
        # if model.Status == GRB.OPTIMAL:
        #     status = SolveStatus.OPTIMAL
        # elif model.Status == GRB.INFEASIBLE:
        #     status = SolveStatus.INFEASIBLE
        # elif model.SolCount > 0:
        #     status = SolveStatus.FEASIBLE
        # else:
        #     status = SolveStatus.UNKNOWN

        if status == SolveStatus.OPTIMAL or status == SolveStatus.FEASIBLE:
            # print(list((i, c, pyo.value(model.x[i, c])) for i in range(len(nodes)) for c in range(max_colors) if pyo.value(model.x[i, c]) > -0.5))
            # print('bbbb')
            raw = {
                nodes[i]: next(c for c in range(max_colors) if pyo.value(model.x[i, c]) > 0.5)
                for i in range(len(nodes))
            }
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
