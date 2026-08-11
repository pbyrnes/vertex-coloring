import itertools
import math
import random
from collections import defaultdict
from time import perf_counter

import networkx as nx

from vertex_coloring import ColoringResult, ColoringInstance, SolveStatus
from vertex_coloring.solvers import BaseColoringSolver
from vertex_coloring.solvers.base import normalize_coloring, greedy_clique_lower_bound


def normalize(solution):
    color_dict = defaultdict(list)
    for i, c in enumerate(solution):
        color_dict[c].append(i)
    color_blocks = sorted(list(color_dict.values()), key=lambda x: x[0])
    return hash(tuple(tuple(b) for b in color_blocks))


class ViolationColoringSolver(BaseColoringSolver):
    name = "violation"
    package_name = "vertex-coloring-framework"


    def violation_search(self, node_count, edges, num_colors, solution):
        def time_exceeded() -> bool:
            limit = self.config.time_limit_seconds
            return limit is not None and perf_counter() - self.started >= limit

        num_colors = min(num_colors, 105)
        neighbor_dict = defaultdict(set)
        for e1, e2 in edges:
            neighbor_dict[e1].add(e2)
            neighbor_dict[e2].add(e1)
        # solution = heuristic_solution(node_count, edges)
        for idx in range(node_count):
            if solution[idx] >= num_colors:
                solution[idx] = random.choice(range(num_colors))
        violation_count = sum(
            solution[i] == solution[j] and (i, j) in edges for i, j in itertools.combinations(range(node_count), 2))
        print(f'initial violation count is {violation_count}')
        temperature = 0.1
        iteration_count = 0
        violation_dict = defaultdict(set)
        while violation_count > 0:
            if time_exceeded():
                return None, True
            iteration_count += 1
            if iteration_count % 1_000_000 == 0:
                temperature *= 0.9
            if iteration_count % 100_000 == 0:
                violation_dict_entries = sum(len(v) for v in violation_dict.values())
                print(f'{iteration_count=}, {num_colors=}, {violation_count=}, {violation_dict_entries=}, increase violation by 1 prob: {math.exp(-1 / temperature)}')
                for i in range(10):
                    print(f'number solutions with {i} violation: {len(violation_dict[i])}')
            node = random.choice(range(node_count))
            current_violations = sum(solution[node] == solution[n] for n in neighbor_dict[node])
            new_color = random.choice(range(num_colors))
            new_violations = sum(new_color == solution[n] for n in neighbor_dict[node])
            # if solution_tuple in violation_dict:
            new_violation_count = violation_count + new_violations - current_violations
            if new_violation_count < 10:
                normalized_solution = normalize(tuple(solution[i] if i != node else new_color for i in range(node_count)))
                violation_dict[new_violation_count].add(normalized_solution)
            if new_violations < current_violations or random.random() < math.exp(
                    (current_violations - new_violations) / temperature):
                violation_count -= current_violations - new_violations
                if iteration_count < -1_000:
                    print(f'{iteration_count=:_}, {node_count=}, {num_colors=}: moving to solution with {violation_count=}')
                solution[node] = new_color
        print(f'{solution=}')
        return solution, False

    def _solve(self, instance: ColoringInstance) -> ColoringResult:
        self.started = perf_counter()
        graph = instance.graph
        edges = list(graph.edges)
        nodes = list(graph.nodes)
        node_index = {node: i for i, node in enumerate(nodes)}
        edges = [(node_index[u], node_index[v]) for u, v in edges]
        initial = nx.coloring.greedy_color(graph, strategy="saturation_largest_first")
        best_coloring = normalize_coloring({str(k): int(v) for k, v in initial.items()})
        solution = {node_index[node]: i for node, i in best_coloring.items()}
        # print(best_coloring)
        # print(initial)
        best_num_colors = len(set(best_coloring.values()))
        lower_bound = greedy_clique_lower_bound(instance)
        timed_out = False


        while best_num_colors > lower_bound and solution is not None:
            solution, timed_out = self.violation_search(len(graph.nodes), edges, best_num_colors-1, solution)
            if solution is not None and not timed_out:
                best_coloring = normalize_coloring({nodes[i]: c for i, c in solution.items()})
                best_num_colors = max(best_coloring.values())+1

        optimal = not timed_out or best_num_colors == lower_bound
        return ColoringResult(
            solver_name=self.name,
            status=SolveStatus.OPTIMAL if optimal else SolveStatus.FEASIBLE,
            coloring=best_coloring,
            num_colors=best_num_colors,
            lower_bound=best_num_colors if optimal else lower_bound,
            upper_bound=best_num_colors,
        )


