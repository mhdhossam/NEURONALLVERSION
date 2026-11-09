"""Compare NEURON's real indexed graph propagation with dense adjacency-matrix scanning.

This is a controlled structural microbenchmark, NOT a Qwen inference benchmark.
Both methods use the same directed random graph, seed, hop limit, damping and
edge strength. The dense baseline stores an N x N byte adjacency matrix and
scans every possible destination for each active source. NEURON uses its actual
NeuronGraph/ Synapse adjacency indexes and bounded spread_activation().
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuron_core import NeuronGraph  # noqa: E402


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * p))]


def build_case(n: int, degree: int, seed: int):
    rng = random.Random(seed)
    targets: list[list[int]] = [[] for _ in range(n)]
    for source in range(n):
        choices = set()
        while len(choices) < min(degree, n - 1):
            target = rng.randrange(n)
            if target != source:
                choices.add(target)
        targets[source] = sorted(choices)

    # Actual repository implementation, not a reimplementation of NEURON.
    graph = NeuronGraph()
    ids = [graph.add_concept(f"node_{i}", kind="benchmark_node") for i in range(n)]
    for source, row in enumerate(targets):
        for target in row:
            # With no evidence, confidence prior is 0.5; effective edge strength
            # is therefore 0.75 * 0.5 = 0.375.
            graph.connect(ids[source], ids[target], "benchmark_edge", weight=0.75)

    # Compact dense adjacency baseline: one byte per possible directed edge.
    matrix = [bytearray(n) for _ in range(n)]
    for source, row in enumerate(targets):
        for target in row:
            matrix[source][target] = 1
    return graph, ids, matrix, targets


def dense_matrix_propagate(matrix, seed: int, hops: int, damping: float,
                           min_activation: float) -> set[int]:
    """Dense row scan; same max-score threshold semantics for uniform edges."""
    n = len(matrix)
    frontier = {seed: 1.0}
    visited = {seed}
    edge_strength = 0.75 * 0.5  # matches NeuronGraph default confidence prior
    for _ in range(hops):
        next_frontier: dict[int, float] = {}
        for source, score in frontier.items():
            signal = score * damping * edge_strength
            if signal < min_activation:
                continue
            row = matrix[source]
            # Deliberately scan all N columns, including non-edges.
            for target in range(n):
                if row[target] and target not in visited:
                    if signal > next_frontier.get(target, 0.0):
                        next_frontier[target] = signal
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return visited



def sparse_adjacency_propagate(adjacency, seed, hops, damping, min_activation):
    """Lean integer adjacency-list baseline with the same graph semantics."""
    frontier = {seed: 1.0}
    visited = {seed}
    edge_strength = 0.75 * 0.5
    for _ in range(hops):
        next_frontier = {}
        for source, score in frontier.items():
            signal = score * damping * edge_strength
            if signal < min_activation:
                continue
            for target in adjacency[source]:
                if target not in visited and signal > next_frontier.get(target, 0.0):
                    next_frontier[target] = signal
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return visited


def run_case(n: int, degree: int, repeats: int, hops: int):
    tracemalloc.start()
    build_started = time.perf_counter_ns()
    graph, ids, matrix, adjacency = build_case(n, degree, 20261109 + n)
    build_ms = (time.perf_counter_ns() - build_started) / 1e6
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    seed_index = 0
    seed_id = ids[seed_index]
    damping = 0.75
    floor = 0.001

    # Warm-up and semantic equivalence check before timing.
    graph_result = graph.spread_activation(
        [seed_id], max_hops=hops, damping=damping,
        min_activation=floor, max_nodes=n, commit_snapshot=False,
    )
    dense_result = dense_matrix_propagate(matrix, seed_index, hops, damping, floor)
    id_to_index = {neuron_id: index for index, neuron_id in enumerate(ids)}\n    graph_set = {id_to_index[item.neuron_id] for item in graph_result}
    sparse_result = sparse_adjacency_propagate(adjacency, seed_index, hops, damping, floor)
    if graph_set != dense_result or graph_set != sparse_result:
        raise AssertionError(
            f"semantic mismatch at N={n}: graph={len(graph_set)}, "
            f"dense={len(dense_result)}, sparse={len(sparse_result)}"
        )

    graph_times, dense_times, sparse_times = [], [], []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        graph.spread_activation(
            [seed_id], max_hops=hops, damping=damping,
            min_activation=floor, max_nodes=n, commit_snapshot=False,
        )
        graph_times.append((time.perf_counter_ns() - start) / 1e6)

        start = time.perf_counter_ns()
        dense_matrix_propagate(matrix, seed_index, hops, damping, floor)
        dense_times.append((time.perf_counter_ns() - start) / 1e6)

        start = time.perf_counter_ns()
        sparse_adjacency_propagate(adjacency, seed_index, hops, damping, floor)
        sparse_times.append((time.perf_counter_ns() - start) / 1e6)

    graph_median = statistics.median(graph_times)
    dense_median = statistics.median(dense_times)
    sparse_median = statistics.median(sparse_times)
    return {
        "nodes": n,
        "edges": n * min(degree, n - 1),
        "density_percent": round((n * min(degree, n - 1)) / (n * n) * 100, 4),
        "activated_nodes": len(graph_set),
        "graph_build_ms": round(build_ms, 3),
        "graph_build_peak_python_bytes": peak_bytes,
        "graph_query_median_ms": round(graph_median, 4),
        "graph_query_p95_ms": round(percentile(graph_times, 0.95), 4),
        "dense_matrix_query_median_ms": round(dense_median, 4),
        "dense_matrix_query_p95_ms": round(percentile(dense_times, 0.95), 4),
        "sparse_adjacency_query_median_ms": round(sparse_median, 4),
        "sparse_adjacency_query_p95_ms": round(percentile(sparse_times, 0.95), 4),
        "dense_over_graph_speed_ratio": round(
            dense_median / graph_median if graph_median else 0.0, 3
        ),
        "sparse_over_graph_speed_ratio": round(
            sparse_median / graph_median if graph_median else 0.0, 3
        ),
        "equivalent_reachable_set": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="128,256,512,1024")
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    results = [
        run_case(int(size), args.degree, args.repeats, args.hops)
        for size in args.sizes.split(",") if size.strip()
    ]
    report = {
        "benchmark": "neuron_graph_vs_dense_adjacency_scan",
        "implementation": "repository neuron_core.NeuronGraph.spread_activation",
        "dense_baseline": "Python N x N byte adjacency matrix; full row scan",
        "sparse_baseline": "Python integer adjacency lists; direct neighbor iteration",
        "controls": {
            "same_graph": True,
            "same_seed": 0,
            "same_hops": args.hops,
            "same_damping": 0.75,
            "same_edge_effective_strength": 0.375,
            "same_activation_floor": 0.001,
            "degree": args.degree,
            "repeats": args.repeats,
        },
        "limitations": [
            "The sparse adjacency-list baseline is a lean reference implementation and may outperform the full NEURON object graph.",
            "The dense baseline scans an N x N byte adjacency matrix; this is not optimized BLAS/GPU matrix multiplication.",
            "It does not benchmark Qwen or image generation.",
            "Python object overhead in the full NEURON knowledge graph is included in build memory.",
            "The generated graph is random and uniform-degree; it is not a learned visual or language workload.",
        ],
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\\n", encoding="utf-8")


if __name__ == "__main__":
    main()
