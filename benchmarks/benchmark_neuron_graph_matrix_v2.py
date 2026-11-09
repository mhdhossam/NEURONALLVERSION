"""Equivalent reachability benchmark: NEURON graph, sparse adjacency, dense NumPy matvec.

All edges use the same weight/confidence and all seeds start at score 1.0, so
NEURON's max-score activation reduces to bounded directed reachability. Every
implementation is checked for identical reachable nodes before timing.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuron_core import NeuronGraph  # noqa: E402

DAMPING = 0.75
EDGE_STRENGTH = 0.375  # weight .75 * neutral confidence prior .5
FLOOR = 0.001


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * p))]


def build_case(n, degree, seed):
    rng = random.Random(seed)
    adjacency = []
    for source in range(n):
        choices = set()
        target_degree = min(degree, n - 1)
        while len(choices) < target_degree:
            target = rng.randrange(n)
            if target != source:
                choices.add(target)
        adjacency.append(sorted(choices))

    graph = NeuronGraph()
    ids = [graph.add_concept("node_%d" % i, kind="benchmark_node") for i in range(n)]
    for source, row in enumerate(adjacency):
        for target in row:
            graph.connect(ids[source], ids[target], "benchmark_edge", weight=0.75)

    dense = np.zeros((n, n), dtype=np.float32)
    for source, row in enumerate(adjacency):
        dense[source, row] = 1.0
    return graph, ids, adjacency, dense


def propagate_sparse(adjacency, seed, hops):
    frontier = {seed}
    visited = {seed}
    for _ in range(hops):
        nxt = set()
        for source in frontier:
            for target in adjacency[source]:
                if target not in visited:
                    nxt.add(target)
        if not nxt:
            break
        visited.update(nxt)
        frontier = nxt
    return visited


def propagate_dense_numpy(matrix, seed, hops):
    n = matrix.shape[0]
    frontier = np.zeros(n, dtype=np.float32)
    frontier[seed] = 1.0
    visited = np.zeros(n, dtype=bool)
    visited[seed] = True
    for _ in range(hops):
        # Dense matrix-vector multiply; optimized NumPy/BLAS implementation.
        reached = (matrix.T @ frontier) > 0.0
        reached &= ~visited
        if not reached.any():
            break
        visited |= reached
        frontier = reached.astype(np.float32)
    return set(np.flatnonzero(visited).tolist())


def run_case(n, degree, repeats, hops):
    tracemalloc.start()
    started = time.perf_counter_ns()
    graph, ids, adjacency, dense = build_case(n, degree, 20261109 + n + degree)
    build_ms = (time.perf_counter_ns() - started) / 1e6
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    graph_run = lambda: graph.spread_activation(
        [ids[0]], max_hops=hops, damping=DAMPING,
        min_activation=FLOOR, max_nodes=n, commit_snapshot=False
    )
    graph_result = graph_run()
    id_index = {nid: i for i, nid in enumerate(ids)}
    graph_set = {id_index[item.neuron_id] for item in graph_result}
    sparse_set = propagate_sparse(adjacency, 0, hops)
    dense_set = propagate_dense_numpy(dense, 0, hops)

    if graph_set != sparse_set or graph_set != dense_set:
        raise AssertionError(
            "reachable-set mismatch N=%d degree=%d (NEURON=%d sparse=%d dense=%d)"
            % (n, degree, len(graph_set), len(sparse_set), len(dense_set))
        )

    functions = {
        "neuron_graph": graph_run,
        "sparse_adjacency": lambda: propagate_sparse(adjacency, 0, hops),
        "dense_numpy_matvec": lambda: propagate_dense_numpy(dense, 0, hops),
    }
    timings = {name: [] for name in functions}
    for _ in range(repeats):
        for name, fn in functions.items():
            tick = time.perf_counter_ns()
            fn()
            timings[name].append((time.perf_counter_ns() - tick) / 1e6)

    med = {name: statistics.median(values) for name, values in timings.items()}
    return {
        "nodes": n,
        "degree": degree,
        "edges": sum(map(len, adjacency)),
        "density_percent": round(100 * sum(map(len, adjacency)) / (n * n), 4),
        "activated_nodes": len(graph_set),
        "neuron_graph_median_ms": round(med["neuron_graph"], 6),
        "neuron_graph_p95_ms": round(percentile(timings["neuron_graph"], 0.95), 6),
        "sparse_adjacency_median_ms": round(med["sparse_adjacency"], 6),
        "sparse_adjacency_p95_ms": round(percentile(timings["sparse_adjacency"], 0.95), 6),
        "dense_numpy_matvec_median_ms": round(med["dense_numpy_matvec"], 6),
        "dense_numpy_matvec_p95_ms": round(percentile(timings["dense_numpy_matvec"], 0.95), 6),
        "sparse_over_neuron_ratio": round(med["sparse_adjacency"] / med["neuron_graph"], 3),
        "dense_over_neuron_ratio": round(med["dense_numpy_matvec"] / med["neuron_graph"], 3),
        "graph_build_ms": round(build_ms, 3),
        "graph_build_peak_python_bytes": peak_bytes,
        "dense_matrix_bytes": int(dense.nbytes),
        "same_reachable_set": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="128,256,512,1024")
    parser.add_argument("--degrees", default="4,16,64")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--hops", type=int, default=3)
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    results = [
        run_case(int(size), int(degree), args.repeats, args.hops)
        for size in args.sizes.split(",") if size.strip()
        for degree in args.degrees.split(",") if degree.strip()
    ]
    report = {
        "benchmark": "NEURON graph vs sparse adjacency vs dense NumPy matrix-vector",
        "task": "bounded directed reachability with uniform edge strength",
        "limitations": [
            "Not Qwen inference or image synthesis.",
            "Dense baseline is CPU NumPy/BLAS, not GPU.",
            "Uniform edges isolate representation/traversal cost; this is not a full weighted transformer layer.",
            "NEURON graph construction includes its knowledge/history/index bookkeeping.",
        ],
        "controls": {
            "degrees": [int(x) for x in args.degrees.split(",") if x.strip()],
            "hops": args.hops,
            "repeats": args.repeats,
            "damping": DAMPING,
            "effective_edge_strength": EDGE_STRENGTH,
            "activation_floor": FLOOR,
            "same_seeded_graph": True,
        },
        "results": results,
    }
    output = json.dumps(report, indent=2)
    print(output)
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
