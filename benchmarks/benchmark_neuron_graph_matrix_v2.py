"""Controlled comparison of NEURON graph propagation and matrix/sparse baselines."""
from __future__ import annotations
import argparse, json, random, statistics, sys, time, tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuron_core import NeuronGraph  # noqa: E402

DAMPING = 0.75
EDGE_STRENGTH = 0.375  # weight .75 * neutral confidence prior .5
FLOOR = 0.001

def percentile(values, p):
    values = sorted(values)
    return values[min(len(values)-1, round((len(values)-1)*p))]

def build_case(n, degree, seed):
    rng = random.Random(seed)
    adjacency = []
    for source in range(n):
        choices = set()
        while len(choices) < min(degree, n-1):
            target = rng.randrange(n)
            if target != source:
                choices.add(target)
        adjacency.append(sorted(choices))
    graph = NeuronGraph()
    ids = [graph.add_concept("node_%d" % i, kind="benchmark_node") for i in range(n)]
    for source, row in enumerate(adjacency):
        for target in row:
            graph.connect(ids[source], ids[target], "benchmark_edge", weight=0.75)
    dense = [bytearray(n) for _ in range(n)]
    for source, row in enumerate(adjacency):
        for target in row:
            dense[source][target] = 1
    return graph, ids, adjacency, dense

def propagate_adjacency(adjacency, seed, hops):
    frontier = {seed: 1.0}
    visited = {seed}
    for _ in range(hops):
        nxt = {}
        for source, score in frontier.items():
            signal = score * DAMPING * EDGE_STRENGTH
            if signal < FLOOR:
                continue
            for target in adjacency[source]:
                if target not in visited and signal > nxt.get(target, 0.0):
                    nxt[target] = signal
        visited.update(nxt)
        frontier = nxt
        if not frontier:
            break
    return visited

def propagate_dense(matrix, seed, hops):
    n = len(matrix)
    frontier = {seed: 1.0}
    visited = {seed}
    for _ in range(hops):
        nxt = {}
        for source, score in frontier.items():
            signal = score * DAMPING * EDGE_STRENGTH
            if signal < FLOOR:
                continue
            row = matrix[source]
            for target in range(n):
                if row[target] and target not in visited and signal > nxt.get(target, 0.0):
                    nxt[target] = signal
        visited.update(nxt)
        frontier = nxt
        if not frontier:
            break
    return visited

def run_case(n, degree, repeats, hops):
    tracemalloc.start()
    t0 = time.perf_counter_ns()
    graph, ids, adjacency, dense = build_case(n, degree, 20261109+n)
    build_ms = (time.perf_counter_ns()-t0)/1e6
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    graph_run = lambda: graph.spread_activation(
        [ids[0]], max_hops=hops, damping=DAMPING,
        min_activation=FLOOR, max_nodes=n, commit_snapshot=False)
    graph_result = graph_run()
    id_index = {nid:i for i,nid in enumerate(ids)}
    graph_set = {id_index[item.neuron_id] for item in graph_result}
    sparse_set = propagate_adjacency(adjacency, 0, hops)
    dense_set = propagate_dense(dense, 0, hops)
    if graph_set != sparse_set or graph_set != dense_set:
        raise AssertionError("reachable-set mismatch at N=%d (graph=%d sparse=%d dense=%d)" %
                             (n,len(graph_set),len(sparse_set),len(dense_set)))

    timings = {"neuron_graph": [], "sparse_adjacency": [], "dense_matrix_scan": []}
    functions = {
        "neuron_graph": graph_run,
        "sparse_adjacency": lambda: propagate_adjacency(adjacency,0,hops),
        "dense_matrix_scan": lambda: propagate_dense(dense,0,hops),
    }
    for _ in range(repeats):
        for name, fn in functions.items():
            start = time.perf_counter_ns()
            fn()
            timings[name].append((time.perf_counter_ns()-start)/1e6)

    med = {name: statistics.median(vals) for name, vals in timings.items()}
    return {
        "nodes": n, "edges": sum(map(len, adjacency)),
        "density_percent": round(100*sum(map(len,adjacency))/(n*n),4),
        "activated_nodes": len(graph_set),
        "neuron_graph_median_ms": round(med["neuron_graph"],5),
        "neuron_graph_p95_ms": round(percentile(timings["neuron_graph"],.95),5),
        "sparse_adjacency_median_ms": round(med["sparse_adjacency"],5),
        "sparse_adjacency_p95_ms": round(percentile(timings["sparse_adjacency"],.95),5),
        "dense_matrix_scan_median_ms": round(med["dense_matrix_scan"],5),
        "dense_matrix_scan_p95_ms": round(percentile(timings["dense_matrix_scan"],.95),5),
        "sparse_over_neuron_ratio": round(med["sparse_adjacency"]/med["neuron_graph"],3),
        "dense_over_neuron_ratio": round(med["dense_matrix_scan"]/med["neuron_graph"],3),
        "graph_build_ms": round(build_ms,3),
        "graph_build_peak_python_bytes": peak_bytes,
        "same_reachable_set": True
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--sizes",default="128,256,512,1024")
    p.add_argument("--degree",type=int,default=8)
    p.add_argument("--repeats",type=int,default=30)
    p.add_argument("--hops",type=int,default=3)
    p.add_argument("--json-output",default="")
    a=p.parse_args()
    report={
        "benchmark":"NEURON indexed graph vs sparse adjacency vs dense matrix scan",
        "note":"Not Qwen inference, BLAS/GPU matrix multiplication, or image synthesis.",
        "controls":{"degree":a.degree,"hops":a.hops,"repeats":a.repeats,
                    "damping":DAMPING,"effective_edge_strength":EDGE_STRENGTH,
                    "activation_floor":FLOOR,"same_seeded_graph":True},
        "results":[run_case(int(s),a.degree,a.repeats,a.hops)
                   for s in a.sizes.split(",") if s.strip()]
    }
    output=json.dumps(report,indent=2)
    print(output)
    if a.json_output:
        path=Path(a.json_output); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(output+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
