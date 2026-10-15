"""Small, deterministic benchmark for Neuron's reasoning-path experiment.

Measures discovery of a known multi-hop endpoint while scaling unrelated graph
content. This is a graph-traversal benchmark, not a general language-reasoning
claim. Run from the repository root:
    python benchmarks/benchmark_reasoning_experiment.py
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
import time

# Direct script execution sets sys.path[0] to benchmarks/, not the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuron_core import NeuronGraph, explore_reasoning


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def run_case(unrelated_nodes: int, repetitions: int = 100) -> dict[str, object]:
    graph = NeuronGraph()
    mira = graph.add_concept("Mira", kind="person")
    orion = graph.add_concept("Orion Lab", kind="organization")
    cairo = graph.add_concept("Cairo", kind="city")
    runtime = graph.add_concept("Rust", kind="programming_language")
    graph.connect(mira, orion, "works_at")
    graph.connect(orion, cairo, "located_in")
    graph.connect(cairo, runtime, "uses")

    for index in range(unrelated_nodes):
        graph.add_concept(f"Unrelated concept {index:05d}", kind="distractor")

    durations_ms = []
    final_result = None
    for _ in range(repetitions):
        started = time.perf_counter()
        final_result = explore_reasoning(
            graph, "Mira", max_hops=3, seed_limit=1, min_strength=0.0
        )
        durations_ms.append((time.perf_counter() - started) * 1000)

    assert final_result is not None
    rust_paths = [
        path for path in final_result["paths"]
        if path["end"] == "Rust" and path["hops"] == 3
    ]
    return {
        "unrelated_nodes": unrelated_nodes,
        "total_nodes": len(graph.neurons),
        "total_edges": len(graph.synapses),
        "repetitions": repetitions,
        "expected_three_hop_path_found": bool(rust_paths),
        "paths_explored_last_run": final_result["paths_explored"],
        "mean_latency_ms": round(statistics.mean(durations_ms), 6),
        "p95_latency_ms": round(percentile(durations_ms, 0.95), 6),
        "decision": final_result["decision"],
    }


def main() -> None:
    report = {
        "benchmark": "neuron_reasoning_path_exploration",
        "scope": (
            "Known-path discovery with unrelated nodes; does not measure "
            "open-ended natural-language reasoning or answer correctness."
        ),
        "cases": [run_case(size) for size in (100, 1000, 5000)],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
