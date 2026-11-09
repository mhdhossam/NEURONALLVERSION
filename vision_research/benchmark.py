"""Reproducible microbenchmark for the NEURON Vision Phase-0 prototype.

Measures graph compilation + SVG decoding only. It is not comparable to Qwen
latency or image quality because this prototype is rule-based and untrained.
"""
from __future__ import annotations

import json
import statistics
import time
import tracemalloc

from neuron_vision import compile_prompt, render_svg


CASES = {
    "simple": "a red car",
    "two_objects": "a red car beside a green tree",
    "scene": "a red car beside a tree with a sun and mountains",
    "mixed": "a wooden chair beside a blue house under a cloud",
    "fallback": "a beautiful futuristic city with reflections",
}


def run_case(prompt: str, repeats: int = 500) -> dict:
    # Warm-up avoids including one-time interpreter/import effects.
    graph, scene, _ = compile_prompt(prompt)
    render_svg(scene)

    durations = []
    tracemalloc.start()
    for _ in range(repeats):
        start = time.perf_counter()
        graph, scene, trace = compile_prompt(prompt)
        svg = render_svg(scene)
        durations.append((time.perf_counter() - start) * 1000)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    ordered = sorted(durations)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return {
        "prompt": prompt,
        "repeats": repeats,
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "svg_bytes": len(svg.encode("utf-8")),
        "mean_ms": round(statistics.mean(durations), 4),
        "median_ms": round(statistics.median(durations), 4),
        "p95_ms": round(p95, 4),
        "images_per_second": round(1000 / statistics.mean(durations), 2),
        "tracemalloc_current_kib": round(current / 1024, 2),
        "tracemalloc_peak_kib": round(peak / 1024, 2),
        "activation_hops": len(trace),
    }


def main() -> None:
    results = [run_case(prompt) for prompt in CASES.values()]
    print(json.dumps({
        "benchmark": "NEURON Vision Phase-0 graph compile + SVG render",
        "warning": "Not a Qwen comparison; no neural inference or learned image synthesis.",
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
