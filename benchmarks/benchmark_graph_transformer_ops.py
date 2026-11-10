"""Microbenchmark graph-native sparse projection vs dense reference."""
import argparse
import json
import statistics
import time

from transformer_research.graph_transformer import GraphLinear


def percentile(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * p))]


def measure(fn, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "median_ms": round(statistics.median(samples), 6),
        "p95_ms": round(percentile(samples, 0.95), 6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--fan-in", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    layer = GraphLinear.random_sparse(
        args.hidden, args.hidden, args.fan_in, seed=20261110
    )
    dense = [[0.0] * args.hidden for _ in range(args.hidden)]
    for edge in layer.edges:
        dense[edge.target][edge.source] = edge.weight
    x = [((i * 17) % 101 - 50) / 50 for i in range(args.hidden)]

    def dense_forward():
        return [
            layer.bias[row] + sum(dense[row][col] * x[col]
                                  for col in range(args.hidden))
            for row in range(args.hidden)
        ]

    sparse_out = layer.forward(x)
    dense_out = dense_forward()
    max_error = max(abs(a - b) for a, b in zip(sparse_out, dense_out))
    if max_error > 1e-10:
        raise AssertionError("graph and dense reference outputs differ")

    graph_timing = measure(lambda: layer.forward(x), args.repeats)
    dense_timing = measure(dense_forward, args.repeats)
    report = {
        "benchmark": "single sparse learned projection",
        "hidden_size": args.hidden,
        "fan_in_per_output": args.fan_in,
        "graph_edges": len(layer.edges),
        "dense_parameters": args.hidden * args.hidden,
        "graph_parameter_sparsity_percent": round(
            100 * (1 - len(layer.edges) / (args.hidden * args.hidden)), 3
        ),
        "max_absolute_output_error_same_weights": max_error,
        "graph_edge_execution": graph_timing,
        "dense_zero_filled_reference": dense_timing,
        "dense_over_graph_median_ratio": round(
            dense_timing["median_ms"] / graph_timing["median_ms"], 3
        ),
        "limitations": [
            "Single projection microbenchmark, not a full Transformer or Qwen comparison.",
            "Dense reference uses Python lists, not optimized BLAS or GPU kernels.",
            "Sparse graph is a sparse model, not equivalent to a fully dense pretrained layer.",
            "No training, tokenizer, multi-head/GQA, KV cache, or pretrained weights.",
            "Python object memory is not represented by logical parameter counts.",
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
