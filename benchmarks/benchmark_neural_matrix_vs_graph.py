"""Compare graph-first Neuron retrieval with a pretrained neural embedding matrix.

Both systems see the same synthetic source-relation-target facts and the same
natural-language queries. The neural baseline embeds all relationship facts
once with sentence-transformers/all-MiniLM-L6-v2, then uses dense matrix-vector
similarity for each query. Query-encoder time and matrix-search time are reported
separately, as well as together.

This is a controlled retrieval benchmark, not a general intelligence benchmark.
Run in the optional CI job after installing sentence-transformers.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

# Keep comparisons reproducible and avoid competing BLAS/PyTorch thread pools.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuron_core import NeuronGraph  # noqa: E402


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
RELATIONS = [
    "profession", "specialty", "research_field", "home_city",
    "primary_tool", "affiliation", "discipline", "invention",
]
FIRST_NAMES = [
    "Ada", "Grace", "Alan", "Marie", "Katherine", "Margaret", "Claude",
    "Barbara", "Donald", "Edsger", "Frances", "Emmy", "Rosalind", "Hedy",
    "Lise", "Chien", "Niels", "James", "John", "Mary", "Sarah", "Aisha",
    "Omar", "Youssef", "Layla", "Noor", "Karim", "Dalia", "Amira", "Samir",
    "Hana", "Mina", "Ziad", "Farah", "Tarek", "Salma", "Nadia", "Rania",
    "Dina", "Mona", "Adam", "Leo", "Maya", "Alex", "Robin", "Taylor",
    "Jordan", "Morgan", "Avery", "Riley",
]
LAST_NAMES = [
    "Lovelace", "Hopper", "Turing", "Curie", "Johnson", "Franklin", "Lamarr",
    "Meitner", "Wu", "Bohr", "Maxwell", "Hamilton", "Church", "Dijkstra",
    "Knuth", "Berners", "Torvalds", "Hinton", "LeCun", "Bengio", "Ng",
    "Hassabis", "Goodall", "Darwin", "Newton", "Faraday", "Tesla", "Einstein",
    "Feynman", "Noether", "Shannon", "Wiener", "Backus", "McCarthy", "Ritchie",
    "Thompson", "Stroustrup", "Kernighan", "Matsumoto", "Rossum", "Elgamal",
    "Minsky", "Newell", "Simon", "Li", "Russell", "Pearl", "Mitchell",
    "Sutton", "Barto",
]
TARGET_ADJECTIVES = [
    "quantum", "orbital", "distributed", "symbolic", "adaptive", "cognitive",
    "geometric", "probabilistic", "linguistic", "robotic", "computational",
    "molecular", "celestial", "recursive", "causal", "relational", "temporal",
    "spatial", "neural", "algebraic", "stochastic", "modular", "autonomous",
    "evolutionary", "topological", "acoustic", "visual", " Bayesian".strip(),
    "semantic", "synthetic", "ecological", "mechanical", "cryptographic",
    "numerical", "biological", "electrical", "fluid", "symbolic", "robust",
    "dynamic", "hybrid", "discrete", "continuous", "parallel", "formal",
    "contextual", "hierarchical", "recursive", "spectral", "graphical",
]
TARGET_NOUNS = [
    "reasoning", "cartography", "optimization", "linguistics", "navigation",
    "planning", "compression", "simulation", "inference", "classification",
    "forecasting", "translation", "chemistry", "astronomy", "robotics",
    "database", "security", "architecture", "statistics", "phonetics",
    "logistics", "synthesis", "control", "diagnostics", "economics",
    "ecology", "geometry", "topology", "retrieval", "indexing", "learning",
    "perception", "memory", "causality", "detection", "verification",
    "generation", "alignment", "scheduling", "optimization", "modeling",
    "signal processing", "vision", "mathematics", "acoustics", "biophysics",
    "semantics", "networking", "probability", "algorithms",
]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def summarize_ranks(ranks: list[int | None], latencies_ms: list[float]) -> dict:
    reciprocal = [1.0 / rank if rank else 0.0 for rank in ranks]
    count = max(1, len(ranks))
    return {
        "queries": len(ranks),
        "hit_at_1": round(sum(rank == 1 for rank in ranks) / count, 4),
        "recall_at_3": round(sum(rank is not None and rank <= 3 for rank in ranks) / count, 4),
        "recall_at_5": round(sum(rank is not None and rank <= 5 for rank in ranks) / count, 4),
        "mrr": round(statistics.mean(reciprocal), 4),
        "mean_ms": round(statistics.mean(latencies_ms), 4),
        "p50_ms": round(percentile(latencies_ms, 0.50), 4),
        "p95_ms": round(percentile(latencies_ms, 0.95), 4),
    }


def create_dataset(n_nodes: int) -> tuple[NeuronGraph, list[str], list[str], list[dict], float]:
    if n_nodes < 100 or n_nodes % 2:
        raise ValueError("node count must be an even integer >= 100")
    edge_count = n_nodes // 2
    graph = NeuronGraph()
    sources: list[str] = []
    targets: list[str] = []
    facts: list[str] = []
    started = time.perf_counter()

    for i in range(edge_count):
        source_name = f"{FIRST_NAMES[i % len(FIRST_NAMES)]} {LAST_NAMES[i // len(FIRST_NAMES)]}"
        sources.append(graph.add_concept(source_name, kind="source_concept"))
    for i in range(edge_count):
        target_name = (
            f"{TARGET_ADJECTIVES[i // len(TARGET_NOUNS)]} "
            f"{TARGET_NOUNS[i % len(TARGET_NOUNS)]} {i:04d}"
        )
        targets.append(graph.add_concept(target_name, kind="target_concept"))

    query_cases = []
    for i, (source_id, target_id) in enumerate(zip(sources, targets)):
        source_name = graph.neurons[source_id].name
        target_name = graph.neurons[target_id].name
        relation = RELATIONS[i % len(RELATIONS)]
        relation_label = relation.replace("_", " ")
        graph.connect(source_id, target_id, relation, weight=1.0)
        facts.append(f"{source_name}'s {relation_label} is {target_name}.")
        query_cases.append({
            "query": f"What is {source_name}'s {relation_label}?",
            "source_id": source_id,
            "target_id": target_id,
            "fact_index": i,
        })

    build_ms = (time.perf_counter() - started) * 1000.0
    # Fixed, spread-out query sample. Use the same cases for both engines.
    count = min(100, edge_count)
    step = max(1, edge_count // count)
    selected = query_cases[::step][:count]
    return graph, facts, targets, selected, build_ms


def run_matrix_search(model, fact_embeddings: np.ndarray, query: str, top_k: int):
    start = time.perf_counter_ns()
    query_embedding = model.encode(
        [query], batch_size=1, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )[0]
    encoded_at = time.perf_counter_ns()
    # Both corpus and query embeddings are normalized; cosine is a dot product.
    scores = fact_embeddings @ query_embedding
    if len(scores) <= top_k:
        indices = np.argsort(scores)[::-1]
    else:
        indices = np.argpartition(scores, -top_k)[-top_k:]
        indices = indices[np.argsort(scores[indices])[::-1]]
    done_at = time.perf_counter_ns()
    return (
        [int(index) for index in indices],
        (encoded_at - start) / 1e6,
        (done_at - encoded_at) / 1e6,
        (done_at - start) / 1e6,
    )


def evaluate_size(model, n_nodes: int) -> dict:
    graph, facts, targets, cases, graph_build_ms = create_dataset(n_nodes)

    matrix_build_started = time.perf_counter()
    fact_embeddings = model.encode(
        facts, batch_size=64, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    matrix_build_ms = (time.perf_counter() - matrix_build_started) * 1000.0
    fact_embeddings = np.ascontiguousarray(fact_embeddings, dtype=np.float32)

    graph_ranks: list[int | None] = []
    graph_latencies: list[float] = []
    matrix_ranks: list[int | None] = []
    matrix_full_latencies: list[float] = []
    matrix_encode_latencies: list[float] = []
    matrix_similarity_latencies: list[float] = []

    # Warm both paths once so model initialization / one-time lazy work is excluded.
    if cases:
        graph.search(cases[0]["query"], top_k=TOP_K, max_hops=2, workers=1, record_history=False)
        run_matrix_search(model, fact_embeddings, cases[0]["query"], TOP_K)

    for case in cases:
        started = time.perf_counter_ns()
        graph_rows = graph.search(
            case["query"], top_k=TOP_K, max_hops=2, workers=1, record_history=False
        )
        graph_latencies.append((time.perf_counter_ns() - started) / 1e6)
        graph_ids = [row["neuron_id"] for row in graph_rows]
        graph_ranks.append(
            graph_ids.index(case["target_id"]) + 1 if case["target_id"] in graph_ids else None
        )

        ranked_facts, encode_ms, similarity_ms, full_ms = run_matrix_search(
            model, fact_embeddings, case["query"], TOP_K
        )
        matrix_encode_latencies.append(encode_ms)
        matrix_similarity_latencies.append(similarity_ms)
        matrix_full_latencies.append(full_ms)
        target_ranking = [targets[index] for index in ranked_facts]
        matrix_ranks.append(
            target_ranking.index(case["target_id"]) + 1
            if case["target_id"] in target_ranking else None
        )

    graph_metrics = summarize_ranks(graph_ranks, graph_latencies)
    matrix_metrics = summarize_ranks(matrix_ranks, matrix_full_latencies)
    return {
        "nodes": n_nodes,
        "edges_or_facts": len(facts),
        "queries": len(cases),
        "graph_build_and_preprocess_ms": round(graph_build_ms, 3),
        "matrix_fact_embedding_build_ms": round(matrix_build_ms, 3),
        "matrix_embedding_dimension": int(fact_embeddings.shape[1]),
        "matrix_embedding_bytes": int(fact_embeddings.nbytes),
        "graph": graph_metrics,
        "neural_matrix": {
            **matrix_metrics,
            "query_encoder_mean_ms": round(statistics.mean(matrix_encode_latencies), 4),
            "matrix_similarity_mean_ms": round(statistics.mean(matrix_similarity_latencies), 4),
        },
        "comparison_notes": [
            "Same source-relation-target facts and same fixed queries.",
            "Graph uses prepared token index, typed edges, and relation-aware traversal.",
            "Neural baseline uses pretrained all-MiniLM-L6-v2 query/fact embeddings and dense matrix-vector dot products.",
            "Neural end-to-end query latency includes query embedding generation and similarity ranking, but excludes one-time model download/load and fact embedding build.",
            "Both methods return the same target IDs and are scored with Hit@1, Recall@3/5, and MRR.",
            "Synthetic, template-based relational retrieval; not general reasoning or an independent real-world dataset.",
        ],
    }


def main() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="100,1000,5000")
    parser.add_argument("--json-output", default="neuron-vs-matrix-results.json")
    args = parser.parse_args()
    sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]

    load_started = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.eval()
    model_load_seconds = time.perf_counter() - load_started

    report = {
        "benchmark": "Neuron graph retrieval vs pretrained neural embedding matrix retrieval",
        "model": MODEL_NAME,
        "model_load_seconds_excluded_from_query_latency": round(model_load_seconds, 3),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "sizes": [evaluate_size(model, size) for size in sizes],
    }
    for result in report["sizes"]:
        graph = result["graph"]
        matrix = result["neural_matrix"]
        print(
            f"\n{result['nodes']} nodes / {result['edges_or_facts']} facts / {result['queries']} queries"
        )
        print(
            f"Graph: Hit@1={graph['hit_at_1']:.1%}, Recall@3={graph['recall_at_3']:.1%}, "
            f"MRR={graph['mrr']:.3f}, mean={graph['mean_ms']:.4f} ms, p95={graph['p95_ms']:.4f} ms"
        )
        print(
            f"Neural matrix: Hit@1={matrix['hit_at_1']:.1%}, Recall@3={matrix['recall_at_3']:.1%}, "
            f"MRR={matrix['mrr']:.3f}, end-to-end mean={matrix['mean_ms']:.4f} ms, "
            f"p95={matrix['p95_ms']:.4f} ms"
        )
        print(
            f"  neural query encoder mean={matrix['query_encoder_mean_ms']:.4f} ms; "
            f"dense similarity mean={matrix['matrix_similarity_mean_ms']:.4f} ms"
        )
        print(
            f"  graph build/preprocess={result['graph_build_and_preprocess_ms']:.2f} ms; "
            f"fact embedding build={result['matrix_fact_embedding_build_ms']:.2f} ms; "
            f"dense index={result['matrix_embedding_bytes'] / 1024:.1f} KiB"
        )

    output = Path(args.json_output)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nJSON report written to {output}")


if __name__ == "__main__":
    main()
