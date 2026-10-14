"""Reproducible graph-first retrieval benchmark; standard library only.

Workload: a relation question names a source concept and asks what it uses.
The expected answer is only reachable through a graph edge; its name/content
does not contain the source name. This deliberately measures relational
retrieval, not generic semantic understanding.

Methods:
  lexical          NeuronGraph.find_candidates
  tfidf            sparse TF-IDF cosine over node text (non-neural baseline)
  activation_only  lexical top-1 seed, then rank by propagated activation
  hybrid           NeuronGraph.search (the actual v3 hybrid path)
  graph_path       oracle-seeded shortest path; structural diagnostic only,
                   not an end-to-end query baseline because source ID is given.

Run: python benchmarks/benchmark_retrieval.py
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
import re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuron_core import NeuronGraph  # noqa: E402


TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.casefold()) if len(token) > 1]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


class SparseTfidfIndex:
    """Transparent lexical vector baseline. Not a pretrained embedding model."""

    def __init__(self, graph: NeuronGraph):
        self.documents: dict[str, dict[str, float]] = {}
        document_tokens: dict[str, list[str]] = {}
        df: Counter[str] = Counter()
        for nid, neuron in graph.neurons.items():
            knowledge = graph.knowledge[neuron.knowledge_id]
            attribute_text = " ".join(f"{key} {value}" for key, value in knowledge.attributes.items())
            metadata_text = f"{neuron.metadata}"
            doc = " ".join([
                neuron.name, neuron.kind, neuron.content, attribute_text,
                metadata_text, knowledge.name,
            ])
            doc_tokens = tokens(doc)
            document_tokens[nid] = doc_tokens
            df.update(set(doc_tokens))
        n_docs = max(1, len(document_tokens))
        idf = {term: math.log(1 + n_docs / (1 + count)) + 1.0
               for term, count in df.items()}
        for nid, doc_tokens in document_tokens.items():
            counts = Counter(doc_tokens)
            weights = {term: (1.0 + math.log(count)) * idf[term]
                       for term, count in counts.items()}
            norm = math.sqrt(sum(value * value for value in weights.values()))
            if norm:
                weights = {term: value / norm for term, value in weights.items()}
            self.documents[nid] = weights
        self.idf = idf

    def search(self, query: str, top_k: int = 5) -> list[str]:
        counts = Counter(tokens(query))
        weights = {term: (1.0 + math.log(count)) * self.idf[term]
                   for term, count in counts.items() if term in self.idf}
        norm = math.sqrt(sum(value * value for value in weights.values()))
        if not norm:
            return []
        weights = {term: value / norm for term, value in weights.items()}
        scored = []
        for nid, doc in self.documents.items():
            score = sum(value * doc.get(term, 0.0) for term, value in weights.items())
            if score > 0.0:
                scored.append((nid, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [nid for nid, _ in scored[:top_k]]


def make_workload(n_nodes: int) -> tuple[NeuronGraph, SparseTfidfIndex, list[dict], float, float]:
    if n_nodes < 20 or n_nodes % 2:
        raise ValueError("node count must be an even integer >= 20")
    graph = NeuronGraph()
    half = n_nodes // 2
    source_ids = []
    target_ids = []
    started = time.perf_counter()
    for i in range(half):
        source_ids.append(graph.add_concept(
            f"source{i:05d}", kind="source", content=f"entrykey{i:05d}"
        ))
    for i in range(half):
        target_ids.append(graph.add_concept(
            f"target{i:05d}", kind="target", content=f"payloadkey{i:05d}"
        ))
    for i, (source, target) in enumerate(zip(source_ids, target_ids)):
        graph.connect(source, target, "uses", weight=0.8 + 0.2 * (i % 5) / 4)
    graph_build_seconds = time.perf_counter() - started
    index_started = time.perf_counter()
    vector_index = SparseTfidfIndex(graph)
    vector_build_seconds = time.perf_counter() - index_started

    # Up to 100 fixed, evenly distributed, held-out query cases per size.
    case_count = min(100, half)
    step = max(1, half // case_count)
    cases = []
    for i in range(0, half, step):
        if len(cases) >= case_count:
            break
        cases.append({
            "query": f"source{i:05d} uses",
            "source_id": source_ids[i],
            "target_id": target_ids[i],
        })
    return graph, vector_index, cases, graph_build_seconds, vector_build_seconds


def method_functions(graph: NeuronGraph, index: SparseTfidfIndex, top_k: int):
    def lexical(query: str) -> list[str]:
        return [nid for nid, _score in graph.find_candidates(query, top_k=top_k)]

    def vector(query: str) -> list[str]:
        return index.search(query, top_k=top_k)

    def activation_only(query: str) -> list[str]:
        seed = graph.find_candidates(query, top_k=1)
        if not seed:
            return []
        results = graph.spread_activation(
            {seed[0][0]: max(0.25, seed[0][1])},
            max_hops=3, damping=0.75, max_nodes=top_k * 5,
        )
        return [item.neuron_id for item in results[:top_k]]

    def hybrid(query: str) -> list[str]:
        return [row["neuron_id"] for row in graph.search(
            query, top_k=top_k, max_hops=3, damping=0.75
        )]

    return {
        "lexical": lexical,
        "tfidf_vector": vector,
        "seeded_activation_only": activation_only,
        "neuron_hybrid": hybrid,
    }


def run_method(fn, cases: list[dict], top_k: int) -> dict:
    # Warm-up, excluded from measured latency.
    if cases:
        fn(cases[0]["query"])
    latencies = []
    ranks = []
    hits_at_1 = hits_at_3 = hits_at_k = 0
    for case in cases:
        started = time.perf_counter_ns()
        ranked = fn(case["query"])
        latencies.append((time.perf_counter_ns() - started) / 1e6)
        try:
            rank = ranked.index(case["target_id"]) + 1
        except ValueError:
            rank = None
        ranks.append(rank)
        if rank == 1:
            hits_at_1 += 1
        if rank is not None and rank <= 3:
            hits_at_3 += 1
        if rank is not None and rank <= top_k:
            hits_at_k += 1
    n = max(1, len(cases))
    reciprocal_ranks = [1.0 / rank if rank else 0.0 for rank in ranks]
    return {
        "queries": len(cases),
        "target_hit_rate_at_1": round(hits_at_1 / n, 4),
        "target_recall_at_3": round(hits_at_3 / n, 4),
        f"target_recall_at_{top_k}": round(hits_at_k / n, 4),
        "mrr": round(statistics.mean(reciprocal_ranks), 4),
        "mean_ms_per_query": round(statistics.mean(latencies), 4),
        "p50_ms_per_query": round(percentile(latencies, 0.50), 4),
        "p95_ms_per_query": round(percentile(latencies, 0.95), 4),
    }


def transient_peak_bytes(fn, cases: list[dict]) -> int:
    # Objects/index already exist before tracing; this is extra query-time
    # allocated memory, not total model/index memory.
    tracemalloc.start()
    tracemalloc.reset_peak()
    for case in cases:
        fn(case["query"])
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def run_benchmark(sizes: list[int], top_k: int = 5) -> dict:
    report = {
        "benchmark": "Neuron v3 relational retrieval microbenchmark",
        "version": 1,
        "python": sys.version.split()[0],
        "top_k": top_k,
        "method_notes": {
            "lexical": "Actual NeuronGraph.find_candidates token-overlap ranking.",
            "tfidf_vector": "Sparse TF-IDF cosine over the same node text; not a pretrained semantic embedding.",
            "seeded_activation_only": "Seeded by the lexical top-1 result, then ranked only by graph activation.",
            "neuron_hybrid": "Actual NeuronGraph.search: lexical seed candidates plus bounded graph activation and blended scoring.",
            "graph_path": "Oracle-seeded structural diagnostic; source node ID is supplied, so not an end-to-end text retrieval baseline."
        },
        "sizes": [],
    }
    for size in sizes:
        graph, index, cases, graph_build_seconds, vector_build_seconds = make_workload(size)
        methods = method_functions(graph, index, top_k)
        method_results = {}
        for name, fn in methods.items():
            result = run_method(fn, cases, top_k)
            result["transient_peak_bytes"] = transient_peak_bytes(fn, cases)
            method_results[name] = result

        path_latencies = []
        path_hits = 0
        for case in cases:
            started = time.perf_counter_ns()
            path = graph.find_path(case["source_id"], case["target_id"], max_depth=3)
            path_latencies.append((time.perf_counter_ns() - started) / 1e6)
            if path and path[-1]["target_id"] == case["target_id"]:
                path_hits += 1

        # Estimate retained size of sparse TF-IDF vectors only. Python object
        # accounting is approximate and intentionally labelled accordingly.
        approx_vector_index_bytes = sys.getsizeof(index.documents) + sys.getsizeof(index.idf)
        for document in index.documents.values():
            approx_vector_index_bytes += sys.getsizeof(document)
            approx_vector_index_bytes += sum(
                sys.getsizeof(term) + sys.getsizeof(value) for term, value in document.items()
            )
        approx_vector_index_bytes += sum(
            sys.getsizeof(term) + sys.getsizeof(value) for term, value in index.idf.items()
        )

        report["sizes"].append({
            "nodes": size,
            "edges": len(graph.synapses),
            "queries": len(cases),
            "graph_build_seconds": round(graph_build_seconds, 6),
            "tfidf_index_build_seconds": round(vector_build_seconds, 6),
            "approx_tfidf_index_bytes": approx_vector_index_bytes,
            "methods": method_results,
            "structural_diagnostic": {
                "description": "Shortest path with the correct source ID already known.",
                "path_success_rate": round(path_hits / max(1, len(cases)), 4),
                "mean_ms_per_path": round(statistics.mean(path_latencies), 4),
                "p95_ms_per_path": round(percentile(path_latencies, 0.95), 4),
            },
        })
    return report


def print_report(report: dict) -> None:
    print("\nNEURON V3 BENCHMARK — relational retrieval")
    print("Task: query gives a source name + relation; answer is a linked target whose text omits the source name.")
    print("TF-IDF is a sparse lexical-vector baseline, NOT a neural embedding model.")
    print("Graph path is reported separately because it receives the source ID as an oracle seed.\n")
    print("NODES  METHOD                    HIT@1   RECALL@3  MRR     MEAN ms  P95 ms")
    print("-" * 78)
    for size in report["sizes"]:
        for name, result in size["methods"].items():
            print(f"{size['nodes']:>5}  {name:<25} "
                  f"{result['target_hit_rate_at_1']:>5.1%}   "
                  f"{result['target_recall_at_3']:>7.1%}  "
                  f"{result['mrr']:>5.3f}  "
                  f"{result['mean_ms_per_query']:>7.3f}  "
                  f"{result['p95_ms_per_query']:>6.3f}")
        diag = size["structural_diagnostic"]
        print(f"       {'graph_path (oracle seed)':<25} "
              f"{diag['path_success_rate']:>5.1%}   {'n/a':>7}  {'n/a':>5}  "
              f"{diag['mean_ms_per_path']:>7.3f}  {diag['p95_ms_per_path']:>6.3f}")
        print(f"       build={size['graph_build_seconds']:.3f}s; "
              f"TF-IDF index={size['tfidf_index_build_seconds']:.3f}s; "
              f"approx TF-IDF index memory={size['approx_tfidf_index_bytes']/1024:.1f} KiB")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="100,1000,5000",
                        help="Comma-separated even node counts (default: 100,1000,5000)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()
    sizes = [int(item.strip()) for item in args.sizes.split(",") if item.strip()]
    report = run_benchmark(sizes, top_k=args.top_k)
    print_report(report)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON report written to {output}")


if __name__ == "__main__":
    main()
