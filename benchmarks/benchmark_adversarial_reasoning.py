"""Adversarial, less-template-driven evaluation of Neuron vs neural fact retrieval.

Covers ambiguous queries, multi-hop questions, unseen relation paraphrases, and
contradictory claims. The neural baseline retrieves facts using a pretrained
sentence embedding model; it does not get a separate language model to reason.
The graph baseline is the repository's existing NeuronGraph.search method.
Both methods see the same graph facts, including evidence metadata rendered into
the neural fact documents.

This is an evaluation harness, not a claim that dense retrieval alone is a
complete neural reasoner. CI reports category-specific answer accuracy and
latency. Run after installing sentence-transformers:
  python benchmarks/benchmark_adversarial_reasoning.py --sizes 100,1000,5000
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuron_core import NeuronGraph  # noqa: E402

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


class ScenarioWorld:
    def __init__(self):
        self.graph = NeuronGraph()
        self.node_ids: dict[str, str] = {}
        self.facts: list[str] = []
        self.fact_targets: list[str] = []
        self.cases: list[dict] = []

    def node(self, name: str, kind: str = "concept") -> str:
        if name not in self.node_ids:
            self.node_ids[name] = self.graph.add_concept(name, kind=kind)
        return self.node_ids[name]

    def fact(
        self, source: str, relation: str, target: str, *,
        source_kind: str = "concept", target_kind: str = "concept",
        weight: float = 1.0, positive_evidence: int = 0,
        negative_evidence: int = 0, evidence_note: str = "",
    ) -> str:
        source_id = self.node(source, source_kind)
        target_id = self.node(target, target_kind)
        edge_id = self.graph.connect(source_id, target_id, relation, weight=weight)
        for index in range(positive_evidence):
            self.graph.add_edge_evidence(
                edge_id, positive=True, source=f"corroborating_source_{index + 1}",
                note=evidence_note or "independently corroborated",
            )
        for index in range(negative_evidence):
            self.graph.add_edge_evidence(
                edge_id, positive=False, source=f"contradicting_source_{index + 1}",
                note=evidence_note or "claim disputed",
            )
        # Serialize information available to both approaches, including the evidence
        # ledger. This lets a semantic retriever see the evidence text, while graph
        # ranking can use its numerical edge confidence.
        edge = self.graph.synapses[edge_id]
        evidence_text = (
            f" Evidence: {positive_evidence} supporting reports, "
            f"{negative_evidence} contradicting reports. {evidence_note}"
        )
        self.facts.append(
            f"{source} --{relation.replace('_', ' ')}--> {target}.{evidence_text}"
        )
        self.fact_targets.append(target_id)
        return edge_id

    def query(self, name: str, category: str, text: str, targets: list[str]) -> None:
        self.cases.append({
            "name": name,
            "category": category,
            "query": text,
            "target_ids": [self.node_ids[target] for target in targets],
            "target_names": targets,
        })


def build_world(fact_count: int) -> ScenarioWorld:
    world = ScenarioWorld()

    # Ambiguity is intentional: "based" does not say whether it means residence,
    # workplace, or birthplace. A sound evaluator should expose multiple plausible
    # interpretations rather than silently pretend the wording is precise.
    world.fact("Jordan Lee", "lives_in", "Cairo", source_kind="person", target_kind="city")
    world.fact("Jordan Lee", "works_at", "Cairo Robotics", source_kind="person", target_kind="organization")
    world.fact("Jordan Lee", "born_in", "Amman", source_kind="person", target_kind="city")
    world.query(
        "ambiguous_where_is_jordan_based", "ambiguous",
        "Where is Jordan Lee based?",
        ["Cairo", "Cairo Robotics", "Amman"],
    )
    world.query(
        "ambiguous_jordan_location", "ambiguous",
        "Tell me Jordan Lee's location.",
        ["Cairo", "Cairo Robotics", "Amman"],
    )

    # Multi-hop: person -> institute/lab -> city. The target city is not adjacent
    # to the person, so a single fact retrieval can return the intermediate lab.
    world.fact("Nia Patel", "directs", "Helix Research Lab", source_kind="person", target_kind="organization")
    world.fact("Helix Research Lab", "located_in", "Alexandria", source_kind="organization", target_kind="city")
    world.query(
        "two_hop_lab_city", "multi_hop",
        "Which city is home to the research center under Nia Patel's direction?",
        ["Alexandria"],
    )
    world.fact("Omar Selim", "employed_by", "Northstar Systems", source_kind="person", target_kind="organization")
    world.fact("Northstar Systems", "maintains", "Atlas Runtime", source_kind="organization", target_kind="software")
    world.fact("Atlas Runtime", "written_in", "Rust", source_kind="software", target_kind="programming_language")
    world.query(
        "three_hop_employer_language", "multi_hop",
        "Which programming language underpins the service at Omar Selim's employer?",
        ["Rust"],
    )

    # Held-out phrasing: these are deliberately not registered as relation aliases.
    world.fact("Ada Lovelace", "profession", "Analytical Engine Programmer", source_kind="person", target_kind="profession")
    world.query(
        "unseen_pay_the_bills", "unseen_paraphrase",
        "How does Ada Lovelace earn her keep?",
        ["Analytical Engine Programmer"],
    )
    world.fact("Grace Hopper", "profession", "Compiler Pioneer", source_kind="person", target_kind="profession")
    world.query(
        "unseen_put_food_on_table", "unseen_paraphrase",
        "What vocation puts food on Grace Hopper's table?",
        ["Compiler Pioneer"],
    )
    world.fact("Alan Turing", "primary_tool", "Bombe Machine", source_kind="person", target_kind="device")
    world.query(
        "unseen_go_to_instrument", "unseen_paraphrase",
        "What instrument did Alan Turing tend to reach for?",
        ["Bombe Machine"],
    )

    # Contradictory claim. The expected answer is defined by the evidence ledger:
    # the verified claim has corroboration; the attractive-looking alternative is
    # disputed. This also tests whether direct relation ranking respects confidence.
    world.fact(
        "Alex Kim", "profession", "Senior Platform Engineer",
        source_kind="person", target_kind="profession",
        positive_evidence=5, negative_evidence=0,
        evidence_note="verified in an official staff directory",
    )
    world.fact(
        "Alex Kim", "profession", "Aardvark Consultant",
        source_kind="person", target_kind="profession",
        positive_evidence=0, negative_evidence=4,
        evidence_note="unverified rumor, contradicted by an official staff directory",
    )
    world.query(
        "contradiction_weighted_job", "contradiction",
        "What is Alex Kim's job?",
        ["Senior Platform Engineer"],
    )
    world.fact(
        "Mira Chen", "home_city", "Port Said",
        source_kind="person", target_kind="city",
        positive_evidence=4, evidence_note="confirmed by recent first-party records",
    )
    world.fact(
        "Mira Chen", "home_city", "Aswan",
        source_kind="person", target_kind="city",
        negative_evidence=4, evidence_note="outdated claim explicitly corrected by first-party records",
    )
    world.query(
        "contradiction_corrected_residence", "contradiction",
        "Where does Mira Chen live?",
        ["Port Said"],
    )

    # Fill the remaining fact budget with noisy but plausible entities and
    # relationships. Noise uses overlapping vocabulary to make retrieval scale
    # less trivial than a corpus of unique, query-disjoint tokens.
    relation_pool = [
        "works_at", "uses", "specialty", "affiliation", "lives_in",
        "research_field", "located_in", "maintains", "profession",
    ]
    nouns = [
        "Vector", "Cedar", "Orbit", "Signal", "Meridian", "Delta", "Vertex",
        "Copper", "Nimbus", "Harbor", "Pioneer", "Lattice", "Prism", "Summit",
    ]
    adjectives = [
        "Applied", "Advanced", "Regional", "Open", "Central", "Distributed",
        "Secure", "Research", "Digital", "Autonomous", "Adaptive", "Scientific",
    ]
    index = 0
    while len(world.facts) < fact_count:
        source = f"Researcher {index:05d}"
        target = f"{adjectives[index % len(adjectives)]} {nouns[(index * 7) % len(nouns)]} {index:05d}"
        relation = relation_pool[(index * 5 + 3) % len(relation_pool)]
        world.fact(source, relation, target, source_kind="person", target_kind="organization")
        index += 1
    return world


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {"queries": 0, "hit_at_1": 0.0, "hit_at_5": 0.0, "mean_ms": 0.0, "p95_ms": 0.0}
    hit1 = [bool(row["top1_hit"]) for row in rows]
    hit5 = [bool(row["top5_hit"]) for row in rows]
    latency = [row["latency_ms"] for row in rows]
    return {
        "queries": len(rows),
        "hit_at_1": round(sum(hit1) / len(hit1), 4),
        "hit_at_5": round(sum(hit5) / len(hit5), 4),
        "mean_ms": round(statistics.mean(latency), 4),
        "p95_ms": round(percentile(latency, 0.95), 4),
    }


def evaluate_size(model, fact_count: int) -> dict:
    started = time.perf_counter()
    world = build_world(fact_count)
    graph_build_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    matrix = model.encode(
        world.facts, batch_size=64, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    matrix_build_ms = (time.perf_counter() - started) * 1000.0

    graph_rows: list[dict] = []
    matrix_rows: list[dict] = []
    # Warm-up each path; exclude first-call setup costs from query latency.
    first = world.cases[0]
    world.graph.search(first["query"], top_k=TOP_K, max_hops=3, record_history=False)
    model.encode([first["query"]], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    detailed: list[dict] = []
    for case in world.cases:
        expected = set(case["target_ids"])

        t0 = time.perf_counter_ns()
        rows = world.graph.search(
            case["query"], top_k=TOP_K, max_hops=3, damping=0.75,
            workers=1, record_history=False,
        )
        graph_ms = (time.perf_counter_ns() - t0) / 1e6
        graph_ids = [row["neuron_id"] for row in rows]
        graph_result = {
            "case": case["name"], "category": case["category"],
            "top1_hit": bool(graph_ids and graph_ids[0] in expected),
            "top5_hit": bool(expected.intersection(graph_ids)),
            "latency_ms": graph_ms,
        }
        graph_rows.append(graph_result)

        t0 = time.perf_counter_ns()
        q_vec = model.encode(
            [case["query"]], batch_size=1, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )[0]
        encode_ms = (time.perf_counter_ns() - t0) / 1e6
        t0 = time.perf_counter_ns()
        scores = matrix @ q_vec
        k = min(TOP_K, len(scores))
        ids = np.argpartition(scores, -k)[-k:]
        ids = ids[np.argsort(scores[ids])[::-1]]
        sim_ms = (time.perf_counter_ns() - t0) / 1e6
        neural_ids = []
        for index in ids.tolist():
            target_id = world.fact_targets[int(index)]
            if target_id not in neural_ids:
                neural_ids.append(target_id)
        matrix_result = {
            "case": case["name"], "category": case["category"],
            "top1_hit": bool(neural_ids and neural_ids[0] in expected),
            "top5_hit": bool(expected.intersection(neural_ids)),
            "latency_ms": encode_ms + sim_ms,
        }
        matrix_rows.append(matrix_result)
        detailed.append({
            "case": case["name"],
            "category": case["category"],
            "query": case["query"],
            "expected": case["target_names"],
            "graph_top5": [world.graph.neurons[nid].name for nid in graph_ids],
            "neural_fact_targets_top5": [
                world.graph.neurons[nid].name for nid in neural_ids
            ],
            "graph_hit1": graph_result["top1_hit"],
            "graph_hit5": graph_result["top5_hit"],
            "neural_hit1": matrix_result["top1_hit"],
            "neural_hit5": matrix_result["top5_hit"],
            "neural_encode_ms": round(encode_ms, 4),
            "neural_similarity_ms": round(sim_ms, 4),
        })

    categories = {}
    for category in sorted({case["category"] for case in world.cases}):
        categories[category] = {
            "graph": metrics([r for r in graph_rows if r["category"] == category]),
            "neural_matrix": metrics([r for r in matrix_rows if r["category"] == category]),
        }

    # For ambiguity, top-1 is not enough: report how many plausible interpretations
    # each method retrieves in its top five. A query is not marked as "unambiguous"
    # just because a system ranks one interpretation first.
    ambiguous_cases = [c for c in world.cases if c["category"] == "ambiguous"]
    ambiguous_coverage = {"graph_mean_plausible_recall_at_5": None, "neural_mean_plausible_recall_at_5": None}
    if ambiguous_cases:
        graph_coverages, neural_coverages = [], []
        for case in ambiguous_cases:
            accepted = set(case["target_ids"])
            g = world.graph.search(case["query"], top_k=TOP_K, max_hops=3, record_history=False)
            g_ids = {r["neuron_id"] for r in g}
            q_vec = model.encode([case["query"]], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
            scores = matrix @ q_vec
            k = min(TOP_K, len(scores))
            indices = np.argpartition(scores, -k)[-k:]
            n_ids = {world.fact_targets[int(i)] for i in indices}
            graph_coverages.append(len(accepted & g_ids) / len(accepted))
            neural_coverages.append(len(accepted & n_ids) / len(accepted))
        ambiguous_coverage = {
            "graph_mean_plausible_recall_at_5": round(statistics.mean(graph_coverages), 4),
            "neural_mean_plausible_recall_at_5": round(statistics.mean(neural_coverages), 4),
        }

    return {
        "fact_count": fact_count,
        "entity_count": len(world.graph.neurons),
        "edge_count": len(world.graph.synapses),
        "graph_build_ms": round(graph_build_ms, 3),
        "neural_embedding_build_ms": round(matrix_build_ms, 3),
        "dense_matrix_mib": round(matrix.nbytes / (1024 * 1024), 3),
        "overall": {
            "graph": metrics(graph_rows),
            "neural_matrix": metrics(matrix_rows),
        },
        "by_category": categories,
        "ambiguous_query_analysis": ambiguous_coverage,
        "cases": detailed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="100,1000,5000", help="number of facts per test world")
    parser.add_argument("--json-output", default="", help="optional JSON output path")
    args = parser.parse_args()
    sizes = [int(value.strip()) for value in args.sizes.split(",") if value.strip()]
    if not sizes or any(value < 20 for value in sizes):
        raise SystemExit("sizes must contain fact counts >= 20")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    results = {
        "benchmark": "adversarial_graph_vs_neural_fact_retrieval",
        "model": MODEL_NAME,
        "top_k": TOP_K,
        "method_notes": [
            "Graph is the existing NeuronGraph.search implementation (max_hops=3).",
            "Neural baseline ranks dense embeddings of the same fact/evidence text; it returns targets of retrieved facts and does not compose chains itself.",
            "Unseen paraphrases are not added as relation aliases.",
            "Contradiction cases include evidence counts/notes in the corpus and in the graph evidence ledger.",
            "Ambiguous questions accept multiple plausible targets and separately report their recall@5.",
        ],
        "sizes": [evaluate_size(model, size) for size in sizes],
    }
    rendered = json.dumps(results, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
