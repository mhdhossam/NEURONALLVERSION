"""Stage-wise diagnostic for Neuron relation retrieval.

Runs against the repository's actual NeuronGraph implementation. All methods use
the same graph facts and queries. Held-out phrasings are not used in the alias
rules or embedding prototypes.

Stages:
1) relation interpretation (which relation is being asked about?),
2) candidate discovery (does the correct fact reach top-k?),
3) answer ranking (Hit@1, MRR, and Hit@1 conditional on Recall@k).

Controls:
- source-only lexical retrieval exposes the single-fact-per-person shortcut;
- oracle-relation graph lookup measures the ceiling when interpretation is given.

This is a controlled synthetic retrieval diagnostic, not a general reasoning test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from collections import Counter, defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuron_core import NeuronGraph  # noqa: E402
from neuron_core.engine import DEFAULT_RELATION_ALIASES, _tokens  # noqa: E402

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RELATIONS = [
    "profession", "specialty", "research_field", "home_city",
    "primary_tool", "affiliation", "discipline", "invention",
]
SHIPPED_TEMPLATES = {
    "profession": "What kind of work does {source} do?",
    "specialty": "Which subject is {source} especially skilled in?",
    "research_field": "What area does {source} study?",
    "home_city": "Where does {source} live?",
    "primary_tool": "Which tool does {source} mainly use?",
    "affiliation": "Which organization is {source} connected with?",
    "discipline": "What subject does {source} work in?",
    "invention": "What did {source} create?",
}
# These phrasings are deliberately not among Neuron's configured aliases.
HELDOUT_TEMPLATES = {
    "profession": "What livelihood sustains {source}?",
    "specialty": "Which capability distinguishes {source}?",
    "research_field": "What questions does {source} investigate?",
    "home_city": "Where has {source} put down roots?",
    "primary_tool": "Which implement does {source} reach for?",
    "affiliation": "Which institution counts {source} among its own?",
    "discipline": "Which scholarly branch frames {source}'s work?",
    "invention": "Which novelty is attributable to {source}?",
}
# Prototypes are authored only from the known relation meanings; no held-out
# query text is included. The query is masked before relation classification.
RELATION_PROTOTYPES = {
    "profession": "a person's occupation or job role",
    "specialty": "the topic or area of expertise a person knows particularly well",
    "research_field": "the area of research and study a person investigates",
    "home_city": "the place where a person lives or resides",
    "primary_tool": "the main tool someone uses to do their work",
    "affiliation": "an organization a person belongs to or is associated with",
    "discipline": "an academic discipline or subject area",
    "invention": "a thing a person creates, designs, or builds",
}
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
ADJECTIVES = [
    "Quantum", "Orbital", "Distributed", "Symbolic", "Adaptive", "Cognitive",
    "Geometric", "Probabilistic", "Linguistic", "Robotic", "Computational",
    "Molecular", "Celestial", "Recursive", "Causal", "Relational", "Temporal",
    "Spatial", "Neural", "Algebraic", "Stochastic", "Modular", "Autonomous",
    "Evolutionary", "Topological", "Acoustic", "Visual", "Semantic", "Synthetic",
    "Ecological", "Mechanical", "Cryptographic", "Numerical", "Biological",
    "Electrical", "Fluid", "Robust", "Dynamic", "Hybrid", "Discrete",
    "Continuous", "Parallel", "Formal", "Contextual", "Hierarchical", "Spectral",
    "Graphical", "Statistical", "Computational", "Causal",
]
NOUNS = [
    "Reasoning", "Cartography", "Optimization", "Linguistics", "Navigation",
    "Planning", "Compression", "Simulation", "Inference", "Classification",
    "Forecasting", "Translation", "Chemistry", "Astronomy", "Robotics",
    "Database", "Security", "Architecture", "Statistics", "Phonetics",
    "Logistics", "Synthesis", "Control", "Diagnostics", "Economics", "Ecology",
    "Geometry", "Topology", "Retrieval", "Indexing", "Learning", "Perception",
    "Memory", "Causality", "Detection", "Verification", "Generation", "Alignment",
    "Scheduling", "Modeling", "Vision", "Mathematics", "Acoustics", "Biophysics",
    "Semantics", "Networking", "Probability", "Algorithms",
]
TARGET_KINDS = {
    "profession": "profession", "specialty": "topic",
    "research_field": "topic", "home_city": "city",
    "primary_tool": "tool", "affiliation": "organization",
    "discipline": "discipline", "invention": "artifact",
}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    pos = (len(vals) - 1) * p
    low, high = math.floor(pos), math.ceil(pos)
    return vals[low] * (high - pos) + vals[high] * (pos - low)


def rank_metrics(ranks: list[int | None], latencies_ms: list[float], top_k: int) -> dict:
    n = max(1, len(ranks))
    hit1 = sum(rank == 1 for rank in ranks) / n
    recall = sum(rank is not None and rank <= top_k for rank in ranks) / n
    reciprocal = [1.0 / rank if rank else 0.0 for rank in ranks]
    return {
        "n": len(ranks),
        "hit_at_1": round(hit1, 4),
        "recall_at_k": round(recall, 4),
        "mrr": round(statistics.mean(reciprocal), 4) if reciprocal else 0.0,
        "hit1_given_candidate_in_top_k": round(hit1 / recall, 4) if recall else None,
        "mean_ms": round(statistics.mean(latencies_ms), 4) if latencies_ms else None,
        "p95_ms": round(percentile(latencies_ms, 0.95), 4) if latencies_ms else None,
    }


def build_world(n_facts: int, all_relations_per_person: bool) -> dict:
    graph = NeuronGraph()
    sources: list[str] = []
    targets: list[str] = []
    source_names: list[str] = []
    facts: list[dict] = []
    source_to_id: dict[str, str] = {}

    if all_relations_per_person:
        people_count = n_facts // len(RELATIONS)
        relation_plan = [(person, relation) for person in range(people_count) for relation in RELATIONS]
    else:
        people_count = n_facts
        relation_plan = [(person, RELATIONS[person % len(RELATIONS)]) for person in range(people_count)]

    for person in range(people_count):
        name = f"{FIRST_NAMES[person % len(FIRST_NAMES)]} {LAST_NAMES[(person // len(FIRST_NAMES)) % len(LAST_NAMES)]} P{person:04d}"
        source_names.append(name)
        source_to_id[name] = graph.add_concept(name, kind="person")
        sources.append(source_to_id[name])

    for fact_index, (person, relation) in enumerate(relation_plan):
        source_name = source_names[person]
        target_name = (
            f"{ADJECTIVES[(fact_index * 7) % len(ADJECTIVES)]} "
            f"{NOUNS[(fact_index * 11) % len(NOUNS)]} T{fact_index:04d}"
        )
        target_id = graph.add_concept(target_name, kind=TARGET_KINDS[relation])
        sources_id = source_to_id[source_name]
        graph.connect(sources_id, target_id, relation, weight=1.0)
        targets.append(target_id)
        relation_label = relation.replace("_", " ")
        facts.append({
            "index": fact_index,
            "source": source_name,
            "source_id": sources_id,
            "relation": relation,
            "target": target_name,
            "target_id": target_id,
            "document": f"{source_name}'s {relation_label} is {target_name}.",
        })

    cases_by_mode: dict[str, list[dict]] = {"canonical": [], "shipped": [], "heldout": []}
    for fact in facts:
        for mode, templates in (
            ("canonical", None),
            ("shipped", SHIPPED_TEMPLATES),
            ("heldout", HELDOUT_TEMPLATES),
        ):
            query = (
                f"What is {fact['source']}'s {fact['relation'].replace('_', ' ')}?"
                if mode == "canonical"
                else templates[fact["relation"]].format(source=fact["source"])
            )
            masked = query.replace(fact["source"], "PERSON")
            cases_by_mode[mode].append({
                **fact,
                "query": query,
                "masked_query": masked,
                "mode": mode,
            })

    return {
        "graph": graph, "facts": facts, "sources": sources, "targets": targets,
        "cases_by_mode": cases_by_mode, "all_relations_per_person": all_relations_per_person,
        "people_count": people_count,
    }


def relation_rule_matches(query: str) -> set[str]:
    q = _tokens(query)
    matched: set[str] = set()
    for relation in RELATIONS:
        phrases = set(DEFAULT_RELATION_ALIASES.get(relation, ()))
        phrases.add(relation.replace("_", " "))
        custom = getattr(NeuronGraph(), "_custom_relation_aliases", {}).get(relation, set())
        phrases.update(custom)
        for phrase in phrases:
            pt = _tokens(phrase)
            if pt and pt.issubset(q):
                matched.add(relation)
                break
    return matched


def tokenize(text: str) -> list[str]:
    import re
    return [t for t in re.findall(r"\w+", text.casefold()) if len(t) > 1]


def build_sparse_retrievers(documents: list[str]) -> tuple[dict, dict]:
    doc_tokens = [tokenize(doc) for doc in documents]
    tf = [Counter(tokens) for tokens in doc_tokens]
    df = Counter(token for row in tf for token in row)
    n = len(documents)
    idf = {term: math.log((n + 1) / (freq + 1)) + 1.0 for term, freq in df.items()}
    tfidf_docs = []
    tfidf_norms = []
    tfidf_postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
    lengths = [sum(row.values()) for row in tf]
    avgdl = statistics.mean(lengths) if lengths else 0.0
    bm25_postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
    bm25_idf = {
        term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
        for term, freq in df.items()
    }
    k1, b = 1.5, 0.75

    for index, row in enumerate(tf):
        weights = {term: count * idf[term] for term, count in row.items()}
        norm = math.sqrt(sum(value * value for value in weights.values())) or 1.0
        tfidf_docs.append({term: value / norm for term, value in weights.items()})
        tfidf_norms.append(norm)
        dl = lengths[index]
        for term, count in row.items():
            tfidf_postings[term].append((index, tfidf_docs[-1][term]))
            denom = count + k1 * (1.0 - b + b * dl / max(avgdl, 1e-9))
            score = bm25_idf[term] * (count * (k1 + 1.0)) / denom
            bm25_postings[term].append((index, score))

    return {
        "tf": tf, "idf": idf, "tfidf_docs": tfidf_docs, "tfidf_postings": tfidf_postings,
        "bm25_postings": bm25_postings, "doc_count": n,
    }, {"avgdl": avgdl, "lengths": lengths, "k1": k1, "b": b, "bm25_idf": bm25_idf}


def sparse_rank(query: str, index: dict, mode: str, top_k: int) -> list[int]:
    terms = tokenize(query)
    scores: dict[int, float] = defaultdict(float)
    if mode == "tfidf":
        counts = Counter(terms)
        q_weights = {term: count * index["idf"].get(term, 0.0) for term, count in counts.items()}
        q_norm = math.sqrt(sum(value * value for value in q_weights.values())) or 1.0
        for term, q_weight in q_weights.items():
            for doc_idx, doc_weight in index["tfidf_postings"].get(term, ()):
                scores[doc_idx] += q_weight * doc_weight
        rows = [(i, value / q_norm) for i, value in scores.items()]
    else:
        for term in set(terms):
            for doc_idx, term_score in index["bm25_postings"].get(term, ()):
                scores[doc_idx] += term_score
        rows = list(scores.items())
    rows.sort(key=lambda item: (-item[1], item[0]))
    return [index for index, _ in rows[:top_k]]


def source_only_rank(case: dict, facts: list[dict], top_k: int) -> list[int]:
    source_terms = set(tokenize(case["source"]))
    scored = []
    for fact in facts:
        overlap = len(source_terms.intersection(tokenize(fact["source"])))
        tie = hashlib.sha256(f"{case['source']}|{fact['index']}".encode()).hexdigest()
        scored.append((overlap, tie, fact["index"]))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [row[2] for row in scored[:top_k]]


def rank_of_target(target_ids: list[str], expected_target: str) -> int | None:
    try:
        return target_ids.index(expected_target) + 1
    except ValueError:
        return None


def evaluate_world(label: str, world: dict, model, top_k: int) -> dict:
    graph: NeuronGraph = world["graph"]
    facts: list[dict] = world["facts"]
    docs = [fact["document"] for fact in facts]
    retriever, _bm_params = build_sparse_retrievers(docs)
    prototypes = [RELATION_PROTOTYPES[r] for r in RELATIONS]
    prototype_vectors = model.encode(
        prototypes, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False, batch_size=32,
    ).astype(np.float32)

    result = {
        "dataset": label,
        "fact_count": len(facts),
        "person_count": world["people_count"],
        "all_relations_per_person": world["all_relations_per_person"],
        "modes": {},
        "data_leakage_control": {},
    }
    target_id_to_fact = {fact["target_id"]: fact["index"] for fact in facts}
    target_ids = set(target_id_to_fact)

    for mode, cases in world["cases_by_mode"].items():
        if not cases:
            continue
        full_queries = [case["query"] for case in cases]
        masked_queries = [case["masked_query"] for case in cases]
        # Batch embeddings are for ranking metrics; single-query timings are sampled
        # separately so the latency report is not a batch-throughput measurement.
        q_masked = model.encode(
            masked_queries, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False, batch_size=64,
        ).astype(np.float32)
        similarities = q_masked @ prototype_vectors.T
        predicted_relations = [RELATIONS[int(i)] for i in similarities.argmax(axis=1)]
        top2_relation_ids = np.argsort(similarities, axis=1)[:, -2:]
        relation_top2 = [
            case["relation"] in {RELATIONS[int(j)] for j in row}
            for case, row in zip(cases, top2_relation_ids)
        ]

        graph_ranks: list[int | None] = []
        graph_raw_node_hit1: list[bool] = []
        graph_latency: list[float] = []
        matcher_ranks: list[int | None] = []
        matcher_latency_sample: list[float] = []
        bm25_ranks: list[int | None] = []
        bm25_latency: list[float] = []
        tfidf_ranks: list[int | None] = []
        tfidf_latency: list[float] = []
        source_ranks: list[int | None] = []
        oracle_ranks: list[int | None] = []
        rule_gold_covered = []
        rule_exact = []
        rule_false_positive_counts = []
        embed_pred_correct = []
        embed_pred_top2 = []
        bm_relation_correct = []
        tfidf_relation_correct = []
        examples = []

        # Warmup the native graph path once.
        if cases:
            graph.search(cases[0]["query"], top_k=top_k, max_hops=2, workers=1, record_history=False)

        for i, case in enumerate(cases):
            expected_fact_index = case["index"]
            expected_target_id = case["target_id"]
            gold_relation = case["relation"]

            matched = relation_rule_matches(case["masked_query"])
            rule_gold_covered.append(gold_relation in matched)
            rule_exact.append(matched == {gold_relation})
            rule_false_positive_counts.append(len(matched - {gold_relation}))
            embed_pred_correct.append(predicted_relations[i] == gold_relation)
            embed_pred_top2.append(relation_top2[i])

            start = time.perf_counter_ns()
            graph_rows = graph.search(
                case["query"], top_k=top_k, max_hops=2, workers=1, record_history=False
            )
            graph_latency.append((time.perf_counter_ns() - start) / 1e6)
            raw_top = graph_rows[0]["neuron_id"] if graph_rows else None
            graph_raw_node_hit1.append(raw_top == expected_target_id)
            ranked_graph_targets = [
                row["neuron_id"] for row in graph_rows if row["neuron_id"] in target_ids
            ]
            graph_rank = rank_of_target(ranked_graph_targets, expected_target_id)
            graph_ranks.append(graph_rank)

            # Graph + semantic relation matcher: resolve the subject from Neuron's
            # lexical index, classify the masked relation phrase, then follow that
            # typed outgoing edge. It does not train on any held-out query wording.
            t0 = time.perf_counter_ns()
            seed_rows = graph.find_candidates(case["query"], top_k=10)
            person_seeds = [
                (nid, score) for nid, score in seed_rows
                if graph.neurons[nid].kind == "person"
            ]
            chosen_source = person_seeds[0][0] if person_seeds else None
            target_id = None
            if chosen_source is not None:
                predicted_relation = predicted_relations[i]
                outgoing = graph._outgoing_by_relation.get(chosen_source, {}).get(predicted_relation, [])
                if outgoing:
                    edge = graph.synapses[outgoing[0]]
                    target_id = edge.target_id
            matcher_latency_sample.append((time.perf_counter_ns() - t0) / 1e6)
            matcher_ranks.append(1 if target_id == expected_target_id else None)

            # BM25 and TF-IDF rank exactly the same fact documents.
            t0 = time.perf_counter_ns()
            bm_rows = sparse_rank(case["query"], retriever, "bm25", top_k)
            bm25_latency.append((time.perf_counter_ns() - t0) / 1e6)
            bm25_ranks.append(bm_rows.index(expected_fact_index) + 1 if expected_fact_index in bm_rows else None)
            bm_relation_correct.append(facts[bm_rows[0]]["relation"] == gold_relation if bm_rows else False)

            t0 = time.perf_counter_ns()
            tf_rows = sparse_rank(case["query"], retriever, "tfidf", top_k)
            tfidf_latency.append((time.perf_counter_ns() - t0) / 1e6)
            tfidf_ranks.append(tf_rows.index(expected_fact_index) + 1 if expected_fact_index in tf_rows else None)
            tfidf_relation_correct.append(facts[tf_rows[0]]["relation"] == gold_relation if tf_rows else False)

            source_rows = source_only_rank(case, facts, top_k)
            source_ranks.append(source_rows.index(expected_fact_index) + 1 if expected_fact_index in source_rows else None)

            # Oracle ceiling: relation interpretation is supplied correctly; the
            # graph then selects the matching typed edge from the recovered person.
            source_id = case["source_id"]
            gold_edges = graph._outgoing_by_relation.get(source_id, {}).get(gold_relation, [])
            oracle_target = graph.synapses[gold_edges[0]].target_id if gold_edges else None
            oracle_ranks.append(1 if oracle_target == expected_target_id else None)

            if len(examples) < 8 and mode == "heldout":
                examples.append({
                    "query": case["query"],
                    "gold_relation": gold_relation,
                    "rule_matches": sorted(matched),
                    "embedding_relation": predicted_relations[i],
                    "expected_target": case["target"],
                    "graph_target_rank": graph_rank,
                    "bm25_fact_rank": bm25_ranks[-1],
                    "tfidf_fact_rank": tfidf_ranks[-1],
                })

        # End-to-end latency for neural interpretation includes per-query encoding.
        # Measure a stable subset, not the batched metric path above.
        embed_e2e_sample = []
        timing_n = min(100, len(cases))
        for case in cases[:timing_n]:
            t0 = time.perf_counter_ns()
            qv = model.encode(
                [case["masked_query"]], normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )[0]
            _pred = int(np.argmax(prototype_vectors @ qv))
            embed_e2e_sample.append((time.perf_counter_ns() - t0) / 1e6)

        def relation_accuracy(correct: list[bool]) -> float:
            return round(sum(correct) / max(1, len(correct)), 4)

        def compact(ranks: list[int | None], latencies: list[float]) -> dict:
            return rank_metrics(ranks, latencies, top_k)

        result["modes"][mode] = {
            "query_count": len(cases),
            "stage_1_relation_interpretation": {
                "neuron_alias_gold_relation_covered": relation_accuracy(rule_gold_covered),
                "neuron_alias_exactly_one_correct_relation": relation_accuracy(rule_exact),
                "mean_false_positive_relations_per_query": round(statistics.mean(rule_false_positive_counts), 4),
                "embedding_prototype_top1_accuracy": relation_accuracy(embed_pred_correct),
                "embedding_prototype_top2_accuracy": relation_accuracy(embed_pred_top2),
                "bm25_top1_fact_relation_accuracy": relation_accuracy(bm_relation_correct),
                "tfidf_top1_fact_relation_accuracy": relation_accuracy(tfidf_relation_correct),
            },
            "stage_2_and_3_answer_retrieval": {
                "neuron_graph": compact(graph_ranks, graph_latency),
                "graph_plus_embedding_relation_matcher": compact(matcher_ranks, matcher_latency_sample),
                "bm25_fact_retrieval": compact(bm25_ranks, bm25_latency),
                "tfidf_fact_retrieval": compact(tfidf_ranks, tfidf_latency),
                "source_only_control": compact(source_ranks, []),
                "oracle_relation_graph_upper_bound": compact(oracle_ranks, []),
                "neuron_raw_node_top1_is_expected_target": round(
                    sum(graph_raw_node_hit1) / max(1, len(graph_raw_node_hit1)), 4
                ),
                "embedding_matcher_mean_ms_single_query_sample": round(statistics.mean(embed_e2e_sample), 4),
                "embedding_matcher_p95_ms_single_query_sample": round(percentile(embed_e2e_sample, 0.95), 4),
            },
            "heldout_examples": examples,
        }

    # The single-fact-per-person setting can be solved by reading the person's only
    # target, regardless of which relation is asked. Quantify this explicitly.
    if not world["all_relations_per_person"]:
        result["data_leakage_control"] = {
            "warning": "S1 has exactly one fact per person; source-only lookup can recover the benchmark's target without interpreting the requested relation.",
            "source_only_hit1_per_mode": {
                mode: result["modes"][mode]["stage_2_and_3_answer_retrieval"]["source_only_control"]["hit_at_1"]
                for mode in result["modes"]
            },
        }
    else:
        result["data_leakage_control"] = {
            "warning": "S2 places all eight relations on every person, so subject-only lookup cannot uniquely identify the answer.",
            "source_only_hit1_per_mode": {
                mode: result["modes"][mode]["stage_2_and_3_answer_retrieval"]["source_only_control"]["hit_at_1"]
                for mode in result["modes"]
            },
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="100,1000,5000", help="S1 graph node counts; each fact uses one source and one target node")
    parser.add_argument("--s2-facts", type=int, default=800, help="S2 facts; must be divisible by the number of relations")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-output", default="stage-diagnostic-results.json")
    args = parser.parse_args()
    if args.s2_facts % len(RELATIONS):
        raise SystemExit(f"--s2-facts must be divisible by {len(RELATIONS)}")
    sizes = [int(value.strip()) for value in args.sizes.split(",") if value.strip()]
    if not sizes or any(value < 100 or value % 2 for value in sizes):
        raise SystemExit("--sizes must be even node counts >= 100")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    load_started = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.eval()
    load_seconds = time.perf_counter() - load_started

    results = {
        "benchmark": "Neuron stage-wise relation interpretation, candidate discovery, and answer ranking diagnostic",
        "model": MODEL_NAME,
        "model_load_seconds_excluded_from_query_latency": round(load_seconds, 3),
        "top_k": args.top_k,
        "method_notes": [
            "All methods see the same source-relation-target facts in each world.",
            "S1 has one fact per person; the source-only control measures whether this makes relation interpretation unnecessary.",
            "S2 has all eight relations per person and is the principal relation-selection diagnostic.",
            "Held-out templates are not present in Neuron's shipped alias table or the embedding relation prototypes.",
            "Graph+embedding matcher first resolves a person through Neuron's lexical index, predicts a relation from masked query wording, then follows that typed edge.",
            "BM25 and TF-IDF rank the same fact documents; they are retrieval baselines, not reasoning engines.",
            "Oracle-relation graph scores are an upper bound with the correct relation supplied, not an independently deployable method.",
            "Latency is measured on this CPU runner; semantic matcher end-to-end latency uses a sampled single-query encode, while metrics use batched embeddings.",
            "This is a synthetic retrieval diagnostic, not a general language understanding or LLM comparison.",
        ],
        "s1_one_fact_per_person": [],
        "s2_all_relations_per_person": None,
    }
    for node_count in sizes:
        n_facts = node_count // 2
        world = build_world(n_facts, all_relations_per_person=False)
        results["s1_one_fact_per_person"].append(
            evaluate_world(f"S1_{node_count}_nodes_{n_facts}_facts", world, model, args.top_k)
        )
    s2_world = build_world(args.s2_facts, all_relations_per_person=True)
    results["s2_all_relations_per_person"] = evaluate_world(
        f"S2_{args.s2_facts}_facts_{len(RELATIONS)}_relations_per_person",
        s2_world, model, args.top_k,
    )

    rendered = json.dumps(results, indent=2, ensure_ascii=False)
    print(rendered)
    output = Path(args.json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(f"\nJSON report written to {output}")


if __name__ == "__main__":
    main()
