"""Compare baseline Neuron with feedback-reconstructed activation/path exploration.

Training uses a separate synthetic graph and known-answer feedback. Held-out meaning
cases are evaluated on a fresh graph; test labels never update the policy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuron_core import NeuronGraph, ReconstructedNeuron  # noqa: E402
from benchmarks.benchmark_meaning_understanding import build_world  # noqa: E402


def build_training_world():
    graph = NeuronGraph()
    rows = [
        ("Train Ada", "lives_in", "Aswan", "city", "Where does Train Ada live?"),
        ("Train Bo", "works_at", "Orion Lab", "organization", "Where does Train Bo work?"),
        ("Train Cy", "born_in", "Alexandria", "city", "Where was Train Cy born?"),
        ("Train Dee", "profession", "Platform Engineer", "profession", "What is Train Dee's job?"),
        ("Train Eli", "owns", "Bicycle", "object", "What does Train Eli own?"),
        ("Train Fay", "lives_in", "Cairo", "city", "Where does Train Fay live?"),
        ("Train Gil", "works_at", "Delta Clinic", "organization", "Where does Train Gil work?"),
        ("Train Hal", "profession", "Data Engineer", "profession", "What is Train Hal's job?"),
    ]
    for person, relation, answer, kind, _ in rows:
        p = graph.add_concept(person, kind="person")
        a = graph.add_concept(answer, kind=kind)
        distractor = graph.add_concept("Distractor " + answer, kind=kind)
        graph.connect(p, a, relation)
        graph.connect(p, distractor, relation)
    for relation, aliases in {
        "lives_in": ["live", "lives", "reside"],
        "works_at": ["work", "works", "work at"],
        "born_in": ["born", "birthplace"],
        "profession": ["job", "occupation"],
        "owns": ["own", "owned"],
    }.items():
        for alias in aliases:
            graph.add_relation_alias(relation, alias)
    return graph, [
        {"query": query, "expected": answer}
        for _, _, answer, _, query in rows
    ]


def evaluate(graph, cases, policy=None):
    rows = []
    for case in cases:
        start = time.perf_counter_ns()
        if policy is None:
            results = graph.search(case["query"], top_k=5, max_hops=3,
                                   workers=1, record_history=False)
        else:
            results = policy.search(case["query"], top_k=5, max_hops=3)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        names = [row["name"] for row in results]
        expected, mode = set(case["expected"]), case["mode"]
        if mode == "abstain":
            hit1 = hit5 = not results
        elif mode == "all_targets":
            hit1 = bool(names) and names[0] in expected
            hit5 = expected.issubset(set(names))
        else:
            hit1 = bool(names) and names[0] in expected
            hit5 = bool(expected.intersection(names))
        rows.append({"id": case["id"], "category": case["category"],
                     "hit1": bool(hit1), "hit5": bool(hit5),
                     "latency_ms": round(elapsed, 5), "returned": names})
    n = max(1, len(rows))
    return {
        "cases": len(rows),
        "hit_at_1": round(sum(r["hit1"] for r in rows) / n, 4),
        "hit_at_5": round(sum(r["hit5"] for r in rows) / n, 4),
        "mean_latency_ms": round(sum(r["latency_ms"] for r in rows) / n, 5),
        "details": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", default="")
    parser.add_argument("--episodes", type=int, default=120)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    train_graph, training = build_training_world()
    learner = ReconstructedNeuron(train_graph)
    for episode in range(args.episodes):
        item = training[episode % len(training)]
        results = learner.search(item["query"], top_k=8, max_hops=2)
        chosen = next((row for row in results if row["name"] == item["expected"]), None)
        if chosen is not None:
            learner.observe(item["query"], chosen["neuron_id"], 1.0)
        elif results:
            learner.observe(item["query"], results[0]["neuron_id"], -1.0)

    test_graph, cases = build_world()
    baseline = evaluate(test_graph, cases)
    reconstructed = evaluate(test_graph, cases, ReconstructedNeuron(test_graph))
    report = {
        "benchmark": "neuron_v4_feedback_reconstructed_activation_and_paths",
        "training": {
            "separate_synthetic_graph": True,
            "episodes": args.episodes,
            "verified_feedback_updates": learner.feedback_count,
            "learned_activation_weights": learner.activation_weights,
            "learned_path_weights": learner.path_weights,
        },
        "test": {
            "held_out_labels_never_update_policy": True,
            "same_fresh_graph_and_queries": True,
            "baseline_neuron": {k: v for k, v in baseline.items() if k != "details"},
            "reconstructed_neuron": {k: v for k, v in reconstructed.items() if k != "details"},
            "case_details": [
                {"id": b["id"], "category": b["category"],
                 "baseline_hit1": b["hit1"], "reconstructed_hit1": r["hit1"],
                 "baseline_top5": b["returned"], "reconstructed_top5": r["returned"]}
                for b, r in zip(baseline["details"], reconstructed["details"])
            ],
        },
        "limitations": [
            "This is a small synthetic benchmark, not evidence of general intelligence.",
            "Feedback is supplied by a known-answer oracle on a separate synthetic graph.",
            "The graph's lexical candidate discovery is unchanged; missing seed concepts remain missing.",
            "The reconstructed policy changes activation scores and edge exploration, but does not alter stored facts.",
            "Cross-language understanding is not solved by this experiment.",
            "Timing is noisy on shared CI and should be treated as approximate.",
        ],
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print("Baseline Neuron:", json.dumps(report["test"]["baseline_neuron"], sort_keys=True))
    print("Reconstructed Neuron:", json.dumps(report["test"]["reconstructed_neuron"], sort_keys=True))
    print("Feedback updates:", learner.feedback_count)
    if args.json_output:
        path = Path(args.json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
