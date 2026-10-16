"""Compare baseline Neuron ranking with an isolated RL-style decision layer.

Training feedback comes from a separate synthetic world. Existing meaning
understanding cases are held out; their labels never update the ranker.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuron_core import NeuronGraph  # noqa: E402
from neuron_core.decision import ContextualBanditRanker  # noqa: E402
from benchmarks.benchmark_meaning_understanding import build_world  # noqa: E402

def build_training_world():
    graph = NeuronGraph()
    pairs = [
        ("Training Person A", "lives_in", "Cairo", "city", "Where does Training Person A live?"),
        ("Training Person B", "works_at", "Delta Clinic", "organization", "Where does Training Person B work at?"),
        ("Training Person C", "born_in", "Alexandria", "city", "Where was Training Person C born?"),
        ("Training Person D", "profession", "Platform Engineer", "profession", "What is Training Person D's job?"),
        ("Training Person E", "owns", "Bicycle", "object", "What does Training Person E own?"),
        ("Training Person F", "lives_in", "Aswan", "city", "Where does Training Person F live?"),
        ("Training Person G", "works_at", "Orion Lab", "organization", "Where does Training Person G work at?"),
        ("Training Person H", "profession", "Data Engineer", "profession", "What is Training Person H's job?"),
    ]
    for person, relation, target, kind, _ in pairs:
        person_id = graph.add_concept(person, kind="person")
        target_id = graph.add_concept(target, kind=kind)
        distractor_id = graph.add_concept(f"Unrelated {kind} {person[-1]}", kind=kind)
        graph.connect(person_id, target_id, relation)
        graph.connect(person_id, distractor_id, relation)
    for relation, aliases in {
        "lives_in": ["live", "lives", "reside"], "works_at": ["work at", "works at"],
        "born_in": ["born", "birthplace"], "profession": ["job", "occupation"],
        "owns": ["own", "owned"],
    }.items():
        for alias in aliases:
            graph.add_relation_alias(relation, alias)
    training = []
    for person, relation, target, kind, query in pairs:
        training.append({"query": query, "expected": target,
                         "candidates": graph.search(query, top_k=5, max_hops=2, record_history=False)})
    return training

def metrics(rows):
    if not rows:
        return {"cases": 0, "hit_at_1": 0.0, "hit_at_5": 0.0, "mean_latency_ms": 0.0}
    return {"cases": len(rows),
            "hit_at_1": round(sum(row["hit1"] for row in rows) / len(rows), 4),
            "hit_at_5": round(sum(row["hit5"] for row in rows) / len(rows), 4),
            "mean_latency_ms": round(sum(row["latency_ms"] for row in rows) / len(rows), 4)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-output', default='')
    args = parser.parse_args()
    ranker = ContextualBanditRanker(exploration=0.25, seed=7)
    training = build_training_world()
    feedback = []
    for episode in range(120):
        item = training[episode % len(training)]
        chosen, reward = ranker.choose_and_observe(
            item["query"], item["candidates"],
            lambda candidate, expected=item["expected"]: 1.0 if candidate.get("name") == expected else -1.0,
            explore=True)
        feedback.append({"episode": episode, "chosen": chosen, "reward": reward})

    test_graph, cases = build_world()
    baseline_rows, policy_rows, details = [], [], []
    for case in cases:
        started = time.perf_counter_ns()
        baseline = test_graph.search(case["query"], top_k=5, max_hops=3, workers=1, record_history=False)
        baseline_ms = (time.perf_counter_ns() - started) / 1e6
        started = time.perf_counter_ns()
        policy = ranker.rank(case["query"], baseline, explore=False)
        # Include graph retrieval as well as reranking for fair end-to-end latency.
        policy_ms = baseline_ms + (time.perf_counter_ns() - started) / 1e6
        expected, mode = set(case["expected"]), case["mode"]
        base_names, policy_names = [x["name"] for x in baseline], [x["name"] for x in policy]
        if mode == "abstain":
            b1 = b5 = not baseline
            p1 = p5 = not policy
        elif mode == "all_targets":
            b1 = bool(base_names) and base_names[0] in expected
            b5 = expected.issubset(set(base_names))
            p1 = bool(policy_names) and policy_names[0] in expected
            p5 = expected.issubset(set(policy_names))
        else:
            b1 = bool(base_names) and base_names[0] in expected
            b5 = bool(expected.intersection(base_names))
            p1 = bool(policy_names) and policy_names[0] in expected
            p5 = bool(expected.intersection(policy_names))
        baseline_rows.append({"hit1": b1, "hit5": b5, "latency_ms": baseline_ms})
        policy_rows.append({"hit1": p1, "hit5": p5, "latency_ms": policy_ms})
        details.append({"id": case["id"], "category": case["category"], "query": case["query"],
                        "expected": case["expected"], "baseline_top5": base_names,
                        "policy_top5": policy_names, "baseline_hit1": bool(b1), "policy_hit1": bool(p1)})
    report = {
        "benchmark": "neuron_v4_rl_decision_layer_vs_baseline",
        "training": {"separate_synthetic_world": True, "episodes": len(feedback),
                     "ranker_feedback_observations": ranker.observations,
                     "state_features": list(ranker.feature_names), "third_party_dependencies": []},
        "test": {"cases": len(cases), "held_out_from_policy_updates": True,
                 "same_test_world_and_queries_for_both_modes": True,
                 "baseline_neuron_search": metrics(baseline_rows),
                 "rl_decision_layer": metrics(policy_rows), "case_details": details},
        "limitations": [
            "The policy ranks only candidates returned by Neuron and cannot recover a missing answer.",
            "Training rewards are synthetic and supplied by a known-label oracle.",
            "The held-out set is small and synthetic; results do not establish open-ended reasoning.",
            "The diagonal LinUCB approximation is deliberately tiny; reward and feature quality bound performance.",
            "Ranking adds CPU work, so latency is measured rather than assumed to improve.",
            "The baseline graph retrieval implementation is unchanged.",
        ]}
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print("Baseline:", json.dumps(report["test"]["baseline_neuron_search"], sort_keys=True))
    print("RL policy:", json.dumps(report["test"]["rl_decision_layer"], sort_keys=True))
    print("Training feedback observations:", ranker.observations)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\\n", encoding="utf-8")
    else:
        print(rendered)

if __name__ == "__main__":
    main()
