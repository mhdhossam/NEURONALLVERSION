"""Measure first-attempt genetic learning on Neuron's synthetic meaning benchmark."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuron_core.genetic import GeneticRanker
from benchmarks.benchmark_meaning_understanding import build_world


def top1(case, rows):
    names = [r["name"] for r in rows]
    if case["mode"] == "abstain":
        return not names
    return bool(names) and names[0] in set(case["expected"])


def passed(case, rows):
    names, expected = {r["name"] for r in rows}, set(case["expected"])
    if case["mode"] == "abstain":
        return not names
    if case["mode"] == "all_targets":
        return expected.issubset(names)
    return bool(expected.intersection(names))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--population", type=int, default=36)
    parser.add_argument("--generations", type=int, default=16)
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    graph, cases = build_world()
    learner = GeneticRanker(population_size=args.population, generations=args.generations, seed=17)
    details, grouped, updates = [], defaultdict(list), 0

    for case in cases:
        baseline = graph.search(case["query"], top_k=args.top_k, max_hops=3,
                                workers=1, record_history=False)
        start = time.perf_counter_ns()
        candidates = graph.search(case["query"], top_k=max(args.top_k, 20), max_hops=3,
                                  workers=1, record_history=False)
        first = learner.rank(case["query"], candidates)[:args.top_k]
        first_ms = (time.perf_counter_ns() - start) / 1e6
        # Labels are revealed only after first attempt. Correct targets must exist in
        # retrieval output; otherwise this ranker cannot fix the case.
        if case["mode"] == "abstain":
            if candidates:
                # This exposes a limitation: current genome has no abstention mechanism.
                # Do not pretend the wrong retrieved answer is a correct target.
                pass
        else:
            expected = set(case["expected"])
            available = [r["name"] for r in candidates if r["name"] in expected]
            if available:
                learner.learn_from_mistake(case["query"], candidates, available)
                updates += 1
        start = time.perf_counter_ns()
        retry = learner.rank(case["query"], candidates)[:args.top_k]
        retry_ms = (time.perf_counter_ns() - start) / 1e6
        item = {
            "id": case["id"], "category": case["category"], "query": case["query"],
            "expected": case["expected"], "baseline_top1": top1(case, baseline),
            "genetic_first_top1": top1(case, first), "genetic_retry_top1": top1(case, retry),
            "genetic_first_pass": passed(case, first), "genetic_retry_pass": passed(case, retry),
            "first_results": [r["name"] for r in first], "retry_results": [r["name"] for r in retry],
            "first_latency_ms": round(first_ms, 5), "retry_latency_ms": round(retry_ms, 5),
            "feedback_examples_after_case": len(learner.examples),
        }
        details.append(item)
        grouped[case["category"]].append(item)

    def summary(items):
        n = max(1, len(items))
        return {
            "cases": len(items),
            "baseline_hit_at_1": round(sum(x["baseline_top1"] for x in items) / n, 4),
            "genetic_first_hit_at_1": round(sum(x["genetic_first_top1"] for x in items) / n, 4),
            "genetic_retry_hit_at_1": round(sum(x["genetic_retry_top1"] for x in items) / n, 4),
            "genetic_first_pass_rate": round(sum(x["genetic_first_pass"] for x in items) / n, 4),
            "genetic_retry_pass_rate": round(sum(x["genetic_retry_pass"] for x in items) / n, 4),
            "mean_first_latency_ms": round(sum(x["first_latency_ms"] for x in items) / n, 5),
            "mean_retry_latency_ms": round(sum(x["retry_latency_ms"] for x in items) / n, 5),
        }

    report = {
        "benchmark": "neuron_first_attempt_genetic_learning",
        "learning_type": "genetic selection, crossover, mutation; no RL optimizer",
        "offline_training_examples": 0, "feedback_updates": updates,
        "genetic_state": learner.state_dict(), "overall": summary(details),
        "by_category": {k: summary(v) for k, v in sorted(grouped.items())},
        "details": details,
        "limitations": [
            "Hand-authored benchmark labels act as verifier feedback only after first attempt.",
            "This component only reorders retrieved candidates; missing facts remain missing.",
            "No learned abstention mechanism is implemented in this first experiment.",
            "Sequential results are sensitive to case order and do not prove broad generalization.",
        ],
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print("Genetic-learning summary:", json.dumps({"overall": report["overall"],
          "feedback_updates": updates, "by_category": report["by_category"]}, sort_keys=True))
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
