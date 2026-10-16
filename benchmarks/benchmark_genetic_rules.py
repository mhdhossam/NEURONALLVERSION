"""Compare baseline, genetic weights, and genetic symbolic rules.

This is an online feedback experiment on the existing small synthetic benchmark.
Labels are revealed only after each first attempt. Because the cases are processed
in their defined order, results are order-sensitive and do not establish broad
generalization. The report separately measures later paraphrase cases as transfer.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuron_core.genetic import GeneticRanker
from neuron_core.genetic_rules import GeneticRuleRanker
from benchmarks.benchmark_meaning_understanding import build_world


def top1(case, rows):
    names = [r["name"] for r in rows]
    if case["mode"] == "abstain":
        return not names
    return bool(names) and names[0] in set(case["expected"])


def main():
    graph, cases = build_world()
    weights = GeneticRanker(population_size=36, generations=16, seed=17)
    rules = GeneticRuleRanker(population_size=36, generations=16, seed=29)
    rows = []
    for case in cases:
        baseline = graph.search(case["query"], top_k=5, max_hops=3, workers=1, record_history=False)
        candidates = graph.search(case["query"], top_k=20, max_hops=3, workers=1, record_history=False)
        # First attempts are measured before exposing labels for this case.
        w_first = weights.rank(case["query"], candidates)[:5]
        r_first = rules.rank(case["query"], candidates)[:5]
        b_ok, w_ok, r_ok = top1(case, baseline), top1(case, w_first), top1(case, r_first)
        expected = set(case["expected"])
        available = [c["name"] for c in candidates if c["name"] in expected]
        if case["mode"] != "abstain":
            if not w_ok and available:
                weights.learn_from_mistake(case["query"], candidates, available)
            if not r_ok and available:
                rules.learn_from_mistake(case["query"], candidates, available)
        # Retry only after feedback; this measures immediate correction separately.
        w_retry = weights.rank(case["query"], candidates)[:5]
        r_retry = rules.rank(case["query"], candidates)[:5]
        rows.append({
            "id": case["id"], "category": case["category"], "query": case["query"],
            "baseline_hit1": b_ok, "weight_first_hit1": w_ok, "rule_first_hit1": r_ok,
            "weight_retry_hit1": top1(case, w_retry), "rule_retry_hit1": top1(case, r_retry),
            "weight_retry_names": [x["name"] for x in w_retry],
            "rule_retry_names": [x["name"] for x in r_retry],
            "rule_count_after_case": len(rules.rules),
            "weight_feedback_examples": len(weights.examples),
            "rule_feedback_examples": len(rules.examples),
        })

    def rate(items, key):
        return round(sum(bool(x[key]) for x in items) / max(1, len(items)), 4)
    # Treat held-out paraphrase category as a transfer probe after previous
    # feedback categories; its first-attempt score is before current-case feedback.
    groups = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    report = {
        "benchmark": "neuron_genetic_symbolic_rule_vs_weight",
        "cases": len(rows),
        "overall": {
            "baseline_hit1": rate(rows, "baseline_hit1"),
            "weight_first_hit1": rate(rows, "weight_first_hit1"),
            "rule_first_hit1": rate(rows, "rule_first_hit1"),
            "weight_retry_hit1": rate(rows, "weight_retry_hit1"),
            "rule_retry_hit1": rate(rows, "rule_retry_hit1"),
        },
        "by_category": {k: {
            "cases": len(v), "baseline_hit1": rate(v, "baseline_hit1"),
            "weight_first_hit1": rate(v, "weight_first_hit1"),
            "rule_first_hit1": rate(v, "rule_first_hit1"),
            "weight_retry_hit1": rate(v, "weight_retry_hit1"),
            "rule_retry_hit1": rate(v, "rule_retry_hit1"),
        } for k, v in sorted(groups.items())},
        "final_rule_count": len(rules.rules),
        "final_rules": rules.rules,
        "weight_feedback_examples": len(weights.examples),
        "rule_feedback_examples": len(rules.examples),
        "limitations": [
            "Verifier labels are supplied by the benchmark after first attempts; the system does not detect its own mistakes.",
            "This is a 31-case hand-authored synthetic benchmark and is sensitive to order and candidate retrieval.",
            "Rules only reorder retrieved candidates; they cannot recover missing facts or safely learn abstention.",
            "Category scores are diagnostics, not a clean held-out generalization estimate.",
        ],
        "details": rows,
    }
    print("Genetic-rule comparison:", json.dumps({"overall": report["overall"],
          "by_category": report["by_category"], "final_rule_count": report["final_rule_count"]}, sort_keys=True))
    Path("neuron-genetic-rule-results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")



