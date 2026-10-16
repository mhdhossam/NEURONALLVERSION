"""Clean held-out query-transfer and regression test for Neuron genetic rankers.

No held-out paraphrase query or label is passed to either learner. Training feedback
is supplied only after a wrong first attempt on non-held-out cases. This is a
synthetic query-level transfer test, not evidence of unseen-fact reasoning.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuron_core.genetic import GeneticRanker
from neuron_core.genetic_rules import GeneticRuleRanker
from benchmarks.benchmark_meaning_understanding import build_world


HELD_OUT_CATEGORY = "heldout_paraphrase"


def is_correct(case, rows):
    names = [row["name"] for row in rows]
    if case["mode"] == "abstain":
        return not names
    return bool(names) and names[0] in set(case["expected"])


def retrieve(graph, case):
    return graph.search(
        case["query"], top_k=20, max_hops=3, workers=1, record_history=False
    )


def score_cases(cases, candidate_map, weights, rules):
    details = []
    for case in cases:
        candidates = candidate_map[case["id"]]
        w_rows = weights.rank(case["query"], candidates)[:5]
        r_rows = rules.rank(case["query"], candidates)[:5]
        details.append({
            "id": case["id"],
            "query": case["query"],
            "expected": list(case["expected"]),
            "weight_hit1": is_correct(case, w_rows),
            "rule_hit1": is_correct(case, r_rows),
            "weight_top1": w_rows[0]["name"] if w_rows else None,
            "rule_top1": r_rows[0]["name"] if r_rows else None,
            "candidate_coverage": (
                bool(set(case["expected"]) & {x["name"] for x in candidates})
                if case["mode"] != "abstain" else not candidates
            ),
        })
    return details


def rate(rows, key):
    return round(sum(bool(row[key]) for row in rows) / max(1, len(rows)), 4)


def main():
    graph, all_cases = build_world()
    heldout = [c for c in all_cases if c["category"] == HELD_OUT_CATEGORY]
    train = [c for c in all_cases if c["category"] != HELD_OUT_CATEGORY]

    # Retrieval results are cached by case ID for a fair before/after comparison.
    candidates = {case["id"]: retrieve(graph, case) for case in all_cases}
    weights = GeneticRanker(population_size=36, generations=16, seed=17)
    rules = GeneticRuleRanker(population_size=36, generations=16, seed=29)

    # Freeze a pre-learning snapshot of both held-out and regression queries.
    heldout_before = score_cases(heldout, candidates, weights, rules)
    regression_before = score_cases(train, candidates, weights, rules)

    feedback_log = []
    for case in train:
        pool = candidates[case["id"]]
        if case["mode"] == "abstain":
            continue
        expected = set(case["expected"])
        available = [row["name"] for row in pool if row["name"] in expected]
        if not available:
            continue

        w_first = weights.rank(case["query"], pool)[:5]
        r_first = rules.rank(case["query"], pool)[:5]
        w_wrong = not is_correct(case, w_first)
        r_wrong = not is_correct(case, r_first)
        # The expected labels are revealed only after each first attempt.
        if w_wrong:
            weights.learn_from_mistake(case["query"], pool, available)
        if r_wrong:
            rules.learn_from_mistake(case["query"], pool, available)
        if w_wrong or r_wrong:
            feedback_log.append({
                "training_case_id": case["id"],
                "weight_feedback": w_wrong,
                "rule_feedback": r_wrong,
                "target_available": True,
            })

    heldout_after = score_cases(heldout, candidates, weights, rules)
    regression_after = score_cases(train, candidates, weights, rules)

    before_by_id = {row["id"]: row for row in regression_before}
    after_by_id = {row["id"]: row for row in regression_after}
    regressions = {
        "weights": [
            case_id for case_id, before in before_by_id.items()
            if before["weight_hit1"] and not after_by_id[case_id]["weight_hit1"]
        ],
        "rules": [
            case_id for case_id, before in before_by_id.items()
            if before["rule_hit1"] and not after_by_id[case_id]["rule_hit1"]
        ],
    }

    report = {
        "benchmark": "neuron_clean_heldout_query_transfer_and_regression",
        "split": {
            "training_cases": len(train),
            "heldout_cases": len(heldout),
            "heldout_category": HELD_OUT_CATEGORY,
            "heldout_queries_or_labels_sent_to_learners": 0,
            "heldout_labels_used_for_evaluation_only": True,
        },
        "feedback_updates": {
            "weights": len(weights.examples),
            "rules": len(rules.examples),
            "cases_with_any_feedback": len(feedback_log),
            "log": feedback_log,
        },
        "heldout_transfer": {
            "cases": len(heldout),
            "candidate_coverage_rate": rate(heldout_before, "candidate_coverage"),
            "weights_hit1_before": rate(heldout_before, "weight_hit1"),
            "weights_hit1_after": rate(heldout_after, "weight_hit1"),
            "rules_hit1_before": rate(heldout_before, "rule_hit1"),
            "rules_hit1_after": rate(heldout_after, "rule_hit1"),
            "details_before": heldout_before,
            "details_after": heldout_after,
        },
        "regression_check": {
            "cases": len(train),
            "weights_hit1_before": rate(regression_before, "weight_hit1"),
            "weights_hit1_after": rate(regression_after, "weight_hit1"),
            "rules_hit1_before": rate(regression_before, "rule_hit1"),
            "rules_hit1_after": rate(regression_after, "rule_hit1"),
            "weight_regressions": regressions["weights"],
            "rule_regressions": regressions["rules"],
        },
        "limitations": [
            "Synthetic 31-case world; the held-out split tests unseen query wording, not unseen facts.",
            "The graph and candidate retriever already contain the underlying facts; learners only reorder retrieved candidates.",
            "If a target is absent from candidates, ranking cannot recover it.",
            "Only wrong first attempts with a retrieved expected candidate can trigger feedback.",
            "Small sample size means percentage changes are descriptive, not statistically conclusive.",
        ],
    }
    print(json.dumps({
        "benchmark": report["benchmark"],
        "split": report["split"],
        "feedback_updates": report["feedback_updates"],
        "heldout_transfer": {k: v for k, v in report["heldout_transfer"].items()
                             if not k.startswith("details")},
        "regression_check": report["regression_check"],
    }, indent=2, ensure_ascii=False))
    Path("neuron-genetic-transfer-results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
