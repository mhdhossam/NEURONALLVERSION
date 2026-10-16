"""Measure first-attempt Neuron, one-feedback online learning, and retry.

No offline RL episodes are used. Each case is attempted before feedback; a verifier
then supplies one reward signal, and the same policy retries. The baseline is also
measured on the same query before any policy update. This is a synthetic diagnostic,
not a claim of general language understanding.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuron_core import NeuronGraph, ReconstructedNeuron  # noqa: E402
from benchmarks.benchmark_meaning_understanding import build_world  # noqa: E402


def names(rows):
    return [row["name"] for row in rows]


def score(case, rows):
    found = names(rows)
    expected = set(case["expected"])
    if case["mode"] == "abstain":
        return not found
    if case["mode"] == "all_targets":
        return expected.issubset(set(found))
    return bool(expected.intersection(found))


def top1(case, rows):
    found = names(rows)
    if case["mode"] == "abstain":
        return not found
    return bool(found) and found[0] in set(case["expected"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", default="")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    graph, cases = build_world()
    policy = ReconstructedNeuron(graph)
    details = []
    grouped = defaultdict(list)
    total_updates = 0

    for case in cases:
        # The first attempt happens before this case's feedback update.
        start = time.perf_counter_ns()
        baseline_rows = graph.search(case["query"], top_k=args.top_k, max_hops=3,
                                     workers=1, record_history=False)
        baseline_ms = (time.perf_counter_ns() - start) / 1e6

        start = time.perf_counter_ns()
        first_rows = policy.search(case["query"], top_k=args.top_k, max_hops=3)
        first_ms = (time.perf_counter_ns() - start) / 1e6
        first_ok = top1(case, first_rows)
        full_ok = score(case, first_rows)

        # One verifier signal after the first attempt. If a supported expected
        # target was found anywhere in the trace, reward that target's path.
        # Otherwise punish the selected wrong result. For abstention cases,
        # punish a false-positive result; an already-correct abstention has no
        # candidate path to update.
        feedback_id = None
        reward = None
        expected_rows = [
            row for row in policy.search(case["query"], top_k=max(args.top_k, 20), max_hops=3)
            if row["name"] in set(case["expected"])
        ] if case["mode"] != "abstain" else []
        if expected_rows:
            feedback_id = expected_rows[0]["neuron_id"]
            reward = 1.0
        elif first_rows:
            feedback_id = first_rows[0]["neuron_id"]
            reward = -1.0

        if feedback_id is not None:
            policy.observe(case["query"], feedback_id, reward)
            total_updates += 1

        # Retry after exactly one feedback event for this query (when available).
        start = time.perf_counter_ns()
        retry_rows = policy.search(case["query"], top_k=args.top_k, max_hops=3)
        retry_ms = (time.perf_counter_ns() - start) / 1e6
        retry_ok = top1(case, retry_rows)

        row = {
            "id": case["id"], "category": case["category"], "query": case["query"],
            "baseline_top1": top1(case, baseline_rows),
            "first_attempt_top1": first_ok,
            "first_attempt_pass": full_ok,
            "retry_top1": retry_ok,
            "retry_pass": score(case, retry_rows),
            "feedback_reward": reward,
            "feedback_applied": feedback_id is not None,
            "baseline_results": names(baseline_rows),
            "first_attempt_results": names(first_rows),
            "retry_results": names(retry_rows),
            "baseline_latency_ms": round(baseline_ms, 5),
            "first_attempt_latency_ms": round(first_ms, 5),
            "retry_latency_ms": round(retry_ms, 5),
        }
        details.append(row)
        grouped[case["category"]].append(row)

    def summary(rows):
        n = max(1, len(rows))
        return {
            "cases": len(rows),
            "top1_rate": round(sum(r["baseline_top1"] for r in rows) / n, 4),
            "first_attempt_top1_rate": round(sum(r["first_attempt_top1"] for r in rows) / n, 4),
            "retry_top1_rate": round(sum(r["retry_top1"] for r in rows) / n, 4),
            "first_attempt_pass_rate": round(sum(r["first_attempt_pass"] for r in rows) / n, 4),
            "retry_pass_rate": round(sum(r["retry_pass"] for r in rows) / n, 4),
            "mean_first_attempt_latency_ms": round(sum(r["first_attempt_latency_ms"] for r in rows) / n, 5),
            "mean_retry_latency_ms": round(sum(r["retry_latency_ms"] for r in rows) / n, 5),
        }

    # Baseline uses the baseline_top1 field, while the policy first attempt is
    # explicitly measured before the current case's reward update.
    overall = summary(details)
    by_category = {}
    for category, rows in sorted(grouped.items()):
        by_category[category] = summary(rows)

    report = {
        "benchmark": "neuron_online_learning_from_first_attempt",
        "offline_rl_episodes": 0,
        "feedback_updates": total_updates,
        "policy_state_after_online_run": policy.state_dict(),
        "overall": overall,
        "by_category": by_category,
        "details": details,
        "limitations": [
            "Synthetic, hand-labeled retrieval benchmark; not general language understanding.",
            "Verifier feedback uses benchmark labels after the first attempt; labels are never available to that attempt.",
            "The expected answer must be present in the explored trace for positive feedback to teach its path.",
            "The retry metric is same-query correction, not independent generalization; inspect later cases and held-out tests separately.",
            "Graph candidate discovery and stored facts are unchanged.",
        ],
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print("First-attempt online learning summary:", json.dumps({
        "overall": overall, "feedback_updates": total_updates,
        "offline_rl_episodes": 0, "by_category": by_category,
    }, sort_keys=True))
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
