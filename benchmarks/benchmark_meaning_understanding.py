"""Meaning-preservation probe for the existing Neuron graph search.

This is a diagnostic benchmark, not a claim that retrieval equals understanding.
It scores target retrieval, ranking, abstention on unsupported/negative claims, and
whether contradictory candidates are kept visible. No architecture changes or
external model downloads are required.
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

from neuron_core import NeuronGraph  # noqa: E402


def build_world() -> tuple[NeuronGraph, list[dict]]:
    graph = NeuronGraph()
    ids: dict[str, str] = {}

    def node(name: str, kind: str = "concept") -> str:
        if name not in ids:
            ids[name] = graph.add_concept(name, kind=kind)
        return ids[name]

    def fact(subject: str, relation: str, obj: str,
             subject_kind: str = "concept", object_kind: str = "concept") -> None:
        graph.connect(node(subject, subject_kind), node(obj, object_kind), relation)

    # Small, explicit world. The benchmark intentionally does not teach every
    # test phrasing as an alias: held-out phrasing must stand on its own.
    fact("Mira", "pushes", "cart", "person", "object")
    fact("Mira", "pushes", "box", "person", "object")
    fact("cart", "hits", "crate", "object", "object")
    fact("crate", "breaks", "crate fragments", "object", "object")
    fact("Ali", "owns", "bicycle", "person", "object")
    fact("Ali", "owns", "scooter", "person", "object")
    fact("Sara", "lives_in", "Cairo", "person", "city")
    fact("Sara", "works_at", "Delta Clinic", "person", "organization")
    fact("Sara", "born_in", "Alexandria", "person", "city")
    fact("Jordan Lee", "lives_in", "Cairo", "person", "city")
    fact("Jordan Lee", "works_at", "Cairo Robotics", "person", "organization")
    fact("Jordan Lee", "born_in", "Amman", "person", "city")
    fact("Alex Kim", "profession", "Senior Platform Engineer", "person", "profession")
    fact("Alex Kim", "profession", "Aardvark Consultant", "person", "profession")

    # Aliases are deliberately limited to common forms; the benchmark reports
    # how far the existing explicit vocabulary carries the query.
    for relation, aliases in {
        "pushes": ["push", "pushed", "pushes"],
        "owns": ["own", "owned", "owns"],
        "lives_in": ["live", "lives", "resides", "reside"],
        "works_at": ["work at", "works at", "office"],
        "born_in": ["born", "birthplace"],
        "profession": ["job", "occupation"],
        "hits": ["hit", "strikes"],
    }.items():
        for alias in aliases:
            graph.add_relation_alias(relation, alias)

    cases = [
        # Direct / known paraphrase retrieval
        ("known_alias", "What does Mira push?", ["cart", "box"], "targets"),
        ("known_alias", "What did Mira push?", ["cart", "box"], "targets"),
        ("known_alias", "What does Ali own?", ["bicycle", "scooter"], "targets"),
        ("known_alias", "What is Ali's job?", ["Senior Platform Engineer", "Aardvark Consultant"], "targets"),
        ("known_alias", "Where does Sara live?", ["Cairo"], "targets"),
        ("known_alias", "Where was Sara born?", ["Alexandria"], "targets"),
        ("known_alias", "Which office does Sara work at?", ["Delta Clinic"], "targets"),
        ("known_alias", "Where does Jordan Lee reside?", ["Cairo"], "targets"),

        # Related meaning, but phrasing is not registered as an alias
        ("heldout_paraphrase", "What object did Mira propel?", ["cart", "box"], "targets"),
        ("heldout_paraphrase", "What vehicle belongs to Ali?", ["bicycle", "scooter"], "targets"),
        ("heldout_paraphrase", "Which city is Sara a resident of?", ["Cairo"], "targets"),
        ("heldout_paraphrase", "What is Sara's place of employment?", ["Delta Clinic"], "targets"),
        ("heldout_paraphrase", "What city is Jordan Lee originally from?", ["Amman"], "targets"),
        ("heldout_paraphrase", "What occupation does Alex Kim have?", ["Senior Platform Engineer", "Aardvark Consultant"], "targets"),

        # Negation must not be interpreted as an affirmative assertion.
        # The current API is retrieval-only; any returned target is a diagnostic
        # false-positive candidate, not a final natural-language answer.
        ("negation", "What did Mira not push?", [], "abstain"),
        ("negation", "Sara does not live in Cairo; where does she live?", [], "abstain"),
        ("negation", "Ali does not own a bicycle. Is it his?", [], "abstain"),
        ("negation", "Jordan Lee was not born in Amman. Where was he born?", [], "abstain"),

        # Role reversal: do not assert the inverse of a directed relation.
        ("role_reversal", "Did the cart push Mira?", [], "abstain"),
        ("role_reversal", "Does Cairo live in Sara?", [], "abstain"),
        ("role_reversal", "Does the bicycle own Ali?", [], "abstain"),
        ("role_reversal", "Does Delta Clinic work at Sara?", [], "abstain"),

        # Cross-language meaning probes. No Arabic aliases/nodes are added:
        # these test current capability rather than a preconfigured translation.
        ("cross_language", "أين تعيش سارة؟", ["Cairo"], "targets"),
        ("cross_language", "ماذا يملك علي؟", ["bicycle", "scooter"], "targets"),
        ("cross_language", "أين وُلدت سارة؟", ["Alexandria"], "targets"),
        ("cross_language", "ما وظيفة أليكس كيم؟", ["Senior Platform Engineer", "Aardvark Consultant"], "targets"),

        # No supporting facts: a safe retrieval layer should return no candidates.
        ("unsupported", "What caused Atlantis to disappear?", [], "abstain"),
        ("unsupported", "Which planet does Zorblax inhabit?", [], "abstain"),
        ("unsupported", "Why did Mira push the cart?", [], "abstain"),
        ("unsupported", "What is the password of the moon?", [], "abstain"),

        # Contradictory/competing facts: preserve both candidates rather than
        # silently presenting one as uniquely established.
        ("contradiction", "What is Alex Kim's job?", ["Senior Platform Engineer", "Aardvark Consultant"], "all_targets"),
    ]
    return graph, [
        {"id": f"case_{i + 1:02d}", "category": category, "query": query,
         "expected": expected, "mode": mode}
        for i, (category, query, expected, mode) in enumerate(cases)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")

    graph, cases = build_world()
    rows = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        start = time.perf_counter_ns()
        results = graph.search(
            case["query"], top_k=args.top_k, max_hops=3, workers=1,
            record_history=False,
        )
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        names = [item["name"] for item in results]
        expected = case["expected"]
        if case["mode"] == "abstain":
            passed = not results
            top1_correct = not results
            topk_correct = not results
        elif case["mode"] == "all_targets":
            present = [name for name in expected if name in names]
            passed = len(present) == len(expected)
            top1_correct = bool(names) and names[0] in expected
            topk_correct = passed
        else:
            present = [name for name in expected if name in names]
            passed = bool(present)
            top1_correct = bool(names) and names[0] in expected
            topk_correct = passed
        row = {
            "id": case["id"], "category": case["category"], "query": case["query"],
            "expected": expected, "mode": case["mode"], "passed": passed,
            "top1_correct": top1_correct, "topk_correct": topk_correct,
            "returned": names, "elapsed_ms": round(elapsed_ms, 4),
        }
        rows.append(row)
        grouped[case["category"]].append(row)

    category_summary = {}
    for category, items in sorted(grouped.items()):
        category_summary[category] = {
            "cases": len(items),
            "pass_rate": round(sum(x["passed"] for x in items) / len(items), 4),
            "top1_rate": round(sum(x["top1_correct"] for x in items) / len(items), 4),
            "topk_rate": round(sum(x["topk_correct"] for x in items) / len(items), 4),
            "mean_latency_ms": round(sum(x["elapsed_ms"] for x in items) / len(items), 4),
        }

    result = {
        "benchmark": "neuron_v4_meaning_preservation_diagnostic",
        "branch_intent": "Measure existing behavior before architecture changes",
        "cases": len(rows),
        "top_k": args.top_k,
        "graph_counts": {"neurons": len(graph.neurons), "synapses": len(graph.synapses)},
        "overall": {
            "pass_rate": round(sum(x["passed"] for x in rows) / len(rows), 4),
            "top1_rate": round(sum(x["top1_correct"] for x in rows) / len(rows), 4),
            "topk_rate": round(sum(x["topk_correct"] for x in rows) / len(rows), 4),
            "mean_latency_ms": round(sum(x["elapsed_ms"] for x in rows) / len(rows), 4),
        },
        "by_category": category_summary,
        "cases_detail": rows,
        "limitations": [
            "This measures retrieval behavior, not full natural-language answer generation.",
            "Expected labels are hand-authored for this small synthetic world.",
            "Arabic cases are deliberately not translated or aliased in the graph.",
            "An abstention case passes only when search returns no candidates; returning a candidate is counted as a false-positive retrieval.",
            "A high score on this small benchmark would not establish general understanding.",
        ],
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
