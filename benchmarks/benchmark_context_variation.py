"""Compare Neuron's intermediate state under context variations.

This benchmark is intentionally not an answer-accuracy benchmark. It records
seeds, relation interpretations, activation by hop, paths, and evidence signals
for multiple phrasings over the same graph. It highlights which components
change with context and which remain invariant in the current architecture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuron_core import NeuronGraph, trace_processing  # noqa: E402


def build_world() -> tuple[NeuronGraph, dict[str, str]]:
    graph = NeuronGraph()
    ids: dict[str, str] = {}

    def node(name: str, kind: str) -> str:
        if name not in ids:
            ids[name] = graph.add_concept(name, kind=kind)
        return ids[name]

    def fact(source: str, relation: str, target: str, *, source_kind="concept",
             target_kind="concept", support=0, contradiction=0, note=""):
        edge = graph.connect(
            node(source, source_kind), node(target, target_kind), relation
        )
        for i in range(support):
            graph.add_edge_evidence(
                edge, positive=True, source=f"supporting-record-{i + 1}", note=note
            )
        for i in range(contradiction):
            graph.add_edge_evidence(
                edge, positive=False, source=f"contradicting-record-{i + 1}", note=note
            )
        return edge

    # Same facts; only the query wording changes.
    fact("Jordan Lee", "lives_in", "Cairo", source_kind="person", target_kind="city")
    fact("Jordan Lee", "works_at", "Cairo Robotics",
         source_kind="person", target_kind="organization")
    fact("Jordan Lee", "born_in", "Amman", source_kind="person", target_kind="city")

    # Configure a small, transparent lexicon so we can isolate context gating
    # separately from out-of-vocabulary paraphrase understanding.
    graph.add_relation_alias("lives_in", "live")
    graph.add_relation_alias("lives_in", "lives")
    graph.add_relation_alias("lives_in", "resides")
    graph.add_relation_alias("works_at", "office")
    graph.add_relation_alias("works_at", "work at")
    graph.add_relation_alias("works_at", "workplace")
    graph.add_relation_alias("born_in", "born")
    graph.add_relation_alias("born_in", "birthplace")

    variants = [
        {"id": "ambiguous", "query": "Where is Jordan Lee based?", "expected_relations": []},
        {"id": "residence", "query": "Where does Jordan Lee live?", "expected_relations": ["lives_in"]},
        {"id": "workplace", "query": "Which office does Jordan Lee work at?", "expected_relations": ["works_at"]},
        {"id": "birthplace", "query": "Where was Jordan Lee born?", "expected_relations": ["born_in"]},
    ]

    # Explicit contradictory evidence for trace inspection, not for a winner test.
    fact(
        "Alex Kim", "profession", "Senior Platform Engineer",
        source_kind="person", target_kind="profession", support=5,
        note="verified in a first-party directory",
    )
    fact(
        "Alex Kim", "profession", "Aardvark Consultant",
        source_kind="person", target_kind="profession", contradiction=4,
        note="disputed claim",
    )
    graph.add_relation_alias("profession", "job")
    variants.extend([
        {"id": "conflict", "query": "What is Alex Kim's job?", "expected_relations": ["profession"]},
    ])

    # A multi-hop chain; relation interpretation is intentionally not a plan.
    fact("Nia Patel", "directs", "Helix Research Lab",
         source_kind="person", target_kind="organization")
    fact("Helix Research Lab", "located_in", "Alexandria",
         source_kind="organization", target_kind="city")
    variants.append({
        "id": "multi_hop",
        "query": "Which city is home to the research center under Nia Patel's direction?",
        "expected_relations": [],
    })
    fact("Omar Selim", "employed_by", "Northstar Systems",
         source_kind="person", target_kind="organization")
    fact("Northstar Systems", "maintains", "Atlas Runtime",
         source_kind="organization", target_kind="software")
    fact("Atlas Runtime", "written_in", "Rust",
         source_kind="software", target_kind="programming_language")
    variants.append({
        "id": "three_hop",
        "query": "Which programming language underpins the service at Omar Selim's employer?",
        "expected_relations": [],
    })
    return graph, {"variants": variants}


def names_from_trace(trace: dict) -> set[str]:
    names = set()
    for items in trace["activation_by_hop"].values():
        for item in items:
            names.add(item["name"])
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()
    graph, data = build_world()

    traces = []
    for variant in data["variants"]:
        started = time.perf_counter_ns()
        trace = trace_processing(
            graph, variant["query"], seed_limit=10, max_hops=3,
            max_nodes=100, workers=1,
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        relation_names = sorted({
            item["relation"] for item in trace["relation_hypotheses"]
        })
        active_names = names_from_trace(trace)
        traces.append({
            "id": variant["id"],
            "category_expected_as_probe": variant["expected_relations"],
            "elapsed_ms": round(elapsed_ms, 4),
            "trace": trace,
            "derived_observations": {
                "matched_relations": relation_names,
                "active_names": sorted(active_names),
                "active_count": len(active_names),
            },
        })

    by_id = {item["id"]: item for item in traces}
    comparisons = []
    for left, right in [
        ("ambiguous", "residence"),
        ("ambiguous", "workplace"),
        ("ambiguous", "birthplace"),
        ("residence", "workplace"),
        ("conflict", "multi_hop"),
        ("multi_hop", "three_hop"),
    ]:
        a, b = by_id[left], by_id[right]
        names_a = set(a["derived_observations"]["active_names"])
        names_b = set(b["derived_observations"]["active_names"])
        union = names_a | names_b
        jaccard = len(names_a & names_b) / len(union) if union else 1.0
        relations_a = set(a["derived_observations"]["matched_relations"])
        relations_b = set(b["derived_observations"]["matched_relations"])
        comparisons.append({
            "left": left,
            "right": right,
            "same_active_name_set": names_a == names_b,
            "active_name_jaccard": round(jaccard, 4),
            "left_relations": sorted(relations_a),
            "right_relations": sorted(relations_b),
            "relation_interpretations_changed": relations_a != relations_b,
        })

    result = {
        "benchmark": "neuron_context_variation_processing_trace",
        "purpose": (
            "Observe whether changing query context changes intermediate graph processing. "
            "This is not a human brain simulation and does not make or score final decisions."
        ),
        "graph_counts": {"neurons": len(graph.neurons), "synapses": len(graph.synapses)},
        "traces": traces,
        "context_comparisons": comparisons,
        "interpretation_notes": [
            "Activation is currently seeded mainly by lexical node matches and spreads generically along outgoing edges.",
            "Relation alias matching is observed separately; this benchmark checks whether it currently gates activation.",
            "Evidence confidence is emitted as state for inspection, not converted into a chosen answer.",
            "Configured relation aliases isolate the effect of known contextual cues and do not demonstrate general paraphrase understanding.",
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
