"""Experimental graph-path reasoning for Neuron.

This module explores possible relational chains without selecting a final answer.
It is intentionally separate from NeuronGraph.search() so experiments cannot
silently change the established retrieval/decision behavior.
"""
from __future__ import annotations

from typing import Any, Iterable

from .engine import NeuronGraph


def explore_reasoning(
    graph: NeuronGraph,
    query: str,
    *,
    max_hops: int = 3,
    seed_limit: int = 10,
    max_paths: int = 500,
    damping: float = 0.75,
    min_strength: float = 0.01,
    allowed_relations: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Enumerate bounded, evidence-carrying graph paths from lexical query seeds.

    The result is an exploration trace, not an answer. Multiple paths to the
    same concept are retained; positive and contradictory edge evidence are
    reported side by side. No path is declared true or selected as the winner.

    Query seeding remains Neuron's current lexical candidate lookup. This
    experiment tests relational chain exploration, not unrestricted NLP.
    """
    if max_hops < 0:
        raise ValueError("max_hops must be >= 0")
    if seed_limit < 1:
        raise ValueError("seed_limit must be >= 1")
    if max_paths < 1:
        raise ValueError("max_paths must be >= 1")
    if not 0.0 <= damping <= 1.0:
        raise ValueError("damping must be between 0 and 1")
    if not 0.0 <= min_strength <= 1.0:
        raise ValueError("min_strength must be between 0 and 1")

    relation_filter = None if allowed_relations is None else set(allowed_relations)
    seeds = graph.find_candidates(query, top_k=seed_limit)
    seed_rows = [
        {
            "neuron_id": neuron_id,
            "name": graph.knowledge[graph.neurons[neuron_id].knowledge_id].name,
            "lexical_score": round(score, 8),
        }
        for neuron_id, score in seeds
    ]

    # A frontier entry is one distinct route. Do not collapse by endpoint:
    # alternative paths are exactly what this experiment is meant to expose.
    frontier: list[dict[str, Any]] = [
        {
            "start_id": neuron_id,
            "node_ids": [neuron_id],
            "edge_ids": [],
            "strength": max(0.25, score),
        }
        for neuron_id, score in seeds
    ]
    explored: list[dict[str, Any]] = []
    paths_truncated = False

    for depth in range(1, max_hops + 1):
        next_frontier: list[dict[str, Any]] = []
        for partial in frontier:
            current_id = partial["node_ids"][-1]
            outgoing_ids = graph._outgoing.get(current_id, ())
            for synapse_id in outgoing_ids:
                edge = graph.synapses[synapse_id]
                if relation_filter is not None and edge.relation not in relation_filter:
                    continue
                if edge.target_id in partial["node_ids"]:
                    continue

                strength = partial["strength"] * damping * edge.effective_strength
                if strength < min_strength:
                    continue
                if len(explored) >= max_paths:
                    paths_truncated = True
                    break

                node_ids = partial["node_ids"] + [edge.target_id]
                edge_ids = partial["edge_ids"] + [synapse_id]
                node_names = [
                    graph.knowledge[graph.neurons[nid].knowledge_id].name
                    for nid in node_ids
                ]
                edge_trace = []
                for path_edge_id in edge_ids:
                    path_edge = graph.synapses[path_edge_id]
                    edge_trace.append({
                        "synapse_id": path_edge.id,
                        "source": graph.knowledge[
                            graph.neurons[path_edge.source_id].knowledge_id
                        ].name,
                        "relation": path_edge.relation,
                        "target": graph.knowledge[
                            graph.neurons[path_edge.target_id].knowledge_id
                        ].name,
                        "weight": path_edge.weight,
                        "effective_strength": path_edge.effective_strength,
                        "support": path_edge.support,
                        "contradiction": path_edge.contradiction,
                        "confidence": path_edge.confidence,
                        "evidence": [
                            {
                                "positive": evidence.positive,
                                "source": evidence.source,
                                "strength": evidence.strength,
                                "note": evidence.note,
                            }
                            for evidence in path_edge.evidence
                        ],
                    })

                route = {
                    "start": node_names[0],
                    "end": node_names[-1],
                    "hops": depth,
                    "nodes": node_names,
                    "relations": [item["relation"] for item in edge_trace],
                    "edges": edge_trace,
                    "exploration_strength": round(strength, 8),
                }
                explored.append(route)
                next_frontier.append({
                    "start_id": partial["start_id"],
                    "node_ids": node_ids,
                    "edge_ids": edge_ids,
                    "strength": strength,
                })
            if paths_truncated:
                break
        frontier = next_frontier
        if paths_truncated or not frontier:
            break

    return {
        "query": query,
        "seed_candidates": seed_rows,
        "paths": explored,
        "paths_explored": len(explored),
        "max_hops": max_hops,
        "paths_truncated": paths_truncated,
        "relation_filter": sorted(relation_filter) if relation_filter is not None else None,
        "ordering": "breadth-first exploration order; not a correctness ranking",
        "decision": None,
        "note": (
            "Exploratory reasoning trace only. Distinct paths and conflicting "
            "evidence are preserved; no answer is selected. Query seeds are "
            "lexical candidates, not unrestricted semantic interpretations."
        ),
    }
